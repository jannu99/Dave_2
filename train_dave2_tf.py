#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train with tf.data (optimized streaming pipeline)
Author: Adapted from Tawn Kramer by ChatGPT
"""
from __future__ import print_function
import os
import sys
import fnmatch
import argparse
import random
import json

import numpy as np
import tensorflow as tf
from tensorflow import keras

# ============================================================
# Global seed for reproducibility
# ============================================================
GLOBAL_SEED = 42

def set_global_seed(seed=42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

set_global_seed(GLOBAL_SEED)

# ============================================================
# GPU setup
# ============================================================
gpus = tf.config.list_physical_devices("GPU")
for g in gpus:
    try:
        tf.config.experimental.set_memory_growth(g, True)
        print(f"✅ Memory growth enabled on {g.name}")
    except Exception as e:
        print(f"⚠️ Could not set memory growth on {g.name}: {e}")

tf.config.optimizer.set_experimental_options({"layout_optimizer": False})

# ============================================================
# Import project modules
# ============================================================
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

import models.conf as conf
from models.models_small_input import get_nvidia_model

# ============================================================
# Matplotlib setup (optional)
# ============================================================
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    do_plot = True
except:
    do_plot = False

# ============================================================
# Constants
# ============================================================
CROP_TOP = 204
CROP_BOTTOM = 35
TARGET_H, TARGET_W = 66, 200

JSON_SEARCH_DIRS = [
    "/media/davidejannussi/New Volume/fortua/dataset_out_front_nominal",
    "/media/davidejannussi/New Volume/fortua/dataset_out_front_recovery",
    "/media/davidejannussi/New Volume/fortua/dataset_out_front_recovery_turns",
]

# ============================================================
# Utility
# ============================================================
def shuffle(samples):
    s = samples[:]
    random.shuffle(s)
    return s

def get_files(filemask):
    filemask = os.path.expanduser(filemask)
    path, mask = os.path.split(filemask)
    matches = []
    for root, _, filenames in os.walk(path):
        for filename in fnmatch.filter(filenames, mask):
            matches.append(os.path.join(root, filename))
    return matches

def find_json_for_augmented_image(img_name):
    """Find the corresponding JSON for augmented image."""
    name_noext = os.path.splitext(img_name)[0]
    parts = name_noext.split("_")
    try:
        date_number = parts[-3]
        frame_number = parts[-1]
    except IndexError:
        return None

    json_name = f"record_{date_number}_frame_{frame_number}.json"
    for root in JSON_SEARCH_DIRS:
        json_path = os.path.join(root, json_name)
        if os.path.exists(json_path):
            return json_path
    return None

# ============================================================
# JSON label reader (called inside tf.py_function)
# ============================================================
def read_json_label(py_path):
    import os, json, numpy as np

    # Convert to Python string safely
    if isinstance(py_path, (bytes, bytearray)):
        py_path = py_path.decode("utf-8")
    elif hasattr(py_path, "numpy"):
        py_path = py_path.numpy().decode("utf-8")
    else:
        py_path = str(py_path)

    fname = os.path.basename(py_path)
    if "dataset_aug" in py_path:
        json_path = find_json_for_augmented_image(fname)
    else:
        parts = fname.split("_")
        if len(parts) < 3:
            return np.zeros((conf.num_outputs,), dtype=np.float32)
        date_number = parts[0]
        frame_number = parts[2].split(".")[0]
        json_path = os.path.join(
            os.path.dirname(py_path),
            f"record_{date_number}_frame_{frame_number}.json"
        )

    if not json_path or not os.path.exists(json_path):
        return np.zeros((conf.num_outputs,), dtype=np.float32)

    try:
        with open(json_path, "r") as f:
            data = json.load(f)
        steering = float(data["user/angle"])
        throttle = float(data["user/throttle"])
        if conf.num_outputs == 2:
            return np.array([steering, throttle], dtype=np.float32)
        else:
            return np.array([steering], dtype=np.float32)
    except Exception:
        return np.zeros((conf.num_outputs,), dtype=np.float32)

# ============================================================
# Parse image + label
# ============================================================
def parse_image_and_label(path):
    img_bytes = tf.io.read_file(path)
    img = tf.image.decode_png(img_bytes, channels=3)
    shape = tf.shape(img)
    h = shape[0]
    w = shape[1]

    # crop + resize
    img = tf.image.crop_to_bounding_box(img, CROP_TOP, 0, h - CROP_TOP - CROP_BOTTOM, w)
    img = tf.image.resize(img, [TARGET_H, TARGET_W])
    img = tf.cast(img, tf.float32)

    label = tf.py_function(read_json_label, [path], Tout=tf.float32)
    label.set_shape([conf.num_outputs])
    return img, label

# ============================================================
# Create tf.data pipelines
# ============================================================
def make_tf_datasets(train_samples, val_samples, batch_size):
    def flip_image(img, label):
        flipped = tf.image.flip_left_right(img)
        if conf.num_outputs == 2:
            flipped_label = tf.stack([-label[0], label[1]])
        else:
            flipped_label = tf.stack([-label[0]])
        return flipped, flipped_label

    # Train dataset
    train_ds = tf.data.Dataset.from_tensor_slices(train_samples)
    train_ds = train_ds.shuffle(buffer_size=len(train_samples), seed=GLOBAL_SEED)
    train_ds = train_ds.map(parse_image_and_label, num_parallel_calls=tf.data.AUTOTUNE)

    flipped_ds = train_ds.map(flip_image, num_parallel_calls=tf.data.AUTOTUNE)
    train_ds = train_ds.concatenate(flipped_ds)
    train_ds = train_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    # Validation dataset
    val_ds = tf.data.Dataset.from_tensor_slices(val_samples)
    val_ds = val_ds.map(parse_image_and_label, num_parallel_calls=tf.data.AUTOTUNE)
    val_ds = val_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return train_ds, val_ds

# ============================================================
# Dataset split
# ============================================================
def train_test_split(lines, test_perc=0.2, split_file=None, seed=GLOBAL_SEED):
    random.seed(seed)
    if split_file and os.path.exists(split_file):
        with open(split_file, "r") as f:
            data = json.load(f)
        old_train = set(data["train"])
        old_val = set(data["val"])
        known = old_train | old_val

        # Detect new samples not present before
        new_samples = [l for l in lines if l not in known]
        if new_samples:
            print(f"🆕 Found {len(new_samples)} new samples to split...")
            new_train, new_val = [], []
            for s in new_samples:
                (new_val if random.random() < test_perc else new_train).append(s)

            # Merge old + new
            train = list(old_train | set(new_train))
            val = list(old_val | set(new_val))
            print(f"✅ Updated split: {len(train)} train, {len(val)} val (including new data)")

            # Save updated file
            data = {"train": train, "val": val, "seed": seed}
            with open(split_file, "w") as f:
                json.dump(data, f, indent=2)
            print(f"💾 Saved updated split to {split_file}")
        else:
            print(f"✅ No new data, using existing split file: {split_file}")
            train, val = list(old_train), list(old_val)
    else:
        # Create new split from scratch
        train, val = [], []
        for line in lines:
            (val if random.random() < test_perc else train).append(line)
        if split_file:
            with open(split_file, "w") as f:
                json.dump({"train": train, "val": val, "seed": seed}, f, indent=2)
            print(f"💾 Created new split file: {split_file}")

    return train, val

# ============================================================
# Build sample lists
# ============================================================
def collect_samples(inputs, inputs2=None, inputs3=None, augmentation=None, limit=None):
    lines = get_files(inputs)
    print(f"Found {len(lines)} images in dataset 1")

    if inputs2:
        lines += get_files(inputs2)
        print(f"Added dataset 2 — total: {len(lines)}")

    if inputs3:
        lines += get_files(inputs3)
        print(f"Added dataset 3 — total: {len(lines)}")

    if limit:
        lines = lines[:limit]
        print(f"Limiting to {len(lines)} images")

    split_file = "dataset_split_combined.json"
    train_samples, val_samples = train_test_split(lines, 0.2, split_file, seed=GLOBAL_SEED)

    # Augmentation set
    if augmentation:
        aug_files = get_files(os.path.join(augmentation, "*.png"))
        if aug_files:
            aug_train, aug_val = train_test_split(aug_files, 0.2, None, seed=GLOBAL_SEED)
            train_samples.extend(aug_train)
            val_samples.extend(aug_val)
            print(f"Added {len(aug_files)} augmented images")

    print(f"Final: {len(train_samples)} train, {len(val_samples)} val")
    return train_samples, val_samples

# ============================================================
# Training
# ============================================================
def go(model_name, epochs=50, inputs="./log/*.png", limit=None, resume=False, augmentation=None):
    print(f"🚀 Training model: {model_name}")

    if resume and os.path.exists(model_name):
        print(f"Resuming from checkpoint: {model_name}")
        model = keras.models.load_model(model_name)
        model.compile(optimizer=keras.optimizers.Adam(1e-4), loss="mse", metrics=["mae"])
    else:
        print("Creating new model...")
        model = get_nvidia_model(conf.num_outputs)
        model.compile(optimizer=keras.optimizers.Adam(1e-4), loss="mse", metrics=["mae"])

    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=conf.training_patience),
        keras.callbacks.ModelCheckpoint(model_name, monitor="val_loss", save_best_only=True),
    ]

    bs = conf.training_batch_size
    train_samples, val_samples = collect_samples(inputs, args.inputs2, args.inputs3, augmentation, limit)
    train_ds, val_ds = make_tf_datasets(train_samples, val_samples, bs)

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        verbose=1,
        callbacks=callbacks,
    )

    if do_plot:
        plt.plot(history.history["loss"])
        plt.plot(history.history["val_loss"])
        plt.title("Model loss")
        plt.ylabel("Loss")
        plt.xlabel("Epoch")
        plt.legend(["train", "val"], loc="upper left")
        plt.savefig("loss.png")
        plt.show()

    print(f"✅ Finished training. Model saved as {model_name}")

# ============================================================
# Entry point
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train with tf.data streaming")
    parser.add_argument("--model", default="./checkpoints/trained_on_trained_on_nominal_recovery_turns_tuesday_morning.h5", type=str)
    parser.add_argument("--epochs", type=int, default=conf.training_default_epochs)
    parser.add_argument("--inputs", default="/media/davidejannussi/New Volume/fortua/dataset_out_front_nominal/*.png")
    parser.add_argument("--inputs2", default="/media/davidejannussi/New Volume/fortua/dataset_out_front_recovery/*.png")
    parser.add_argument("--inputs3", default="/media/davidejannussi/New Volume/fortua/dataset_out_front_recovery_turns/*.png")
    parser.add_argument("--augmentation", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    go(args.model, epochs=args.epochs, limit=args.limit, inputs=args.inputs,
       resume=args.resume, augmentation=args.augmentation)
