#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train
Train your neural network
Author: Tawn Kramer (modified for deterministic splits + JSON logging + augmentation split)
"""
from __future__ import print_function
import os
import sys
import fnmatch
import argparse
import random
import json
from glob import glob

import numpy as np
from PIL import Image
from tensorflow import keras
import tensorflow as tf

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
# GPU memory growth: evita freeze e frammentazione VRAM
# ============================================================
gpus = tf.config.list_physical_devices('GPU')
for g in gpus:
    try:
        tf.config.experimental.set_memory_growth(g, True)
        print(f"✅ Memory growth attivo su {g.name}")
    except Exception as e:
        print(f"⚠️ Impossibile impostare memory growth su {g.name}: {e}")


tf.config.optimizer.set_experimental_options({
    'layout_optimizer': False
})

# ============================================================
# Import project modules
# ============================================================
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

import models.conf as conf
from models.models_small_input import get_nvidia_model

# ============================================================
# Matplotlib setup
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

# Directory dove trovare i JSON originali (per immagini augmentate)
JSON_SEARCH_DIRS = [
    "/media/davidejannussi/New Volume/fortua/dataset_out_front_nominal",
    "/media/davidejannussi/New Volume/fortua/dataset_out_front_recovery",
    "/media/davidejannussi/New Volume/fortua/dataset_out_front_recovery_turns",
]

# ============================================================
# Utility
# ============================================================
def shuffle(samples):
    shuffled = samples[:]
    random.shuffle(shuffled)
    return shuffled

def load_json(filename):
    with open(filename, "rt") as fp:
        return json.load(fp)

def get_files(filemask):
    """Recursively find all matching files."""
    filemask = os.path.expanduser(filemask)
    path, mask = os.path.split(filemask)
    matches = []
    for root, _, filenames in os.walk(path):
        for filename in fnmatch.filter(filenames, mask):
            matches.append(os.path.join(root, filename))
    return matches

def find_json_for_augmented_image(img_name):
    """
    Trova il JSON corretto per un'immagine augmentata.
    Gestisce nomi del tipo 'canny_edges_mapping_2025-10-02-09-10-20_frame_001352.png'
    """
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
# Data generator
# ============================================================
def generator(samples, batch_size=32):
    num_samples = len(samples)
    while True:
        samples = shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset + batch_size]
            images, controls = [], []

            for fullpath in batch_samples:
                try:
                    fname = os.path.basename(fullpath)

                    # --- cerca JSON ---
                    if "dataset_aug" in fullpath:
                        json_filename = find_json_for_augmented_image(fname)
                    else:
                        date_number = fname.split("_")[0]
                        frame_number = fname.split("_")[2].split(".")[0]
                        json_filename = os.path.join(
                            os.path.dirname(fullpath),
                            f"record_{date_number}_frame_{frame_number}.json"
                        )
                    if not json_filename or not os.path.exists(json_filename):
                        continue

                    data = load_json(json_filename)
                    steering = float(data["user/angle"])
                    throttle = float(data["user/throttle"])

                    with Image.open(fullpath) as im:
                        im = im.convert("RGB")       # forza sempre 3 canali
                        width, height = im.size
                        im = im.crop((0, CROP_TOP, width, height - CROP_BOTTOM))
                        im = im.resize((TARGET_W, TARGET_H))
                        img = np.asarray(im, dtype=np.float32)

                    images.append(img)
                    if conf.num_outputs == 2:
                        controls.append([steering, throttle])
                    else:
                        controls.append([steering])

                    # Flipped version
                    flipped = np.flip(img, axis=1).copy()
                    images.append(flipped)
                    if conf.num_outputs == 2:
                        controls.append([-steering, throttle])
                    else:
                        controls.append([-steering])

                except Exception as e:
                    print("Skipping", fullpath, "due to", e)
                    continue

            if not images:
                continue

            batch_x = np.array(images, dtype=np.float32)
            batch_y = np.array(controls, dtype=np.float32)

            # ✅ Rilascio memoria delle liste temporanee
            del images, controls
            yield batch_x, batch_y

# ============================================================
# Deterministic split
# ============================================================
def train_test_split(lines, test_perc, split_file=None, seed=GLOBAL_SEED):
    random.seed(seed)
    if split_file and os.path.exists(split_file):
        with open(split_file, "r") as f:
            data = json.load(f)
        if "seed" in data and data["seed"] == seed:
            print(f"✅ Using existing split file: {split_file}")
            return data["train"], data["val"]
        else:
            print(f"⚠️ Seed changed — regenerating split using seed {seed}")

    train, val = [], []
    for line in lines:
        (val if random.uniform(0, 1) < test_perc else train).append(line)

    if split_file:
        with open(split_file, "w") as f:
            json.dump({"train": train, "val": val, "seed": seed}, f, indent=2)
        print(f"💾 Saved new split (seed {seed}) to {split_file}")

    return train, val

# ============================================================
# Generator builder with augment split
# ============================================================
def make_generators(inputs, inputs2=None, inputs3=None, augmentation=None, limit=None, batch_size=64):
    """Build generators, preserving previous split and adding augmentation data separately."""
    # --- Carica dataset base ---
    lines = get_files(inputs)
    print(f"Found {len(lines)} files in first dataset")

    if inputs2:
        lines += get_files(inputs2)
        print(f"Added second dataset: now {len(lines)} files")

    if inputs3:
        lines += get_files(inputs3)
        print(f"Added third dataset: now {len(lines)} files")

    if limit:
        lines = lines[:limit]
        print(f"Limiting to {len(lines)} files")

    # --- Split base ---
    split_file = "dataset_split_combined.json"
    if os.path.exists(split_file):
        with open(split_file, "r") as f:
            base_split = json.load(f)
        train_samples = base_split["train"]
        val_samples = base_split["val"]
        print(f"✅ Loaded base split ({len(train_samples)} train, {len(val_samples)} val)")
    else:
        print("⚠️ No base split found — generating new one")
        train_samples, val_samples = train_test_split(lines, 0.2, split_file, seed=GLOBAL_SEED)

    # --- Aggiungi dati augmentati ---
    if augmentation:
        aug_files = get_files(os.path.join(augmentation, "*.png"))
        print(f"Found {len(aug_files)} augmented images")

        if aug_files:
            aug_train, aug_val = train_test_split(aug_files, 0.2, None, seed=GLOBAL_SEED)
            print(f"Aug split: {len(aug_train)} train, {len(aug_val)} val")

            train_samples.extend(aug_train)
            val_samples.extend(aug_val)

            # salva nuova combinazione
            new_split_file = "dataset_split_combined_augmented.json"
            with open(new_split_file, "w") as f:
                json.dump({
                    "train": train_samples,
                    "val": val_samples,
                    "seed": GLOBAL_SEED,
                    "augmented_added": len(aug_files)
                }, f, indent=2)
            print(f"💾 Saved augmented split: {new_split_file}")

    print(f"Final train={len(train_samples)}, val={len(val_samples)}")

    return (
        generator(train_samples, batch_size),
        generator(val_samples, batch_size),
        len(train_samples) * 2,
        len(val_samples) * 2,
    )

# ============================================================
# Training
# ============================================================
def go(model_name, epochs=50, inputs="./log/*.*", limit=None, resume=False, augmentation=None):
    print("Working on model", model_name)

    if resume and os.path.exists(model_name):
        print(f"Resuming from checkpoint: {model_name}")
        model = keras.models.load_model(model_name)
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0001), loss="mse", metrics=["mae"])
    else:
        print("Creating new model...")
        model = get_nvidia_model(conf.num_outputs)

    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=conf.training_patience),
        keras.callbacks.ModelCheckpoint(model_name, monitor="val_loss", save_best_only=True),
    ]

    bs = conf.training_batch_size
    train_gen, val_gen, n_train, n_val = make_generators(
        inputs, args.inputs2, args.inputs3, augmentation, limit, bs
    )

    if n_train == 0:
        print("❌ No training data found")
        return

    print(f"steps_per_epoch={n_train // bs}, validation_steps={n_val // bs}")

    history = model.fit(
        train_gen,
        steps_per_epoch=n_train // bs,
        validation_data=val_gen,
        validation_steps=n_val // bs,
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

    print(f"✅ Finished training. Saved model {model_name}")

# ============================================================
# Entry point
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train with deterministic split + optional augmentation")
    parser.add_argument("--model", default="./checkpoints/trained_on_real.h5", type=str)
    parser.add_argument("--epochs", type=int, default=conf.training_default_epochs)
    parser.add_argument("--inputs", default="/media/davidejannussi/New Volume/fortua/dataset_out_front_nominal/*.png")
    parser.add_argument("--inputs2", default="/media/davidejannussi/New Volume/fortua/dataset_out_front_recovery/*.png")
    parser.add_argument("--inputs3", default="/media/davidejannussi/New Volume/fortua/dataset_out_front_recovery_turns/*.png")
    parser.add_argument("--augmentation", default=None, help="augmentation dataset folder")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    go(args.model, epochs=args.epochs, limit=args.limit, inputs=args.inputs, resume=args.resume, augmentation=args.augmentation)
