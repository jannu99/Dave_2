"""
Train
Train your nerual network
Author: Tawn Kramer
"""
from __future__ import print_function
import os
import sys
import fnmatch
import argparse
import random
from glob import glob
import json

import numpy as np
from PIL import Image
from tensorflow import keras

import tensorflow as tf

tf.config.optimizer.set_experimental_options({
    'layout_optimizer': False
})


# Add the root directory of your project to sys.path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

# Now you can use absolute imports
import models.conf as conf
from models.models_small_input import get_nvidia_model

"""
matplotlib can be a pain to setup. So handle the case where it is absent. When present,
use it to generate a plot of training results.
"""
try:
    import matplotlib

    # Force matplotlib to not use any Xwindows backend.
    matplotlib.use("Agg")

    import matplotlib.pyplot as plt

    do_plot = True
except:
    do_plot = False



CROP_TOP = 204
CROP_BOTTOM = 35
TARGET_H, TARGET_W = 66, 200

def shuffle(samples):
    """
    Shuffle a list and return a new shuffled list without modifying the original.
    """
    shuffled = samples[:]  # make a copy
    random.shuffle(shuffled)
    return shuffled



def load_json(filename):
    with open(filename, "rt") as fp:
        data = json.load(fp)
    return data


def generator(samples, batch_size=32):
    num_samples = len(samples)
    shown_preview = False  # show only once

    while True:
        samples = shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            controls = []

            for fullpath in batch_samples:
                try:
                    date_number = os.path.basename(fullpath).split("_")[0]
                    frame_number = os.path.basename(fullpath).split("_")[2].split(".")[0]
                    json_filename = os.path.join(
                        os.path.dirname(fullpath),
                        f"record_{date_number}_frame_{frame_number}.json"
                    )

                    if not os.path.exists(json_filename):
                        continue
                    data = load_json(json_filename)
                    steering = float(data["user/angle"])
                    throttle = float(data["user/throttle"])

                    # --- Load and crop image using PIL ---
                    img = Image.open(fullpath)
                    width, height = img.size
                    img = img.crop((0, CROP_TOP, width, height - CROP_BOTTOM))  # crop
                    img = img.resize((TARGET_W, TARGET_H))  # resize
                    img = np.asarray(img, dtype=np.float32) / 255.0  # normalize


                    images.append(img)
                    if conf.num_outputs == 2:
                        controls.append([steering, throttle])
                    else:
                        controls.append([steering])

                    # --- Flipped augmentation ---
                    flipped = np.flip(img, axis=1)
                    images.append(flipped)
                    if conf.num_outputs == 2:
                        controls.append([-steering, throttle])
                    else:
                        controls.append([-steering])

                except Exception as e:
                    print("skipping", fullpath, "due to", e)
                    continue

            if not images:
                continue

            X_train = np.array(images, dtype=np.float32)
            y_train = np.array(controls, dtype=np.float32)
            yield X_train, y_train



def get_files(filemask):
    """
    use a filemask and search a path recursively for matches
    """
    # matches = glob.glob(os.path.expanduser(filemask))
    # return matches
    filemask = os.path.expanduser(filemask)
    path, mask = os.path.split(filemask)

    matches = []
    for root, dirnames, filenames in os.walk(path):
        for filename in fnmatch.filter(filenames, mask):
            matches.append(os.path.join(root, filename))
    return matches


def train_test_split(lines, test_perc):
    """
    split a list into two parts, percentage of test used to seperate
    """
    train = []
    test = []

    for line in lines:
        if random.uniform(0.0, 1.0) < test_perc:
            test.append(line)
        else:
            train.append(line)

    return train, test


def make_generators(inputs, inputs2=None, limit=None, batch_size=64):
    """
    Load the job spec from the csv and create some generator for training.
    Supports combining two datasets (inputs + inputs2).
    """

    # get the image/steering pairs from the first folder
    lines = get_files(inputs)
    print("Found %d files in first dataset" % len(lines))

    if inputs2:
        lines2 = get_files(inputs2)
        print("Found %d files in second dataset" % len(lines2))
        lines.extend(lines2)

    print("Total combined files: %d" % len(lines))

    if limit is not None:
        lines = lines[:limit]
        print("Limiting to %d files" % len(lines))

    # now split for validation AFTER combining both datasets
    train_samples, validation_samples = train_test_split(lines, test_perc=0.2)

    print("num train/val", len(train_samples), len(validation_samples))

    train_generator = generator(train_samples, batch_size=batch_size)
    validation_generator = generator(validation_samples, batch_size=batch_size)

    n_train = len(train_samples) * 2
    n_val = len(validation_samples) * 2

    return train_generator, validation_generator, n_train, n_val



def go(model_name, epochs=50, inputs="./log/*.*", limit=None, resume=False):
    print("working on model", model_name)

    """
    modify config.json to select the model to train.
    """
    if resume and os.path.exists(model_name):
        print(f"Resuming training from checkpoint: {model_name}")
        model = keras.models.load_model(model_name, compile=False)
        # ricompila con lo stesso optimizer e loss
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0001),
                      loss="mse", metrics=["mae"])
    else:
        print("Creating new model...")
        model = get_nvidia_model(conf.num_outputs)

    

    """
    display layer summary and weights info
    """
    # show_model_summary(model)

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=conf.training_patience, verbose=0
        ),
        keras.callbacks.ModelCheckpoint(
            model_name, monitor="val_loss", save_best_only=True, verbose=0
        ),
    ]

    batch_size = conf.training_batch_size

    # Train on session images
    train_generator, validation_generator, n_train, n_val = make_generators(
        inputs, inputs2=args.inputs2, limit=limit, batch_size=batch_size
    )
    if n_train == 0:
        print("no training data found")
        return

    steps_per_epoch = n_train // batch_size
    validation_steps = n_val // batch_size

    print("steps_per_epoch", steps_per_epoch, "validation_steps", validation_steps)

    history = model.fit(
        train_generator,
        steps_per_epoch=steps_per_epoch,
        validation_data=validation_generator,
        validation_steps=validation_steps,
        epochs=epochs,
        verbose=1,
        callbacks=callbacks,
    )

    try:
        if do_plot:
            # summarize history for loss
            plt.plot(history.history["loss"])
            plt.plot(history.history["val_loss"])
            plt.title("model loss")
            plt.ylabel("loss")
            plt.xlabel("epoch")
            plt.legend(["train", "test"], loc="upper left")
            plt.savefig("loss.png")
            plt.show()
    except:
        print("problems with loss graph")

    print(f"Finished training. Saved model {model_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="train script")
    parser.add_argument("--model",default="./checkpoints/trained_on_real.h5", type=str, help="model name")
    parser.add_argument(
        "--epochs",
        type=int,
        default=conf.training_default_epochs,
        help="number of epochs",
    )
    parser.add_argument(
        "--inputs", default="/media/davidejannussi/New Volume/fortua/dataset_out_front_nominal/*.png", help="input mask to gather images"
    )

    parser.add_argument(
        "--inputs2", default="/media/davidejannussi/New Volume/fortua/dataset_out_front_recovery/*.png",
        help="second input mask to gather images (e.g., recovery dataset)"
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="max number of images to train with"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="resume training from existing checkpoint if available",
    )
    args = parser.parse_args()

    go(args.model, epochs=args.epochs, limit=args.limit, inputs=args.inputs, resume=args.resume)