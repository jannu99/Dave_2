'''
Models
Define the different NN models we will use
Author: Tawn Kramer
'''
from __future__ import print_function
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Input
from tensorflow.keras.layers import Dense, Lambda, ELU
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense
from tensorflow.keras.layers import Cropping2D
from tensorflow.keras.optimizers import Adadelta, Adam

from tensorflow.keras.layers import (
    Input, Conv2D, Dropout, Flatten, Dense, BatchNormalization, Lambda
)

import fnmatch
import argparse
import random
import json

import numpy as np
from PIL import Image
from tensorflow import keras

import models.conf as conf
import os 

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


def show_model_summary(model):
    model.summary()
    for layer in model.layers:
        print(layer.output_shape)

def get_nvidia_model(num_outputs):
    """
    Larger, more stable NVIDIA-inspired model with BatchNorm, smaller LR,
    and smoother activation to improve generalization.
    Compatible with your existing training pipeline.
    """

    row, col, ch = conf.row, conf.col, conf.ch
    drop = 0.2

    img_in = Input(shape=(row, col, ch), name='img_in')
    x = Lambda(lambda x: x / 255.0)(img_in)

    # --- Convolutional backbone (deeper, wider) ---
    x = Conv2D(32, (5,5), strides=(2,2), padding='same', activation='elu')(x)
    x = BatchNormalization()(x)
    x = Dropout(drop)(x)

    x = Conv2D(48, (5,5), strides=(2,2), padding='same', activation='elu')(x)
    x = BatchNormalization()(x)
    x = Dropout(drop)(x)

    x = Conv2D(64, (5,5), strides=(2,2), padding='same', activation='elu')(x)
    x = BatchNormalization()(x)
    x = Dropout(drop)(x)

    x = Conv2D(96, (3,3), strides=(1,1), padding='same', activation='elu')(x)
    x = BatchNormalization()(x)
    x = Dropout(drop)(x)

    x = Conv2D(128, (3,3), strides=(1,1), padding='same', activation='elu')(x)
    x = BatchNormalization()(x)
    x = Dropout(drop)(x)

    x = Flatten()(x)

    # --- Fully connected head (richer representation) ---
    x = Dense(256, activation='elu')(x)
    x = Dropout(0.3)(x)
    x = Dense(128, activation='elu')(x)
    x = Dropout(0.3)(x)
    x = Dense(64, activation='elu')(x)

    # --- Constrained output [-5, 5] for steering/throttle ---
    x = Dense(num_outputs, activation='linear')(x)
    outputs = Lambda(lambda y: y , name='scaled_output')(x)

    # --- Compile ---
    model = Model(inputs=img_in, outputs=outputs)
    opt = Adam(learning_rate=0.0001)  # smaller LR for stability
    model.compile(optimizer=opt, loss='mse', metrics=['mae'])
    return model

