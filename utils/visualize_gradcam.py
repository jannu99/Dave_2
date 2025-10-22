#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GradCAM++ visualization for your DAVE-2 regression model
Optimized: loads model once, supports single image or folder.
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from tensorflow import keras
from tensorflow.keras.preprocessing import image
import tensorflow as tf
import glob
import os
from tf_keras_vis.gradcam_plus_plus import GradcamPlusPlus
from tf_keras_vis.utils.model_modifiers import ReplaceToLinear


# ============================================================
# Load image exactly as in training (raw 0–255)
# ============================================================
def load_and_preprocess(img_path, target_size=(503, 800)):
    # corretto, perché height=503, width=800
    img = image.load_img(img_path, target_size=target_size)
    x = image.img_to_array(img).astype(np.float32)
    x = np.expand_dims(x, axis=0)
    return x, np.array(img, dtype=np.uint8)


# ============================================================
# Find last conv layer
# ============================================================
def find_last_conv_layer(model):
    for i in reversed(range(len(model.layers))):
        if isinstance(model.layers[i], tf.keras.layers.Conv2D):
            return i
    raise ValueError("❌ No Conv2D layer found in model!")


# ============================================================
# Score function (for regression)
# ============================================================
def score(output):
    """Focus on steering neuron (index 0)."""
    return output[:, 0]


# ============================================================
# Single-image GradCAM++ (model already loaded)
# ============================================================
def visualize_gradcam(model, img_path, out_path=None, target_size=(503, 800)):
    print(f"🖼️ Processing image: {img_path}")
    x, orig_img = load_and_preprocess(img_path, target_size)

    replace2linear = ReplaceToLinear()
    gradcam = GradcamPlusPlus(model, model_modifier=replace2linear, clone=True)

    penultimate_layer = find_last_conv_layer(model)
    layer_name = model.layers[penultimate_layer].name

    cam = gradcam(score, x, penultimate_layer=penultimate_layer)
    heatmap = np.uint8(cm.jet(cam[0])[..., :3] * 255)
    overlay = np.uint8(0.5 * orig_img + 0.5 * heatmap)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(overlay)
    ax.axis("off")
    ax.set_title(f"GradCAM++ — layer: {layer_name}", fontsize=14)

    if out_path:
        plt.savefig(out_path, bbox_inches="tight")
        plt.close(fig)
        print(f"💾 Saved: {out_path}")
    else:
        plt.show()


# ============================================================
# Folder loop (loads model once)
# ============================================================
def visualize_folder(model_path, folder, out_dir="gradcam_out", target_size=(503, 800)):
    print(f"\n📂 Loading model once: {model_path}")
    model = keras.models.load_model(model_path, compile=False)

    os.makedirs(out_dir, exist_ok=True)
    image_paths = sorted(glob.glob(os.path.join(folder, "*.png")))
    print(f" Found {len(image_paths)} images in {folder}")

    for img_path in image_paths:
        filename = os.path.basename(img_path)
        out_path = os.path.join(out_dir, filename.replace(".png", "_gradcam.png"))
        try:
            visualize_gradcam(model, img_path, out_path, target_size)
        except Exception as e:
            print(f"⚠️ Skipping {filename} due to: {e}")


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GradCAM++ visualization for DAVE-2")
    parser.add_argument("--model", default="/home/davide/Desktop/Dave_2/trained_on_trained_on_nominal_recovery_turns_tuesday_morning.h5", help="Path to trained model (.h5)")
    parser.add_argument("--image", help="Path to a single image")
    parser.add_argument("--folder", help="Path to folder with images", default="/media/davidejannussi/New Volume/fortua/2025-10-02-09-10-20/images")
    parser.add_argument("--out", help="Output path for single image")
    parser.add_argument("--out_dir", default="gradcam_out", help="Output directory for batch")
    parser.add_argument("--width", type=int, default=800)
    parser.add_argument("--height", type=int, default=503)
    args = parser.parse_args()

    target_size = (args.height, args.width)

    if args.folder:
        visualize_folder(args.model, args.folder, args.out_dir, target_size)
    elif args.image:
        print(f"\n📂 Loading model: {args.model}")
        model = keras.models.load_model(args.model, compile=False)
        visualize_gradcam(model, args.image, args.out, target_size)
    else:
        print("❌ Please specify either --image or --folder")
