#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analisi distribuzione angoli di sterzo
--------------------------------------
Legge i file .png nei tre dataset e i relativi JSON.
Stampa min, max e genera istogramma della distribuzione.
Gestisce anche dataset_aug, i cui JSON sono nelle altre cartelle.
"""

import os
import fnmatch
import json
import numpy as np
import matplotlib.pyplot as plt

# === Percorsi dataset ===
DATASETS = [
    "/media/davidejannussi/New Volume/fortua/dataset_out_front_nominal",
    "/media/davidejannussi/New Volume/fortua/dataset_out_front_recovery",
    "/media/davidejannussi/New Volume/fortua/dataset_out_front_recovery_turns",
    #"/media/davidejannussi/New Volume/fortua/dataset_aug",
]

# Directory in cui cercare i JSON originali (per dataset_aug)
JSON_SEARCH_DIRS = DATASETS[:3]  # le prime tre


# ============================================================
# Utility
# ============================================================
def get_files(path, ext=".png"):
    """Ritorna lista di file con estensione specifica."""
    matches = []
    for root, _, filenames in os.walk(path):
        for filename in filenames:
            if filename.endswith(ext):
                matches.append(os.path.join(root, filename))
    return matches


def load_json(filename):
    with open(filename, "rt") as fp:
        return json.load(fp)

def find_json_for_augmented_image(img_name):
    """
    Trova il file JSON corrispondente a un'immagine augmentata.
    Gestisce nomi del tipo 'canny_edges_mapping_2025-10-02-09-10-20_frame_001352.png'
    estraendo la data e il frame number a partire dalla fine del nome.
    """
    name_noext = os.path.splitext(img_name)[0]  # rimuove .png
    parts = name_noext.split("_")

    # dalla fine: [-3] → data completa, [-1] → numero frame
    try:
        date_number = parts[-3]
        frame_number = parts[-1]
    except IndexError:
        return None

    json_name = f"record_{date_number}_frame_{frame_number}.json"
    
    # cerca nei dataset originali
    for root in JSON_SEARCH_DIRS:
        json_path = os.path.join(root, json_name)
        if os.path.exists(json_path):
            return json_path

    return None

def extract_steering_from_dataset(dataset_path):
    """Estrae tutti i valori di steering da un dataset."""
    files = get_files(dataset_path)
    print(f"Trovati {len(files)} file in {dataset_path}")
    steer_vals = []

    for fullpath in files:
        try:
            fname = os.path.basename(fullpath)

            if dataset_path=="/media/davidejannussi/New Volume/fortua/dataset_aug":  # dataset_aug
                json_filename = find_json_for_augmented_image(fname)
            else:
                base_name = os.path.splitext(fname)[0]
                json_filename = os.path.join(
                    os.path.dirname(fullpath),
                    f"record_{base_name}.json"
                )

            if not json_filename or not os.path.exists(json_filename):
                continue

            data = load_json(json_filename)
            steer = float(data["user/angle"])
            steer_vals.append(steer)

        except Exception as e:
            print("⚠️ Skipping", fullpath, ":", e)
            continue

    return np.array(steer_vals, dtype=np.float32)


# ============================================================
# MAIN
# ============================================================
def main():
    all_steering = []

    for dataset in DATASETS:
        vals = extract_steering_from_dataset(dataset)
        if vals.size > 0:
            print(f"→ {dataset}")
            print(f"   Min: {vals.min():.4f}, Max: {vals.max():.4f}, Mean: {vals.mean():.4f}")
            all_steering.extend(vals.tolist())

    if len(all_steering) == 0:
        print("❌ Nessun dato trovato. Controlla i percorsi e i JSON.")
        return

    all_steering = np.array(all_steering)
    print("\n=== STATISTICHE GLOBALI ===")
    print(f"Totale campioni: {len(all_steering)}")
    print(f"Min globale: {all_steering.min():.4f}")
    print(f"Max globale: {all_steering.max():.4f}")
    print(f"Media globale: {all_steering.mean():.4f}")

    # Plot distribuzione
    plt.figure(figsize=(8,5))
    plt.hist(all_steering, bins=100, color='steelblue', edgecolor='black')
    plt.title("Distribuzione globale angoli di sterzo")
    plt.xlabel("Steering (user/angle)")
    plt.ylabel("Frequenza")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
