#!/usr/bin/env python3
import rospy
import csv
import matplotlib.pyplot as plt
from std_msgs.msg import Float64
import bisect
import numpy as np

# --- Config ---
REAL_TOPIC = "/vehicle/steering_pct"
SHADOW_TOPIC = "/cmd/steering_target"
UNCERT_TOPIC = "/cmd/steering_uncertainty"   # <-- nuovo topic per la std dev
OUTPUT_FILE = "commands_log.csv"
MAX_DT = 0.05   # massimo sfasamento accettato (50 ms)

# --- Storage ---
real_buf = []        # [(t, val), ...]
shadow_buf = []      # [(t, val), ...]
uncert_buf = []      # [(t, std), ...]
timestamps = []
real_cmds = []
shadow_cmds = []
shadow_uncert = []

def nearest_idx(ts_list, t_query):
    """Trova l’indice con timestamp più vicino a t_query."""
    if not ts_list:
        return None
    i = bisect.bisect_left(ts_list, t_query)
    cand = []
    if i < len(ts_list):
        cand.append(i)
    if i > 0:
        cand.append(i - 1)
    return min(cand, key=lambda k: abs(ts_list[k] - t_query)) if cand else None

def try_match():
    """Prova ad accoppiare real, shadow e uncertainty vicini nel tempo."""
    if not real_buf or not shadow_buf:
        return

    t_r, val_r = real_buf[-1]
    shadow_ts = [tt for tt, _ in shadow_buf]
    k = nearest_idx(shadow_ts, t_r)
    if k is None:
        return
    t_s, val_s = shadow_buf[k]
    if abs(t_s - t_r) > MAX_DT:
        return

    # Cerca anche incertezza più vicina (se disponibile)
    uncert_val = None
    if uncert_buf:
        uncert_ts = [tt for tt, _ in uncert_buf]
        ku = nearest_idx(uncert_ts, t_s)
        if ku is not None and abs(uncert_ts[ku] - t_s) <= MAX_DT:
            uncert_val = uncert_buf[ku][1]

    timestamps.append(t_r)
    real_cmds.append(val_r)
    shadow_cmds.append(val_s)
    shadow_uncert.append(uncert_val if uncert_val is not None else 0.0)

def real_cb(msg):
    t = rospy.Time.now().to_sec()
    val = msg.data
    real_buf.append((t, val))
    try_match()

def shadow_cb(msg):
    t = rospy.Time.now().to_sec()
    val = msg.data
    shadow_buf.append((t, val))
    try_match()

def uncert_cb(msg):
    t = rospy.Time.now().to_sec()
    val = msg.data
    uncert_buf.append((t, val))

def save_and_plot():
    # --- CSV ---
    with open(OUTPUT_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["time", "real", "shadow", "uncertainty"])
        for t, r, s, u in zip(timestamps, real_cmds, shadow_cmds, shadow_uncert):
            writer.writerow([t, r, s, u])
    print(f"Dati salvati in {OUTPUT_FILE}")

    # --- Metriche ---
    diffs = [abs(r - s) for r, s in zip(real_cmds, shadow_cmds)]
    mae = sum(diffs) / len(diffs) if diffs else 0.0
    print(f"Mean Absolute Error (MAE): {mae:.6f}")

    # --- Plot ---
    plt.figure(figsize=(10, 5))
    plt.plot(timestamps, real_cmds, label="Real Command", color="black")
    plt.plot(timestamps, shadow_cmds, label="Predicted Mean", color="orange")
    # aggiungi fascia ±1σ
    shadow_cmds_arr = np.array(shadow_cmds)
    shadow_uncert_arr = np.array(shadow_uncert)
    plt.fill_between(
        timestamps,
        shadow_cmds_arr - shadow_uncert_arr,
        shadow_cmds_arr + shadow_uncert_arr,
        color="orange",
        alpha=0.2,
        label="±1σ"
    )
    plt.xlabel("Time [s]")
    plt.ylabel("Steering command")
    plt.legend()
    plt.title("Real vs Predicted (with uncertainty)")
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    rospy.init_node("command_logger_nearest", anonymous=True)
    rospy.Subscriber(REAL_TOPIC, Float64, real_cb)
    rospy.Subscriber(SHADOW_TOPIC, Float64, shadow_cb)
    rospy.Subscriber(UNCERT_TOPIC, Float64, uncert_cb)  # <-- aggiunto

    print("Logging con nearest-neighbor matching + uncertainty... CTRL+C per fermare")

    try:
        rospy.spin()
    except KeyboardInterrupt:
        pass
    finally:
        save_and_plot()
