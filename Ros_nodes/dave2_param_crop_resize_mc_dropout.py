#!/usr/bin/env python3
import rospy
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import cv2
from PIL import Image
from sensor_msgs.msg import Image as SensorImage
from std_msgs.msg import Float64
from cv_bridge import CvBridge
import time
import argparse

# --- CONFIG ---
MODEL_PATH = "/home/davide/catkin_ws/src/ROS-small-scale-vehicle/mixed_reality/nodes/ads/trained_on_trained_on_nominal_recovery_turns_tuesday_morning.h5"
FIXED_THROTTLE = True
STEERING = 0
THROTTLE = 1
N_MC = 5           # Number of Monte Carlo dropout iterations
MAX_FPS = 20       # Inference rate limit (~50ms)

model = None
pub_steer = None
pub_uncert = None
bridge = CvBridge()

crop_top = 204
crop_bottom = 35
target_h, target_w = 66, 200



prev_time = None
last_steer = 0.0
alpha = 0.0
steer_cap = 10.0


# ============================================================
# Utility functions
# ============================================================
def smooth_steer(new_pred):
    """Exponential moving average smoothing."""
    global last_steer, alpha
    smoothed = alpha * last_steer + (1 - alpha) * new_pred
    last_steer = smoothed
    return smoothed


def mc_dropout_predict(model, image, n_iter=5):
    """Perform multiple stochastic forward passes with dropout active."""
    preds = []
    for _ in range(n_iter):
        y = model(image, training=True)  # dropout active during inference
        preds.append(y.numpy().squeeze())
    preds = np.array(preds)
    mean = preds.mean(axis=0)
    std = preds.std(axis=0)
    return mean, std


################# CROPPING NOT APPLIED ########################
def preprocess_image(img):
    """
    Crop e resize come nel training.
    Input:  OpenCV BGR image (H=503, W=800)
    Output: float32 array (1, 66, 200, 3)
    """
    # --- Converti BGR (da OpenCV) → RGB (come PIL) ---
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # --- Converte a PIL (come nel training) ---
    pil_img = Image.fromarray(img)

    # --- Applica lo stesso crop e resize ---
    width, height = pil_img.size
    pil_img = pil_img.crop((0, crop_top, width, height - crop_bottom))
    pil_img = pil_img.resize((target_w, target_h), Image.BILINEAR)

    # --- Converti in array numpy float32 ---
    x = np.asarray(pil_img, dtype=np.float32)  # ⚠️ niente /255.0, normalizza già il modello

    # --- Aggiungi batch dimension ---
    x = np.expand_dims(x, axis=0)

    return x


def parse_model_outputs(outputs):
    return [float(x) for x in outputs[0]]


# ============================================================
# ROS callback
# ============================================================
def new_image(msg):
    global prev_time, model, pub_steer, pub_uncert, steer_cap

    # --- Throttle inference rate ---
    if prev_time is None:
        prev_time = time.time()
    elif time.time() - prev_time < 1.0 / MAX_FPS:
        return
    prev_time = time.time()

    # --- Convert ROS → NumPy ---
    img = bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
    x = preprocess_image(img)

    # --- Monte Carlo Dropout Prediction ---
    try:
        mean_pred, std_pred = mc_dropout_predict(model, x, n_iter=N_MC)
    except Exception as e:
        rospy.logerr(f"Prediction error: {e}")
        return

    steering = mean_pred[STEERING]
    std_steer = std_pred[STEERING]
    throttle = 1.0 if FIXED_THROTTLE else mean_pred[THROTTLE] if len(mean_pred) > 1 else 1.0

    # --- Smooth & clip steering ---
    steering = smooth_steer(steering)
    steering = np.clip(steering, -steer_cap, steer_cap)

    # --- Publishers ---
    if pub_steer is None:
        pub_steer = rospy.Publisher("/cmd/steering_target", Float64, queue_size=1)
    if pub_uncert is None:
        pub_uncert = rospy.Publisher("/cmd/steering_uncertainty", Float64, queue_size=1)

    pub_steer.publish(steering)
    pub_uncert.publish(std_steer)

    rospy.loginfo(f"[MC-Dropout] steer={steering:.3f} ±{std_steer:.3f}, alpha={alpha:.2f}, cap={steer_cap}")


# ============================================================
# Main node
# ============================================================
def e2e_model():
    global model, alpha, steer_cap, N_MC

    parser = argparse.ArgumentParser(description="End-to-End model with MC Dropout")
    parser.add_argument("--model", type=str, default=MODEL_PATH, help="Path to .h5 model")
    parser.add_argument("--alpha", type=float, default=alpha, help="Smoothing factor (0–1)")
    parser.add_argument("--steer_cap", type=float, default=steer_cap, help="Steering cap")
    parser.add_argument("--n_mc", type=int, default=N_MC, help="Number of Monte Carlo passes")
    args, _ = parser.parse_known_args()

    N_MC = args.n_mc
    alpha = args.alpha
    steer_cap = args.steer_cap

    print("\n🧠 Parameters:")
    print(f"  • MODEL_PATH = {args.model}")
    print(f"  • alpha      = {alpha}")
    print(f"  • steer_cap  = {steer_cap}")
    print(f"  • N_MC       = {N_MC}\n")

    print("🚗 Starting e2e_model with MC Dropout...")
    print(f"Loading model: {args.model}")

    model = load_model(args.model, compile=False)
    model.compile(loss="mse", metrics=["mae"])

    rospy.init_node("e2e_model_mc_dropout", anonymous=True)
    rospy.Subscriber("/gmsl_camera/front_narrow/image_raw", SensorImage, new_image)
    rospy.spin()


if __name__ == "__main__":
    try:
        e2e_model()
    except rospy.ROSInterruptException:
        pass