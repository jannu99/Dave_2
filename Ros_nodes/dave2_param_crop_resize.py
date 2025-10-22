#!/usr/bin/env python3
import rospy
import tensorflow as tf
from tensorflow import keras
from keras.models import load_model
import numpy as np
import cv2
from PIL import Image
from sensor_msgs.msg import Image as SensorImage
from std_msgs.msg import Float64
from cv_bridge import CvBridge
import time
import argparse





# --- Globals ---
#### have to specify right parameters
MODEL_PATH = "/home/davide/catkin_ws/src/ROS-small-scale-vehicle/mixed_reality/nodes/ads/trained_on_trained_on_nominal_recovery_turns_tuesday_morning.h5"
FIXED_THROTTLE = True
STEERING = 0
THROTTLE = 1

model = None
pub_throttle_steering = None
bridge = CvBridge()

prev_time = None
last_steer = 0.0
alpha = 0.90        # smoothing factor
steer_cap = 2.0     # steering cap

# --- TensorFlow preprocessing ops (matching training pipeline) ---

crop_top = 204
crop_bottom = 35
target_h, target_w = 66, 200


def smooth_steer(new_pred):
    global last_steer, alpha
    smoothed = alpha * last_steer + (1 - alpha) * new_pred
    last_steer = smoothed
    return smoothed


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
    res = []
    for i in range(outputs.shape[1]):
        res.append(outputs[0][i])
    return res


def new_image(msg):
    global prev_time, model, pub_throttle_steering, alpha, steer_cap

    # --- Throttling to avoid overload ---
    if prev_time is None:
        prev_time = time.time()
    elif time.time() - prev_time < 0.05:
        return
    prev_time = time.time()

    # --- Convert ROS → OpenCV ---
    img = bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

    # --- Preprocess to match training ---
    x = preprocess_image(img)

    # --- Prediction ---
    try:
        if isinstance(model.input, list) and len(model.input) == 2:
            speed = np.array([[0.0]], dtype=np.float32)
            outputs = model.predict([x, speed], verbose=0)
        else:
            outputs = model.predict(x, verbose=0)
    except Exception as e:
        rospy.logerr(f"Prediction error: {e}")
        return

    # --- Parse outputs ---
    parsed = parse_model_outputs(outputs)
    steering = parsed[STEERING] if len(parsed) > 0 else 0.0
    throttle = parsed[THROTTLE] if len(parsed) > 1 else 0.0

    if FIXED_THROTTLE:
        throttle = 1.0

    # --- Smooth & clip steering ---
    steering = smooth_steer(steering)
    steering = np.clip(steering, -steer_cap, steer_cap)

    rospy.loginfo(f"[MODEL] steering={steering:.3f}, alpha={alpha:.2f}, cap={steer_cap}")

    # --- Publish steering ---
    if pub_throttle_steering is None:
        pub_throttle_steering = rospy.Publisher("/cmd/steering_target", Float64, queue_size=1)
    pub_throttle_steering.publish(steering)


def e2e_model():
    global MODEL_PATH, model, alpha, steer_cap

    parser = argparse.ArgumentParser(description="End-to-End cropped/resized model ROS node")
    parser.add_argument("--model", type=str, default=MODEL_PATH, help="Path to .h5 model")
    parser.add_argument("--alpha", type=float, default=alpha, help="Smoothing factor (0–1)")
    parser.add_argument("--steer_cap", type=float, default=steer_cap, help="Steering cap")

    args, _ = parser.parse_known_args()
    MODEL_PATH = args.model
    alpha = args.alpha
    steer_cap = args.steer_cap

    print(f"\n🧠 Parameters:")
    print(f"  • MODEL_PATH = {MODEL_PATH}")
    print(f"  • alpha      = {alpha}")
    print(f"  • steer_cap  = {steer_cap}\n")

    print("🚗 Starting e2e_model node...")
    print(f"Loading model: {MODEL_PATH}")

    model = load_model(MODEL_PATH, compile=False)
    model.compile(loss="sgd", metrics=["mse"])

    rospy.init_node("e2e_model_200x66", anonymous=True)
    rospy.Subscriber("/gmsl_camera/front_narrow/image_raw", SensorImage, new_image)
    rospy.spin()


if __name__ == '__main__':
    try:
        e2e_model()
    except rospy.ROSInterruptException:
        pass
