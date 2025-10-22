#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ROS Bag Visualizer + Slicer GUI (ROS1 native)
--------------------------------------------
✔ Works only with ROS1 .bag files (recorded via rosbag record)
✔ Displays image topic + steering overlay
✔ Lets you select start/end and save segment
✔ Writes using rosbag API directly (no rosbags lib)
✔ Output is bit-identical and playable
"""

import os
os.environ["QT_QPA_PLATFORM"] = "xcb"
if not os.environ.get("DISPLAY"):
    os.environ["QT_QPA_PLATFORM"] = "offscreen"

import sys
import cv2
import rospy
import rosbag
import bisect
import numpy as np
from datetime import datetime
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QFileDialog, QSlider, QMessageBox
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPixmap
from pathlib import Path
from sensor_msgs.msg import Image
from std_msgs.msg import Float64


class BagSlicerGUI(QWidget):
    def __init__(self, bag_path=None):
        super().__init__()
        self.setWindowTitle("ROS Bag Slicer GUI (ROS1 Native)")
        self.resize(1000, 700)

        layout = QVBoxLayout(self)
        self.img_label = QLabel("Open a ROS1 bag to begin")
        self.img_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.img_label)

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setEnabled(False)
        layout.addWidget(self.slider)

        btns = QHBoxLayout()
        self.open_btn = QPushButton("📂 Open bag")
        self.start_btn = QPushButton("⏱️ Mark Start")
        self.end_btn = QPushButton("🏁 Mark End")
        self.save_btn = QPushButton("💾 Save Segment")
        for b in [self.open_btn, self.start_btn, self.end_btn, self.save_btn]:
            btns.addWidget(b)
        layout.addLayout(btns)

        self.open_btn.clicked.connect(self.open_bag)
        self.start_btn.clicked.connect(self.mark_start)
        self.end_btn.clicked.connect(self.mark_end)
        self.save_btn.clicked.connect(self.save_segment)
        self.slider.valueChanged.connect(self.update_frame)

        self.reader_path = None
        self.frames = []
        self.timestamps = []
        self.seg_start = None
        self.seg_end = None
        self.image_topic = None
        self.steering_topic = None
        self.steering_data = []
        self.max_dt = 0.1  # seconds tolerance

        if bag_path:
            self.load_bag(Path(bag_path))

    def nearest_idx(self, ts_list, t_query):
        if not ts_list:
            return None
        i = bisect.bisect_left(ts_list, t_query)
        cand = []
        if i < len(ts_list):
            cand.append(i)
        if i > 0:
            cand.append(i - 1)
        return min(cand, key=lambda k: abs(ts_list[k] - t_query))

    def open_bag(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open ROS1 bag", "", "Bag files (*.bag)")
        if path:
            self.load_bag(Path(path))

    def load_bag(self, bag_path: Path):
        self.reader_path = bag_path
        self.frames.clear()
        self.timestamps.clear()
        self.steering_data.clear()

        with rosbag.Bag(bag_path, "r") as bag:
            topics = bag.get_type_and_topic_info()[1].keys()
            img_topics = [t for t in topics if "image" in t.lower()]
            self.image_topic = img_topics[0] if img_topics else None
            self.steering_topic = "/vehicle/steering_pct" if "/vehicle/steering_pct" in topics else None

            if not self.image_topic:
                QMessageBox.warning(self, "Error", "No image topic found.")
                return

            print(f"[INFO] Using image topic: {self.image_topic}")
            if self.steering_topic:
                print(f"[INFO] Found steering topic: {self.steering_topic}")

            for topic, msg, t in bag.read_messages(topics=[self.image_topic]):
                img = np.frombuffer(msg.data, dtype=np.uint8).reshape((msg.height, msg.width, -1))
                if msg.encoding == "rgb8":
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                self.frames.append(img)
                self.timestamps.append(t.to_sec())

            if self.steering_topic:
                for topic, msg, t in bag.read_messages(topics=[self.steering_topic]):
                    self.steering_data.append((t.to_sec(), msg.data))

        self.slider.setMaximum(len(self.frames) - 1)
        self.slider.setEnabled(True)
        self.update_frame(0)
        print(f"[INFO] Loaded {len(self.frames)} frames")

    def update_frame(self, idx):
        if not self.frames:
            return
        img = self.frames[idx].copy()
        t_img = self.timestamps[idx]
        steering_val = None
        if self.steering_data:
            k = self.nearest_idx([t for t, _ in self.steering_data], t_img)
            if k is not None:
                ts_s, val = self.steering_data[k]
                if abs(ts_s - t_img) <= self.max_dt:
                    steering_val = val
        overlay = f"t={t_img:.2f}s"
        if steering_val is not None:
            overlay += f" | steering={steering_val:.3f}"
        cv2.putText(img, overlay, (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, ch = img_rgb.shape
        qimg = QImage(img_rgb.data, w, h, ch * w, QImage.Format_RGB888)
        self.img_label.setPixmap(QPixmap.fromImage(qimg).scaled(
            self.img_label.width(), self.img_label.height(), Qt.KeepAspectRatio))

    def mark_start(self):
        self.seg_start = self.slider.value()
        QMessageBox.information(self, "Start", f"Segment start at frame {self.seg_start}")

    def mark_end(self):
        self.seg_end = self.slider.value()
        QMessageBox.information(self, "End", f"Segment end at frame {self.seg_end}")

    def save_segment(self):
        if self.seg_start is None or self.seg_end is None:
            QMessageBox.warning(self, "Error", "Mark start and end first!")
            return
        if not self.reader_path:
            return

        t_start = self.timestamps[min(self.seg_start, self.seg_end)]
        t_end = self.timestamps[max(self.seg_start, self.seg_end)]
        dt_str = datetime.utcfromtimestamp(t_start).strftime("%Y-%m-%d-%H-%M-%S")
        default_path = str(self.reader_path.parent / f"{dt_str}.bag")

        out_path, _ = QFileDialog.getSaveFileName(self, "Save segment", default_path, "Bag files (*.bag)")
        if not out_path:
            return

        n = 0
        with rosbag.Bag(self.reader_path, "r") as inbag, rosbag.Bag(out_path, "w") as outbag:
            for topic, msg, t in inbag.read_messages():
                ts = t.to_sec()
                if t_start <= ts <= t_end:
                    outbag.write(topic, msg, t)
                    n += 1

        QMessageBox.information(self, "Saved", f"✅ Saved {n} messages to {out_path}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    bag_file = sys.argv[1] if len(sys.argv) > 1 else None
    gui = BagSlicerGUI(bag_file)
    gui.show()
    sys.exit(app.exec_())

