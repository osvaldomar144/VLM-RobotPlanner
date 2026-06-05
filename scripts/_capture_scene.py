#!/usr/bin/env python3
"""
_capture_scene.py — Runs INSIDE the Docker container.

Subscribes to /wrist_camera/image_raw, captures one frame, and saves it
as /workspace/data/scene.png (bind-mounted from the host).

Decodes the ROS2 Image message directly with numpy + PIL — does NOT use
cv_bridge to avoid the NumPy 1.x/2.x ABI incompatibility.

Usage (called by run_vlm_host.py via docker exec):
    docker exec vlm_ros2 bash -c "source ... && python3 /workspace/scripts/_capture_scene.py"

Exit codes:
    0 — image saved successfully
    1 — timeout or error
"""

from __future__ import annotations

import sys
import time

import numpy as np
from PIL import Image as PilImage

import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image

_OUTPUT_PATH  = "/workspace/data/scene.png"
_TIMEOUT_SEC  = 8.0

# Primary: wrist camera (eye-in-hand) — matches real robot deployment.
# Note: in Gazebo the wrist camera plugin publishes on /wrist_camera/image_raw,
# NOT /camera/color/image_raw (which is listed in DDS but has no publisher).
# Arm must be in "scan" pose before calling this script (see _pre_scan.py).
# Fallback: overview camera (when scan pose not reachable or wrist cam offline).
_TOPICS = [
    "/wrist_camera/image_raw",
    "/overview_camera/image_raw",
]

# Supported encodings: (channels, dtype, bgr_flag)
# Keys are lowercase; Gazebo uses R8G8B8 which maps to rgb8.
_ENCODINGS = {
    "rgb8":   (3, np.uint8,  False),
    "r8g8b8": (3, np.uint8,  False),  # Gazebo Classic alias for rgb8
    "bgr8":   (3, np.uint8,  True),
    "rgba8":  (4, np.uint8,  False),
    "bgra8":  (4, np.uint8,  True),
    "mono8":  (1, np.uint8,  False),
}


def _ros_image_to_pil(msg: Image) -> PilImage.Image:
    enc = msg.encoding.lower()
    if enc not in _ENCODINGS:
        raise ValueError(f"Unsupported encoding: {msg.encoding}")
    channels, dtype, is_bgr = _ENCODINGS[enc]
    arr = np.frombuffer(bytes(msg.data), dtype=dtype).reshape(msg.height, msg.width, channels)
    if is_bgr:
        arr = arr[:, :, ::-1].copy()  # BGR → RGB  (or BGRA → RGBA)
    if channels == 1:
        arr = np.repeat(arr, 3, axis=2)  # mono → RGB
    elif channels == 4:
        arr = arr[:, :, :3]  # drop alpha
    return PilImage.fromarray(arr, "RGB")


_PRIMARY_TOPIC_TIMEOUT = 5.0   # seconds to wait for primary before trying fallbacks


class _CaptureNode(Node):
    def __init__(self) -> None:
        super().__init__("_scene_capture")
        self.saved            = False
        self._primary         = _TOPICS[0]
        self._primary_deadline = time.time() + _PRIMARY_TOPIC_TIMEOUT

        for topic in _TOPICS:
            self.create_subscription(
                Image, topic,
                lambda msg, t=topic: self._cb(msg, t),
                qos_profile_sensor_data,   # BEST_EFFORT — compatible with all Gazebo camera plugins
            )
            self.get_logger().info(f"Listening on {topic}")

    def _cb(self, msg: Image, topic: str) -> None:
        if self.saved:
            return
        # Give the primary topic priority for _PRIMARY_TOPIC_TIMEOUT seconds.
        # After that, accept any topic (fallback to overview camera).
        if topic != self._primary and time.time() < self._primary_deadline:
            return
        try:
            img = _ros_image_to_pil(msg)
            img.save(_OUTPUT_PATH)
            self.saved = True
            label = "primary" if topic == self._primary else "fallback"
            print(f"[OK] Scene saved: {_OUTPUT_PATH} ({img.width}×{img.height}) "
                  f"[{label}: {topic}]")
        except Exception as exc:
            print(f"[ERROR] Failed to save image: {exc}", file=sys.stderr)
            sys.exit(1)


def main() -> None:
    rclpy.init()
    node     = _CaptureNode()
    executor = SingleThreadedExecutor()
    executor.add_node(node)

    t0 = time.time()
    while not node.saved and (time.time() - t0) < _TIMEOUT_SEC:
        executor.spin_once(timeout_sec=0.1)

    node.destroy_node()
    rclpy.shutdown()

    if not node.saved:
        print(
            f"[ERROR] No frame received within {_TIMEOUT_SEC}s "
            "(is the simulation running and camera active?)",
            file=sys.stderr,
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
