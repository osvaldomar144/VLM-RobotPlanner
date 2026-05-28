#!/usr/bin/env python3
"""
test_moveit_arm.py — MoveIt2 smoke test using pymoveit2 (run INSIDE the container).

Coordinate convention
─────────────────────
  Robot spawned at world (0.20, 0, 0.77) → base on table surface.
  panda_link0 frame ≡ robot base:
    table surface   → panda_link0 z = 0.00 m
    objects on table → panda_link0 (x≈0.30, y≈±0.10, z≈+0.06)
    test goals (25 cm above objects) → z = 0.31

Usage (after sourcing ROS 2 + workspace, with simulation.launch.py running):
  python3 /workspace/scripts/test_moveit_arm.py
"""

from __future__ import annotations

import sys
import time
import threading

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup
from geometry_msgs.msg import Pose
from moveit_msgs.msg import CollisionObject
from shape_msgs.msg import SolidPrimitive

try:
    from vlm_robot_planner.moveit2_client import MoveIt2Client as MoveIt2
except ImportError as e:
    sys.exit(
        f"[ERROR] vlm_robot_planner not found — run inside the container "
        f"with the ROS 2 workspace sourced.\n{e}"
    )


# ── Constants ──────────────────────────────────────────────────────────────────

ARM_JOINT_NAMES = [
    "panda_joint1", "panda_joint2", "panda_joint3",
    "panda_joint4", "panda_joint5", "panda_joint6", "panda_joint7",
]

# Gripper pointing straight down: 180° rotation around x → quat [1,0,0,0]
_QUAT_DOWN = [1.0, 0.0, 0.0, 0.0]  # [qx, qy, qz, qw]

# Cartesian goals in panda_link0 frame
GOALS = [
    # (label, x, y, z)
    ("above red_cup",  0.30,  0.10, 0.31),
    ("above blue_box", 0.30, -0.10, 0.31),
    ("above shelf_b",  0.30, -0.25, 0.31),
]

# "ready" joint configuration from moveit_resources_panda SRDF
READY_JOINTS = [0.0, -0.7854, 0.0, -2.3562, 0.0, 1.5708, 0.7854]


# ── Planning scene ─────────────────────────────────────────────────────────────

def setup_planning_scene(node: Node) -> None:
    """
    Publish the table as a collision box to MoveIt2's /collision_object topic.

    Robot base at world z=0.77 m → table surface = panda_link0 z=0.
    Box covers solid table body: z range -0.77 m … 0.00 m.
    """
    pub = node.create_publisher(CollisionObject, "/collision_object", 10)
    time.sleep(0.3)

    co = CollisionObject()
    co.header.frame_id = "panda_link0"
    co.header.stamp    = node.get_clock().now().to_msg()
    co.id              = "table"

    box            = SolidPrimitive()
    box.type       = SolidPrimitive.BOX
    box.dimensions = [1.20, 1.00, 0.77]

    pose              = Pose()
    pose.position.x   = 0.30    # world x=0.50 − robot x=0.20 = 0.30 m
    pose.position.y   = 0.00
    pose.position.z   = -0.385  # centre of 0.77 m slab (top face at z=0)
    pose.orientation.w = 1.0

    co.primitives       = [box]
    co.primitive_poses  = [pose]
    co.operation        = CollisionObject.ADD

    for _ in range(5):
        pub.publish(co)
        time.sleep(0.1)

    print("[OK]   Table collision object published.")
    time.sleep(0.5)


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    rclpy.init()
    node = Node("moveit_arm_test")

    # pymoveit2 requires the node to be spinning in a MultiThreadedExecutor
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    print("\n=== MoveIt2 arm test (pymoveit2) ===\n")
    print("[INFO] Waiting 2 s for move_group to settle…")
    time.sleep(2.0)

    print("[INFO] Publishing table collision object…")
    setup_planning_scene(node)

    print("[INFO] Creating MoveIt2 instance…")
    cb_group = ReentrantCallbackGroup()
    moveit2  = MoveIt2(
        node              = node,
        joint_names       = ARM_JOINT_NAMES,
        base_link_name    = "panda_link0",
        end_effector_name = "panda_hand",
        group_name        = "panda_arm",
        callback_group    = cb_group,
    )
    moveit2.max_velocity     = 0.3
    moveit2.max_acceleration = 0.3
    print("[OK]   MoveIt2 ready.\n")

    # Re-publish scene after MoveIt2 connects so its planning scene is populated
    setup_planning_scene(node)

    results: dict[str, bool] = {}

    # ── Cartesian goal tests ──────────────────────────────────────────────────
    for label, x, y, z in GOALS:
        print(f"--- Test: move to '{label}' (x={x}, y={y}, z={z}) ---")
        moveit2.move_to_pose(
            position  = [x, y, z],
            quat_xyzw = _QUAT_DOWN,
            cartesian = False,
        )
        ok            = bool(moveit2.wait_until_executed())
        results[label] = ok
        tag = "[OK]  " if ok else "[FAIL]"
        print(f"  {tag} {label}\n")
        time.sleep(1.0)

    # ── Return to "ready" ─────────────────────────────────────────────────────
    label = "return to 'ready'"
    print(f"--- Test: {label} ---")
    moveit2.move_to_configuration(READY_JOINTS)
    ok            = bool(moveit2.wait_until_executed())
    results[label] = ok
    tag = "[OK]  " if ok else "[FAIL]"
    print(f"  {tag} {label}\n")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("=== Results ===")
    all_pass = True
    for lbl, passed in results.items():
        t = "PASS" if passed else "FAIL"
        print(f"  {t}  {lbl}")
        if not passed:
            all_pass = False

    print(f"\n{'ALL TESTS PASSED' if all_pass else 'SOME TESTS FAILED'}\n")

    executor.shutdown(timeout_sec=2.0)
    spin_thread.join(timeout=3.0)
    node.destroy_node()
    rclpy.shutdown()
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
