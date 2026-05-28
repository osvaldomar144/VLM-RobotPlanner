"""
Shared MoveIt2 helpers for all arm primitives — uses pymoveit2.

All primitives (pick, place, look_at) inherit from ArmPrimitive.
A single pymoveit2.MoveIt2 instance is created in the Orchestrator
and passed to each primitive to avoid multiple competing planning
interfaces.

The node's executor MUST be MultiThreadedExecutor for pymoveit2 to work.
Gripper control goes through the GripperCommand action directly on
/panda_hand_controller/gripper_cmd, using threading.Event callbacks
so it is safe to call from any background thread.
"""

from __future__ import annotations

import threading
from copy import deepcopy

from action_msgs.msg import GoalStatus
from control_msgs.action import GripperCommand
from geometry_msgs.msg import Pose, Quaternion
from rclpy.action import ActionClient
from rclpy.node import Node

# Panda joint names for the panda_arm group (must match SRDF)
ARM_JOINT_NAMES = [
    "panda_joint1",
    "panda_joint2",
    "panda_joint3",
    "panda_joint4",
    "panda_joint5",
    "panda_joint6",
    "panda_joint7",
]

# Named joint configurations (radians) — sourced from moveit_resources_panda SRDF
_NAMED_CONFIGS = {
    "ready": [0.0, -0.7854, 0.0, -2.3562, 0.0, 1.5708, 0.7854],
}

ARM_GROUP  = "panda_arm"
BASE_FRAME = "panda_link0"
EEF_LINK   = "panda_hand"

_GRIPPER_OPEN   = 0.04   # metres per finger (8 cm total opening)
# Grip position tuned for red_cup (radius = 0.025 m → diameter = 0.05 m).
# 2 mm inside the cup surface per finger → firm contact without physics penetration.
_GRIPPER_CLOSED = 0.023
_GRIPPER_EFFORT = 20.0   # N — enough for lightweight objects

# Top-down grasp orientation: 180° rotation around x → end-effector points down.
_TOP_DOWN_QUAT = Quaternion(x=1.0, y=0.0, z=0.0, w=0.0)


class ArmPrimitive:
    """
    Base class for arm primitives.

    Args:
        node:    rclpy Node (the Orchestrator).
        moveit2: pymoveit2.MoveIt2 instance (shared across all primitives).
    """

    def __init__(self, node: Node, moveit2) -> None:
        self._node    = node
        self._moveit2 = moveit2
        self._gripper_client = ActionClient(
            node, GripperCommand, "/panda_hand_controller/gripper_cmd"
        )

    # ── Arm motions ───────────────────────────────────────────────────────────

    def move_to_pose(
        self,
        pose:       Pose,
        frame_id:   str   = BASE_FRAME,
        timeout_sec: float = 15.0,
    ) -> bool:
        """Plan and execute a Cartesian goal (OMPL, free-space path)."""
        self._moveit2.move_to_pose(
            position=[pose.position.x, pose.position.y, pose.position.z],
            quat_xyzw=[
                pose.orientation.x,
                pose.orientation.y,
                pose.orientation.z,
                pose.orientation.w,
            ],
            cartesian=False,
        )
        result = self._moveit2.wait_until_executed(timeout=timeout_sec)
        if not result:
            self._node.get_logger().warn("ArmPrimitive: move_to_pose failed.")
            return False
        return True

    def move_to_pose_linear(
        self,
        pose:        Pose,
        timeout_sec: float = 15.0,
    ) -> bool:
        """Plan a smooth Cartesian pose motion (PILZ PTP, OMPL fallback).

        Primary: PILZ PTP — deterministic, smooth joint-space interpolation
        that produces near-straight Cartesian paths for small displacements.
        Fallback: OMPL — used if PILZ rejects the goal.
        """
        quat = [pose.orientation.x, pose.orientation.y,
                pose.orientation.z, pose.orientation.w]
        pos  = [pose.position.x, pose.position.y, pose.position.z]

        self._moveit2.move_to_pose_linear(position=pos, quat_xyzw=quat)
        result = self._moveit2.wait_until_executed(timeout=timeout_sec)

        if not result:
            self._node.get_logger().warn(
                "ArmPrimitive: PILZ PTP failed — falling back to OMPL."
            )
            self._moveit2.move_to_pose(position=pos, quat_xyzw=quat)
            result = self._moveit2.wait_until_executed(timeout=timeout_sec)
            if not result:
                self._node.get_logger().warn(
                    "ArmPrimitive: move_to_pose_linear (OMPL fallback) also failed."
                )
                return False

        return True

    def move_to_named(self, config_name: str) -> bool:
        """Move to a named joint configuration defined in _NAMED_CONFIGS."""
        joint_positions = _NAMED_CONFIGS.get(config_name)
        if joint_positions is None:
            self._node.get_logger().warn(
                f"ArmPrimitive: unknown named config '{config_name}' — skipping."
            )
            return False
        self._moveit2.move_to_configuration(joint_positions)
        result = self._moveit2.wait_until_executed()
        if not result:
            self._node.get_logger().warn(
                f"ArmPrimitive: move_to_named('{config_name}') failed."
            )
            return False
        return True

    # ── Gripper motions ───────────────────────────────────────────────────────

    def open_gripper(self) -> bool:
        return self._send_gripper_goal(position=_GRIPPER_OPEN, max_effort=0.0)

    def close_gripper(self, effort: float = _GRIPPER_EFFORT) -> bool:
        return self._send_gripper_goal(position=_GRIPPER_CLOSED, max_effort=effort)

    def _send_gripper_goal(self, position: float, max_effort: float) -> bool:
        """
        Send a GripperCommand action goal.

        Uses threading.Event + callbacks so it is safe to call from any
        background thread without conflicting with the MultiThreadedExecutor.
        """
        if not self._gripper_client.wait_for_server(timeout_sec=3.0):
            self._node.get_logger().warn(
                "ArmPrimitive: gripper action server not ready."
            )
            return False

        goal = GripperCommand.Goal()
        goal.command.position   = position
        goal.command.max_effort = max_effort

        done           = threading.Event()
        result_holder: list = [None]

        def _on_result(future):
            result_holder[0] = future.result()
            done.set()

        def _on_goal(future):
            gh = future.result()
            if gh is None or not gh.accepted:
                self._node.get_logger().warn(
                    "ArmPrimitive: gripper goal rejected."
                )
                done.set()
                return
            gh.get_result_async().add_done_callback(_on_result)

        self._gripper_client.send_goal_async(goal).add_done_callback(_on_goal)
        done.wait(timeout=10.0)

        res = result_holder[0]
        success = res is not None and res.status == GoalStatus.STATUS_SUCCEEDED
        if not success:
            self._node.get_logger().warn(
                "ArmPrimitive: gripper action did not succeed."
            )
        return success

    # ── Utility ───────────────────────────────────────────────────────────────

    def _make_pre_grasp_pose(self, grasp_pose: Pose, lift_m: float = 0.12) -> Pose:
        """Return a pose `lift_m` above the grasp pose (pre-grasp approach)."""
        pre = deepcopy(grasp_pose)
        pre.position.z += lift_m
        return pre

    def _log(self, msg: str) -> None:
        self._node.get_logger().info(f"[{self.__class__.__name__}] {msg}")
