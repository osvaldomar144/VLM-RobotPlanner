"""
'pick' primitive: grasp an object from the table via MoveIt2.

Motion sequence:
  1. open_gripper()              — ensure gripper is fully open
  2. move_to_pose(pre_grasp)    — move above the object (12 cm clearance)
  3. move_to_pose(grasp)        — descend to grasp pose
  4. close_gripper()            — grasp with 20 N effort
  5. move_to_pose(pre_grasp)    — lift the object (retreat)

The grasp pose is the object's centre pose from the GazeboOracle, with
the end-effector oriented top-down (wrist rotated so fingers point down).

For a real Franka, the orientation should come from a grasp planner
(e.g. GraspNet, GPD). In Phase 1 we use a fixed top-down orientation
which works for upright cylinders and boxes on a flat table.
"""

from __future__ import annotations

import math

from geometry_msgs.msg import Pose
from rclpy.node import Node

from vlm_robot_planner.primitives.base import ArmPrimitive, _TOP_DOWN_QUAT

# Grasp approach height above the object centre
_APPROACH_HEIGHT_M = 0.15

# How far above the final grasp position the gripper starts (approach clearance)
_GRASP_OFFSET_Z_M = 0.02   # slight offset above object center to avoid collision


class PickPrimitive(ArmPrimitive):
    """
    Grasps an object given its symbolic name and 3D pose from the oracle.

    Args:
        node:   rclpy Node (Orchestrator).
        moveit: MoveItPy instance shared with all other primitives.
    """

    def __init__(self, node: Node, moveit) -> None:
        super().__init__(node, moveit)

    def execute(self, object_name: str, pose_data: dict) -> bool:
        """
        Execute a top-down pick on the named object.

        Args:
            object_name: Symbolic object name (for logging).
            pose_data:   Pose dict from GazeboOracle:
                         {"position": Point, "orientation": Quaternion}

        Returns:
            True if the full pick sequence completed successfully.
        """
        pos = pose_data["position"]
        self._log(f"pick('{object_name}'): pos=({pos.x:.3f}, {pos.y:.3f}, {pos.z:.3f})")

        grasp_pose = self._build_top_down_pose(pose_data)
        pre_grasp  = self._make_pre_grasp_pose(grasp_pose, lift_m=_APPROACH_HEIGHT_M)

        # ── 1. Open gripper ────────────────────────────────────────────────
        if not self.open_gripper():
            self._log("open_gripper failed — aborting pick")
            return False

        # ── 2. Pre-grasp (above object) ────────────────────────────────────
        self._log(f"  → moving to pre-grasp (z={pre_grasp.position.z:.3f})")
        if not self.move_to_pose(pre_grasp):
            self._log("pre-grasp planning failed — aborting pick")
            return False

        # ── 3. Descend to grasp ────────────────────────────────────────────
        self._log(f"  → descending to grasp (z={grasp_pose.position.z:.3f})")
        if not self.move_to_pose(grasp_pose):
            self._log("grasp descend failed — aborting pick")
            return False

        # ── 4. Close gripper ───────────────────────────────────────────────
        if not self.close_gripper():
            self._log("close_gripper failed — object may have slipped")
            # Continue anyway: attempt retreat even if grasp uncertain

        # ── 5. Lift (retreat to pre-grasp) ────────────────────────────────
        self._log("  → lifting object")
        if not self.move_to_pose(pre_grasp):
            self._log("lift failed — object may be stuck")
            return False

        self._log(f"pick('{object_name}'): SUCCESS")
        return True

    def _build_top_down_pose(self, pose_data: dict) -> Pose:
        """Build a top-down Pose from oracle pose data (Point + Quaternion)."""
        pos = pose_data["position"]
        pose = Pose()
        pose.position.x = pos.x
        pose.position.y = pos.y
        pose.position.z = pos.z + _GRASP_OFFSET_Z_M
        pose.orientation = _TOP_DOWN_QUAT
        return pose
