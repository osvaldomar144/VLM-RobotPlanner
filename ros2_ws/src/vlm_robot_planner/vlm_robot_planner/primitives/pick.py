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

# Grasp approach height above the grasp pose (pre-grasp clearance)
_APPROACH_HEIGHT_M = 0.15

# Height of panda_hand above the object centre at the grasp pose.
# Franka finger length below panda_hand frame ≈ 0.133 m (58 mm joint offset + 75 mm finger).
# panda_hand palm housing extends ~58 mm in +Z (toward fingertips) before the finger joints.
# With _GRASP_OFFSET_Z_M = 0.13 and red_cup centre at z=0.83 m (world):
#   panda_hand z = 0.96 m → palm housing bottom ≈ 0.902 m (above cup top at 0.89 m) ✓
#   finger tips z ≈ 0.827 m ≈ cup centre — good grasp position
#   pre-grasp z = 0.96 + 0.15 = 1.11 m (above any table object)
# On real robot this also ensures the palm never contacts the object top.
_GRASP_OFFSET_Z_M = 0.13


class PickPrimitive(ArmPrimitive):
    """
    Grasps an object given its symbolic name and 3D pose from the oracle.

    Args:
        node:   rclpy Node (Orchestrator).
        moveit: MoveIt2Client instance shared with all other primitives.
        attach: Optional GazeboAttach for simulated object attachment.
                If provided, the object will follow the EEF during the lift.
    """

    def __init__(self, node: Node, moveit, attach=None) -> None:
        super().__init__(node, moveit)
        self._attach = attach

    def execute(
        self,
        object_name: str,
        pose_data: dict,
        support_surface: str | None = None,
    ) -> bool:
        """
        Execute a top-down pick on the named object.

        Args:
            object_name:     Symbolic object name (for logging).
            pose_data:       Pose dict from GazeboOracle.
            support_surface: MoveIt2 collision object name of the surface the
                             item rests on (e.g. "table", "shelf_b").
                             Passed to attach_object() as touch_link so MoveIt2
                             allows the initial ACO-surface overlap during lift.
                             Identical pattern on real robot.

        Returns:
            True if the full pick sequence completed successfully.
        """
        pos = pose_data["position"]
        self._log(f"pick('{object_name}'): pos=({pos.x:.3f}, {pos.y:.3f}, {pos.z:.3f})")

        grasp_pose = self._build_top_down_pose(pose_data)
        pre_grasp  = self._make_pre_grasp_pose(grasp_pose, lift_m=_APPROACH_HEIGHT_M)

        # ── 0. Clean up stale state from previous incomplete operations ──────
        # Covers two failure modes:
        #   a) ACO left attached in MoveIt2 (blocks collision planning)
        #   b) GazeboAttach timer still running (object keeps teleporting)
        # Both are silent — the robot appears fine but the next pick fails.
        self.detach_object()
        if self._attach is not None:
            self._attach.detach()   # stops 100 Hz teleport timer if still running

        # ── 1. Open gripper ────────────────────────────────────────────────
        if not self.open_gripper():
            self._log("open_gripper failed — aborting pick")
            return False

        # ── 2. Pre-grasp (above object) — Cartesian approach (OMPL fallback) ─
        self._log(f"  → moving to pre-grasp (z={pre_grasp.position.z:.3f})")
        if not self.move_to_pose_cartesian(pre_grasp):
            self._log("pre-grasp planning failed — aborting pick")
            return False

        # ── 3. Descend to grasp — PILZ LIN (straight vertical line) ───────
        self._log(f"  → descending to grasp (z={grasp_pose.position.z:.3f})")
        if not self.move_to_pose_linear(grasp_pose):
            self._log("grasp descend failed — aborting pick")
            return False

        # ── 4. Close gripper ───────────────────────────────────────────────
        if not self.close_gripper():
            self._log("close_gripper failed — object may have slipped")

        # ── 4b. Notify MoveIt2 that the object is now held (W5) ───────────
        # Pass support_surface so MoveIt2 allows ACO-surface overlap during lift.
        self.attach_object(object_name, support_surface=support_surface)

        # ── 4c. Simulation-only: physics joint attachment (Boeing plugin) ─
        if self._attach is not None:
            self._attach.attach(object_name, grasp_offset_z=_GRASP_OFFSET_Z_M)

        # ── 5. Lift — PILZ LIN (straight vertical, object follows EEF) ────
        self._log("  → lifting object")
        if not self.move_to_pose_linear(pre_grasp):
            self._log("lift failed — object may be stuck")
            self.detach_object(object_name)
            if self._attach is not None:
                self._attach.detach()
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
