"""
'place' primitive: deposit the held object at a target location via MoveIt2.

Motion sequence:
  1. move_to_pose(pre_place)        — OMPL free-space approach above target
  2. move_to_pose_linear(place)     — PILZ LIN straight descent to release height
  3. open_gripper()                 — release object
  4. move_to_pose_linear(pre_place) — PILZ LIN straight retreat upward
  5. move_to_named("ready")         — PILZ PTP return to neutral

The place pose uses the same top-down orientation as the pick.
"""

from __future__ import annotations

from geometry_msgs.msg import Pose, Quaternion
from rclpy.node import Node

from vlm_robot_planner.primitives.base import ArmPrimitive, _TOP_DOWN_QUAT

# Height of panda_hand above the target surface centre at the release pose.
# shelf_b top at z≈0.02 m (panda_link0) → cup bottom must clear this.
# With _RELEASE_HEIGHT_M = 0.18 and shelf centre z=0.01:
#   panda_hand z = 0.19 m → cup center z = 0.09 m → cup bottom = 0.03 m
#   Cup bottom (0.03 m) > shelf top (0.02 m) → 1 cm gap, falls gently ✓
_RELEASE_HEIGHT_M  = 0.18
# Approach height above the release pose (pre-place clearance)
_APPROACH_HEIGHT_M = 0.15


class PlacePrimitive(ArmPrimitive):
    """
    Deposits the held object at a target location.

    Args:
        node:   rclpy Node (Orchestrator).
        moveit: MoveIt2Client instance shared with all other primitives.
        attach: Optional GazeboAttach shared with PickPrimitive.
                If provided, the object continues following the EEF during
                the approach, is detached after open_gripper, then Gazebo
                physics drops it the remaining ~1 cm onto the surface.
    """

    def __init__(self, node: Node, moveit, attach=None) -> None:
        super().__init__(node, moveit)
        self._attach = attach

    def execute(self, location_name: str, pose_data: dict) -> bool:
        """
        Execute a top-down place at the named location.

        Args:
            location_name: Symbolic location name (for logging).
            pose_data:     Pose dict from GazeboOracle:
                           {"position": Position, "orientation": Orientation}

        Returns:
            True if the full place sequence completed successfully.
        """
        pos = pose_data["position"]
        self._log(f"place('{location_name}'): pos=({pos.x:.3f}, {pos.y:.3f}, {pos.z:.3f})")

        place_pose = self._build_release_pose(pose_data)
        pre_place  = self._make_pre_grasp_pose(place_pose, lift_m=_APPROACH_HEIGHT_M)

        # ── 1. Move above target — Cartesian approach (OMPL fallback) ────
        self._log(f"  → pre-place (z={pre_place.position.z:.3f})")
        if not self.move_to_pose_cartesian(pre_place):
            self._log("pre-place planning failed — aborting place")
            return False

        # ── 2. Descend to release height — PILZ LIN (straight vertical) ──
        self._log(f"  → descending to release (z={place_pose.position.z:.3f})")
        if not self.move_to_pose_linear(place_pose):
            self._log("place descend failed — aborting place")
            return False

        # ── 3. Open gripper (release object) ──────────────────────────────
        if not self.open_gripper():
            self._log("open_gripper failed during place — object may not be released")

        # ── 3b. Detach simulated object — Gazebo physics drops it ~1 cm ──
        if self._attach is not None:
            self._attach.detach()

        # ── 4. Retreat upward — PILZ LIN ──────────────────────────────────
        self._log("  → retreating")
        if not self.move_to_pose_linear(pre_place):
            self._log("retreat after place failed")
            return False

        # ── 5. Return to ready pose (PILZ PTP joint-space) ────────────────
        self.move_to_named("ready")

        self._log(f"place('{location_name}'): SUCCESS")
        return True

    def _build_release_pose(self, pose_data: dict) -> Pose:
        """Build release pose: target surface + clearance, top-down orientation."""
        pos = pose_data["position"]
        pose = Pose()
        pose.position.x = pos.x
        pose.position.y = pos.y
        pose.position.z = pos.z + _RELEASE_HEIGHT_M
        pose.orientation = _TOP_DOWN_QUAT
        return pose
