"""
'place' primitive: deposit the held object at a target location via MoveIt2.

Motion sequence:
  1. move_to_pose(pre_place)  — move above the target location
  2. move_to_pose(place)      — descend to release height
  3. open_gripper()           — release object
  4. move_to_pose(pre_place)  — retreat upward
  5. move_to_named("ready")   — return arm to neutral pose

The place pose uses the same top-down orientation as the pick.
The release height is slightly above the surface so the object lands
gently rather than being dropped.
"""

from __future__ import annotations

from geometry_msgs.msg import Pose, Quaternion
from rclpy.node import Node

from vlm_robot_planner.primitives.base import ArmPrimitive, _TOP_DOWN_QUAT

# Height above the target surface centre at which the gripper opens
_RELEASE_HEIGHT_M  = 0.04   # 4 cm above surface — gentle drop
# Approach height above the release point
_APPROACH_HEIGHT_M = 0.15


class PlacePrimitive(ArmPrimitive):
    """
    Deposits the held object at a target location.

    Args:
        node:   rclpy Node (Orchestrator).
        moveit: MoveItPy instance shared with all other primitives.
    """

    def __init__(self, node: Node, moveit) -> None:
        super().__init__(node, moveit)

    def execute(self, location_name: str, pose_data: dict) -> bool:
        """
        Execute a top-down place at the named location.

        Args:
            location_name: Symbolic location name (for logging).
            pose_data:     Pose dict from GazeboOracle:
                           {"position": [x,y,z], "orientation": [qx,qy,qz,qw]}

        Returns:
            True if the full place sequence completed successfully.
        """
        pos = pose_data["position"]
        self._log(f"place('{location_name}'): pos=({pos.x:.3f}, {pos.y:.3f}, {pos.z:.3f})")

        place_pose = self._build_release_pose(pose_data)
        pre_place  = self._make_pre_grasp_pose(place_pose, lift_m=_APPROACH_HEIGHT_M)

        # ── 1. Move above target ───────────────────────────────────────────
        self._log(f"  → pre-place (z={pre_place.position.z:.3f})")
        if not self.move_to_pose(pre_place):
            self._log("pre-place planning failed — aborting place")
            return False

        # ── 2. Descend to release height ───────────────────────────────────
        self._log(f"  → descending to release (z={place_pose.position.z:.3f})")
        if not self.move_to_pose(place_pose):
            self._log("place descend failed — aborting place")
            return False

        # ── 3. Open gripper (release object) ──────────────────────────────
        if not self.open_gripper():
            self._log("open_gripper failed during place — object may not be released")

        # ── 4. Retreat upward ──────────────────────────────────────────────
        self._log("  → retreating")
        if not self.move_to_pose(pre_place):
            self._log("retreat after place failed")
            return False

        # ── 5. Return to ready pose ────────────────────────────────────────
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
