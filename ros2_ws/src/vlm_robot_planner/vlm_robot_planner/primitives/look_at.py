"""
'look_at' primitive: orient the wrist camera toward a target object.

Phase 1 implementation: moves the arm to a fixed "observation" configuration
that places the eye-in-hand camera above and in front of the robot workspace,
looking down at the table. Full per-object camera pointing (computing IK for
camera-on-target) is a Phase 2 task once hand-eye calibration is done.

The PDDL action is look-at (?i - item), called before pick to ensure
the camera has a clear view of the object before grasping.
"""

from __future__ import annotations

from rclpy.node import Node

from vlm_robot_planner.primitives.base import ArmPrimitive


# Named SRDF configuration that places the arm in an observation pose.
# This is the "ready" pose from moveit_resources_panda_moveit_config SRDF,
# which is defined as: [0, -π/4, 0, -3π/4, 0, π/2, π/4] (radians).
# It positions the camera to look down at the table from ~0.5 m above.
_OBSERVATION_CONFIG = "ready"


class LookAtPrimitive(ArmPrimitive):
    """
    Moves the arm to the observation configuration so the wrist camera
    has a clear view of the table workspace.

    Phase 1: fixed observation pose regardless of target object.
    Phase 2: compute IK so camera_optical_frame is aligned with target
             (requires hand-eye calibration with the RealSense D435i).
    """

    def __init__(self, node: Node, moveit) -> None:
        super().__init__(node, moveit)

    def execute(self, target_name: str, pose_data: dict | None = None) -> bool:
        """
        Move arm to observation pose.

        Args:
            target_name: Name of the object to look at (used for logging;
                         Phase 2 will use this to compute camera pointing).
            pose_data:   Object pose (unused in Phase 1).

        Returns:
            True on success.
        """
        self._log(
            f"look_at('{target_name}'): moving to observation pose '{_OBSERVATION_CONFIG}'"
        )
        success = self.move_to_named(_OBSERVATION_CONFIG)
        if success:
            self._log(f"look_at('{target_name}'): camera positioned — ready for perception")
        else:
            self._log(f"look_at('{target_name}'): failed to reach observation pose")
        return success
