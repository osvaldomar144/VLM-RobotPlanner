"""
'pick' primitive: move the arm to an object and grasp it via MoveIt.
Stub for Phase 1 — implement MoveIt action client calls here.
"""

from __future__ import annotations


class PickPrimitive:
    """
    Grasps an object given its symbolic name.
    Requires the object pose to be resolved externally (oracle or perception).
    """

    def __init__(self, node):
        self._node = node  # rclpy Node

    def execute(self, object_name: str, pose: dict) -> bool:
        """
        Args:
            object_name: Name of the object to pick.
            pose: 6-DOF pose dict {"position": [x,y,z], "orientation": [qx,qy,qz,qw]}.

        Returns:
            True if successful, False otherwise.
        """
        raise NotImplementedError("Wire up MoveIt MoveGroup action client")
