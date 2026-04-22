"""
'place' primitive: move the arm to a target location and release the object.
Stub for Phase 1 — implement MoveIt action client calls here.
"""

from __future__ import annotations


class PlacePrimitive:

    def __init__(self, node):
        self._node = node

    def execute(self, location_name: str, pose: dict) -> bool:
        """
        Args:
            location_name: Symbolic name of the target location.
            pose: Target 6-DOF pose.

        Returns:
            True if successful, False otherwise.
        """
        raise NotImplementedError("Wire up MoveIt MoveGroup action client")
