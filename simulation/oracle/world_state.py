"""
Oracle world state: provides ground-truth object poses from the simulator (Gazebo).

Role in the pipeline:
- NOT used as VLM input (the VLM reasons from images directly).
- Used at execution time: when a primitive needs a 3D pose (e.g. MoveIt pick),
  the oracle resolves the symbolic object name to an actual pose in robot frame.
"""

from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class Pose:
    position: list[float]       # [x, y, z] in meters, robot base frame
    orientation: list[float]    # [qx, qy, qz, qw]


@dataclass
class ObjectState:
    name: str
    pose: Pose
    location: str = ""          # symbolic label (e.g. "table_a") — optional


@dataclass
class WorldState:
    objects: list[ObjectState] = field(default_factory=list)
    gripper_empty: bool = True

    def get_pose(self, object_name: str) -> Pose | None:
        """Resolve a symbolic object name to its current 3D pose."""
        for obj in self.objects:
            if obj.name == object_name:
                return obj.pose
        return None

    def to_pddl_init(self) -> list[str]:
        """PDDL :init facts — used if symbolic planner validation is active."""
        facts = []
        for obj in self.objects:
            if obj.location:
                facts.append(f"(on {obj.name} {obj.location})")
        if self.gripper_empty:
            facts.append("(gripper-empty)")
        return facts


class GazeboOracle:
    """
    Queries Gazebo for ground-truth object poses via ROS 2 service calls.
    Requires a running Gazebo instance with the target models spawned.
    """

    def get_world_state(self, tracked_objects: list[str]) -> WorldState:
        """
        Args:
            tracked_objects: Gazebo model names to track.

        Returns:
            WorldState with current poses in robot base frame.
        """
        raise NotImplementedError("Implement with rclpy + gazebo_msgs/GetEntityState")
