"""
Oracle world state: provides ground-truth object poses from the simulator (Gazebo).

Role in the pipeline:
  - NOT used as VLM input (the VLM reasons from images directly).
  - Used at execution time only: when a primitive needs a 3D pose
    (e.g. MoveIt pick target), the oracle resolves the symbolic object name
    to an actual pose in the robot base frame (panda_link0).

GazeboOracle queries the Gazebo Classic service /gazebo/get_entity_state.
The service is exposed by the libgazebo_ros_state.so plugin (loaded in
tabletop.world with <namespace>/gazebo</namespace>).

WorldState is the pure-Python data container used by the Orchestrator.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field


# ── Lightweight pose types (no ROS2 dependency) ───────────────────────────────

@dataclass
class Position:
    x: float
    y: float
    z: float


@dataclass
class Orientation:
    x: float
    y: float
    z: float
    w: float


@dataclass
class Pose:
    position:    Position
    orientation: Orientation

    def as_geometry_msg(self):
        """Convert to geometry_msgs.msg.Pose (lazy import — only in ROS2 context)."""
        from geometry_msgs.msg import Pose as RosPose, Point, Quaternion
        p              = RosPose()
        p.position     = Point(x=self.position.x, y=self.position.y, z=self.position.z)
        p.orientation  = Quaternion(
            x=self.orientation.x, y=self.orientation.y,
            z=self.orientation.z, w=self.orientation.w,
        )
        return p


# ── World state container ─────────────────────────────────────────────────────

@dataclass
class ObjectState:
    name:     str
    pose:     Pose
    location: str = ""   # symbolic label (e.g. "table_a") — filled by planner


@dataclass
class WorldState:
    objects:       list[ObjectState] = field(default_factory=list)
    gripper_empty: bool              = True

    def get_pose(self, object_name: str) -> Pose | None:
        """Resolve a symbolic object name to its current 3D pose."""
        for obj in self.objects:
            if obj.name == object_name:
                return obj.pose
        return None

    def to_pddl_init(self) -> list[str]:
        """PDDL :init facts derived from world state (for plan validation)."""
        facts = []
        for obj in self.objects:
            if obj.location:
                facts.append(f"(on {obj.name} {obj.location})")
        if self.gripper_empty:
            facts.append("(gripper-empty)")
        return facts


# ── Gazebo oracle ─────────────────────────────────────────────────────────────

class GazeboOracle:
    """
    Queries Gazebo Classic for ground-truth object poses via ROS 2 service.

    Requires the libgazebo_ros_state.so plugin in the world file with
    <namespace>/gazebo</namespace>, which exposes:
      /gazebo/get_entity_state  (gazebo_msgs/srv/GetEntityState)

    The reference_frame should match the robot base link so that MoveIt2
    can use the pose directly without an extra TF lookup.
    """

    SERVICE = "/gazebo/get_entity_state"

    def __init__(self, node, reference_frame: str = "panda_link0") -> None:
        from gazebo_msgs.srv import GetEntityState

        self._node   = node
        self._frame  = reference_frame
        self._client = node.create_client(GetEntityState, self.SERVICE)

        node.get_logger().info(f"GazeboOracle: waiting for {self.SERVICE} ...")
        if not self._client.wait_for_service(timeout_sec=10.0):
            node.get_logger().warn(
                f"GazeboOracle: {self.SERVICE} not available. "
                "Is the gazebo_ros_state plugin loaded in the world file?"
            )
        else:
            node.get_logger().info("GazeboOracle: service ready.")

    def get_pose(self, object_name: str) -> Pose | None:
        """
        Query Gazebo for the current pose of a named model.

        Uses threading.Event + callbacks so it is safe to call from any
        background thread with a MultiThreadedExecutor already spinning.

        Args:
            object_name: Gazebo model name (e.g. "red_cup", "shelf_b").

        Returns:
            Pose in the robot base frame, or None if the model is not found.
        """
        from gazebo_msgs.srv import GetEntityState

        req                  = GetEntityState.Request()
        req.name             = object_name
        req.reference_frame  = self._frame

        done          = threading.Event()
        result_holder: list = [None]

        def _on_response(future):
            result_holder[0] = future.result()
            done.set()

        self._client.call_async(req).add_done_callback(_on_response)
        done.wait(timeout=5.0)

        resp = result_holder[0]
        if resp is None:
            self._node.get_logger().warn(
                f"GazeboOracle: service call timed out for '{object_name}'"
            )
            return None

        if not resp.success:
            self._node.get_logger().warn(
                f"GazeboOracle: '{object_name}' not found in Gazebo. "
                f"Error: {resp.status_message}"
            )
            return None

        p = resp.state.pose.position
        q = resp.state.pose.orientation
        return Pose(
            position    = Position(x=p.x, y=p.y, z=p.z),
            orientation = Orientation(x=q.x, y=q.y, z=q.z, w=q.w),
        )

    def get_world_state(self, tracked_objects: list[str]) -> WorldState:
        """
        Query all tracked objects and return a WorldState snapshot.

        Args:
            tracked_objects: List of Gazebo model names to query.

        Returns:
            WorldState with current poses. Objects not found are skipped.
        """
        object_states = []
        for name in tracked_objects:
            pose = self.get_pose(name)
            if pose is not None:
                object_states.append(ObjectState(name=name, pose=pose))
            else:
                self._node.get_logger().warn(
                    f"GazeboOracle: skipping '{name}' — pose unavailable."
                )
        return WorldState(objects=object_states, gripper_empty=True)
