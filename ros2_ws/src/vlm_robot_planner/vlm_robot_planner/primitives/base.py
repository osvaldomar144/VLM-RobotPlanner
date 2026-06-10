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
    # Scan pose: arm forward-low with camera aimed at the table centre.
    # j1=-0.70 (forward tilt), j3=-2.10 (elbow down), j5=1.40 (wrist).
    # Used as fallback when look_at has no target pose.
    "scan":  [0.0, -0.70, 0.0, -2.10, 0.0, 1.40, 0.7854],

    # Safe retreat: arm pulled up and back, EEF well above table surface.
    # Used as intermediate waypoint after place/pick before returning to "ready".
    # Prevents PILZ PTP from taking paths that dip near table-level objects.
    # Geometry rationale: joint2=-0.9 + joint4=-1.8 → EEF at z≈0.65m above
    # panda_link0 (table surface), safely above any tabletop object.
    # This config is general — does NOT depend on specific scene objects.
    "safe_retreat": [0.0, -0.9, 0.0, -1.8, 0.0, 0.9, 0.7854],
}

ARM_GROUP  = "panda_arm"
BASE_FRAME = "panda_link0"
EEF_LINK   = "panda_hand"

_GRIPPER_OPEN   = 0.04   # metres per finger (8 cm total opening)
# Grip position: 0.020 m per finger = 4 cm total opening.
# Covers Phase 2 objects (hammer handle ~4cm, cup ~5cm, cylinder ~4cm).
# GazeboAttach handles the actual attachment regardless of exact closure.
_GRIPPER_CLOSED = 0.020
_GRIPPER_EFFORT = 20.0   # N — enough for lightweight objects

# Top-down grasp orientation: 180° rotation around x → end-effector points down.
_TOP_DOWN_QUAT = Quaternion(x=1.0, y=0.0, z=0.0, w=0.0)


# ── W5: AttachedCollisionObject parameters ────────────────────────────────────
# Conservative geometry for any grasped object — NOT specific to scene objects.
# Real robot Phase 2: replace with object geometry from PerceptionModule.
# The cylinder approximates "something held in the gripper" in a general way.
_HELD_OBJ_RADIUS = 0.04   # m — covers graspable objects up to ~8 cm diameter
_HELD_OBJ_HEIGHT = 0.14   # m — covers objects up to ~14 cm tall
_HELD_OBJ_LINK   = "panda_hand"
# Object centre offset in panda_hand frame.
# panda_hand convention: +Z = approach direction = toward fingertips/object.
# During top-down grasp: +Z points downward in world frame.
# The grasped object is _GRASP_OFFSET_Z_M (0.10m) from panda_hand in +Z.
_HELD_OBJ_Z_IN_HAND = 0.13   # m — in panda_hand +Z (toward object, away from arm)


class ArmPrimitive:
    # Class-level shared state: which object is currently held in the gripper.
    # Shared across ALL primitive instances (pick and place are separate objects
    # but must coordinate on what is attached in the MoveIt2 planning scene).
    _HELD_ACO_ID: str | None = None
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
        # Publisher for AttachedCollisionObject — W5
        from moveit_msgs.msg import AttachedCollisionObject as _ACO
        self._aco_pub = node.create_publisher(
            _ACO, "/attached_collision_object", 10
        )
        # Publisher to remove objects from the world collision scene.
        # Needed because MoveIt2 re-adds a detached ACO as a world object
        # at the gripper position, causing false collisions during retreat.
        from moveit_msgs.msg import CollisionObject as _ColObj
        self._col_pub = node.create_publisher(_ColObj, "/collision_object", 10)

        self._held_object_id: str | None = None   # instance-level (legacy compat)

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

    def move_to_pose_cartesian(
        self,
        pose:        Pose,
        timeout_sec: float = 15.0,
    ) -> bool:
        """Move to pose via computeCartesianPath (geometrically straight line).

        Falls back to PILZ PTP → OMPL when Cartesian fraction < 90% (e.g. large
        workspace moves or paths through kinematic singularities).
        """
        self._moveit2.move_cartesian_waypoints([pose])
        result = self._moveit2.wait_until_executed(timeout=timeout_sec)
        if not result:
            self._node.get_logger().warn(
                "ArmPrimitive: Cartesian path incomplete — falling back to PILZ PTP."
            )
            return self.move_to_pose_linear(pose, timeout_sec=timeout_sec)
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

    # ── W5: AttachedCollisionObject ───────────────────────────────────────────

    def attach_object(
        self,
        object_id: str,
        support_surface: str | None = None,
    ) -> None:
        """Notify MoveIt2 that the robot is now holding an object (W5).

        Adds a conservative cylinder attached to panda_hand so MoveIt2
        includes the held object in all subsequent collision checks.
        This is the standard MoveIt2 mechanism — identical on real robot.

        Args:
            object_id:       PDDL/Gazebo name of the grasped object.
            support_surface: Name of the MoveIt2 collision object the item was
                             resting on (e.g. "table", "shelf_b").  Added to
                             touch_links so MoveIt2 doesn't block the initial
                             lift due to ACO-surface overlap at grasp height.
                             This is the standard MoveIt2 pick pattern and
                             applies identically on the real robot.

        Phase 2: replace the fixed cylinder geometry with the actual object
        shape from the PerceptionModule (e.g. from a point cloud segment).
        """
        from moveit_msgs.msg import AttachedCollisionObject, CollisionObject
        from shape_msgs.msg import SolidPrimitive
        from geometry_msgs.msg import Pose as _Pose

        aco = AttachedCollisionObject()
        aco.link_name              = _HELD_OBJ_LINK
        aco.object.id              = object_id
        aco.object.header.frame_id = _HELD_OBJ_LINK

        cyl            = SolidPrimitive()
        cyl.type       = SolidPrimitive.CYLINDER
        cyl.dimensions = [_HELD_OBJ_HEIGHT, _HELD_OBJ_RADIUS]

        p              = _Pose()
        p.position.z   = _HELD_OBJ_Z_IN_HAND
        p.orientation.w = 1.0

        aco.object.primitives      = [cyl]
        aco.object.primitive_poses = [p]
        aco.object.operation       = CollisionObject.ADD
        # Allow physical contact with gripper and wrist links.
        # panda_link6/7/8 are wrist links that may touch the held object
        # in certain arm configurations (especially during tight grasps).
        gripper_links = [
            "panda_hand", "panda_leftfinger", "panda_rightfinger",
            "panda_link6", "panda_link7", "panda_link8",
        ]
        surface_links = [support_surface] if support_surface else []
        aco.touch_links = gripper_links + surface_links
        self._held_object_id = object_id          # instance-level (compat)
        ArmPrimitive._HELD_ACO_ID = object_id     # class-level (shared with place)
        self._aco_pub.publish(aco)
        self._node.get_logger().debug(
            f"W5 AttachedCollisionObject: '{object_id}' attached to {_HELD_OBJ_LINK}"
        )

    def detach_object(self, object_id: str | None = None) -> None:
        """Notify MoveIt2 that the robot has released the held object (W5).

        Uses class-level _HELD_ACO_ID so PlacePrimitive can detach even though
        it is a different instance from the PickPrimitive that attached the object.

        Args:
            object_id: explicit ID to detach; falls back to class-level shared state.
        """
        from moveit_msgs.msg import AttachedCollisionObject, CollisionObject

        # Resolve ID: explicit > instance > class-level shared
        oid = object_id or self._held_object_id or ArmPrimitive._HELD_ACO_ID
        if oid is None:
            return
        aco                  = AttachedCollisionObject()
        aco.link_name        = _HELD_OBJ_LINK
        aco.object.id        = oid
        aco.object.operation = CollisionObject.REMOVE
        self._aco_pub.publish(aco)

        # MoveIt2 re-adds the detached object as a world collision object at
        # the gripper position (ACO detach behaviour).  Immediately remove it
        # from the world scene so it doesn't block the retreat trajectory.
        from moveit_msgs.msg import CollisionObject as _CO
        import time as _t
        _t.sleep(0.05)   # brief pause so ACO detach is processed first
        world_rm          = _CO()
        world_rm.id       = oid
        world_rm.operation = _CO.REMOVE
        self._col_pub.publish(world_rm)

        self._held_object_id     = None
        ArmPrimitive._HELD_ACO_ID = None          # clear shared state too
        self._node.get_logger().debug(
            f"W5 AttachedCollisionObject: '{oid}' detached + world object removed"
        )

    # ── Utility ───────────────────────────────────────────────────────────────

    def _make_pre_grasp_pose(self, grasp_pose: Pose, lift_m: float = 0.12) -> Pose:
        """Return a pose `lift_m` above the grasp pose (pre-grasp approach)."""
        pre = deepcopy(grasp_pose)
        pre.position.z += lift_m
        return pre

    def _log(self, msg: str) -> None:
        self._node.get_logger().info(f"[{self.__class__.__name__}] {msg}")
