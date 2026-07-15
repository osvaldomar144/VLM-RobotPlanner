"""
Minimal MoveIt2 Python client — wraps the /move_group action server directly.

Uses only moveit_msgs (already installed with ros-humble-moveit).
No additional pip packages required.

Public interface mirrors pymoveit2.MoveIt2 so the rest of the code
is agnostic of the underlying implementation:
  - move_to_pose(position, quat_xyzw, cartesian=False)   — non-blocking
  - move_to_configuration(joint_positions)               — non-blocking
  - wait_until_executed(timeout=60.0) → bool             — blocking
  - max_velocity, max_acceleration                       — float attributes

Requires the node to be running under MultiThreadedExecutor.
"""

from __future__ import annotations

import threading
from typing import List, Optional

from action_msgs.msg import GoalStatus
from geometry_msgs.msg import Pose
from moveit_msgs.action import ExecuteTrajectory, MoveGroup
from moveit_msgs.msg import (
    BoundingVolume,
    Constraints,
    JointConstraint,
    MotionPlanRequest,
    OrientationConstraint,
    PlanningOptions,
    PositionConstraint,
    WorkspaceParameters,
)
from moveit_msgs.srv import GetCartesianPath
from rclpy.action import ActionClient
from rclpy.callback_groups import CallbackGroup
from rclpy.node import Node
from shape_msgs.msg import SolidPrimitive

_SUCCESS = 1   # MoveItErrorCodes.SUCCESS


class MoveIt2Client:
    """
    Drop-in replacement for pymoveit2.MoveIt2.

    Communicates with move_group via the moveit_msgs/action/MoveGroup
    action server — no C++ compilation or extra pip packages needed.
    """

    def __init__(
        self,
        node:               Node,
        joint_names:        List[str],
        base_link_name:     str,
        end_effector_name:  str,
        group_name:         str,
        callback_group:     Optional[CallbackGroup] = None,
    ) -> None:
        self._node         = node
        self._joint_names  = joint_names
        self._base_link    = base_link_name
        self._eef_link     = end_effector_name
        self._group_name   = group_name

        # Motion scaling — adjust before calling move_to_*
        self.max_velocity     = 0.3
        self.max_acceleration = 0.3

        # The move_group node exposes the action under the name "move_action"
        # (resolves to /move_action in the root namespace).
        self._client = ActionClient(
            node,
            MoveGroup,
            "move_action",
            callback_group=callback_group,
        )
        self._cartesian_client = node.create_client(
            GetCartesianPath,
            "/compute_cartesian_path",
            callback_group=callback_group,
        )
        self._execute_client = ActionClient(
            node,
            ExecuteTrajectory,
            "/execute_trajectory",
            callback_group=callback_group,
        )
        self._lock        = threading.Lock()
        self._done_event  = threading.Event()
        self._last_success = False
        self._active_goal_handle = None   # for cancellation on timeout

    # ── Public API ────────────────────────────────────────────────────────────

    def move_to_pose(
        self,
        position:   List[float],
        quat_xyzw:  List[float],
        cartesian:  bool = False,
    ) -> None:
        """Start a Cartesian-goal motion (non-blocking). Uses OMPL."""
        self._send_goal_async(self._build_pose_goal(position, quat_xyzw))

    def move_to_pose_linear(
        self,
        position:  List[float],
        quat_xyzw: List[float],
    ) -> None:
        """Start a smooth Cartesian-goal motion using PILZ PTP (non-blocking).

        PILZ PTP solves IK for the target pose and interpolates in joint space,
        producing smooth, deterministic, near-straight Cartesian paths for small
        displacements (e.g. 15 cm descend/lift in pick/place).

        Note: PILZ LIN (geometrically straight line) fails with error_code=-1
        for Cartesian constraint goals — PTP is used as the reliable alternative.
        The caller (ArmPrimitive.move_to_pose_linear) falls back to OMPL if this
        also fails.
        """
        goal = self._build_pose_goal(position, quat_xyzw)
        goal.request.pipeline_id   = "pilz_industrial_motion_planner"
        goal.request.planner_id    = "PTP"
        goal.request.num_planning_attempts = 1  # PILZ is deterministic
        self._send_goal_async(goal)

    def move_to_configuration(self, joint_positions: List[float]) -> None:
        """Start a joint-space-goal motion (non-blocking)."""
        self._send_goal_async(self._build_joint_goal(joint_positions))

    def wait_until_executed(self, timeout: float = 60.0) -> bool:
        """Block until the current motion finishes. Returns True on success."""
        signalled = self._done_event.wait(timeout=timeout)
        if not signalled:
            self._node.get_logger().warn(
                "MoveIt2Client: wait_until_executed timed out."
            )
            # Cancel the active goal so the trajectory_execution_manager is
            # freed and subsequent moves can execute normally.
            gh = self._active_goal_handle
            if gh is not None:
                self._active_goal_handle = None
                try:
                    gh.cancel_goal_async()
                except Exception:
                    pass
        return self._last_success

    # ── Goal builders ─────────────────────────────────────────────────────────

    def _build_pose_goal(
        self, position: List[float], quat_xyzw: List[float]
    ) -> MoveGroup.Goal:
        request = self._base_request()

        # Tolerance sphere around target position
        sphere       = SolidPrimitive()
        sphere.type  = SolidPrimitive.SPHERE
        sphere.dimensions = [0.002]   # 2 mm radius

        centre           = Pose()
        centre.position.x = float(position[0])
        centre.position.y = float(position[1])
        centre.position.z = float(position[2])
        centre.orientation.w = 1.0

        bv                  = BoundingVolume()
        bv.primitives       = [sphere]
        bv.primitive_poses  = [centre]

        pos_c                    = PositionConstraint()
        pos_c.header.frame_id    = self._base_link
        pos_c.link_name          = self._eef_link
        pos_c.constraint_region  = bv
        pos_c.weight             = 1.0

        ori_c                            = OrientationConstraint()
        ori_c.header.frame_id            = self._base_link
        ori_c.link_name                  = self._eef_link
        ori_c.orientation.x              = float(quat_xyzw[0])
        ori_c.orientation.y              = float(quat_xyzw[1])
        ori_c.orientation.z              = float(quat_xyzw[2])
        ori_c.orientation.w              = float(quat_xyzw[3])
        ori_c.absolute_x_axis_tolerance  = 0.1
        ori_c.absolute_y_axis_tolerance  = 0.1
        ori_c.absolute_z_axis_tolerance  = 0.1
        ori_c.weight                     = 1.0

        goal_c = Constraints()
        goal_c.position_constraints    = [pos_c]
        goal_c.orientation_constraints = [ori_c]
        request.goal_constraints = [goal_c]

        return self._wrap_request(request)

    def _build_joint_goal(self, joint_positions: List[float]) -> MoveGroup.Goal:
        request = self._base_request()
        request.pipeline_id = "pilz_industrial_motion_planner"
        request.planner_id  = "PTP"

        goal_c = Constraints()
        for name, pos in zip(self._joint_names, joint_positions):
            jc                 = JointConstraint()
            jc.joint_name      = name
            jc.position        = float(pos)
            jc.tolerance_above = 0.001
            jc.tolerance_below = 0.001
            jc.weight          = 1.0
            goal_c.joint_constraints.append(jc)

        request.goal_constraints = [goal_c]
        return self._wrap_request(request)

    def _base_request(self) -> MotionPlanRequest:
        r = MotionPlanRequest()
        r.group_name                       = self._group_name
        r.num_planning_attempts            = 5
        r.allowed_planning_time            = 10.0
        r.max_velocity_scaling_factor      = float(self.max_velocity)
        r.max_acceleration_scaling_factor  = float(self.max_acceleration)
        r.start_state.is_diff              = True   # use current robot state

        ws                  = WorkspaceParameters()
        ws.header.frame_id  = self._base_link
        ws.min_corner.x     = -1.5
        ws.min_corner.y     = -1.5
        ws.min_corner.z     = -1.5
        ws.max_corner.x     =  1.5
        ws.max_corner.y     =  1.5
        ws.max_corner.z     =  1.5
        r.workspace_parameters = ws
        return r

    @staticmethod
    def _wrap_request(request: MotionPlanRequest) -> MoveGroup.Goal:
        opts              = PlanningOptions()
        opts.plan_only    = False
        opts.replan       = False
        opts.replan_attempts = 0

        goal                    = MoveGroup.Goal()
        goal.request            = request
        goal.planning_options   = opts
        return goal

    # ── Cartesian path planning + execution ───────────────────────────────────

    def move_cartesian_waypoints(
        self,
        waypoints:    List[Pose],
        max_step:     float = 0.005,
        min_fraction: float = 0.90,
    ) -> None:
        """Plan a straight Cartesian path through waypoints and execute it.

        Uses GetCartesianPath service (planning) + ExecuteTrajectory action
        (execution). If the computed fraction < min_fraction, sets
        _last_success=False immediately so the caller can fall back to
        move_to_pose_linear / OMPL.

        Non-blocking: call wait_until_executed() afterwards.
        """
        with self._lock:
            self._done_event.clear()
            self._last_success = False

        threading.Thread(
            target=self._run_cartesian,
            args=(waypoints, max_step, min_fraction),
            daemon=True,
        ).start()

    def _run_cartesian(
        self,
        waypoints:    List[Pose],
        max_step:     float,
        min_fraction: float,
    ) -> None:
        if not self._cartesian_client.wait_for_service(timeout_sec=5.0):
            self._node.get_logger().error(
                "MoveIt2Client: /compute_cartesian_path service not available."
            )
            self._done_event.set()
            return

        req = GetCartesianPath.Request()
        req.header.frame_id      = self._base_link
        req.start_state.is_diff  = True
        req.group_name           = self._group_name
        req.link_name            = self._eef_link
        req.waypoints            = list(waypoints)
        req.max_step             = float(max_step)
        # 5.0 rad: rejects path segments where a joint jumps > 5x the average
        # step. With TRAC-IK Distance solver this rarely fires in practice, but
        # catches any residual IK flips before they reach ExecuteTrajectory.
        req.jump_threshold       = 5.0
        req.avoid_collisions     = True
        # Note: max_velocity/acceleration_scaling_factor not available in
        # GetCartesianPath.Request on ROS2 Humble — velocity is applied
        # by the ExecuteTrajectory action using the existing scaling factors.

        svc_done   = threading.Event()
        svc_result = [None]

        def _on_svc(future):
            svc_result[0] = future.result()
            svc_done.set()

        self._cartesian_client.call_async(req).add_done_callback(_on_svc)
        if not svc_done.wait(timeout=15.0):
            self._node.get_logger().warn(
                "MoveIt2Client: GetCartesianPath timed out."
            )
            self._done_event.set()
            return

        response = svc_result[0]
        fraction  = response.fraction if response else 0.0
        if response is None or fraction < min_fraction:
            self._node.get_logger().warn(
                f"MoveIt2Client: Cartesian path {fraction:.0%} "
                f"(need ≥{min_fraction:.0%}) — caller should fall back."
            )
            self._done_event.set()
            return

        self._node.get_logger().info(
            f"MoveIt2Client: Cartesian path {fraction:.0%} — executing."
        )

        if not self._execute_client.wait_for_server(timeout_sec=5.0):
            self._node.get_logger().error(
                "MoveIt2Client: /execute_trajectory action not available."
            )
            self._done_event.set()
            return

        exec_goal = ExecuteTrajectory.Goal()
        exec_goal.trajectory = response.solution
        self._execute_client.send_goal_async(exec_goal).add_done_callback(
            self._on_exec_goal
        )

    def _on_exec_goal(self, future) -> None:
        gh = future.result()
        if gh is None or not gh.accepted:
            self._node.get_logger().warn(
                "MoveIt2Client: ExecuteTrajectory goal rejected."
            )
            self._done_event.set()
            return
        self._active_goal_handle = gh
        gh.get_result_async().add_done_callback(self._on_exec_result)

    def _on_exec_result(self, future) -> None:
        self._active_goal_handle = None
        response = future.result()
        if response is None:
            self._node.get_logger().warn(
                "MoveIt2Client: ExecuteTrajectory null result."
            )
        else:
            code = response.result.error_code.val
            self._last_success = (
                response.status == GoalStatus.STATUS_SUCCEEDED
                and code == _SUCCESS
            )
            if not self._last_success:
                self._node.get_logger().warn(
                    f"MoveIt2Client: ExecuteTrajectory failed — "
                    f"status={response.status}, code={code}"
                )
        self._done_event.set()

    # ── Async dispatch ────────────────────────────────────────────────────────

    def _send_goal_async(self, goal: MoveGroup.Goal) -> None:
        with self._lock:
            self._done_event.clear()
            self._last_success = False

        if not self._client.wait_for_server(timeout_sec=10.0):
            self._node.get_logger().error(
                "MoveIt2Client: /move_action action server not available "
                "(is move_group running?)."
            )
            self._done_event.set()
            return

        self._client.send_goal_async(goal).add_done_callback(
            self._on_goal_response
        )

    def _on_goal_response(self, future) -> None:
        gh = future.result()
        if gh is None or not gh.accepted:
            self._node.get_logger().warn(
                "MoveIt2Client: goal rejected by move_group."
            )
            self._done_event.set()
            return
        self._active_goal_handle = gh
        gh.get_result_async().add_done_callback(self._on_result)

    def _on_result(self, future) -> None:
        self._active_goal_handle = None
        response = future.result()
        if response is None:
            self._node.get_logger().warn(
                "MoveIt2Client: null result from move_group."
            )
        else:
            code = response.result.error_code.val
            self._last_success = (
                response.status == GoalStatus.STATUS_SUCCEEDED
                and code == _SUCCESS
            )
            if not self._last_success:
                self._node.get_logger().warn(
                    f"MoveIt2Client: motion failed — "
                    f"status={response.status}, error_code={code}"
                )
        self._done_event.set()
