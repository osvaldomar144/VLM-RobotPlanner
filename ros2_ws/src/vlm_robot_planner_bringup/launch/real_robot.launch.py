"""
real_robot.launch.py — Bringup for the real Franka Panda (no Gazebo).

Starts:
  1. robot_state_publisher   (reads joint states from /joint_states bridged from ROS 1)
  2. Static TF: world → panda_link0
  3. MoveIt 2 move_group     (motion planning server — real time, no sim_time)
  4. RViz2                   (optional, set rviz:=false to skip)
  5. Orchestrator node       (VLM→PDDL→primitive dispatch)

Prerequisites (must be running before this launch file):
  - Robot PC: roscore + franka_ros + RealSense driver
  - ROS 1 ↔ ROS 2 bridge: bin/start_bridge.sh --robot-ip <IP>
    (bridges /joint_states, /camera/*, franka_state_controller/franka_states)

Usage:
  ros2 launch vlm_robot_planner_bringup real_robot.launch.py robot_ip:=192.168.1.100
  ros2 launch vlm_robot_planner_bringup real_robot.launch.py robot_ip:=192.168.1.100 rviz:=false
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, TimerAction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch.substitutions import Command, FindExecutable
from moveit_configs_utils import MoveItConfigsBuilder
from ament_index_python.packages import get_package_share_path


# Auto-load overview camera config saved by scripts/setup_overview_camera.py.
# The path resolves to <repo_root>/data/overview_camera_setup.json whether the
# launch file runs in Docker (/workspace) or directly on the host.
_CAM_CFG_PATH = (
    Path(__file__).resolve()
    .parent.parent.parent.parent.parent  # repo root
    / "data" / "overview_camera_setup.json"
)
_cam_cfg: dict = {}
if _CAM_CFG_PATH.exists():
    try:
        with open(_CAM_CFG_PATH) as _f:
            _cam_cfg = json.load(_f)
        print(f"[launch] overview camera config: {_CAM_CFG_PATH.name}")
    except Exception as _e:
        print(f"[launch] WARNING: could not read {_CAM_CFG_PATH.name}: {_e}")


def generate_launch_description() -> LaunchDescription:

    # ── Arguments ────────────────────────────────────────────────────────────
    robot_ip_arg = DeclareLaunchArgument(
        "robot_ip",
        description="IP address of the robot PC (needed by franka_ros on the robot side).",
    )
    rviz_arg = DeclareLaunchArgument(
        "rviz",
        default_value="true",
        description="Launch RViz2 for visualization",
    )
    # Overview camera position relative to panda_link0.
    # Default values come from data/overview_camera_setup.json (written by
    # scripts/setup_overview_camera.py). Fall back to hardcoded estimates if
    # the file does not exist yet.
    overview_x_arg     = DeclareLaunchArgument(
        "overview_x",     default_value=str(_cam_cfg.get("x",     0.65)))
    overview_y_arg     = DeclareLaunchArgument(
        "overview_y",     default_value=str(_cam_cfg.get("y",     0.70)))
    overview_z_arg     = DeclareLaunchArgument(
        "overview_z",     default_value=str(_cam_cfg.get("z",     0.73)))
    overview_roll_arg  = DeclareLaunchArgument(
        "overview_roll",  default_value=str(_cam_cfg.get("roll",  0.0)))
    overview_pitch_arg = DeclareLaunchArgument(
        "overview_pitch", default_value=str(_cam_cfg.get("pitch", 0.68)))
    overview_yaw_arg   = DeclareLaunchArgument(
        "overview_yaw",   default_value=str(_cam_cfg.get("yaw",   -2.19)))

    # ── Paths ─────────────────────────────────────────────────────────────────
    bringup_share = get_package_share_directory("vlm_robot_planner_bringup")
    urdf_path     = os.path.join(bringup_share, "urdf",   "panda_real.urdf.xacro")
    rviz_path     = os.path.join(bringup_share, "config", "moveit.rviz")

    # Fall back to sim URDF if a real-specific one isn't available yet.
    if not os.path.exists(urdf_path):
        urdf_path = os.path.join(bringup_share, "urdf", "panda_sim.urdf.xacro")

    # ── Robot description ─────────────────────────────────────────────────────
    robot_description_content = ParameterValue(
        Command([FindExecutable(name="xacro"), " ", urdf_path]),
        value_type=str,
    )
    robot_description = {"robot_description": robot_description_content}

    # ── MoveIt 2 configuration (real robot — use_sim_time = False) ────────────
    moveit_config = (
        MoveItConfigsBuilder(
            robot_name="panda",
            package_name="moveit_resources_panda_moveit_config",
        )
        .robot_description(file_path=urdf_path)
        .robot_description_semantic(
            file_path=str(
                get_package_share_path("moveit_resources_panda_moveit_config")
                / "config" / "panda.srdf"
            )
        )
        .robot_description_kinematics(
            file_path=os.path.join(bringup_share, "config", "kinematics.yaml")
        )
        .to_moveit_configs()
    )

    moveit_params = moveit_config.to_dict()
    # Real robot: do NOT set use_sim_time — wall clock is required.
    moveit_params["use_sim_time"] = False

    # ── 1. robot_state_publisher ──────────────────────────────────────────────
    robot_state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        output="screen",
        parameters=[robot_description, {"use_sim_time": False}],
    )

    # ── 2. Static TF: world → panda_link0 ────────────────────────────────────
    # Adjust translation to match the real robot's base position on the table.
    static_tf = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="static_tf_world_to_panda",
        arguments=["0.20", "0", "0.77", "0", "0", "0", "world", "panda_link0"],
        output="screen",
        parameters=[{"use_sim_time": False}],
    )

    # ── 2b. Static TF: panda_link0 → overview_camera_optical_frame ───────────
    # Position measured physically when mounting the overview RealSense D435i.
    # Pass the measured values via launch arguments (see declarations above).
    # _capture_scene.py reads this TF and saves data/overview_camera_pose.json
    # automatically — no manual JSON editing needed.
    static_tf_overview = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="static_tf_panda_to_overview_cam",
        arguments=[
            LaunchConfiguration("overview_x"),
            LaunchConfiguration("overview_y"),
            LaunchConfiguration("overview_z"),
            LaunchConfiguration("overview_roll"),
            LaunchConfiguration("overview_pitch"),
            LaunchConfiguration("overview_yaw"),
            "panda_link0",
            "overview_camera_optical_frame",
        ],
        output="screen",
        parameters=[{"use_sim_time": False}],
    )

    # ── 3. MoveIt 2 move_group ────────────────────────────────────────────────
    move_group = Node(
        package="moveit_ros_move_group",
        executable="move_group",
        output="screen",
        parameters=[moveit_params],
    )

    # ── 4. RViz2 (optional) ───────────────────────────────────────────────────
    rviz2 = Node(
        package="rviz2",
        executable="rviz2",
        output="screen",
        arguments=["-d", rviz_path] if os.path.exists(rviz_path) else [],
        parameters=[moveit_params],
        condition=IfCondition(LaunchConfiguration("rviz")),
    )

    # ── 5. Orchestrator node ──────────────────────────────────────────────────
    orchestrator = TimerAction(
        period=8.0,
        actions=[
            Node(
                package="vlm_robot_planner",
                executable="orchestrator",
                output="screen",
                parameters=[moveit_params],
            )
        ],
    )

    return LaunchDescription([
        robot_ip_arg,
        rviz_arg,
        overview_x_arg, overview_y_arg, overview_z_arg,
        overview_roll_arg, overview_pitch_arg, overview_yaw_arg,
        robot_state_publisher,
        static_tf,
        static_tf_overview,
        move_group,
        rviz2,
        orchestrator,
    ])
