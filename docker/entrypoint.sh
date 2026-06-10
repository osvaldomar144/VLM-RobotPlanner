#!/bin/bash
set -e

source /opt/ros/humble/setup.bash

# Overlay source-built gazebo_ros2_control (fixes long-URDF bug in Humble apt package)
if [ -f /opt/gcr2c_ws/install/setup.bash ]; then
    source /opt/gcr2c_ws/install/setup.bash
fi

# Boeing gazebo_model_attachment_plugin (physics-based object grasping)
if [ -f /opt/boeing_ws/install/setup.bash ]; then
    source /opt/boeing_ws/install/setup.bash
fi

# Rebuild the Python package so entry point scripts are always current.
# ament_python packages have no C++ — this takes ~2s.
if [ -d /workspace/ros2_ws/src ]; then
    cd /workspace/ros2_ws
    colcon build \
        --symlink-install \
        --packages-select vlm_robot_planner vlm_robot_planner_bringup \
        --event-handlers console_direct- \
        2>&1 | grep -E '(Summary|ERROR|error:)' || true

    # colcon-ros ament_python puts console_scripts in bin/ via pip editable install,
    # but ros2 launch's executable finder looks in lib/<pkg_name>/.
    # Create the missing symlink manually.
    BIN=/workspace/ros2_ws/install/vlm_robot_planner/bin/orchestrator
    LIBEXEC=/workspace/ros2_ws/install/vlm_robot_planner/lib/vlm_robot_planner/orchestrator
    if [ -f "$BIN" ] && [ ! -e "$LIBEXEC" ]; then
        mkdir -p "$(dirname "$LIBEXEC")"
        ln -s "$BIN" "$LIBEXEC"
    fi

    cd /
fi

# Source workspace overlay
if [ -f /workspace/ros2_ws/install/setup.bash ]; then
    source /workspace/ros2_ws/install/setup.bash
fi

exec "$@"
