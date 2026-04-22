# Base image: ROS 2 Humble + Gazebo + MoveIt 2
# Ubuntu 22.04 inside Docker, independent of host Ubuntu 20.04

FROM osrf/ros:humble-desktop-full

# System dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-colcon-common-extensions \
    ros-humble-moveit \
    ros-humble-gazebo-ros-pkgs \
    ros-humble-franka-description \
    ros-humble-rviz2 \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies (non-ROS)
COPY requirements.txt /tmp/requirements.txt
RUN pip3 install --no-cache-dir -r /tmp/requirements.txt

# ROS 2 workspace
WORKDIR /workspace
COPY ros2_ws/ ros2_ws/
RUN bash -c "source /opt/ros/humble/setup.bash && \
    cd ros2_ws && \
    colcon build --symlink-install"

# Copy project source (non-ROS modules)
COPY vlm/ vlm/
COPY pddl/ pddl/
COPY planner/ planner/
COPY simulation/ simulation/
COPY scripts/ scripts/
COPY config/ config/

# Entrypoint
COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]
CMD ["bash"]
