# ROS 2 Humble + Gazebo Classic 11 + MoveIt 2 + Franka
# Ubuntu 22.04 inside Docker — host stays on Ubuntu 20.04
# VLM weights and GPU inference run on the HOST (not in this container).
# This container handles only: ROS 2, Gazebo, MoveIt 2, robot control.

FROM osrf/ros:humble-desktop-full

# Avoid interactive prompts during apt installs
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    # Build tools
    python3-pip \
    python3-colcon-common-extensions \
    python3-vcstool \
    python3-rosdep \
    # Fast Downward build dependencies
    cmake \
    g++ \
    libglib2.0-dev \
    # MoveIt 2 (arm motion planning)
    ros-humble-moveit \
    ros-humble-moveit-ros-planning-interface \
    ros-humble-moveit-resources-panda-moveit-config \
    ros-humble-moveit-configs-utils \
    ros-humble-moveit-ros-move-group \
    # Gazebo Classic 11 + ros2_control bridge
    ros-humble-gazebo-ros-pkgs \
    ros-humble-gazebo-ros2-control \
    # ros2_control stack
    ros-humble-ros2-controllers \
    ros-humble-controller-manager \
    ros-humble-joint-state-broadcaster \
    ros-humble-joint-trajectory-controller \
    ros-humble-gripper-controllers \
    ros-humble-position-controllers \
    # moveit_resources panda description (URDF + meshes, used by moveit_config)
    ros-humble-moveit-resources-panda-description \
    # Visualization & utilities
    ros-humble-rviz2 \
    ros-humble-xacro \
    ros-humble-robot-state-publisher \
    ros-humble-joint-state-publisher-gui \
    ros-humble-tf2-ros \
    ros-humble-tf2-tools \
    # cv_bridge for camera image conversion
    ros-humble-cv-bridge \
    ros-humble-image-transport \
    # Mesa software rendering — fixes libGL/libGLX version mismatch that causes
    # Gazebo/OGRE black screen after Docker image rebuild.
    # The rebuild updates libGLX_mesa (Oct 2024) but leaves libGL.so at Jan 2022,
    # causing silent OGRE rendering failure.
    # Explicitly installing all Mesa packages together ensures consistent versions.
    libglvnd0 \
    libgl1 \
    libglx0 \
    libglx-mesa0 \
    libgl1-mesa-dri \
    libgl1-mesa-glx \
    mesa-utils \
    x11-utils \
    # Utils
    git wget vim curl \
    && rm -rf /var/lib/apt/lists/*

# Python packages needed inside the container (non-VLM)
RUN pip3 install --no-cache-dir \
    "numpy<2" \
    opencv-python-headless \
    pillow \
    transforms3d

# Build gazebo_ros2_control from source with a targeted patch that fixes the
# Humble bug where rcl_parse_arguments chokes on --param robot_description:=<xml>
# (long URDFs contain characters that break the RCL YAML arg parser).
# The patch writes robot_description to a temp YAML file and uses --params-file.
COPY docker/patch_gazebo_plugin.py /tmp/patch_gazebo_plugin.py

RUN bash -c "\
    source /opt/ros/humble/setup.bash && \
    git clone --branch humble --depth 1 \
      https://github.com/ros-controls/gazebo_ros2_control.git \
      /opt/gcr2c_ws/src/gazebo_ros2_control && \
    python3 /tmp/patch_gazebo_plugin.py && \
    cd /opt/gcr2c_ws && \
    colcon build \
      --symlink-install \
      --cmake-args -DCMAKE_BUILD_TYPE=Release \
      --packages-select gazebo_ros2_control \
      --event-handlers console_direct+"

# Boeing gazebo_model_attachment_plugin — physics-based object grasping.
# Replaces the teleport-based GazeboAttach (which oscillates due to ODE gravity
# fighting kinematic updates). Creates a real Gazebo fixed joint at grasp time.
RUN bash -c "\
    source /opt/ros/humble/setup.bash && \
    git clone --branch humble --depth 1 \
      https://github.com/Boeing/gazebo_model_attachment_plugin.git \
      /opt/boeing_ws/src/gazebo_model_attachment_plugin && \
    cd /opt/boeing_ws && \
    rosdep install --from-paths src --ignore-src -r -y && \
    colcon build \
      --symlink-install \
      --cmake-args -DCMAKE_BUILD_TYPE=Release \
      --event-handlers console_direct+"

# Fast Downward PDDL planner — build from source (takes ~2-3 min)
RUN git clone --depth 1 https://github.com/aibasel/downward.git /opt/fast-downward \
 && cd /opt/fast-downward && python3 build.py \
 && ln -sf /opt/fast-downward/fast-downward.py /usr/local/bin/fast-downward

WORKDIR /workspace

# Build ROS 2 workspace (our vlm_robot_planner + vlm_robot_planner_bringup packages).
# MoveIt2 motion planning uses pymoveit2 (pure Python, pip-installed above) —
# no C++ moveit_py compilation required.
COPY ros2_ws/ ros2_ws/
RUN bash -c "source /opt/ros/humble/setup.bash && \
    cd /workspace/ros2_ws && \
    rosdep install --from-paths src --ignore-src -r -y && \
    colcon build --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=Release"

# Copy pure-Python planning modules (accessible from container for debugging)
COPY vlm/      vlm/
COPY pddl/     pddl/
COPY planner/  planner/
COPY simulation/ simulation/
COPY scripts/  scripts/

# Auto-source ROS 2 overlays in every interactive shell (docker exec bash).
# The workspace overlay is built at container startup by entrypoint.sh, so
# guard it with [ -f ] to avoid errors if it doesn't exist yet.
RUN echo "source /opt/ros/humble/setup.bash" >> /root/.bashrc \
 && echo "[ -f /opt/gcr2c_ws/install/setup.bash ] && source /opt/gcr2c_ws/install/setup.bash" >> /root/.bashrc \
 && echo "[ -f /opt/boeing_ws/install/setup.bash ] && source /opt/boeing_ws/install/setup.bash" >> /root/.bashrc \
 && echo "[ -f /workspace/ros2_ws/install/setup.bash ] && source /workspace/ros2_ws/install/setup.bash" >> /root/.bashrc

# Entrypoint sources both ROS 2 overlays
COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]
CMD ["bash"]
