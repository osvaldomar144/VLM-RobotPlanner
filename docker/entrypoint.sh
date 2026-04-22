#!/bin/bash
set -e

source /opt/ros/humble/setup.bash
source /workspace/ros2_ws/install/setup.bash

exec "$@"
