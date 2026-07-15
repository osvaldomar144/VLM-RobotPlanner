#!/bin/bash
# start_real.sh — Launch the real-robot execution environment.
#
# This script starts the full stack for operating the real Franka Panda:
#   1. ROS 1 ↔ ROS 2 bridge (reads joint states and camera from the robot,
#      forwards MoveIt 2 trajectory goals back to franka_ros)
#   2. MoveIt 2 for the real robot (no Gazebo — real_robot.launch.py)
#
# Usage:
#   bin/start_real.sh --robot-ip <ROBOT_PC_IP>
#
# Prerequisites:
#   - Robot PC is running: roscore + franka_ros + RealSense driver
#   - Both machines are on the same LAN; ICMP (ping) works
#   - Docker image is built: docker compose build ros2

set -e

ROBOT_IP=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --robot-ip)
            ROBOT_IP="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 --robot-ip <IP>"
            exit 0
            ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

if [[ -z "$ROBOT_IP" ]]; then
    echo "Error: --robot-ip is required."
    echo "Usage: $0 --robot-ip <IP>"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "=== Real robot stack ==="
echo "Robot IP: $ROBOT_IP"
echo ""

# Step 1: start the bridge in the background.
echo "[1/2] Starting ROS 1 ↔ ROS 2 bridge..."
"$SCRIPT_DIR/start_bridge.sh" --robot-ip "$ROBOT_IP" &
BRIDGE_PID=$!

# Give the bridge a few seconds to connect before launching MoveIt 2.
sleep 5

# Step 2: launch MoveIt 2 in the ros2 container (real robot config, no Gazebo).
echo "[2/2] Launching MoveIt 2 for real robot..."
cd "$REPO_ROOT"
docker compose run --rm ros2 \
    ros2 launch vlm_robot_planner_bringup real_robot.launch.py \
        robot_ip:="$ROBOT_IP"

# If MoveIt 2 exits, also stop the bridge.
kill "$BRIDGE_PID" 2>/dev/null || true
