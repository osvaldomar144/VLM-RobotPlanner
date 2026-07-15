#!/bin/bash
# start_bridge.sh — Build and start the ROS 1 ↔ ROS 2 bridge container.
#
# Usage:
#   bin/start_bridge.sh --robot-ip <IP>           # dynamic bridge (default)
#   bin/start_bridge.sh --robot-ip <IP> --mode static
#   bin/start_bridge.sh --robot-ip <IP> --mode pairs   # diagnostic only
#
# The bridge connects the real Franka robot (ROS 1 Noetic on the robot PC)
# to the ROS 2 Humble planning stack running on this machine.

set -e

ROBOT_IP=""
BRIDGE_MODE="dynamic"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --robot-ip)
            ROBOT_IP="$2"
            shift 2
            ;;
        --mode)
            BRIDGE_MODE="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 --robot-ip <IP> [--mode dynamic|static|pairs]"
            exit 0
            ;;
        *)
            echo "Unknown argument: $1"
            echo "Usage: $0 --robot-ip <IP> [--mode dynamic|static|pairs]"
            exit 1
            ;;
    esac
done

if [[ -z "$ROBOT_IP" ]]; then
    echo "Error: --robot-ip is required."
    echo "Usage: $0 --robot-ip <IP> [--mode dynamic|static|pairs]"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "Robot IP      : $ROBOT_IP"
echo "Bridge mode   : $BRIDGE_MODE"
echo "ROS_MASTER_URI: http://${ROBOT_IP}:11311"

export ROS_MASTER_URI="http://${ROBOT_IP}:11311"
export ROS_IP="$(hostname -I | awk '{print $1}')"
export ROS_DOMAIN_ID=42

cd "$REPO_ROOT"
docker compose --profile real build ros1_bridge
docker compose --profile real run --rm ros1_bridge "$BRIDGE_MODE"
