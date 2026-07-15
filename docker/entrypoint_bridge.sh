#!/bin/bash
# entrypoint_bridge.sh — Startup script for the ros1_bridge container.
#
# Usage (set via docker-compose CMD or command override):
#   dynamic  — bridge all topics whose message types exist on both sides (default)
#   static   — bridge only the topics listed in bridge_topics.yaml
#   pairs    — print matched/unmatched topic type pairs and exit (diagnostic)

set -e

source /opt/ros/noetic/setup.bash
source /opt/ros/foxy/setup.bash

MODE="${1:-dynamic}"

echo "[bridge] ROS_MASTER_URI = ${ROS_MASTER_URI}"
echo "[bridge] ROS_DOMAIN_ID  = ${ROS_DOMAIN_ID}"
echo "[bridge] Mode           = ${MODE}"

# Wait for roscore to be reachable before starting the bridge.
echo "[bridge] Waiting for roscore..."
until rostopic list > /dev/null 2>&1; do
    sleep 1
done
echo "[bridge] roscore reachable — starting bridge."

case "$MODE" in
    dynamic)
        exec ros2 run ros1_bridge dynamic_bridge --bridge-all-topics
        ;;
    static)
        PARAMS_FILE="/workspace/docker/bridge_topics.yaml"
        exec ros2 run ros1_bridge parameter_bridge \
            --ros-args --params-file "$PARAMS_FILE"
        ;;
    pairs)
        exec ros2 run ros1_bridge dynamic_bridge --print-pairs
        ;;
    *)
        echo "[bridge] Unknown mode: $MODE (use: dynamic | static | pairs)"
        exit 1
        ;;
esac
