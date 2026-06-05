#!/bin/bash
# start_sim.sh — Avvia il container Docker e la simulazione Gazebo + MoveIt2.
# Uso: ./start_sim.sh [rviz:=true]

set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RVIZ_ARG="${1:-rviz:=false}"

echo "=== VLM-RobotPlanner: avvio simulazione ==="

# 1. Permessi X11 per il container
xhost +local: > /dev/null 2>&1

# 2. Avvia (o riavvia) il container
cd "$REPO_ROOT/docker"
docker compose up -d

# 3. Lancia simulazione — rimane in foreground (Ctrl+C per fermare)
echo "Lancio Gazebo + MoveIt2 + orchestratore... (Ctrl+C per fermare)"
echo ""
exec docker exec -it vlm_ros2 bash -c \
  "source /opt/ros/humble/setup.bash && \
   source /workspace/ros2_ws/install/setup.bash && \
   ros2 launch vlm_robot_planner_bringup simulation.launch.py ${RVIZ_ARG}"
