# VLM-RobotPlanner

Master's thesis project: Vision-Language Model based task planning for a Franka Emika Panda arm mounted on a mobile base.

Extends "Look Before You Leap: Unveiling the Power of GPT-4V in Robotic Vision-Language Planning" using fully open-source tools.

## Architecture

```
User command + images
        │
        ▼
   VLM Module (Qwen2.5-VL)
   └─ generates structured high-level plan
        │
        ▼
   Symbolic Planner (PDDL + Fast Downward)
   └─ validates & sequences primitives
        │
        ▼
   Primitives Layer (ROS 2 / MoveIt)
   └─ pick, place, open_gripper, ...
        │
        ▼
   Franka Panda (simulation → real)
```

## Repository Structure

```
├── vlm/            # VLM inference module (no ROS dependency)
├── pddl/           # PDDL domain and problem files
├── planner/        # Symbolic planner interface (Fast Downward wrapper)
├── simulation/     # Gazebo worlds + oracle world state
├── ros2_ws/        # ROS 2 workspace (MoveIt, primitives, Franka sim)
├── tests/          # Unit and integration tests
├── scripts/        # Entry points and utilities
├── config/         # Robot URDF/SRDF and scene configs
└── docker/         # Docker setup for reproducible environment
```

## Phases

| Phase | Focus | Perception | Mobile Base |
|-------|-------|-----------|-------------|
| 1 | VLM → PDDL → arm primitives (sim) | Oracle (ground truth) | No |
| 2 | Add real perception module | Camera images → detection | No |
| 3 | Add mobile base navigation | Perception | Yes |
| 4 | Sim-to-real transfer | Real camera | Yes |

## Stack

- **VLM**: Qwen2.5-VL-7B (open-source)
- **Symbolic planner**: Fast Downward (PDDL)
- **Motion planning**: MoveIt 2
- **Navigation**: Nav2 (Phase 3+)
- **Perception**: Grounded SAM 2 + OWL-ViT v2 (Phase 2+)
- **Simulator**: Gazebo Fortress (via Docker)
- **Framework**: ROS 2 Humble

## Getting Started

```bash
# Build and start the Docker environment
cd docker/
docker compose up -d

# Run the pipeline (Phase 1 — oracle mode)
python scripts/run_pipeline.py --task "pick the red cup and place it on the shelf" --oracle
```
