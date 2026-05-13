# VLM-RobotPlanner — Project Context for Claude Code

## What this is

Master's thesis project. Goal: use a Vision-Language Model (VLM) to plan robot manipulation tasks from images + natural language commands, applied to a Franka Emika Panda arm mounted on an external mobile base. Fully open-source stack.

Reference paper: "Look Before You Leap: Unveiling the Power of GPT-4V in Robotic Vision-Language Planning".

---

## Hardware

**Linux lab PC** (primary development and execution machine):
- OS: Ubuntu 20.04 LTS
- GPU: NVIDIA GeForce RTX 3090 Ti (24 GB VRAM)
- CPU: Intel Xeon E5-2620 v4 @ 2.10GHz × 16 cores
- RAM: 62.7 GiB

**Robot**: Franka Emika Panda arm (fixed) + external mobile base.

---

## Architecture

```
User command (text) + Scene image(s)
        │
        ▼
VLMPlanner  [vlm/planner.py]
└─ Qwen2.5-VL-7B-Instruct reasons over images → outputs JSON plan
        │
        ▼
ProblemGenerator  [planner/problem_generator.py]
└─ Generates PDDL problem dynamically from VLMPlan
        │
        ▼
FastDownwardPlanner  [planner/fast_downward.py]
└─ Validates plan symbolically, returns verified action sequence
        │
        ▼
parse_plan()  [planner/plan_parser.py]
└─ Converts PDDL strings → PrimitiveCall list
        │
        ▼
Primitives (pick, place, ...)  [ros2_ws/src/vlm_robot_planner/primitives/]
└─ MoveIt 2 action clients (stubs — not yet implemented)
```

The **oracle** (`simulation/oracle/world_state.py`) provides ground-truth 3D poses from Gazebo **only at execution time** — it is NOT fed to the VLM. The VLM reasons from images directly.

---

## Development phases

| Phase | Focus | Status |
|-------|-------|--------|
| 1 | VLM → PDDL → arm primitives in simulation (arm stationary, no base) | **In progress** |
| 2 | Add real perception module (camera → object detection) | Not started |
| 3 | Add mobile base navigation | Not started |
| 4 | Sim-to-real on real Franka | Not started |

In Phase 1: no mobile base, no perception module. The VLM receives images directly and acts as both perception and planner. The oracle provides 3D poses for primitive execution.

---

## Tech stack (all open-source)

| Layer | Tool |
|-------|------|
| VLM | Qwen2.5-VL-7B-Instruct (HuggingFace) |
| Symbolic planner | Fast Downward (PDDL) |
| Motion planning | MoveIt 2 |
| Simulator | Gazebo Fortress |
| ROS framework | ROS 2 Humble (via Docker on Ubuntu 20.04 host) |
| Python deps | See `requirements.txt` |

**ROS 2 note**: ROS 2 Humble requires Ubuntu 22.04. Use Docker (`docker/`) to run it on the Ubuntu 20.04 host. The VLM/planning layer (pure Python) can run directly on the host with a venv.

---

## Repository structure

```
vlm/                    # VLM module — pure Python, no ROS
  planner.py            # VLMPlanner class: images + command → VLMPlan
  model_loader.py       # Loads Qwen2.5-VL from HuggingFace
  prompts/              # System prompt (constrained JSON output format)

pddl/
  domain/manipulation.pddl   # PDDL domain: pick, place, predicates
  problems/                  # Static example problems

planner/                # Pure Python, no ROS
  fast_downward.py      # Subprocess wrapper for Fast Downward
  plan_parser.py        # PDDL action strings → PrimitiveCall
  problem_generator.py  # VLMPlan → PDDL problem file (dynamic)

simulation/
  oracle/world_state.py # WorldState + GazeboOracle stub

ros2_ws/src/
  vlm_robot_planner/    # Main ROS 2 package
    orchestrator.py     # ROS 2 node: coordinates full pipeline
    primitives/         # pick.py, place.py — MoveIt stubs

scripts/
  run_pipeline.py           # Standalone pipeline test (no ROS needed)
  test_vlm_inference.py     # Full GPU inference test (run this first on Linux)

tests/                  # 32 unit tests, all pass without GPU or ROS
  conftest.py           # Synthetic image fixtures
  test_vlm_parser.py
  test_vlm_image_input.py
  test_problem_generator.py
  test_pddl.py
  test_world_state.py

docker/
  Dockerfile            # ROS 2 Humble + MoveIt + Gazebo
  docker-compose.yml
```

---

## Key design decisions

1. **VLM input = images only** — no text description of the scene is passed to the VLM. It reasons from images directly, matching the real deployment scenario.

2. **Separation of AI layer and ROS layer** — everything in `vlm/`, `planner/`, `simulation/` is pure Python and testable without ROS. Only `ros2_ws/` requires ROS.

3. **Dynamic PDDL generation** — the PDDL problem file is generated at runtime from the VLM's output, not written by hand. This is what `planner/problem_generator.py` does.

4. **Small primitive set** — intentionally limited to 5 primitives (pick, place, open_gripper, close_gripper, look_at) to stress-test VLM high-level reasoning before expanding.

5. **Oracle is not VLM input** — `WorldState` provides 3D poses to MoveIt at execution time only.

---

## Setup on this machine (Ubuntu 20.04 + RTX 3090 Ti)

### Python environment (VLM + planning layer)
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Run all unit tests (no GPU needed)
```bash
source .venv/bin/activate
python -m pytest tests/ -v
```

### Test VLM inference end-to-end (GPU required — run this first)
```bash
source .venv/bin/activate
python scripts/test_vlm_inference.py --synthetic
```

### Run full pipeline dry-run (no GPU, no ROS)
```bash
source .venv/bin/activate
python scripts/run_pipeline.py \
  --task "pick the red cup and place it on the shelf" \
  --images tests/images/scene.jpg \
  --dry-run --skip-pddl
```

### ROS 2 + Gazebo (via Docker)
```bash
cd docker/
docker compose up -d
```

---

## Current implementation status

| Component | Status | Notes |
|-----------|--------|-------|
| `VLMPlanner._to_pil()` | Done + tested | |
| `VLMPlanner._build_messages()` | Done + tested | |
| `VLMPlanner._run_inference()` | Done | Not yet tested on GPU |
| `VLMPlanner._parse_output()` | Done + tested | |
| `ProblemGenerator` | Done + tested | |
| `FastDownwardPlanner` | Done | Needs fast-downward installed |
| `plan_parser` | Done + tested | |
| `GazeboOracle` | Stub | Needs ROS 2 + Gazebo |
| `PickPrimitive` / `PlacePrimitive` | Stub | Needs MoveIt 2 |
| `Orchestrator` (ROS 2 node) | Partial | Wiring done, primitives not impl. |
| Gazebo simulation (Franka) | Not started | Next major step |

---

## Immediate next steps

1. `python scripts/test_vlm_inference.py --synthetic` — validate GPU inference works
2. Set up Gazebo simulation with Franka Panda URDF + a tabletop scene
3. Implement `GazeboOracle.get_world_state()` via ROS 2 service call
4. Implement `PickPrimitive.execute()` and `PlacePrimitive.execute()` with MoveIt 2
5. Wire full pipeline: real image from Gazebo camera → VLM → PDDL → MoveIt
