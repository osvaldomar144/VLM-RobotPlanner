#!/usr/bin/env python3
"""
run_vlm_host.py — HOST-SIDE script: VLM inference → plan injection.

Runs on the Ubuntu host (GPU available). Performs VLM inference with
Qwen3-VL-8B-Instruct, then pipes the resulting VLMPlan JSON into the
Docker container via stdin so the ROS 2 Orchestrator can execute it.

Prerequisites:
  - .venv activated (pip install -r requirements.txt)
  - Docker container 'vlm_ros2' running with the simulation launched

Usage:
    source .venv/bin/activate
    python3 scripts/run_vlm_host.py \\
        --task "pick the red cup and place it on the shelf" \\
        --image path/to/scene.jpg

    # Dry-run: show plan without publishing (no Docker required)
    python3 scripts/run_vlm_host.py --task "..." --dry-run

    # Synthetic image (for testing without a real scene)
    python3 scripts/run_vlm_host.py --task "..." --synthetic
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))


def _make_synthetic_image():
    """Return a minimal PIL Image for testing (no real camera needed)."""
    from PIL import Image as PilImage
    import numpy as np
    rng = np.random.default_rng(42)
    arr = rng.integers(0, 255, (224, 224, 3), dtype=np.uint8)
    return PilImage.fromarray(arr)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run VLM on host and inject plan into ROS 2 container."
    )
    parser.add_argument("--task", required=True, help="Natural language task")
    parser.add_argument(
        "--image", nargs="*", default=[], metavar="PATH",
        help="Scene image file paths (can specify multiple)"
    )
    parser.add_argument(
        "--synthetic", action="store_true",
        help="Use a synthetic noise image instead of a real scene image"
    )
    parser.add_argument(
        "--no-vlm", action="store_true",
        help="Skip VLM inference; inject a hardcoded pick+place plan (for testing injection only)"
    )
    parser.add_argument(
        "--container", default="vlm_ros2",
        help="Docker container name (default: vlm_ros2)"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print JSON plan without publishing to the container"
    )
    args = parser.parse_args()

    # ── Build plan ────────────────────────────────────────────────────────────
    from vlm.planner import VLMPlan, PlanStep

    if args.no_vlm:
        # Hardcoded plan for testing the injection mechanism without GPU
        print("[INFO] --no-vlm: using hardcoded pick+place plan.")
        plan = VLMPlan(
            goal=args.task,
            steps=[
                PlanStep(primitive="pick",  args={"object": "red_cup"}),
                PlanStep(primitive="place", args={"object": "red_cup", "location": "shelf_b"}),
            ],
            raw_output="[mock — no VLM inference]",
            domain_template="manipulation_base",
        )
    else:
        # ── Load images ───────────────────────────────────────────────────────
        images = []
        if args.synthetic:
            print("[INFO] Using synthetic image.")
            images = [_make_synthetic_image()]
        elif args.image:
            from PIL import Image as PilImage
            for p in args.image:
                img = PilImage.open(p).convert("RGB")
                images.append(img)
                print(f"[INFO] Loaded image: {p} ({img.size[0]}×{img.size[1]})")
        else:
            print("[WARN] No image provided — VLM will reason without visual context.")

        # ── VLM inference ─────────────────────────────────────────────────────
        print(f"\n[INFO] Loading VLM (Qwen3-VL-8B-Instruct)…")
        from vlm.planner import VLMPlanner
        vlm = VLMPlanner()
        vlm.load()
        print("[OK]   VLM loaded.\n")

        print(f"[INFO] Running inference for: '{args.task}'")
        plan = vlm.plan(args.task, images)

    print(f"[OK]   Goal: {plan.goal}")
    print(f"       Domain template: {plan.domain_template}")
    print(f"       Steps ({len(plan.steps)}):")
    for i, step in enumerate(plan.steps, 1):
        print(f"         {i}. {step.primitive}({step.args})")

    # ── Serialize ─────────────────────────────────────────────────────────────
    payload = json.dumps({
        "command":  args.task,
        "vlm_plan": json.loads(plan.to_json()),
    })
    print(f"\n[INFO] Serialized plan: {len(payload)} chars")

    if args.dry_run:
        print("\n[DRY-RUN] Plan JSON (not published):")
        print(payload)
        return

    # ── Inject via docker exec ────────────────────────────────────────────────
    # The container has /opt/ros/humble and /workspace/ros2_ws sourced via
    # entrypoint.sh; we source them explicitly here for robustness.
    bash_cmd = (
        "source /opt/ros/humble/setup.bash && "
        "source /workspace/ros2_ws/install/setup.bash && "
        "python3 /workspace/scripts/_publish_plan.py"
    )
    cmd = ["docker", "exec", "-i", args.container, "bash", "-c", bash_cmd]

    print(f"[INFO] Injecting into container '{args.container}'…")
    result = subprocess.run(
        cmd,
        input=payload.encode(),
        capture_output=True,
    )

    if result.returncode == 0:
        output = result.stdout.decode().strip()
        print(f"[OK]   {output}")
    else:
        stderr = result.stderr.decode().strip()
        stdout = result.stdout.decode().strip()
        print(f"[FAIL] docker exec returned {result.returncode}")
        if stdout:
            print(f"       stdout: {stdout}")
        if stderr:
            print(f"       stderr: {stderr}")
        sys.exit(1)


if __name__ == "__main__":
    main()
