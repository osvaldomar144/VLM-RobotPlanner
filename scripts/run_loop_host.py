#!/usr/bin/env python3
"""
run_loop_host.py — Closed-loop task execution (HOST side).

Implements the closed-loop architecture:
  [scan] -> [capture] -> [VLM next step] -> [inject] -> [wait complete] -> repeat

Each iteration:
  1. Pre-scan: move arm to scan pose via _pre_scan.py (wrist camera view)
  2. Capture: take image from wrist camera via _capture_scene.py
  3. VLM: plan_next_step(task, image, completed_steps) -> single action
  4. Ground: OWL-ViT or bbox grounding -> correct names
  5. Inject: send single-step plan to orchestrator
  6. Wait: _wait_step_complete.py -> get completion signal
  7. If complete: break; else: add step to completed_steps, repeat

Sim-to-real note: the same loop works on the real robot — the only difference
is that Gazebo oracle is replaced by RealSense depth in the PerceptionModule.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))


def _docker(container: str, use_sudo: bool) -> list[str]:
    return (["sudo", "docker"] if use_sudo else ["docker"]) + ["exec", "-i", container]


def _run_in_container(args, bash_cmd: str, timeout: int = 30) -> subprocess.CompletedProcess:
    return subprocess.run(
        _docker(args.container, args.sudo_docker) + ["bash", "-c", bash_cmd],
        capture_output=True, timeout=timeout,
    )


def _pre_scan(args) -> bool:
    """Move arm to scan pose before capture."""
    print("[LOOP] Pre-scan: moving arm to scan pose...")
    bash_cmd = (
        "source /opt/ros/humble/setup.bash && "
        "source /workspace/ros2_ws/install/setup.bash && "
        "python3 /workspace/scripts/_pre_scan.py"
    )
    r = _run_in_container(args, bash_cmd, timeout=40)
    output = r.stdout.decode().strip()
    if output:
        for line in output.splitlines():
            print(f"       {line}")
    if r.returncode == 0:
        print("[OK]   Scan pose reached.")
        return True
    err = r.stderr.decode().strip()
    if err:
        print(f"[WARN] Pre-scan: {err}")
    print("[WARN] Pre-scan failed — continuing anyway (will use fallback camera)")
    return False


def _capture(args) -> Path | None:
    """Capture image from wrist camera."""
    scene_path = _REPO_ROOT / "data" / "scene.png"
    bash_cmd = (
        "source /opt/ros/humble/setup.bash && "
        "source /workspace/ros2_ws/install/setup.bash && "
        "python3 /workspace/scripts/_capture_scene.py"
    )
    r = _run_in_container(args, bash_cmd, timeout=15)
    output = r.stdout.decode().strip()
    if r.returncode != 0:
        print(f"[FAIL] Capture failed: {r.stderr.decode().strip()}")
        return None
    # Print capture output (includes which camera topic was used)
    for line in output.splitlines():
        print(f"       {line}")
    return scene_path


def _get_gazebo_models(args) -> dict:
    """Get Gazebo scene objects and their positions."""
    bash_cmd = (
        "source /opt/ros/humble/setup.bash && "
        "source /workspace/ros2_ws/install/setup.bash && "
        "python3 /workspace/scripts/_get_model_states.py"
    )
    r = _run_in_container(args, bash_cmd, timeout=10)
    if r.returncode == 0:
        try:
            return json.loads(r.stdout.decode().strip()).get("models", {})
        except Exception:
            pass
    return {}


def _wait_step_complete(args, timeout: int = 60) -> dict:
    """Wait for step completion signal from orchestrator."""
    bash_cmd = (
        "source /opt/ros/humble/setup.bash && "
        "source /workspace/ros2_ws/install/setup.bash && "
        f"python3 /workspace/scripts/_wait_step_complete.py --timeout {timeout}"
    )
    r = _run_in_container(args, bash_cmd, timeout=timeout + 5)
    if r.returncode == 0:
        try:
            return json.loads(r.stdout.decode().strip())
        except Exception:
            pass
    return {"success": False, "task_complete": False}


def main() -> None:
    parser = argparse.ArgumentParser(description="Closed-loop task execution")
    parser.add_argument("--task",       required=True)
    parser.add_argument("--max-steps",  type=int, default=10)
    parser.add_argument("--container",  default="vlm_ros2")
    parser.add_argument("--sudo-docker", action="store_true")
    args = parser.parse_args()

    # Load VLM once
    print("[LOOP] Loading VLM (Qwen3-VL-8B-Instruct)…")
    from vlm.planner import VLMPlanner
    from vlm.perception import PerceptionModule
    from PIL import Image as PilImage

    vlm       = VLMPlanner()
    vlm.load()
    perception = PerceptionModule()
    perception.load()
    print("[OK]   VLM + PerceptionModule loaded.\n")

    completed_steps: list[str] = []
    docker_cmd = _docker(args.container, args.sudo_docker)

    for iteration in range(args.max_steps):
        print(f"\n{'─'*60}")
        print(f"  ITERAZIONE {iteration+1} / {args.max_steps}")
        print(f"  Completati: {completed_steps or ['(nessuno)']}")
        print(f"{'─'*60}")

        # 1. Pre-scan — only when gripper is empty.
        # If holding an object, scan pose movement prevents place from succeeding
        # (MoveIt2 can't plan from scan+held_object to pre-place position).
        last_pick  = max((i for i,s in enumerate(completed_steps) if s.startswith("pick")),  default=-1)
        last_place = max((i for i,s in enumerate(completed_steps) if s.startswith("place") or s.startswith("stack")), default=-1)
        holding = last_pick > last_place

        if not holding:
            _pre_scan(args)
            time.sleep(1.0)
        else:
            print("[LOOP] Holding object — skip scan pose, capture from current arm position")

        # 2. Capture
        image_path = _capture(args)
        if image_path is None:
            print("[FAIL] No image — aborting loop")
            break
        image = PilImage.open(image_path).convert("RGB")

        # Save iteration snapshot
        iter_path = _REPO_ROOT / "data" / f"loop_iter_{iteration+1:02d}.png"
        image.save(str(iter_path))
        print(f"[LOOP] Snapshot: {iter_path.name}")

        # 3. Get Gazebo models
        gazebo_poses = _get_gazebo_models(args)
        gazebo_models = list(gazebo_poses.keys())
        print(f"[LOOP] Scene objects: {gazebo_models}")

        # 4. VLM: plan next single step (measure inference time)
        # Strip arrow notation from completed_steps before passing to VLM
        # so it doesn't echo back "cube->red_cup" and cause double-arrows.
        vlm_context = [s.split("->")[-1].rstrip(")") + ")" if "->" in s else s
                       for s in completed_steps
                       if not s.startswith("skip_")]
        print(f"[LOOP] VLM planning next step for: '{args.task}'")
        t_vlm = time.time()
        plan = vlm.plan_next_step(args.task, [image], vlm_context)
        vlm_time = time.time() - t_vlm
        print(f"[LOOP] VLM inference    : {vlm_time:.1f}s")

        # ── VLM plan summary ──────────────────────────────────────────────
        print(f"[LOOP] Domain template  : {plan.domain_template}")

        # Show domain enrichment if the VLM added anything beyond the base template
        da = plan.domain_additions
        enriched = (da.get("new_predicates") or da.get("new_actions") or
                    da.get("new_types") or da.get("modified_preconditions"))
        if enriched:
            print(f"[LOOP] ⚡ DOMAIN ENRICHMENT:")
            if da.get("new_types"):
                print(f"         new_types      : {da['new_types']}")
            if da.get("new_predicates"):
                print(f"         new_predicates : {da['new_predicates']}")
            if da.get("new_actions"):
                for a in da["new_actions"]:
                    print(f"         new_action     : {a.get('name')} "
                          f"({a.get('parameters','')}) "
                          f"pre={a.get('precondition','')} "
                          f"eff={a.get('effect','')}")
            if da.get("modified_preconditions"):
                print(f"         mod_precond    : {da['modified_preconditions']}")
        else:
            print(f"[LOOP] Domain enrichment: none (base template sufficient)")

        if not plan.steps:
            # Save final state image (no bboxes — task is complete)
            try:
                final_path = _REPO_ROOT / "data" / f"loop_iter_{iteration+1:02d}.png"
                image.save(str(final_path))
                print(f"[LOOP] Snapshot finale: {final_path.name}")
            except Exception:
                pass
            print("\n[LOOP] ✅  Task completato secondo VLM!")
            break

        step0 = plan.steps[0]
        print(f"[LOOP] Prossimo step: {step0.primitive}({step0.args})")

        # Prevent phantom place: skip place if the gripper should be empty
        # (no pick in completed_steps since last place/gripper_open)
        if step0.primitive == "place":
            last_pick = max(
                (i for i, s in enumerate(completed_steps) if s.startswith("pick")),
                default=-1
            )
            last_place = max(
                (i for i, s in enumerate(completed_steps) if s.startswith("place")),
                default=-1
            )
            if last_pick < last_place:
                print(f"[WARN] Phantom place detected (no pick since last place) — skipping")
                completed_steps.append(f"skip_place({step0.args.get('object','?')})")
                continue

        # Save debug image with VLM bboxes only
        # NOTE: Gazebo model projections (green circles) are NOT shown here because
        # world_to_pixel uses the fixed overview camera calibration, which is wrong
        # for the moving wrist camera.  FK-based projection is a Phase 2 task.
        try:
            from PIL import ImageDraw
            dbg = image.copy()
            draw = ImageDraw.Draw(dbg)
            for step in plan.steps:
                for key, color in [("bbox","orange"), ("location_bbox","cyan")]:
                    b = step.args.get(key)
                    if b and len(b)==4:
                        x1,y1,x2,y2 = [int(x) for x in b]
                        draw.rectangle([x1,y1,x2,y2], outline=color, width=2)
                        lbl = step.args.get("object" if key=="bbox" else "location","?")
                        draw.text((x1, max(0,y1-12)), lbl, fill=color)
            dbg_path = _REPO_ROOT / "data" / f"loop_iter_{iteration+1:02d}_debug.png"
            dbg.save(str(dbg_path))
            print(f"[LOOP] Debug: {dbg_path.name} (🟠pick bbox 🔵place bbox)")
        except Exception as _e:
            pass

        # 5. Ground names — OWL-ViT (bbox grounding requires camera calibration
        # that changes with arm position — only valid for fixed overview camera)
        plan_grounded = perception.ground_names(
            plan, image,
            known_items=gazebo_models,
            known_locations=gazebo_models,
        )

        # Show PDDL problem for this step
        try:
            from planner.problem_generator import generate_problem
            pddl = generate_problem(plan_grounded)
            print("\n  PDDL PROBLEM:")
            for line in pddl.splitlines():
                print(f"    {line}")
            print()
        except Exception:
            pass

        # 6. Serialize + inject — PDDL validates the step correctly because
        # problem_generator now handles partial plans:
        # - pick-only: goal = (holding obj)
        # - place-only: init = (holding obj), no gripper-empty
        payload = json.dumps({
            "command":  args.task,
            "vlm_plan": json.loads(plan_grounded.to_json()),
        })
        bash_cmd = (
            "source /opt/ros/humble/setup.bash && "
            "source /workspace/ros2_ws/install/setup.bash && "
            "python3 /workspace/scripts/_publish_plan.py"
        )
        inject_result = subprocess.run(
            docker_cmd + ["bash", "-c", bash_cmd],
            input=payload.encode(),
            capture_output=True,
        )
        if inject_result.returncode != 0:
            print(f"[FAIL] Injection failed: {inject_result.stderr.decode().strip()}")
            break
        print(f"[OK]   Step injected.")

        # 7. Wait for step completion
        print("[LOOP] Attendo completamento step...")
        result = _wait_step_complete(args, timeout=60)

        # Build step description — include original VLM name + grounded PDDL name
        # so the VLM can match its own terminology with the completed action.
        s0_orig = plan.steps[0]
        s0_grnd = plan_grounded.steps[0]
        obj_orig = s0_orig.args.get("object", s0_orig.args.get("target", "?"))
        obj_grnd = s0_grnd.args.get("object", s0_grnd.args.get("target", "?"))
        loc_orig = s0_orig.args.get("location", "")
        loc_grnd = s0_grnd.args.get("location", "")
        # E.g.: "place(blue_cube->blue_box, red_cylinder->red_cup)"
        obj_str = f"{obj_orig}->{obj_grnd}" if obj_orig != obj_grnd else obj_grnd
        loc_str = f"{loc_orig}->{loc_grnd}" if loc_orig != loc_grnd else loc_grnd
        step_desc = f"{s0_grnd.primitive}({obj_str}, {loc_str})" if loc_grnd else f"{s0_grnd.primitive}({obj_str})"
        if result.get("success"):
            completed_steps.append(step_desc)
            print(f"[OK]   Step completato: {step_desc}")
        else:
            print(f"[FAIL] Step fallito: {step_desc}")
            break
        # NOTE: task_complete from orchestrator = last step of CURRENT plan done.
        # In closed-loop, task completion is determined by the VLM (next iteration
        # returns complete=true or 0 steps), not by step count.  Do NOT break here.
    else:
        print(f"\n[WARN] Limite massimo di {args.max_steps} step raggiunto.")

    print(f"\n[LOOP] Steps completati: {completed_steps}")


if __name__ == "__main__":
    main()
