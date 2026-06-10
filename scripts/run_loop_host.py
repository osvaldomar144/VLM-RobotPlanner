#!/usr/bin/env python3
"""
run_loop_host.py — Closed-loop task execution (HOST side).

Implements the closed-loop architecture:
  [scan] -> [capture] -> [VLM next step] -> [inject] -> [wait complete] -> repeat

Each iteration:
  1. Pre-scan: move arm to scan pose via _pre_scan.py (wrist camera view)
  2. Capture: take image from wrist camera via _capture_scene.py
  3. VLM: plan_next_step(task, image, completed_steps) -> single action
  4. Ground: GroundingDINO -> correct object names and 3D poses
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


def _publish_perception_pose(
    args, object_name: str, x: float, y: float, z: float
) -> bool:
    """Publish a perception-estimated pose to /perception/object_pose."""
    bash_cmd = (
        "source /opt/ros/humble/setup.bash && "
        "source /workspace/ros2_ws/install/setup.bash && "
        f"python3 /workspace/scripts/_publish_perception_pose.py "
        f"--object {object_name} --x {x:.6f} --y {y:.6f} --z {z:.6f}"
    )
    r = _run_in_container(args, bash_cmd, timeout=10)
    for line in r.stdout.decode().strip().splitlines():
        print(f"       {line}")
    return r.returncode == 0


def _run_perception(
    args,
    target_name: str,
    perception,
    obj_z_base: float = 0.025,
    placed_at: dict | None = None,
    excl_radius: float = 0.10,
) -> bool:
    """
    Phase 2: after look_at, capture wrist-camera frame and estimate 3D pose
    via GroundingDINO. Publishes to /perception/object_pose.
    Orchestrator prefers this over oracle when cache is fresh.
    Phase 4 (real robot): same flow, RealSense depth replaces ray-plane z.
    """
    from vlm.perception import PerceptionModule
    from PIL import Image as PilImage

    print("[LOOP] Phase 2: capturing from scan pose…")
    fresh_path = _capture(args)
    if fresh_path is None:
        print("[WARN] Phase 2: capture failed — oracle will be used as fallback")
        return False
    obs_image = PilImage.open(fresh_path).convert("RGB")

    data_dir = str(_REPO_ROOT / "data")
    cam_data = PerceptionModule.load_camera_data(data_dir)
    if cam_data is None:
        print("[LOOP] Phase 2: camera calibration not available")
        return False

    K, cam_to_base = cam_data
    print(f"[LOOP] Phase 2: GroundingDINO → get_pose('{target_name}')")
    pose = perception.get_pose(target_name, obs_image, K, cam_to_base,
                               obj_z_base=obj_z_base)

    if pose is None:
        print(f"[WARN] Phase 2: '{target_name}' not detected — oracle will be used as fallback")
        return False

    # Apply spatial exclusion (same logic as pre-step)
    if placed_at:
        import numpy as _np3
        for _ph_obj, (_px, _py) in placed_at.items():
            _d = float(_np3.linalg.norm([pose['x']-_px, pose['y']-_py]))
            if _d < excl_radius:
                print(f"[WARN] Phase 2 exclusion: '{target_name}' at "
                      f"({pose['x']:.2f},{pose['y']:.2f}) occupied by "
                      f"'{_ph_obj}' (dist={_d*100:.0f}cm) — treating as not found")
                return False

    print(
        f"[LOOP] Phase 2: '{target_name}' → "
        f"({pose['x']:.3f}, {pose['y']:.3f}, {pose['z']:.3f}) panda_link0"
    )
    ok = _publish_perception_pose(args, target_name, pose["x"], pose["y"], pose["z"])
    if ok:
        print("[OK]   Phase 2: pose published → orchestrator will prefer it over oracle")
    return ok


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

    # Spatial exclusion: track where objects have been placed.
    # After place(X, loc), the location position is marked as occupied by X.
    # New DINO detections within EXCL_RADIUS of an occupied position are skipped
    # to avoid re-picking already-handled objects.
    # Key: VLM object name; Value: (x, y) of the place destination.
    _placed_at: dict[str, tuple[float, float]] = {}
    _last_dino_est: dict[str, tuple[float, float]] = {}  # last DINO estimate per name
    _EXCL_RADIUS = 0.10  # 10cm — objects within this radius are treated as identical

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

        # Persist last scan-pose image + calibration for place location detection.
        # When arm is holding an object, the camera view is distorted by the arm.
        # Using the last FREE scan gives better geometry for location detection.
        if not holding:
            import shutil
            _data = _REPO_ROOT / "data"
            for fname in ("scene.png", "camera_info.json", "camera_pose.json"):
                src = _data / fname
                if src.exists():
                    shutil.copy2(str(src), str(_data / f"last_scan_{fname}"))
            print("[LOOP] Last scan saved (arm free → used for place location detection)")

        # 3. Get Gazebo models — filter scene infrastructure (never pick/place targets)
        _INFRA = frozenset({
            'floor', 'room', 'ground_plane', 'sun', 'robot_pedestal',
            'overview_camera', 'table', 'workbench',
        })
        gazebo_poses = {k: v for k, v in _get_gazebo_models(args).items()
                        if k not in _INFRA}
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

        # Prevent phantom pick: skip pick if already holding an object
        if step0.primitive == "pick":
            last_pick  = max((i for i, s in enumerate(completed_steps) if s.startswith("pick")),  default=-1)
            last_place = max((i for i, s in enumerate(completed_steps) if s.startswith("place") or s.startswith("stack")), default=-1)
            if last_pick > last_place:
                print(f"[WARN] Phantom pick detected (already holding) — skipping")
                completed_steps.append(f"skip_pick({step0.args.get('object','?')})")
                continue

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
                # Count consecutive skip_place to detect stuck loop
                consecutive_skips = sum(
                    1 for s in reversed(completed_steps)
                    if s.startswith("skip_place")
                    ) if completed_steps else 0
                if consecutive_skips >= 2:
                    print(f"[LOOP] ✅ {consecutive_skips} phantom places consecutivi → "
                          "task considerato completato (oggetto già depositato)")
                    break
                print(f"[WARN] Phantom place detected (no pick since last place) — skipping")
                completed_steps.append(f"skip_place({step0.args.get('object','?')})")
                continue

        # Save iteration snapshot (no bbox overlay — bboxes removed from pipeline)
        # The scene image is already saved as loop_iter_NN.png above.

        # Phase 2: no grounding — VLM names used directly for DINO queries.
        # ground_names() was for Phase 1 oracle lookup (Gazebo model names).
        # In Phase 2, DINO takes any natural language query ("glass", "the cup")
        # and finds the object. Consistent names across iterations = no mismatch.
        from copy import deepcopy
        plan_grounded = deepcopy(plan)

        # 5c. Phase 2: DINO stima le pose di TUTTI gli oggetti nel passo corrente
        # dall'immagine scan. Questo copre:
        #   - look_at: pubblica posa → orchestratore passa a LookAtPrimitive → j0 direzionale
        #   - pick: posa già gestita via _run_perception dopo look_at, ma pre-stima utile
        #   - place: posa della location
        # Tutti pubblicati su /perception/object_pose → orchestratore preferisce su oracle.
        _data_dir = str(_REPO_ROOT / "data")
        step0 = plan_grounded.steps[0] if plan_grounded.steps else None
        if step0:
            try:
                from vlm.perception import PerceptionModule
                cam_data = PerceptionModule.load_camera_data(_data_dir)
                if cam_data:
                    K_s, ctb_s = cam_data
                    # Raccogli tutti i nomi oggetto nel passo corrente
                    # For place locations use the LAST SCAN image (arm free, better geometry).
                    # If no last_scan exists yet, fall back to current image.
                    _last_scan_img_path = _REPO_ROOT / "data" / "last_scan_scene.png"
                    _last_scan_info_path = _REPO_ROOT / "data" / "last_scan_camera_info.json"
                    _last_scan_pose_path = _REPO_ROOT / "data" / "last_scan_camera_pose.json"
                    _has_last_scan = (_last_scan_img_path.exists() and
                                      _last_scan_info_path.exists() and
                                      _last_scan_pose_path.exists())

                    if _has_last_scan:
                        _scan_img  = PilImage.open(str(_last_scan_img_path)).convert("RGB")
                        with open(str(_last_scan_info_path)) as _f:
                            import json as _json
                            _ki = _json.load(_f)
                        import numpy as _np
                        K_scan  = _np.array(_ki["K"])
                        with open(str(_last_scan_pose_path)) as _f:
                            ctb_scan = _np.array(_json.load(_f)["cam_to_base"])
                    else:
                        _scan_img, K_scan, ctb_scan = image, K_s, ctb_s

                    # Track name→key mapping to correctly update plan_grounded args
                    names_to_estimate = {}  # {name: key}
                    for _key in ("target", "object", "location"):
                        _n = step0.args.get(_key, "")
                        if _n and _n not in _INFRA and _n not in names_to_estimate:
                            names_to_estimate[_n] = _key

                    for name, name_key in names_to_estimate.items():
                        is_location = (name_key == "location")
                        # Use last_scan for locations (arm free → better angle)
                        det_img  = _scan_img if (is_location and _has_last_scan) else image
                        det_K    = K_scan    if (is_location and _has_last_scan) else K_s
                        det_ctb  = ctb_scan  if (is_location and _has_last_scan) else ctb_s
                        src_label = "last_scan" if (is_location and _has_last_scan) else "current"
                        pose_est = perception.get_pose(
                            name, det_img, det_K, det_ctb,
                            vlm_description=name.replace("_", " "),
                        )
                        if pose_est:
                            # Spatial exclusion: only for PICK targets, NOT for locations.
                            # Locations (e.g. keyboard) can receive multiple objects.
                            import numpy as _np2
                            _excl_match = None
                            if name_key != "location":
                                for _ph_obj, (_px, _py) in _placed_at.items():
                                    _d = float(_np2.linalg.norm(
                                        [pose_est['x']-_px, pose_est['y']-_py]))
                                    if _d < _EXCL_RADIUS:
                                        _excl_match = (_ph_obj, _d)
                                        break
                            if _excl_match:
                                print(f"[LOOP] Spatial exclusion: '{name}' at "
                                      f"({pose_est['x']:.2f},{pose_est['y']:.2f}) "
                                      f"occupied by '{_excl_match[0]}' "
                                      f"(dist={_excl_match[1]*100:.0f}cm) — skip")
                                pose_est = None
                            else:
                                # Track last estimate
                                _last_dino_est[name] = (pose_est['x'], pose_est['y'])

                        if pose_est:
                            print(f"[LOOP] Phase 2 DINO pre-step [{src_label}]: '{name}' → "
                                  f"({pose_est['x']:.3f},{pose_est['y']:.3f},{pose_est['z']:.3f})")
                            _publish_perception_pose(
                                args, name, pose_est['x'], pose_est['y'], pose_est['z'])

                            # Sim-only: GazeboAttach needs actual Gazebo model name.
                            # Resolve VLM name → nearest Gazebo model by position.
                            if name not in gazebo_poses and step0.primitive in ("pick", "place"):
                                import numpy as _np
                                _RBASE_XY = _np.array([0.20, 0.0])
                                _px = _np.array([pose_est['x'], pose_est['y']])
                                _best_gz, _best_d = None, float('inf')
                                for _gz, _gp in gazebo_poses.items():
                                    _gz_xy = _np.array([_gp['x'], _gp['y']]) - _RBASE_XY
                                    _d = float(_np.linalg.norm(_px - _gz_xy))
                                    if _d < _best_d:
                                        _best_d, _best_gz = _d, _gz
                                if _best_gz and _best_d < 0.15:
                                    _publish_perception_pose(
                                        args, _best_gz, pose_est['x'], pose_est['y'], pose_est['z'])
                                    # Update correct key in plan_grounded
                                    step0.args = dict(step0.args)
                                    step0.args[name_key] = _best_gz
                                    print(f"[LOOP] GazeboAttach: '{name}' → '{_best_gz}' "
                                          f"(dist={_best_d*100:.1f}cm, sim-only)")
                        else:
                            print(f"[LOOP] Phase 2 DINO pre-step: '{name}' non rilevato — oracle fallback")
            except Exception as _pe:
                print(f"[WARN] pre-step perception failed: {_pe}")

        # 5b. RIMOSSO — bbox-ground sostituito da DINO-only (step 5c).

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
        # Always use ORIGINAL VLM names in completed_steps context.
        # The Gazebo resolution (glass→coffee_cup) is sim-internal — VLM should
        # see its own names so it recognises completed steps correctly.
        obj_orig = s0_orig.args.get("object", s0_orig.args.get("target", "?"))
        loc_orig = s0_orig.args.get("location", "")
        step_desc = (f"{s0_orig.primitive}({obj_orig}, {loc_orig})"
                     if loc_orig else f"{s0_orig.primitive}({obj_orig})")
        if result.get("success"):
            # Detect look_at loop: if same look_at target already in completed, stop
            if s0_grnd.primitive == "look_at" and step_desc in completed_steps:
                print(f"[WARN] look_at('{obj_orig}') già eseguito — DINO non riesce a trovare "
                      "l'oggetto. Verificare il nome dell'oggetto nella scena.")
                break
            completed_steps.append(step_desc)
            print(f"[OK]   Step completato: {step_desc}")

            # Phase 2: after look_at → DINO estimates 3D pose, publishes to cache
            if s0_grnd.primitive == "look_at":
                target = s0_grnd.args.get("target", obj_orig)
                _run_perception(args, target, perception,
                                placed_at=_placed_at, excl_radius=_EXCL_RADIUS)

            # Track place destinations for spatial exclusion.
            # After place(X, loc), loc's position is recorded as occupied by X.
            # Future DINO detections within _EXCL_RADIUS of that position are skipped.
            if s0_orig.primitive == "place":
                obj_placed = s0_orig.args.get("object", "")
                loc_placed = s0_grnd.args.get("location", "")
                if obj_placed and loc_placed in _last_dino_est:
                    px, py = _last_dino_est[loc_placed]
                    _placed_at[obj_placed] = (px, py)
                    print(f"[LOOP] Spatial exclusion: '{obj_placed}' marked at "
                          f"({px:.2f},{py:.2f}) — excluded from future detections")
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
