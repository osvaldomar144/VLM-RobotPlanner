"""
vlm/perception.py — Perception Module: visual grounding for object name resolution.

Phase 1 (simulation):
  Maps VLM-generated object names to known PDDL names using OWL-ViT open-vocabulary
  object detection.  Runs on the host GPU alongside the VLM.  The GazeboOracle still
  provides 3D poses — this module only corrects the names before PDDL planning.

Phase 2 (real robot — future):
  Extend get_pose() to use RealSense D435i depth instead of the oracle.  The
  grounding logic (OWL-ViT inference) is identical; only the 3D localization backend
  changes.  No other code needs to be modified.

Why this matters:
  The VLM generates object names from the task text ("blue cube", "cylinder", …)
  which may differ from the PDDL names ("blue_box", "red_cup").  Without grounding,
  the oracle lookup fails.  With grounding, names are corrected visually before
  reaching the PDDL pipeline — regardless of how the VLM describes the object.
"""

from __future__ import annotations

import math
from copy import deepcopy
from typing import Optional

import numpy as np
import torch
from PIL import Image

from vlm.planner import PlanStep, VLMPlan


# ── Camera calibration (overview_camera, from tabletop.world SDF) ────────────
# These are constants for Phase 1 simulation.  Phase 2 (real robot) reads
# calibration from the RealSense factory calibration and hand-eye procedure.

_CAM_POS  = np.array([1.0, 0.7, 1.5])        # position in world frame (m)
_CAM_RPY  = (0.0, 0.68, -2.19)               # roll, pitch, yaw (rad) from SDF
_IMG_W, _IMG_H = 640, 480                     # image resolution (px)
_FOV_H    = 1.047                             # horizontal FOV (rad)


def _rpy_to_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """SDF-convention RPY -> 3×3 rotation matrix (R = Rz @ Ry @ Rx)."""
    Rx = np.array([[1,0,0],[0,math.cos(roll),-math.sin(roll)],
                   [0,math.sin(roll),math.cos(roll)]])
    Ry = np.array([[math.cos(pitch),0,math.sin(pitch)],[0,1,0],
                   [-math.sin(pitch),0,math.cos(pitch)]])
    Rz = np.array([[math.cos(yaw),-math.sin(yaw),0],
                   [math.sin(yaw),math.cos(yaw),0],[0,0,1]])
    return Rz @ Ry @ Rx


def _build_camera_matrix() -> tuple[np.ndarray, np.ndarray, float, float, float, float]:
    """
    Build the world-to-OpenCV-camera projection components.

    In Gazebo, the camera link's +X axis is the optical axis.
    OpenCV/ROS convention: +Z forward, +X right, +Y down.
    The transform from Gazebo-link to OpenCV camera frame is:
        R_C_G = [[0,-1,0],[0,0,-1],[1,0,0]]

    Returns: (R, t, fx, fy, cx, cy)
      R, t  — 3×3 rotation and 3-vector translation for world->camera projection
      fx,fy — focal lengths (px)
      cx,cy — principal point (px)
    """
    # Rotation: world -> Gazebo link -> OpenCV camera
    R_W_G  = _rpy_to_matrix(*_CAM_RPY)   # camera frame axes as world columns
    R_G_W  = R_W_G.T                      # world -> Gazebo link
    # Gazebo camera link: +X=forward(optical), +Y=right, +Z=up
    # OpenCV camera frame: +Z=forward, +X=right, +Y=down
    # Mapping: G_x->C_z, G_y->C_x, G_z->C_(-y)
    R_C_G  = np.array([[0, 1, 0],
                        [0, 0,-1],
                        [1, 0, 0]])   # Gazebo link -> OpenCV cam
    R      = R_C_G @ R_G_W               # world -> OpenCV camera

    t      = -R @ _CAM_POS               # translation in OpenCV camera frame

    # Intrinsics from SDF (square pixels, principal point at image centre)
    fx = fy = _IMG_W / (2.0 * math.tan(_FOV_H / 2.0))
    cx, cy  = _IMG_W / 2.0, _IMG_H / 2.0

    return R, t, fx, fy, cx, cy


_R, _T, _FX, _FY, _CX, _CY = _build_camera_matrix()


def world_to_pixel(xyz: np.ndarray) -> tuple[float, float] | None:
    """
    Project a world 3D point to image pixel (u, v).
    Returns None if the point is behind the camera.
    """
    P_cam = _R @ xyz + _T
    if P_cam[2] <= 0:        # behind camera
        return None
    u = _FX * P_cam[0] / P_cam[2] + _CX
    v = _FY * P_cam[1] / P_cam[2] + _CY
    return u, v


def _iou(a: list[float], b: list[float]) -> float:
    """Intersection-over-Union of two [x0, y0, x1, y1] boxes."""
    x0 = max(a[0], b[0]);  y0 = max(a[1], b[1])
    x1 = min(a[2], b[2]);  y1 = min(a[3], b[3])
    inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union  = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


class PerceptionModule:
    """
    Visual grounding module: maps free-form VLM names to known PDDL names.

    Uses OWL-ViT open-vocabulary detection to find each object in the scene
    image.  Two objects are considered the same if their detected bounding boxes
    overlap sufficiently (IoU ≥ IOU_MATCH_THRESHOLD).

    Usage:
        perception = PerceptionModule()
        perception.load()
        corrected_plan = perception.ground_names(plan, image,
                                                  known_items=["red_cup", "blue_box"],
                                                  known_locations=["shelf_b"])
    """

    MODEL_NAME          = "google/owlvit-base-patch32"
    DETECTION_THRESHOLD = 0.05   # low to maximise recall
    IOU_MATCH_THRESHOLD = 0.05   # min IoU to accept a visual match

    def __init__(self) -> None:
        self._processor = None
        self._model     = None
        self._device    = "cuda" if torch.cuda.is_available() else "cpu"

    def load(self) -> None:
        """Load OWL-ViT weights (much lighter than VLM — ~500 MB)."""
        from transformers import OwlViTForObjectDetection, OwlViTProcessor
        print(f"[INFO] Loading PerceptionModule ({self.MODEL_NAME})…")
        self._processor = OwlViTProcessor.from_pretrained(self.MODEL_NAME)
        self._model     = OwlViTForObjectDetection.from_pretrained(
            self.MODEL_NAME
        ).to(self._device)
        self._model.eval()
        print("[OK]   PerceptionModule loaded.")

    # ── Public API ────────────────────────────────────────────────────────────

    def ground_names_with_bbox(
        self,
        plan:           VLMPlan,
        model_poses:    dict[str, dict],   # {"red_cup": {"x":…,"y":…,"z":…}, …}
        img_w:          int = _IMG_W,
        img_h:          int = _IMG_H,
    ) -> VLMPlan:
        """
        Correct object/location names in *plan* using VLM-provided bounding
        boxes and camera projection.  No OWL-ViT required.

        For each step arg that carries a bbox:
          1. Take the bbox center (u, v) from the VLM output.
          2. Project every Gazebo model's 3D position -> pixel (u_m, v_m).
          3. Find the model whose projected pixel is nearest to the bbox center.
          4. Replace the step name with that Gazebo model name.

        Sim-to-real note: on the real robot, step 2 becomes
            depth = realsense.get_depth(u, v) -> 3D point in robot frame
        and no oracle lookup is needed at all.

        Falls back gracefully if no bbox is present in the plan.
        Returns a new VLMPlan; the original is not modified.
        """
        if not model_poses:
            return plan

        # Pre-project all Gazebo models to pixel coordinates
        model_pixels: dict[str, tuple[float, float]] = {}
        for name, pos in model_poses.items():
            px = world_to_pixel(np.array([pos["x"], pos["y"], pos["z"]]))
            if px is not None:
                model_pixels[name] = px

        if not model_pixels:
            return plan

        used_pddl: set[str] = set()

        def _nearest(u: float, v: float) -> str | None:
            best_name: str | None = None
            best_dist = float("inf")
            for mname, (mu, mv) in model_pixels.items():
                if mname in used_pddl:
                    continue
                d = math.hypot(u - mu, v - mv)
                if d < best_dist:
                    best_dist = d
                    best_name = mname
            return best_name

        def _center(bbox) -> tuple[float, float] | None:
            if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
                return None
            x1, y1, x2, y2 = bbox
            if max(x1, y1, x2, y2) <= 1.0:   # normalised -> pixels
                x1, y1, x2, y2 = x1*img_w, y1*img_h, x2*img_w, y2*img_h
            return (x1 + x2) / 2.0, (y1 + y2) / 2.0

        corrected = deepcopy(plan)
        changed   = False

        for step in corrected.steps:
            args = dict(step.args)

            # pick / look_at — "bbox" -> correct "object" or "target"
            c = _center(args.get("bbox"))
            if c:
                for key in ("object", "target"):
                    if key in args:
                        match = _nearest(*c)
                        if match:
                            print(f"[OK]   bbox->'{match}' "
                                  f"(was '{args[key]}', "
                                  f"center=({c[0]:.0f},{c[1]:.0f}))")
                            args[key] = match
                            used_pddl.add(match)
                            changed = True
                        break

            # place — "location_bbox" -> correct "location"
            lc = _center(args.get("location_bbox"))
            if lc and "location" in args:
                match = _nearest(*lc)
                if match:
                    print(f"[OK]   location_bbox->'{match}' "
                          f"(was '{args['location']}', "
                          f"center=({lc[0]:.0f},{lc[1]:.0f}))")
                    args["location"] = match
                    used_pddl.add(match)
                    changed = True

            step.args = args

        # Second pass: propagate name_map to ANY remaining step arg that still
        # holds an old VLM name (e.g. "object" in place step shares the name
        # already corrected in the pick step but has no bbox of its own).
        name_map = {
            old: new
            for step, orig in zip(corrected.steps, plan.steps)
            for key in orig.args
            if isinstance(orig.args.get(key), str)
            and isinstance(step.args.get(key), str)
            and orig.args[key] != step.args[key]
            for old, new in [(orig.args[key], step.args[key])]
        }
        if name_map:
            for step in corrected.steps:
                step.args = {
                    k: (name_map.get(v, v) if isinstance(v, str) else v)
                    for k, v in step.args.items()
                }

        if not changed:
            print("[INFO] PerceptionModule: no bbox in plan "
                  "— falling back to OWL-ViT name grounding.")

        return corrected

    # Words that hint at a model being a location (surface/stand/shelf)
    # rather than a graspable item.
    _LOCATION_HINTS: frozenset[str] = frozenset({
        "shelf", "table", "surface", "platform", "stand",
        "rack", "tray", "bin", "ground", "floor",
    })

    def ground_names(
        self,
        plan:            VLMPlan,
        image:           Image.Image,
        known_items:     list[str],
        known_locations: list[str],
    ) -> VLMPlan:
        """
        Correct object/location names in *plan* using visual grounding.

        Names already present in known_items / known_locations are left unchanged.
        Unknown names are matched using:
          1. OWL-ViT visual detection (bounding-box IoU)
          2. Token-overlap fallback
          3. Semantic role hint (object vs location) to avoid conflicts

        The matching is role-aware: names used as pick targets ("object" key)
        are matched separately from names used as place targets ("location" key).
        This prevents e.g. "green_platform" and "red_cylinder" from both
        resolving to "red_cup".

        Returns a new VLMPlan; the original is not modified.
        """
        if self._model is None:
            raise RuntimeError("Call load() before ground_names()")

        # Deduplicate while preserving order (items before locations)
        seen: set[str] = set()
        known_all: list[str] = []
        for n in known_items + known_locations:
            if n not in seen:
                known_all.append(n)
                seen.add(n)

        # Separate unknown names by their semantic role in the plan
        to_ground_items: set[str] = set()    # used as pick/look_at targets
        to_ground_locs:  set[str] = set()    # used as place destinations

        for step in plan.steps:
            for key, val in step.args.items():
                if not isinstance(val, str) or val in known_all:
                    continue
                if key in ("object", "target"):
                    to_ground_items.add(val)
                elif key == "location":
                    to_ground_locs.add(val)

        all_to_ground = to_ground_items | to_ground_locs
        if not all_to_ground:
            return plan

        print(f"[INFO] PerceptionModule: grounding {all_to_ground} -> {known_all}")

        # Pre-detect all known objects once (visual boxes)
        known_boxes = self._detect(known_all, image)

        name_map: dict[str, str] = {}
        used_pddl: set[str] = set()

        def _match(vlm_name: str, prefer_location: bool) -> str:
            """Match one VLM name, avoiding already-used candidates."""
            available = [c for c in known_all if c not in used_pddl]
            if not available:
                available = known_all

            # Visual match
            vlm_boxes = self._detect([vlm_name], image)
            vis_match = self._best_iou_match(
                vlm_boxes.get(vlm_name, []),
                {k: known_boxes[k] for k in available if k in known_boxes},
            )
            if vis_match:
                return vis_match

            # Token-overlap fallback
            best_tok  = self._token_fallback(vlm_name, available)
            tok_score = len(
                set(vlm_name.lower().replace("_", " ").split()) &
                set(best_tok.lower().replace("_", " ").split())
            )

            # If token score is 0 and we need a location, prefer candidates
            # whose name contains a location-hint word (shelf, table, …).
            if tok_score == 0 and prefer_location:
                loc_scored = [
                    (c, self._location_score(c)) for c in available
                ]
                loc_scored.sort(key=lambda x: x[1], reverse=True)
                if loc_scored[0][1] > 0:
                    return loc_scored[0][0]

            return best_tok

        # Match items first (pick / look_at targets)
        for name in sorted(to_ground_items):
            match = _match(name, prefer_location=False)
            tag   = "visual" if self._best_iou_match(
                self._detect([name], image).get(name, []),
                {k: known_boxes[k] for k in known_all if k in known_boxes},
            ) else "token fallback"
            name_map[name] = match
            used_pddl.add(match)
            icon = "[OK]  " if "visual" in tag else "[WARN]"
            print(f"{icon} '{name}' -> '{match}' ({tag})")

        # Match locations second (prefer candidates not used by items)
        for name in sorted(to_ground_locs):
            match = _match(name, prefer_location=True)
            tag   = "visual" if self._best_iou_match(
                self._detect([name], image).get(name, []),
                {k: known_boxes[k] for k in known_all if k in known_boxes},
            ) else "token fallback"
            name_map[name] = match
            used_pddl.add(match)
            icon = "[OK]  " if "visual" in tag else "[WARN]"
            print(f"{icon} '{name}' -> '{match}' ({tag})")

        # Apply corrections
        corrected = deepcopy(plan)
        corrected.steps = [
            PlanStep(
                primitive=step.primitive,
                args={
                    k: (name_map.get(v, v) if isinstance(v, str) else v)
                    for k, v in step.args.items()
                },
            )
            for step in plan.steps
        ]
        return corrected

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _detect(
        self,
        names:     list[str],
        image:     Image.Image,
        threshold: Optional[float] = None,
    ) -> dict[str, list[list[float]]]:
        """
        Run OWL-ViT and return bounding boxes per name.
        Underscores in names are replaced with spaces for the text query.
        Returns: {name: [[x0, y0, x1, y1], ...]}
        """
        if threshold is None:
            threshold = self.DETECTION_THRESHOLD

        readable = [n.replace("_", " ") for n in names]
        inputs   = self._processor(
            text=[readable], images=image, return_tensors="pt"
        ).to(self._device)

        with torch.no_grad():
            outputs = self._model(**inputs)

        # Version-agnostic post-processing (avoids API differences across
        # transformers versions — works with any OWL-ViT release).
        # outputs.logits:    [1, num_patches, num_queries]
        # outputs.pred_boxes: [1, num_patches, 4]  (cx, cy, w, h, normalised)
        logits    = outputs.logits[0].cpu()      # [patches, queries]
        pred_boxes = outputs.pred_boxes[0].cpu() # [patches, 4]
        W, H = float(image.size[0]), float(image.size[1])

        boxes: dict[str, list] = {n: [] for n in names}
        for q_idx, name in enumerate(names):
            scores = torch.sigmoid(logits[:, q_idx])
            for p_idx in (scores >= threshold).nonzero(as_tuple=False).flatten().tolist():
                cx, cy, w, h = pred_boxes[p_idx].tolist()
                boxes[name].append([
                    (cx - w / 2) * W, (cy - h / 2) * H,
                    (cx + w / 2) * W, (cy + h / 2) * H,
                ])

        return boxes

    def _best_iou_match(
        self,
        query_boxes: list[list[float]],
        known_boxes: dict[str, list[list[float]]],
    ) -> str | None:
        """Return the known name with highest IoU against any query box."""
        best_name: str | None = None
        best_iou  = self.IOU_MATCH_THRESHOLD  # minimum to qualify

        for known_name, kboxes in known_boxes.items():
            for qb in query_boxes:
                for kb in kboxes:
                    score = _iou(qb, kb)
                    if score > best_iou:
                        best_iou = score
                        best_name = known_name
        return best_name

    def _location_score(self, name: str) -> int:
        """How many location-hint words appear in the candidate name."""
        words = set(name.lower().replace("_", " ").split())
        return len(words & self._LOCATION_HINTS)

    # Shape/function synonyms for token matching — MUST be scene-independent.
    # Only geometric/functional equivalences, NOT color-object mappings.
    # Color mappings (e.g. "red"->"cup") would overfit to specific scenes.
    _SYNONYMS: dict[str, list[str]] = {
        "cube":      ["box"],
        "box":       ["cube"],
        "cylinder":  ["cup"],
        "cup":       ["cylinder"],
        "platform":  ["shelf"],
        "shelf":     ["platform", "rack", "stand"],
        "surface":   ["shelf", "platform", "table"],
        "container": ["box", "bin"],
        "object":    [],   # too generic — no expansion
        "thing":     [],   # too generic — no expansion
    }

    @staticmethod
    def _token_fallback(name: str, candidates: list[str]) -> str:
        """Fallback: pick candidate with most token overlap (+ synonym expansion)."""
        # Expand name tokens with synonyms
        raw_tokens = set(name.lower().replace("_", " ").split())
        expanded: set[str] = set(raw_tokens)
        for token in raw_tokens:
            expanded.update(PerceptionModule._SYNONYMS.get(token, []))

        scored = []
        for c in candidates:
            c_tokens = set(c.lower().replace("_", " ").split())
            c_expanded: set[str] = set(c_tokens)
            for token in c_tokens:
                c_expanded.update(PerceptionModule._SYNONYMS.get(token, []))
            score = len(expanded & c_expanded)
            scored.append((score, c))

        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[0][1]
