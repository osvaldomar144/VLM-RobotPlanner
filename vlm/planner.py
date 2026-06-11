"""
VLM planning module.
Input:  user command (str) + scene image(s) (file paths or PIL Images)
Output: VLMPlan — structured list of primitive calls
"""

from __future__ import annotations
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Union

from PIL import Image


@dataclass
class PlanStep:
    primitive: str      # e.g. "pick"
    args: dict          # e.g. {"object": "red_cup"}


_EMPTY_ADDITIONS: dict = {
    "new_types": [],
    "new_predicates": [],
    "new_actions": [],
    "modified_preconditions": {},
}


@dataclass
class VLMPlan:
    goal: str
    steps: list[PlanStep]
    raw_output: str           # raw VLM text (kept for debugging / thesis analysis)
    domain_template: str = "manipulation_base"
    domain_additions: dict = field(default_factory=lambda: {
        "new_types": [],
        "new_predicates": [],
        "new_actions": [],
        "modified_preconditions": {},
    })

    def to_domain_additions(self):
        """Convert domain_additions dict → DomainAdditions for use with DomainEnricher."""
        from planner.domain_enricher import DomainAdditions
        d = self.domain_additions
        return DomainAdditions(
            new_types=d.get("new_types", []),
            new_predicates=d.get("new_predicates", []),
            new_actions=d.get("new_actions", []),
            modified_preconditions=d.get("modified_preconditions", {}),
        )

    def to_json(self) -> str:
        """Serialize to JSON string (for host→container transport)."""
        return json.dumps({
            "goal": self.goal,
            "steps": [{"primitive": s.primitive, "args": s.args} for s in self.steps],
            "raw_output": self.raw_output,
            "domain_template": self.domain_template,
            "domain_additions": self.domain_additions,
        })

    @classmethod
    def from_json(cls, json_str: str) -> "VLMPlan":
        """Deserialize from JSON string."""
        data = json.loads(json_str)
        steps = [PlanStep(primitive=s["primitive"], args=s["args"])
                 for s in data.get("steps", [])]
        return cls(
            goal=data["goal"],
            steps=steps,
            raw_output=data.get("raw_output", ""),
            domain_template=data.get("domain_template", "manipulation_base"),
            domain_additions=data.get("domain_additions", _EMPTY_ADDITIONS.copy()),
        )


ImageInput = Union[str, Path, Image.Image]


class VLMPlanner:
    """
    Wraps Qwen3-VL to produce task plans from images + a natural language command.
    The VLM acts as both perception and planner: it identifies objects in the scene
    and produces a primitive sequence — no separate text description needed.
    """

    SYSTEM_PROMPT_PATH           = Path(__file__).parent / "prompts" / "system_prompt.txt"
    SYSTEM_PROMPT_LOOP_PATH      = Path(__file__).parent / "prompts" / "system_prompt_loop.txt"
    SYSTEM_PROMPT_REPLANNING_PATH = Path(__file__).parent / "prompts" / "system_prompt_replanning.txt"

    def __init__(self, model_name: str = "Qwen/Qwen3-VL-8B-Instruct"):
        self.model_name = model_name
        self._model = None
        self._processor = None

    def load(self) -> None:
        """Load model weights. Call once at startup (heavy operation)."""
        from vlm.model_loader import load_qwen_vl
        self._model, self._processor = load_qwen_vl(self.model_name)

    def plan(
        self,
        command: str,
        images: list[ImageInput],
        scene_context: dict | None = None,
    ) -> VLMPlan:
        """
        Args:
            command:       Natural language task (e.g. "pick the red cup …").
            images:        One or more scene images (file paths or PIL Images).
            scene_context: Optional dict with known PDDL names, e.g.:
                           {"items": ["red_cup", "blue_box"],
                            "locations": ["shelf_b"]}
                           When provided, the names are appended to the user
                           message so the VLM uses them verbatim in its output.
                           This prevents name-mismatch failures between VLM
                           output and the PDDL problem / Gazebo oracle.

        Returns:
            VLMPlan with primitive sequence grounded on what the VLM sees.
        """
        if self._model is None:
            raise RuntimeError("Call load() before plan()")

        pil_images = [self._to_pil(img) for img in images]
        system_prompt = self.SYSTEM_PROMPT_PATH.read_text()

        messages = self._build_messages(system_prompt, command, pil_images, scene_context)
        raw = self._run_inference(messages)
        return self._parse_output(command, raw)

    def plan_next_step(
        self,
        task:            str,
        images:          list[ImageInput],
        completed_steps: list[str],
    ) -> VLMPlan:
        """Closed-loop mode: plan the NEXT SINGLE action given current state.

        Args:
            task:            Overall task description (unchanged throughout loop).
            images:          Current scene image(s) from wrist camera.
            completed_steps: List of already-executed primitives (e.g. ["pick(red_cup)"]).

        Returns:
            VLMPlan with 0 steps (task complete) or 1 step (next action).
        """
        if self._model is None:
            raise RuntimeError("Call load() before plan_next_step()")

        pil_images   = [self._to_pil(img) for img in images]
        system_prompt = self.SYSTEM_PROMPT_LOOP_PATH.read_text()

        # Build user message with task context and completed steps
        completed_str = (
            "\n".join(f"  - {s}" for s in completed_steps)
            if completed_steps else "  (none yet)"
        )
        user_text = (
            f"Task goal: {task}\n\n"
            f"Completed steps:\n{completed_str}\n\n"
            f"What is the NEXT single action?"
        )

        messages = self._build_messages(system_prompt, user_text, pil_images)
        raw = self._run_inference(messages)
        plan = self._parse_output(task, raw)

        # Handle complete=true signal
        try:
            data = __import__("json").loads(
                raw if raw.strip().startswith("{") else
                __import__("re").search(r"\{.*\}", raw, __import__("re").DOTALL).group()
            )
            if data.get("complete"):
                plan.steps = []   # empty steps = task done
        except Exception:
            pass

        return plan

    def plan_remaining(
        self,
        task:            str,
        images:          list[ImageInput],
        completed_steps: list[str],
        failed_step:     str | None = None,
    ) -> VLMPlan:
        """Replanning mode: generate the COMPLETE remaining plan from current state.

        Called either at task start (completed_steps=[]) or after a failure.
        Returns ALL remaining steps to complete the task.

        Args:
            task:            Overall task description.
            images:          Current scene image(s) — overview camera preferred.
            completed_steps: Steps already executed successfully.
            failed_step:     The step that just failed (for replan context), or None.

        Returns:
            VLMPlan with all remaining steps, or empty steps if task is complete.
        """
        if self._model is None:
            raise RuntimeError("Call load() before plan_remaining()")

        pil_images    = [self._to_pil(img) for img in images]
        system_prompt = self.SYSTEM_PROMPT_REPLANNING_PATH.read_text()

        completed_str = (
            "\n".join(f"  - {s}" for s in completed_steps)
            if completed_steps else "  (none yet)"
        )
        failed_str = f"\n\nFailed step (just failed — replan needed):\n  {failed_step}" \
                     if failed_step else ""

        user_text = (
            f"Task goal: {task}\n\n"
            f"Completed steps:\n{completed_str}"
            f"{failed_str}\n\n"
            f"Generate the COMPLETE remaining plan to finish the task."
        )

        messages = self._build_messages(system_prompt, user_text, pil_images)
        raw  = self._run_inference(messages)
        plan = self._parse_output(task, raw)

        try:
            data = __import__("json").loads(
                raw if raw.strip().startswith("{") else
                __import__("re").search(r"\{.*\}", raw, __import__("re").DOTALL).group()
            )
            if data.get("complete"):
                plan.steps = []
            # Store replan metadata in plan for debug logging
            if data.get("replanning") and data.get("replan_reason"):
                plan.raw_output = f"[REPLAN: {data['replan_reason']}]\n{plan.raw_output}"
        except Exception:
            pass

        return plan

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _to_pil(self, img: ImageInput) -> Image.Image:
        if isinstance(img, Image.Image):
            return img
        return Image.open(img).convert("RGB")

    def _build_messages(
        self,
        system_prompt: str,
        command: str,
        images: list[Image.Image],
        scene_context: dict | None = None,
    ) -> list[dict]:
        image_content = [{"type": "image", "image": img} for img in images]
        user_text = command
        if scene_context:
            parts = ["\n\nKnown PDDL names — use these EXACTLY in your JSON output:"]
            if scene_context.get("items"):
                parts.append(f"  Items: {', '.join(scene_context['items'])}")
            if scene_context.get("locations"):
                parts.append(f"  Locations: {', '.join(scene_context['locations'])}")
            user_text += "\n".join(parts)
        return [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": image_content + [{"type": "text", "text": user_text}],
            },
        ]

    def _run_inference(self, messages: list[dict]) -> str:
        import torch
        from qwen_vl_utils import process_vision_info

        text = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages, image_patch_size=16)
        inputs = self._processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self._model.device)

        with torch.no_grad():
            generated = self._model.generate(**inputs, max_new_tokens=512)

        output_ids = generated[:, inputs.input_ids.shape[1]:]
        return self._processor.batch_decode(
            output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

    def _parse_output(self, command: str, raw: str) -> VLMPlan:
        """Extract JSON plan from VLM output. Robust to markdown code fences."""
        match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
        json_str = match.group(1) if match else raw.strip()

        try:
            data = json.loads(json_str)
            steps = [PlanStep(**s) for s in data.get("steps", [])]
            goal = data.get("goal", command)
            domain_template = data.get("domain_template", "manipulation_base")
            domain_additions = data.get("domain_additions", {
                "new_types": [],
                "new_predicates": [],
                "new_actions": [],
                "modified_preconditions": {},
            })
        except (json.JSONDecodeError, TypeError):
            steps = []
            goal = command
            domain_template = "manipulation_base"
            domain_additions = {
                "new_types": [],
                "new_predicates": [],
                "new_actions": [],
                "modified_preconditions": {},
            }

        return VLMPlan(
            goal=goal,
            steps=steps,
            raw_output=raw,
            domain_template=domain_template,
            domain_additions=domain_additions,
        )
