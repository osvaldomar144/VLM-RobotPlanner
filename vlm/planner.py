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

    SYSTEM_PROMPT_PATH = Path(__file__).parent / "prompts" / "system_prompt.txt"

    def __init__(self, model_name: str = "Qwen/Qwen3-VL-8B-Instruct"):
        self.model_name = model_name
        self._model = None
        self._processor = None

    def load(self) -> None:
        """Load model weights. Call once at startup (heavy operation)."""
        from vlm.model_loader import load_qwen_vl
        self._model, self._processor = load_qwen_vl(self.model_name)

    def plan(self, command: str, images: list[ImageInput]) -> VLMPlan:
        """
        Args:
            command: Natural language task (e.g. "pick the red cup and put it on the shelf").
            images:  One or more scene images. Accepts file paths or PIL Images.
                     Typically: [overview_image] or [overview, close_up].

        Returns:
            VLMPlan with primitive sequence grounded on what the VLM sees in the images.
        """
        if self._model is None:
            raise RuntimeError("Call load() before plan()")

        pil_images = [self._to_pil(img) for img in images]
        system_prompt = self.SYSTEM_PROMPT_PATH.read_text()

        messages = self._build_messages(system_prompt, command, pil_images)
        raw = self._run_inference(messages)
        return self._parse_output(command, raw)

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
    ) -> list[dict]:
        image_content = [{"type": "image", "image": img} for img in images]
        return [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": image_content + [{"type": "text", "text": command}],
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
