"""
Loads the VLM model and processor from HuggingFace.
Isolated here so the rest of the codebase does not import torch/transformers directly.
"""

from __future__ import annotations
from typing import Tuple


def load_qwen_vl(
    model_name: str = "Qwen/Qwen3-VL-8B-Instruct",
    device: str = "cuda",
) -> Tuple:
    """
    Load Qwen3-VL model and processor.

    Returns:
        (model, processor) tuple ready for inference.
    """
    from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
    import torch

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map=device,
    )
    processor = AutoProcessor.from_pretrained(model_name)
    return model, processor
