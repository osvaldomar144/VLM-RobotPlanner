"""
Tests for VLM output parsing — no model loading required.
These verify that _parse_output handles VLM responses correctly,
including malformed JSON and markdown code fences.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from vlm.planner import VLMPlanner, PlanStep


def _parser():
    """Return a planner instance without loading weights."""
    p = VLMPlanner.__new__(VLMPlanner)
    return p


def test_clean_json():
    raw = '{"goal": "pick the cup", "steps": [{"primitive": "pick", "args": {"object": "cup"}}]}'
    plan = _parser()._parse_output("pick the cup", raw)
    assert plan.goal == "pick the cup"
    assert len(plan.steps) == 1
    assert plan.steps[0] == PlanStep(primitive="pick", args={"object": "cup"})


def test_json_with_markdown_fences():
    raw = '```json\n{"goal": "pick", "steps": [{"primitive": "look_at", "args": {"object": "box"}}]}\n```'
    plan = _parser()._parse_output("pick", raw)
    assert len(plan.steps) == 1
    assert plan.steps[0].primitive == "look_at"


def test_malformed_json_returns_empty_plan():
    raw = "Sorry, I cannot do that."
    plan = _parser()._parse_output("do something", raw)
    assert plan.steps == []
    assert plan.raw_output == raw


def test_empty_steps_list():
    raw = '{"goal": "impossible task", "steps": [], "error": "no valid primitives"}'
    plan = _parser()._parse_output("impossible task", raw)
    assert plan.steps == []


def test_multi_step_plan():
    raw = """{
        "goal": "pick red cup and place on shelf",
        "steps": [
            {"primitive": "look_at", "args": {"object": "red_cup"}},
            {"primitive": "pick",    "args": {"object": "red_cup"}},
            {"primitive": "place",   "args": {"object": "red_cup", "location": "shelf"}}
        ]
    }"""
    plan = _parser()._parse_output("pick red cup and place on shelf", raw)
    assert len(plan.steps) == 3
    assert plan.steps[1].primitive == "pick"
    assert plan.steps[2].args == {"object": "red_cup", "location": "shelf"}
