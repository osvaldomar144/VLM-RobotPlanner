"""
Tests for PDDL problem generation from VLM plan output.
No GPU, no ROS, no model loading required.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from vlm.planner import VLMPlan, PlanStep
from planner.problem_generator import (
    extract_objects_and_locations,
    infer_init_state,
    infer_goal_state,
    generate_problem,
)


def _plan(*steps: tuple) -> VLMPlan:
    """Helper: build a VLMPlan from (primitive, args_dict) tuples."""
    return VLMPlan(
        goal="test",
        steps=[PlanStep(primitive=p, args=a) for p, a in steps],
        raw_output="",
    )


# ── extract_objects_and_locations ──────────────────────────────────────────────

def test_extract_single_pick_place():
    plan = _plan(
        ("pick",  {"object": "red_cup"}),
        ("place", {"object": "red_cup", "location": "shelf"}),
    )
    objects, locations = extract_objects_and_locations(plan.steps)
    assert objects == {"red_cup"}
    assert locations == {"shelf"}


def test_extract_multiple_objects():
    plan = _plan(
        ("pick",  {"object": "cup"}),
        ("place", {"object": "cup",  "location": "table_b"}),
        ("pick",  {"object": "box"}),
        ("place", {"object": "box",  "location": "shelf"}),
    )
    objects, _ = extract_objects_and_locations(plan.steps)
    assert objects == {"cup", "box"}


def test_extract_ignores_look_at():
    plan = _plan(
        ("look_at", {"object": "cup"}),
        ("pick",    {"object": "cup"}),
        ("place",   {"object": "cup", "location": "shelf"}),
    )
    objects, locations = extract_objects_and_locations(plan.steps)
    assert "cup" in objects
    assert "shelf" in locations


# ── infer_init_state ───────────────────────────────────────────────────────────

def test_init_assigns_source_location():
    plan = _plan(("pick", {"object": "red_cup"}))
    init = infer_init_state(plan.steps)
    assert len(init) == 1
    obj, loc = init[0]
    assert obj == "red_cup"
    assert "red_cup" in loc   # default: source_red_cup


def test_init_no_duplicates_for_same_object():
    plan = _plan(
        ("pick",  {"object": "cup"}),
        ("place", {"object": "cup", "location": "shelf"}),
        ("pick",  {"object": "cup"}),   # second pick of same object
    )
    init = infer_init_state(plan.steps)
    objects_in_init = [o for o, _ in init]
    assert objects_in_init.count("cup") == 1


# ── infer_goal_state ───────────────────────────────────────────────────────────

def test_goal_from_place():
    plan = _plan(
        ("pick",  {"object": "cup"}),
        ("place", {"object": "cup", "location": "shelf"}),
    )
    goal = infer_goal_state(plan.steps)
    assert ("cup", "shelf") in goal


def test_no_goal_if_no_place():
    plan = _plan(("pick", {"object": "cup"}))
    assert infer_goal_state(plan.steps) == []


# ── generate_problem (PDDL string) ─────────────────────────────────────────────

def test_generated_pddl_is_valid_structure():
    plan = _plan(
        ("look_at", {"object": "red_cup"}),
        ("pick",    {"object": "red_cup"}),
        ("place",   {"object": "red_cup", "location": "shelf"}),
    )
    pddl = generate_problem(plan)

    assert "(define (problem" in pddl
    assert "(:domain manipulation)" in pddl
    assert "red_cup - object" in pddl
    assert "- location" in pddl and "shelf" in pddl   # PDDL groups types: "shelf source_red_cup - location"
    assert "(gripper-empty)" in pddl
    assert "(on red_cup shelf)" in pddl     # goal


def test_generated_pddl_multi_goal():
    plan = _plan(
        ("pick",  {"object": "cup"}),
        ("place", {"object": "cup",  "location": "shelf"}),
        ("pick",  {"object": "box"}),
        ("place", {"object": "box",  "location": "table_b"}),
    )
    pddl = generate_problem(plan)
    assert "(and" in pddl
    assert "(on cup shelf)" in pddl
    assert "(on box table_b)" in pddl


def test_reachable_facts_included():
    plan = _plan(
        ("pick",  {"object": "cup"}),
        ("place", {"object": "cup", "location": "shelf"}),
    )
    pddl = generate_problem(plan)
    assert "(reachable shelf)" in pddl
