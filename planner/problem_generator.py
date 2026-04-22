"""
Generates a PDDL problem file dynamically from a VLMPlan.

The VLM identifies objects and their goal state from the images.
This module converts that structured output into a valid PDDL problem
that Fast Downward can solve.
"""

from __future__ import annotations
from pathlib import Path

from vlm.planner import VLMPlan, PlanStep


# Primitives that consume an object from a location
_PICK_PRIMITIVES = {"pick"}
# Primitives that deposit an object at a location
_PLACE_PRIMITIVES = {"place"}


def extract_objects_and_locations(steps: list[PlanStep]) -> tuple[set[str], set[str]]:
    """
    Walk the plan steps and collect all object and location names referenced.

    Returns:
        (objects, locations) — two sets of symbolic names.
    """
    objects: set[str] = set()
    locations: set[str] = set()

    for step in steps:
        args = step.args
        if "object" in args:
            objects.add(args["object"])
        if "location" in args:
            locations.add(args["location"])

    return objects, locations


def infer_init_state(steps: list[PlanStep]) -> list[tuple[str, str]]:
    """
    Infer the minimal initial state needed for the plan to be valid.

    Strategy: for each pick(obj), the object must start somewhere.
    If the VLM plan doesn't provide explicit initial locations,
    we assign a generic source location per object.

    Returns:
        List of (object, location) pairs representing the :init state.
    """
    init: list[tuple[str, str]] = []
    seen: set[str] = set()

    for step in steps:
        if step.primitive in _PICK_PRIMITIVES:
            obj = step.args.get("object", "")
            if obj and obj not in seen:
                # Use "source_<obj>" as default initial location when not specified
                loc = step.args.get("source", f"source_{obj}")
                init.append((obj, loc))
                seen.add(obj)

    return init


def infer_goal_state(steps: list[PlanStep]) -> list[tuple[str, str]]:
    """
    Infer the goal state: objects that end up at a place location.

    Returns:
        List of (object, location) pairs representing the :goal.
    """
    goal: list[tuple[str, str]] = []

    for step in steps:
        if step.primitive in _PLACE_PRIMITIVES:
            obj = step.args.get("object", "")
            loc = step.args.get("location", "")
            if obj and loc:
                goal.append((obj, loc))

    return goal


def generate_problem(
    plan: VLMPlan,
    domain_name: str = "manipulation",
    problem_name: str = "generated_problem",
) -> str:
    """
    Generate a PDDL problem string from a VLMPlan.

    Args:
        plan:         Output of VLMPlanner.plan().
        domain_name:  Must match the domain defined in pddl/domain/.
        problem_name: Label for the generated problem (informational).

    Returns:
        PDDL problem as a string, ready to be written to a .pddl file.
    """
    objects, locations = extract_objects_and_locations(plan.steps)
    init_pairs = infer_init_state(plan.steps)
    goal_pairs = infer_goal_state(plan.steps)

    # Collect all locations (including inferred source locations from init)
    all_locations = locations | {loc for _, loc in init_pairs}

    lines: list[str] = []
    lines.append(f"(define (problem {problem_name})")
    lines.append(f"  (:domain {domain_name})")
    lines.append("")

    # Objects
    obj_str = " ".join(sorted(objects)) + " - object" if objects else ""
    loc_str = " ".join(sorted(all_locations)) + " - location" if all_locations else ""
    lines.append("  (:objects")
    if obj_str:
        lines.append(f"    {obj_str}")
    if loc_str:
        lines.append(f"    {loc_str}")
    lines.append("  )")
    lines.append("")

    # Init
    lines.append("  (:init")
    for obj, loc in init_pairs:
        lines.append(f"    (on {obj} {loc})")
    # Add reachable facts for all known locations
    for loc in sorted(all_locations):
        lines.append(f"    (reachable {loc})")
    lines.append("    (gripper-empty)")
    lines.append("  )")
    lines.append("")

    # Goal
    lines.append("  (:goal")
    if len(goal_pairs) == 1:
        obj, loc = goal_pairs[0]
        lines.append(f"    (on {obj} {loc})")
    elif len(goal_pairs) > 1:
        lines.append("    (and")
        for obj, loc in goal_pairs:
            lines.append(f"      (on {obj} {loc})")
        lines.append("    )")
    else:
        lines.append("    (gripper-empty)  ; no explicit goal inferred")
    lines.append("  )")
    lines.append(")")

    return "\n".join(lines)


def write_problem(
    plan: VLMPlan,
    output_path: str | Path,
    domain_name: str = "manipulation",
) -> Path:
    """
    Generate and write the PDDL problem to a file.

    Args:
        plan:        VLMPlan from the VLM module.
        output_path: Where to write the .pddl file.
        domain_name: Domain name to reference in the problem.

    Returns:
        Path to the written file.
    """
    content = generate_problem(plan, domain_name=domain_name)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    return path
