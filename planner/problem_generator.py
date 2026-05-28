"""
Generates a PDDL problem file dynamically from a VLMPlan.

The VLM identifies objects, their locations, and goal state from images.
This module converts that structured output into a valid PDDL problem
compatible with the domain templates in pddl/domains/.

Limitation (Phase 1): initial state is inferred from plan steps (pick → object
must be somewhere). Explicit scene state from the VLM will be added in a future
iteration when the VLMPlan output is extended with a dedicated scene description.
"""

from __future__ import annotations
from pathlib import Path

from vlm.planner import VLMPlan, PlanStep


# Primitives that consume an object from a source
_PICK_PRIMITIVES = {"pick", "unstack", "pick-from-container"}
# Primitives that deposit an object at a destination
_PLACE_PRIMITIVES = {"place", "stack", "place-in-container"}

# Maps VLMPlan.domain_template → PDDL domain name (must match (define (domain ...)) in file)
DOMAIN_TEMPLATE_TO_NAME: dict[str, str] = {
    "manipulation_base":       "manipulation-base",
    "manipulation_stacking":   "manipulation-stacking",
    "containers_manipulation": "manipulation-containers",
    "navigation_manipulation": "manipulation-navigation",
}


def extract_objects_and_locations(steps: list[PlanStep]) -> tuple[set[str], set[str]]:
    """
    Walk the plan steps and collect all object and location names referenced.

    Returns:
        (objects, locations) — two sets of symbolic names.
    """
    objects:   set[str] = set()
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
    Infer the minimal initial state: for each picked object, it must start
    somewhere. If the VLM plan specifies a 'source' key, use that; otherwise
    generate a default source location name.

    Returns:
        List of (object, location) pairs for the :init (on ...) facts.
    """
    init:  list[tuple[str, str]] = []
    seen:  set[str] = set()

    for step in steps:
        if step.primitive in _PICK_PRIMITIVES:
            obj = step.args.get("object", "")
            if obj and obj not in seen:
                loc = step.args.get("source", f"source_{obj}")
                init.append((obj, loc))
                seen.add(obj)

    return init


def infer_goal_state(steps: list[PlanStep]) -> list[tuple[str, str]]:
    """
    Infer the goal state: objects that end up at a place destination.

    Returns:
        List of (object, location) pairs for the :goal (on ...) facts.
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
    domain_name: str | None = None,
    problem_name: str = "generated_problem",
) -> str:
    """
    Generate a PDDL problem string from a VLMPlan.

    The domain name is resolved automatically from plan.domain_template
    unless overridden by the domain_name parameter.

    Args:
        plan:         Output of VLMPlanner.plan().
        domain_name:  Override the domain name. If None, derived from
                      plan.domain_template via DOMAIN_TEMPLATE_TO_NAME.
        problem_name: Label for the generated problem (informational).

    Returns:
        PDDL problem as a string, ready to be written to a .pddl file.
    """
    if domain_name is None:
        domain_name = DOMAIN_TEMPLATE_TO_NAME.get(
            plan.domain_template, "manipulation-base"
        )

    objects, locations = extract_objects_and_locations(plan.steps)
    init_pairs = infer_init_state(plan.steps)
    goal_pairs = infer_goal_state(plan.steps)

    all_locations = locations | {loc for _, loc in init_pairs}

    lines: list[str] = []
    lines.append(f"(define (problem {problem_name})")
    lines.append(f"  (:domain {domain_name})")
    lines.append("")

    # Objects — use 'item' and 'location' types to match the new domain templates
    obj_str = " ".join(sorted(objects)) + " - item"     if objects       else ""
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
    # All items start clear (nothing stacked on them) — required by stacking template
    for obj in sorted(objects):
        lines.append(f"    (clear {obj})")
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
    domain_name: str | None = None,
) -> Path:
    """
    Generate and write the PDDL problem to a file.

    Returns:
        Path to the written file.
    """
    content = generate_problem(plan, domain_name=domain_name)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    return path
