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


def infer_init_state(steps: list[PlanStep]) -> tuple[list[tuple], set[str]]:
    """
    Infer the minimal initial state.

    Handles three cases:
    - Normal (pick + place): objects start on source locations, gripper empty.
    - Pick-only: objects start on source locations, gripper empty.
    - Place-only (closed-loop iteration 2+): arm is already holding the objects
      that appear in place steps but have no corresponding pick step.

    Returns:
        (on_pairs, held_objects)
        on_pairs    — List of (object, location) for :init (on ...) facts.
        held_objects — Set of objects already held (no (on) fact, no gripper-empty).
    """
    on_pairs:    list[tuple[str, str]] = []
    held_objects: set[str]             = set()
    seen:         set[str]             = set()

    # Objects that are explicitly picked in this plan
    picked = {
        step.args.get("object", "")
        for step in steps
        if step.primitive in _PICK_PRIMITIVES and step.args.get("object", "")
    }

    # Objects placed but NOT picked in this plan → arm must already be holding them
    for step in steps:
        if step.primitive in _PLACE_PRIMITIVES:
            obj = step.args.get("object", "")
            if obj and obj not in picked:
                held_objects.add(obj)

    # Objects that are picked → they start on a surface
    for step in steps:
        if step.primitive in _PICK_PRIMITIVES:
            obj = step.args.get("object", "")
            if obj and obj not in seen:
                loc = step.args.get("source", f"source_{obj}")
                on_pairs.append((obj, loc))
                seen.add(obj)

    return on_pairs, held_objects


def infer_goal_state(steps: list[PlanStep]) -> list[tuple[str, ...]]:
    """
    Infer the goal state.

    Returns a list of goal facts as tuples:
    - ("on", obj, loc)      for place goals
    - ("holding", obj)      for pick-only plans (no place step)

    The "holding" sentinel is used when the plan has picks but no places —
    otherwise Fast Downward would return an empty plan because the default
    goal (gripper-empty) is already satisfied in the initial state.
    """
    goal: list[tuple[str, ...]] = []

    for step in steps:
        if step.primitive in _PLACE_PRIMITIVES:
            obj = step.args.get("object", "")
            loc = step.args.get("location", "")
            if obj and loc:
                goal.append(("on", obj, loc))

    # Pick-only plan: goal = holding the picked object(s)
    if not goal:
        for step in steps:
            if step.primitive in _PICK_PRIMITIVES:
                obj = step.args.get("object", "")
                if obj:
                    goal.append(("holding", obj))

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
    init_pairs, held_objects = infer_init_state(plan.steps)
    goal_facts = infer_goal_state(plan.steps)

    all_locations = locations | {loc for _, loc in init_pairs}
    # Held objects are items but have no (on ...) fact and no source location
    all_objects = objects | held_objects

    lines: list[str] = []
    lines.append(f"(define (problem {problem_name})")
    lines.append(f"  (:domain {domain_name})")
    lines.append("")

    # Objects
    obj_str = " ".join(sorted(all_objects)) + " - item"      if all_objects  else ""
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
    for obj in sorted(held_objects):
        lines.append(f"    (holding {obj})")   # arm already holds these (place-only plan)
    for obj in sorted(all_objects):
        lines.append(f"    (clear {obj})")
    for loc in sorted(all_locations):
        lines.append(f"    (reachable {loc})")
    if not held_objects:
        lines.append("    (gripper-empty)")    # omit if arm is holding something
    lines.append("  )")
    lines.append("")

    # Goal
    lines.append("  (:goal")
    on_goals     = [f for f in goal_facts if f[0] == "on"]
    holding_goals = [f for f in goal_facts if f[0] == "holding"]

    all_goal_facts = []
    for f in on_goals:
        all_goal_facts.append(f"(on {f[1]} {f[2]})")
    for f in holding_goals:
        all_goal_facts.append(f"(holding {f[1]})")

    if not all_goal_facts:
        all_goal_facts = ["(gripper-empty)  ; no explicit goal inferred"]

    if len(all_goal_facts) == 1:
        lines.append(f"    {all_goal_facts[0]}")
    else:
        lines.append("    (and")
        for fact in all_goal_facts:
            lines.append(f"      {fact}")
        lines.append("    )")
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
