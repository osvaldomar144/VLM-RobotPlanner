"""
Parses a Fast Downward plan (list of PDDL action strings)
into a list of primitive calls ready for execution.
"""

from __future__ import annotations
import re
from dataclasses import dataclass


@dataclass
class PrimitiveCall:
    name: str
    args: list[str]


def parse_plan(pddl_actions: list[str]) -> list[PrimitiveCall]:
    """
    Converts PDDL action strings to PrimitiveCalls.

    Example:
        "(pick red_cup table_a)" → PrimitiveCall(name="pick", args=["red_cup", "table_a"])
    """
    calls = []
    for action in pddl_actions:
        # Strip parentheses: "(pick red_cup table_a)" → "pick red_cup table_a"
        inner = re.sub(r"[()]", "", action).strip()
        parts = inner.split()
        if not parts:
            continue
        calls.append(PrimitiveCall(name=parts[0], args=parts[1:]))
    return calls
