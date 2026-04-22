"""
Interface to the Fast Downward PDDL planner.
Assumes fast-downward is installed and available as `fast-downward` on PATH.
"""

from __future__ import annotations
import subprocess
import tempfile
import os
from pathlib import Path


class FastDownwardPlanner:
    """
    Runs Fast Downward given a domain and problem PDDL file.
    Returns the plan as a list of action strings.
    """

    SEARCH_CONFIG = "astar(blind())"   # swap for lama-first in production

    def solve(self, domain_path: str, problem_path: str) -> list[str] | None:
        """
        Args:
            domain_path:  Path to domain.pddl
            problem_path: Path to problem.pddl

        Returns:
            List of action strings (e.g. ["(pick red_cup table_a)", ...])
            or None if no plan was found.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            sas_file = os.path.join(tmpdir, "output.sas")
            plan_file = os.path.join(tmpdir, "plan")

            result = subprocess.run(
                [
                    "fast-downward",
                    "--sas-file", sas_file,
                    "--plan-file", plan_file,
                    domain_path,
                    problem_path,
                    "--search", self.SEARCH_CONFIG,
                ],
                capture_output=True,
                text=True,
            )

            if result.returncode not in (0, 1):
                raise RuntimeError(f"Fast Downward error:\n{result.stderr}")

            plan_path = Path(plan_file)
            if not plan_path.exists():
                return None  # unsolvable

            lines = plan_path.read_text().splitlines()
            return [l.strip() for l in lines if l.strip() and not l.startswith(";")]
