"""
Main pipeline orchestrator (ROS 2 node).

Flow:
  1. Get world state from oracle (Phase 1) or perception (Phase 2+)
  2. Build scene description → call VLM → get high-level plan
  3. Translate VLM plan to PDDL problem → run Fast Downward
  4. Parse validated plan → dispatch primitives
"""

from __future__ import annotations
import sys
import os

# Allow importing non-ROS modules from the repo root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../../../.."))

import rclpy
from rclpy.node import Node

from vlm.planner import VLMPlanner
from planner.fast_downward import FastDownwardPlanner
from planner.plan_parser import parse_plan
from simulation.oracle.world_state import WorldState


class Orchestrator(Node):

    def __init__(self):
        super().__init__("vlm_robot_planner")

        self._vlm = VLMPlanner()
        self._symbolic_planner = FastDownwardPlanner()

        self.get_logger().info("Orchestrator ready.")

    def run(self, command: str, world_state: WorldState) -> None:
        scene_description = world_state.to_scene_description()

        # Step 1: VLM planning
        vlm_plan = self._vlm.plan(command)
        self.get_logger().info(f"VLM plan: {vlm_plan}")

        # Step 2: Symbolic planning (PDDL validation)
        # TODO: auto-generate problem.pddl from world_state + vlm_plan.goal
        # For now, point to a pre-written problem file
        pddl_plan = self._symbolic_planner.solve(
            domain_path="pddl/domain/manipulation.pddl",
            problem_path="pddl/problems/pick_place_example.pddl",
        )
        if pddl_plan is None:
            self.get_logger().error("Symbolic planner found no valid plan.")
            return

        # Step 3: Parse and dispatch primitives
        primitives = parse_plan(pddl_plan)
        for prim in primitives:
            self.get_logger().info(f"Executing: {prim.name}({prim.args})")
            # TODO: dispatch to PickPrimitive / PlacePrimitive / ...


def main():
    rclpy.init()
    node = Orchestrator()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
