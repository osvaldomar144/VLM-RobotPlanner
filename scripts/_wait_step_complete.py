#!/usr/bin/env python3
"""
_wait_step_complete.py — Runs INSIDE the Docker container.

Waits for a single step completion signal on /vlm_planner/step_complete
and outputs the JSON payload to stdout.

Used by run_loop_host.py to synchronize the closed loop:
  host injects step → waits → reads result → decides next step

Output (stdout, JSON):
    {"step": 0, "primitive": "pick", "success": true, "task_complete": false}

Exit codes:
    0 — received completion signal
    1 — timeout
"""

from __future__ import annotations

import argparse
import json
import sys
import time

import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from std_msgs.msg import String


class _WaitNode(Node):
    def __init__(self) -> None:
        super().__init__("_wait_step_complete")
        self.result: dict | None = None
        # TRANSIENT_LOCAL matches publisher QoS — receives last message even if
        # published before this subscriber was created (race condition fix).
        from rclpy.qos import QoSProfile, DurabilityPolicy
        _latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(
            String, "/vlm_planner/step_complete", self._on_step, _latched
        )
        self.create_subscription(
            String, "/vlm_planner/status", self._on_status, 10
        )

    def _on_step(self, msg: String) -> None:
        if self.result is not None:
            return
        try:
            self.result = json.loads(msg.data)
        except Exception:
            pass

    def _on_status(self, msg: String) -> None:
        """Detect orchestrator errors that prevent step_complete from being published."""
        if self.result is not None:
            return
        status = msg.data.lower()
        if status.startswith("error"):
            self.result = {
                "step": -1, "primitive": "unknown",
                "success": False, "task_complete": False,
                "error": msg.data,
            }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timeout", type=float, default=60.0)
    args = parser.parse_args()

    rclpy.init()
    node     = _WaitNode()
    executor = SingleThreadedExecutor()
    executor.add_node(node)

    t0 = time.time()
    while node.result is None and (time.time() - t0) < args.timeout:
        executor.spin_once(timeout_sec=0.1)

    node.destroy_node()
    rclpy.shutdown()

    if node.result is None:
        print(
            f"[ERROR] No step_complete signal within {args.timeout}s",
            file=sys.stderr,
        )
        sys.exit(1)

    print(json.dumps(node.result))


if __name__ == "__main__":
    main()
