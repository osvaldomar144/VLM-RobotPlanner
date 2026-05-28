"""
Orchestrator — ROS 2 node: bridges the VLM-PDDL pipeline with the robot.

Data flow:
  /vlm_planner/task_command  (std_msgs/String)   → triggers a planning + execution cycle
  /camera/color/image_raw    (sensor_msgs/Image)  → latest camera frame, passed to VLM
  /vlm_planner/status        (std_msgs/String)    → publishes pipeline status (busy / ready / error)

On each task command:
  1. Take the most recent camera frame.
  2. Run Pipeline.run(command, [pil_image]).
  3. For each PrimitiveCall: resolve pose via GazeboOracle, dispatch to primitive.

Primitives dispatched:
  pick, place → MoveIt2 (ArmPrimitive subclasses)
  look_at     → observation pose (Phase 1 fixed pose)
  navigate_to → stub (Phase 3: Nav2)
  open/close_gripper, open/close_container, say → inline handlers

VLM weights are loaded at startup in a background thread so the node
remains responsive during the (slow) GPU model load.
"""

from __future__ import annotations

import sys
import os
import threading
from typing import Any

import rclpy
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import Image

# Locate repo root so planner/, vlm/, simulation/ are importable.
# VLMRP_REPO_ROOT can be set externally; inside Docker the layout is /workspace/<modules>.
_REPO_ROOT = os.environ.get("VLMRP_REPO_ROOT") or "/workspace"
sys.path.insert(0, _REPO_ROOT)

from planner.pipeline import Pipeline, PipelineResult
from planner.plan_parser import PrimitiveCall
from simulation.oracle.world_state import GazeboOracle, GazeboAttach


# All Gazebo model names the oracle tracks by default
_TRACKED_OBJECTS = ["red_cup", "blue_box", "shelf_b", "table"]


class Orchestrator(Node):

    def __init__(self) -> None:
        super().__init__("vlm_robot_planner")

        # ── Pipeline (pure Python — no ROS2 dependency) ───────────────────
        self._pipeline = Pipeline(repair_retries=3)
        self._task_lock = threading.Lock()   # prevents concurrent pipeline runs
        self._busy      = False

        # ── Primitive dispatch table ──────────────────────────────────────
        # Populated in _init_primitives() after MoveIt2 is ready.
        self._prim_dispatch: dict[str, Any] = {}

        # ── Oracle + simulated attachment ─────────────────────────────────
        self._oracle  = GazeboOracle(node=self, reference_frame="panda_link0")
        self._attach  = GazeboAttach(node=self, eef_frame="panda_hand")

        # ── Camera image buffer ───────────────────────────────────────────
        self._latest_image: Image | None = None
        self._image_sub = self.create_subscription(
            Image,
            "/camera/color/image_raw",
            self._camera_callback,
            qos_profile=10,
        )

        # ── Task command subscription ─────────────────────────────────────
        self._cmd_sub = self.create_subscription(
            String,
            "/vlm_planner/task_command",
            self._command_callback,
            qos_profile=10,
        )

        # ── Status publisher ──────────────────────────────────────────────
        self._status_pub = self.create_publisher(String, "/vlm_planner/status", 10)
        self._publish_status("ready — VLM loading in background")

        # ── Load VLM weights in background thread ─────────────────────────
        # GPU load takes ~20-30 s; doing it in background keeps the node alive.
        threading.Thread(target=self._load_vlm_async, daemon=True).start()

        self.get_logger().info(
            "Orchestrator ready. Listening on /vlm_planner/task_command. "
            "VLM loading in background..."
        )

    # ── Startup helpers ───────────────────────────────────────────────────────

    def _load_vlm_async(self) -> None:
        """Load VLM weights and initialise MoveIt2 in a background thread.

        VLM loading is best-effort: in Phase 1 the VLM runs on the HOST (GPU),
        not inside the container. A failure here is expected and non-fatal —
        MoveIt2 is initialised regardless so the arm primitives are always ready.
        """
        try:
            self.get_logger().info("Loading VLM weights (GPU)…")
            self._pipeline.load_vlm()
            self.get_logger().info("VLM loaded.")
        except Exception as exc:
            self.get_logger().warn(
                f"VLM not loaded (expected in container — runs on host): {exc}"
            )
            # Do NOT return: MoveIt2 must be initialised even without VLM.

        try:
            self.get_logger().info("Initialising MoveIt2…")
            self._init_primitives()
            self._setup_planning_scene()
            self.get_logger().info("MoveIt2 ready. Orchestrator fully operational.")
            self._publish_status("ready")
        except Exception as exc:
            self.get_logger().error(f"MoveIt2 init failed: {exc}")
            self._publish_status(f"error — MoveIt2 init failed: {exc}")

    def _setup_planning_scene(self) -> None:
        """Add static environment geometry to MoveIt2's planning scene.

        Robot base is at world z=0.77 m (on the table surface).
        In panda_link0 frame:
          table surface  →  z = 0.00 m
          solid table body → z = -0.77 m … 0.00 m  (centre z = -0.385 m, height = 0.77 m)
        """
        from moveit_msgs.msg import CollisionObject
        from shape_msgs.msg import SolidPrimitive
        from geometry_msgs.msg import Pose

        pub = self.create_publisher(CollisionObject, "/collision_object", 10)

        co = CollisionObject()
        co.header.frame_id = "panda_link0"
        co.header.stamp = self.get_clock().now().to_msg()
        co.id = "table"

        box = SolidPrimitive()
        box.type = SolidPrimitive.BOX
        box.dimensions = [1.20, 1.00, 0.77]

        pose = Pose()
        pose.position.x = 0.30   # table centre: world x=0.50 - robot x=0.20 = 0.30 m
        pose.position.y = 0.00
        pose.position.z = -0.385
        pose.orientation.w = 1.0

        co.primitives      = [box]
        co.primitive_poses = [pose]
        co.operation       = CollisionObject.ADD

        import time as _time
        for _ in range(5):
            pub.publish(co)
            _time.sleep(0.1)

        self.get_logger().info("Planning scene: table collision object added.")

    def _init_primitives(self) -> None:
        """Initialise pymoveit2 and wire up all primitives."""
        from vlm_robot_planner.moveit2_client          import MoveIt2Client as MoveIt2
        from vlm_robot_planner.primitives.base        import ARM_JOINT_NAMES
        from vlm_robot_planner.primitives.pick        import PickPrimitive
        from vlm_robot_planner.primitives.place       import PlacePrimitive
        from vlm_robot_planner.primitives.look_at     import LookAtPrimitive
        from vlm_robot_planner.primitives.navigate_to import NavigateToPrimitive

        cb_group = ReentrantCallbackGroup()
        moveit2  = MoveIt2(
            node              = self,
            joint_names       = ARM_JOINT_NAMES,
            base_link_name    = "panda_link0",
            end_effector_name = "panda_hand",
            group_name        = "panda_arm",
            callback_group    = cb_group,
        )
        moveit2.max_velocity     = 0.3
        moveit2.max_acceleration = 0.3

        pick  = PickPrimitive(self, moveit2, attach=self._attach)
        place = PlacePrimitive(self, moveit2, attach=self._attach)
        look  = LookAtPrimitive(self, moveit2)
        nav   = NavigateToPrimitive(self)

        self._prim_dispatch = {
            "pick":            pick.execute,
            "place":           place.execute,
            "look_at":         look.execute,
            "navigate_to":     nav.execute,
            "open_gripper":    lambda name, pose: pick.open_gripper(),
            "close_gripper":   lambda name, pose: pick.close_gripper(),
            "open_container":  self._exec_open_container,
            "close_container": self._exec_close_container,
            "say":             self._exec_say,
        }

    # ── ROS 2 callbacks ───────────────────────────────────────────────────────

    def _camera_callback(self, msg: Image) -> None:
        """Store the latest camera frame (thread-safe — overwrite is atomic)."""
        self._latest_image = msg

    def _command_callback(self, msg: String) -> None:
        """
        Receive a task command and run the full pipeline in a separate thread
        so the ROS2 executor is not blocked during planning/execution.
        """
        command = msg.data.strip()
        if not command:
            return

        if self._busy:
            self.get_logger().warn(
                f"Orchestrator busy — ignoring command: '{command}'"
            )
            self._publish_status("busy — ignoring new command")
            return

        self.get_logger().info(f"Task received: '{command}'")
        threading.Thread(
            target=self._run_task, args=(command,), daemon=True
        ).start()

    # ── Task execution ────────────────────────────────────────────────────────

    def _run_task(self, command: str) -> None:
        with self._task_lock:
            self._busy = True
            self._publish_status(f"busy — planning: {command}")
            success = False
            try:
                success = self.run(command)
            except Exception as exc:
                self.get_logger().error(f"Orchestrator: unhandled exception: {exc}")
                import traceback
                self.get_logger().error(traceback.format_exc())
            finally:
                self._busy = False
                self._publish_status("ready" if success else f"error — task failed: {command}")

    def run(self, command: str) -> bool:
        """
        Execute one full task:
          images → VLM → PDDL pipeline → primitives → robot.

        Returns:
            True if all primitives executed successfully.
        """
        self.get_logger().info(f"Task: '{command}'")

        # Convert latest ROS2 Image to PIL
        images = []
        if self._latest_image is not None:
            pil = self._ros_image_to_pil(self._latest_image)
            if pil is not None:
                images = [pil]
            else:
                self.get_logger().warn("Image conversion failed — running VLM without image.")
        else:
            self.get_logger().warn("No camera frame available — running VLM without image.")

        # Run planning pipeline
        result: PipelineResult = self._pipeline.run(command, images)

        if not result.success:
            self.get_logger().error(
                f"Pipeline failed at '{result.failure_stage}': {result.error}"
            )
            return False

        self.get_logger().info(
            f"Plan validated — {len(result.primitives)} primitives "
            f"(repair_attempts={result.repair_attempts})"
        )

        # Refresh world state once before dispatch
        world_state = self._oracle.get_world_state(_TRACKED_OBJECTS)

        # Dispatch each primitive
        for prim in result.primitives:
            self.get_logger().info(f"  → {prim.name}({prim.args})")
            ok = self._dispatch(prim, world_state)
            if not ok:
                self.get_logger().error(
                    f"Primitive '{prim.name}' failed — aborting task."
                )
                return False

        self.get_logger().info("Task completed successfully.")
        return True

    # ── Primitive dispatch ────────────────────────────────────────────────────

    def _dispatch(self, prim: PrimitiveCall, world_state) -> bool:
        """Route a PrimitiveCall to the correct handler."""
        handler = self._prim_dispatch.get(prim.name)
        if handler is None:
            self.get_logger().warn(f"Unknown primitive '{prim.name}' — skipping.")
            return True   # unknown primitives are non-fatal in Phase 1

        # Resolve object/location pose from oracle
        obj_name  = prim.args[0] if prim.args else ""
        pose_data = None
        if obj_name:
            pose = world_state.get_pose(obj_name)
            if pose is not None:
                pose_data = {
                    "position":    pose.position,
                    "orientation": pose.orientation,
                }
            else:
                self.get_logger().warn(
                    f"Oracle: no pose for '{obj_name}' — "
                    "primitive will use None pose."
                )

        return handler(obj_name, pose_data)

    # ── Inline primitive handlers ─────────────────────────────────────────────

    def _exec_open_container(self, container: str, pose_data: dict | None) -> bool:
        self.get_logger().info(f"    open_container('{container}') — stub")
        return True

    def _exec_close_container(self, container: str, pose_data: dict | None) -> bool:
        self.get_logger().info(f"    close_container('{container}') — stub")
        return True

    def _exec_say(self, text: str, pose_data: dict | None) -> bool:
        self.get_logger().info(f"    say: \"{text}\"")
        return True

    # ── Camera conversion ─────────────────────────────────────────────────────

    def _ros_image_to_pil(self, msg: Image):
        """Convert sensor_msgs/Image → PIL.Image using cv_bridge + numpy."""
        try:
            from cv_bridge import CvBridge
            from PIL import Image as PilImage
            bridge = CvBridge()
            cv_img = bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")
            return PilImage.fromarray(cv_img)
        except Exception as exc:
            self.get_logger().warn(f"_ros_image_to_pil failed: {exc}")
            return None

    # ── Status ────────────────────────────────────────────────────────────────

    def _publish_status(self, status: str) -> None:
        msg = String()
        msg.data = status
        self._status_pub.publish(msg)


def main() -> None:
    rclpy.init()
    node = Orchestrator()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
