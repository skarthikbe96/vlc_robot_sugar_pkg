# robot_sugar/dataset/sugar_dataset_recorder.py
from __future__ import annotations

import os
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from std_msgs.msg import String, Empty
from sensor_msgs.msg import PointCloud2, JointState
from sensor_msgs_py import point_cloud2

from tf2_ros import Buffer, TransformListener
from tf2_ros import TransformException

from robot_sugar.utils.pc_utils import decode_pointcloud2_xyzrgb
from robot_sugar.utils.io_utils import timestamped_dir, save_npy, save_json, save_text, ensure_dir


class SugarDatasetRecorder(Node):
    def __init__(self):
        super().__init__("sugar_dataset_recorder")

        # --- Params ---
        self.declare_parameter("dataset_root", "/tmp/sugar_dataset")
        self.declare_parameter("instruction_topic", "/task_instruction")
        self.declare_parameter("pc_topic", "/sugar/final_4096_cloud")
        self.declare_parameter("joint_state_topic", "/joint_states")

        self.declare_parameter("base_frame", "panda_link0")
        self.declare_parameter("ee_frame", "panda_hand_tcp")  # change in YAML if needed
        self.declare_parameter("tf_timeout_s", 0.5)

        self.declare_parameter("gripper_joint_names", ["panda_finger_joint1", "panda_finger_joint2"])
        self.declare_parameter("open_threshold", 0.035)  # meters-ish if joint is prismatic; adjust for your model
        self.declare_parameter("record_on_gripper_toggle", True)

        # --- TF ---
        self.tf_buffer = Buffer(cache_time=Duration(seconds=10.0))
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # --- State ---
        self.latest_pc: PointCloud2 | None = None
        self.latest_instruction: str = ""
        self.latest_gripper_open: float | None = None
        self.latest_gripper_bin: int | None = None

        self.episode_dir: str | None = None
        self.step_idx: int = 0

        ensure_dir(self.get_parameter("dataset_root").value)

        # --- Subs ---
        qos = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST, depth=10)

        self.sub_instr = self.create_subscription(
            String, self.get_parameter("instruction_topic").value, self.on_instruction, qos
        )
        self.sub_pc = self.create_subscription(
            PointCloud2, self.get_parameter("pc_topic").value, self.on_pc, qos
        )
        self.sub_js = self.create_subscription(
            JointState, self.get_parameter("joint_state_topic").value, self.on_joint_state, qos
        )

        # --- Manual triggers ---
        self.sub_record_step = self.create_subscription(Empty, "/sugar/record_step", self.on_record_step, qos)
        self.sub_end_episode = self.create_subscription(Empty, "/sugar/end_episode", self.on_end_episode, qos)

        self.get_logger().info(
            "SugarDatasetRecorder ready.\n"
            "Publish instruction to start an episode. Then either:\n"
            "  - toggle gripper (auto record), OR\n"
            "  - publish /sugar/record_step (std_msgs/Empty)\n"
            "Finish with /sugar/end_episode (std_msgs/Empty)."
        )

    def on_pc(self, msg: PointCloud2):
        self.latest_pc = msg

    def on_instruction(self, msg: String):
        instr = msg.data.strip()
        if not instr:
            return
        self.latest_instruction = instr

        # Start a new episode folder
        self.episode_dir = timestamped_dir(self.get_parameter("dataset_root").value, prefix="episode")
        self.step_idx = 0
        save_text(os.path.join(self.episode_dir, "instruction.txt"), instr)
        self.get_logger().info(f"Started episode: {self.episode_dir} | instruction={instr!r}")

    def on_joint_state(self, msg: JointState):
        names = list(msg.name)
        pos = list(msg.position)

        joint_names = list(self.get_parameter("gripper_joint_names").value)
        vals = []
        for jn in joint_names:
            if jn in names:
                vals.append(float(pos[names.index(jn)]))

        if not vals:
            return

        # openness: average of finger joints (works for panda fingers)
        openness = float(np.mean(vals))
        self.latest_gripper_open = openness

        thresh = float(self.get_parameter("open_threshold").value)
        gripper_bin = 1 if openness > thresh else 0

        if self.latest_gripper_bin is None:
            self.latest_gripper_bin = gripper_bin
            return

        # Auto record on toggle
        if bool(self.get_parameter("record_on_gripper_toggle").value):
            if gripper_bin != self.latest_gripper_bin:
                self.latest_gripper_bin = gripper_bin
                self.record_step(reason="gripper_toggle")

    def on_record_step(self, _msg: Empty):
        self.record_step(reason="manual")

    def on_end_episode(self, _msg: Empty):
        if self.episode_dir:
            self.get_logger().info(f"Ended episode: {self.episode_dir}")
        self.episode_dir = None
        self.step_idx = 0

    def get_ee_pose_in_base(self):
        base = self.get_parameter("base_frame").value
        ee = self.get_parameter("ee_frame").value
        timeout_s = float(self.get_parameter("tf_timeout_s").value)

        try:
            tf = self.tf_buffer.lookup_transform(
                base, ee, rclpy.time.Time(),
                timeout=Duration(seconds=timeout_s)
            )
        except TransformException as ex:
            raise RuntimeError(f"TF lookup failed {ee} -> {base}: {ex}")

        t = tf.transform.translation
        q = tf.transform.rotation
        pos = [float(t.x), float(t.y), float(t.z)]
        quat = [float(q.x), float(q.y), float(q.z), float(q.w)]
        return pos, quat

    def record_step(self, reason: str):
        if self.episode_dir is None:
            self.get_logger().warn("No active episode. Publish /task_instruction first.")
            return
        if self.latest_pc is None:
            self.get_logger().warn("No pointcloud received yet on pc_topic.")
            return
        if self.latest_gripper_open is None:
            self.get_logger().warn("No gripper state received yet.")
            return

        # Convert final pointcloud (should already be ~4096) -> Nx6
        try:
            xyz, rgb = decode_pointcloud2_xyzrgb(self.latest_pc)
        except Exception as e:
            self.get_logger().error(f"Failed to decode final cloud: {e}")
            return

        # If it's not exactly 4096, keep it (you can enforce later)
        arr = np.concatenate([xyz.astype(np.float32), rgb.astype(np.float32)], axis=1)

        # EE pose
        try:
            pos, quat = self.get_ee_pose_in_base()
        except Exception as e:
            self.get_logger().error(str(e))
            return

        open_val = float(self.latest_gripper_open)

        step_dir = os.path.join(self.episode_dir, f"step_{self.step_idx:03d}")
        os.makedirs(step_dir, exist_ok=True)

        save_npy(os.path.join(step_dir, "pc.npy"), arr)
        save_json(os.path.join(step_dir, "action.json"), {
            "pos": pos,
            "quat": quat,
            "open": open_val,
            "reason": reason,
            "instruction": self.latest_instruction,
            "base_frame": self.get_parameter("base_frame").value,
            "ee_frame": self.get_parameter("ee_frame").value,
        })

        self.get_logger().info(f"Recorded step {self.step_idx:03d} ({reason}) -> {step_dir}")
        self.step_idx += 1


def main():
    rclpy.init()
    node = SugarDatasetRecorder()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
