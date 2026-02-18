# robot_sugar/perception/sugar_perception_node.py
from __future__ import annotations

import os
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from std_msgs.msg import Header, String
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2

from tf2_ros import Buffer, TransformListener

from robot_sugar.utils.pc_utils import (
    decode_pointcloud2_xyzrgb,
    voxel_downsample_xyzrgb,
    sample_points,
    pack_rgb_float,
)
from robot_sugar.utils.tf_utils import (
    lookup_T_base_cam,
    transform_points_cam_to_base,
)
from robot_sugar.utils.io_utils import ensure_empty_dir, ensure_dir, save_npy


def create_cloud_xyzrgb(frame_id: str, stamp, xyz: np.ndarray, rgb01: np.ndarray) -> PointCloud2:
    rgb_f = pack_rgb_float(rgb01)
    pts = np.zeros((xyz.shape[0],), dtype=[("x", np.float32), ("y", np.float32), ("z", np.float32), ("rgb", np.float32)])
    pts["x"] = xyz[:, 0].astype(np.float32)
    pts["y"] = xyz[:, 1].astype(np.float32)
    pts["z"] = xyz[:, 2].astype(np.float32)
    pts["rgb"] = rgb_f

    header = Header()
    header.frame_id = frame_id
    header.stamp = stamp

    fields = [
        point_cloud2.PointField(name="x", offset=0, datatype=point_cloud2.PointField.FLOAT32, count=1),
        point_cloud2.PointField(name="y", offset=4, datatype=point_cloud2.PointField.FLOAT32, count=1),
        point_cloud2.PointField(name="z", offset=8, datatype=point_cloud2.PointField.FLOAT32, count=1),
        point_cloud2.PointField(name="rgb", offset=12, datatype=point_cloud2.PointField.FLOAT32, count=1),
    ]
    return point_cloud2.create_cloud(header, fields, pts)


class SugarPerceptionNode(Node):
    def __init__(self):
        super().__init__("sugar_perception_node")

        # --- Parameters ---
        self.declare_parameter("pc_topic", "/camera_pan_tilt/points_xyzrgb")
        self.declare_parameter("base_frame", "panda_mounted_husky/camera_pan_tilt_link/camera_pan_tilt")
        self.declare_parameter("camera_frame", "panda_mounted_husky/camera_pan_tilt_link/camera_pan_tilt")  # if empty, use cloud.header.frame_id
        self.declare_parameter("tf_timeout_s", 0.5)

        # paper preprocessing
        self.declare_parameter("use_workspace_crop", True)
        self.declare_parameter("crop_x_min", -0.42)
        self.declare_parameter("crop_x_max", 0.80)
        self.declare_parameter("crop_y_min", -0.60)
        self.declare_parameter("crop_y_max", 0.15)
        self.declare_parameter("crop_z_min", 0.30)
        self.declare_parameter("crop_z_max", 0.65)

        self.declare_parameter("voxel_size", 0.005)  # paper: 0.5cm
        self.declare_parameter("num_points", 10000)
        self.declare_parameter("sampling_method", "random")  # random | fps
        self.declare_parameter("publish_debug_clouds", True)
        self.declare_parameter("pc_stats_log_mode", "on_instruction")  # always | once | on_instruction | never
        self.declare_parameter("instruction_topic", "/task_instruction")

        # debug saving
        self.declare_parameter("debug_dir", "/tmp/sugar_debug")
        self.declare_parameter("clear_debug_dir_on_start", True)
        self.declare_parameter("save_debug_npy", True)

        # --- TF ---
        self.tf_buffer = Buffer(cache_time=Duration(seconds=10.0))
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # --- Publishers ---
        self.pub_raw = self.create_publisher(PointCloud2, "/sugar/raw_cloud", 1)
        self.pub_crop = self.create_publisher(PointCloud2, "/sugar/crop_cloud", 1)
        self.pub_voxel = self.create_publisher(PointCloud2, "/sugar/voxel_cloud", 1)
        self.pub_final = self.create_publisher(PointCloud2, "/sugar/final_4096_cloud", 1)

        # --- Subscriber ---
        qos_pc = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self.sub = self.create_subscription(
            PointCloud2,
            self.get_parameter("pc_topic").value,
            self.on_cloud,
            qos_pc,
        )

        # --- Log control ---
        self._pc_stats_logged_once = False
        self._pc_stats_log_on_next = False
        if self.get_parameter("pc_stats_log_mode").value == "on_instruction":
            self.create_subscription(
                String,
                self.get_parameter("instruction_topic").value,
                self._on_instruction,
                10,
            )

        # --- Debug dir ---
        self.debug_dir = self.get_parameter("debug_dir").value
        if bool(self.get_parameter("clear_debug_dir_on_start").value):
            ensure_empty_dir(self.debug_dir, recreate=True)
        else:
            ensure_dir(self.debug_dir)

        self.get_logger().info(f"SugarPerceptionNode ready. Subscribing to {self.get_parameter('pc_topic').value}")

    def _on_instruction(self, msg: String) -> None:
        if msg.data.strip():
            self._pc_stats_log_on_next = True

    def _should_log_pc_stats(self) -> bool:
        mode = str(self.get_parameter("pc_stats_log_mode").value).strip().lower()
        if mode == "always":
            return True
        if mode == "never":
            return False
        if mode == "once":
            if self._pc_stats_logged_once:
                return False
            self._pc_stats_logged_once = True
            return True
        if mode == "on_instruction":
            if not self._pc_stats_log_on_next:
                return False
            self._pc_stats_log_on_next = False
            return True
        self.get_logger().warn(f"Unknown pc_stats_log_mode '{mode}', defaulting to always.")
        return True

    def on_cloud(self, cloud: PointCloud2):
        base_frame = self.get_parameter("base_frame").value
        cam_frame_param = self.get_parameter("camera_frame").value
        cam_frame = cam_frame_param if cam_frame_param else cloud.header.frame_id
        tf_timeout = float(self.get_parameter("tf_timeout_s").value)

        # 1) Decode cloud in camera frame
        try:
            xyz_cam, rgb = decode_pointcloud2_xyzrgb(cloud)
        except Exception as e:
            self.get_logger().error(f"Decode failed: {e}")
            return

        if xyz_cam.shape[0] < 50:
            self.get_logger().warn(f"Too few points decoded: {xyz_cam.shape[0]}")
            return

        # 2) TF: camera -> base (paper uses known extrinsics)
        try:
            R, t = lookup_T_base_cam(self.tf_buffer, base_frame, cam_frame, timeout_s=tf_timeout)
            xyz_base = transform_points_cam_to_base(xyz_cam, R, t)
        except Exception as e:
            self.get_logger().error(str(e))
            return

        stamp = cloud.header.stamp

        # Publish raw (in base frame)
        if bool(self.get_parameter("publish_debug_clouds").value):
            self.pub_raw.publish(create_cloud_xyzrgb(base_frame, stamp, xyz_base.astype(np.float32), rgb))

        # 3) Workspace crop in base frame (paper: fixed box around table)
        xyz_crop = xyz_base
        rgb_crop = rgb
        if bool(self.get_parameter("use_workspace_crop").value):
            x0 = float(self.get_parameter("crop_x_min").value)
            x1 = float(self.get_parameter("crop_x_max").value)
            y0 = float(self.get_parameter("crop_y_min").value)
            y1 = float(self.get_parameter("crop_y_max").value)
            z0 = float(self.get_parameter("crop_z_min").value)
            z1 = float(self.get_parameter("crop_z_max").value)
            m = (
                (xyz_base[:, 0] >= x0) & (xyz_base[:, 0] <= x1) &
                (xyz_base[:, 1] >= y0) & (xyz_base[:, 1] <= y1) &
                (xyz_base[:, 2] >= z0) & (xyz_base[:, 2] <= z1)
            )
            xyz_crop = xyz_base[m]
            rgb_crop = rgb[m]

        if xyz_crop.shape[0] < 50:
            self.get_logger().warn(f"Crop produced too few points: {xyz_crop.shape[0]}. Check crop bounds.")
            return

        if bool(self.get_parameter("publish_debug_clouds").value):
            self.pub_crop.publish(create_cloud_xyzrgb(base_frame, stamp, xyz_crop.astype(np.float32), rgb_crop))

        # 4) Voxel downsample (paper: 0.5cm)
        voxel_size = float(self.get_parameter("voxel_size").value)
        xyz_vox, rgb_vox = voxel_downsample_xyzrgb(xyz_crop.astype(np.float32), rgb_crop.astype(np.float32), voxel_size)

        if xyz_vox.shape[0] < 50:
            self.get_logger().warn(f"Voxel downsample produced too few points: {xyz_vox.shape[0]}.")
            return

        if bool(self.get_parameter("publish_debug_clouds").value):
            self.pub_voxel.publish(create_cloud_xyzrgb(base_frame, stamp, xyz_vox, rgb_vox))

        # 5) Sample to N=4096
        N = int(self.get_parameter("num_points").value)
        method = str(self.get_parameter("sampling_method").value)
        try:
            xyz_fin, rgb_fin = sample_points(xyz_vox, rgb_vox, N, method=method)
        except Exception as e:
            self.get_logger().error(f"Sampling failed: {e}")
            return

        # Publish final
        if bool(self.get_parameter("publish_debug_clouds").value):
            self.pub_final.publish(create_cloud_xyzrgb(base_frame, stamp, xyz_fin, rgb_fin))

        # Save Nx6 npy (paper model input style)
        if bool(self.get_parameter("save_debug_npy").value):
            arr = np.concatenate([xyz_fin, rgb_fin], axis=1).astype(np.float32)
            save_npy(os.path.join(self.debug_dir, "final_4096.npy"), arr)

        if self._should_log_pc_stats():
            self.get_logger().info(
                f"PC: raw={xyz_cam.shape[0]} crop={xyz_crop.shape[0]} voxel={xyz_vox.shape[0]} "
                f"final={xyz_fin.shape[0]} (frame={base_frame})"
            )


def main():
    rclpy.init()
    node = SugarPerceptionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
