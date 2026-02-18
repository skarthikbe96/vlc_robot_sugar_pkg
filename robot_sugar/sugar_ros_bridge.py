#!/usr/bin/env python3
"""
sugar_ros_bridge.py  (ROS 2 Jazzy / Python 3.12)

ROS bridge:
  - subscribes to PointCloud2 (scene)
  - subscribes to /task_instruction (String)
  - calls conda-run standalone inference script
  - publishes /pick_pose and /place_pose in base_frame

This version:
  ✅ uses cloud.header.frame_id (not param) for transforms
  ✅ uses cloud.header.stamp for TF sync (important on moving robot)
  ✅ filters non-finite (inf) points
  ✅ adds debug dumps to /tmp/sugar_debug
  ✅ logs pre/post crop stats + top-red points in cam/base
"""

import json
import os
import struct
import subprocess
import tempfile
import time
from typing import Optional, Tuple

import numpy as np
import rclpy
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2
from std_msgs.msg import String

from tf2_ros import Buffer, TransformListener


class SugarBridge(Node):
    def __init__(self):
        super().__init__("sugar_bridge")

        # --- conda / scripts
        self.declare_parameter("conda_exe", "conda")
        self.declare_parameter("conda_env", "robo3d")
        self.declare_parameter(
            "sugar_script",
            "/home/rebellion/mobile_robotics/vlc_robot_sugar_ws/src/robot_sugar_pkg/robot_sugar/sugar_infer_standalone.py",
        )

        # --- REG (grounding)
        self.declare_parameter(
            "reg_config",
            "/home/rebellion/mobile_robotics/vlc_robot_sugar_ws/src/robot_sugar_pkg/robot_sugar/robo3d/configs/downstream/pct_ocidref.yaml",
        )
        self.declare_parameter(
            "reg_checkpoint",
            "/home/rebellion/mobile_robotics/robot_sugar_ws/src/sugar_policy_ros2/robot_sugar/data3d/experiments/downstreams/ocidref/pc10k-512x32-openclip-init.objaverse4_nolvis.multi.4096.mtgrasp-freeze.enc/ckpts/model_step_80000.pt",
        )
        self.declare_parameter("mask_thresh", 0.5)

        # --- GPS (grasp)
        self.declare_parameter(
            "gps_config",
            "/home/rebellion/mobile_robotics/vlc_robot_sugar_ws/src/robot_sugar_pkg/robot_sugar/robo3d/configs/pretrain/pct_pretrain.yaml",
        )
        self.declare_parameter(
            "gps_checkpoint",
            "/home/rebellion/mobile_robotics/robot_sugar_ws/src/sugar_policy_ros2/robot_sugar/data3d/experiments/pretrain/shapenet/multiobjrandcam1-pc4096.g256.s32-mae.color.0.05-csc.l1.txt.img-openclip-scene.mae.csc.obj.ref-multi.grasp-nodetach-init.shapenet.single/ckpts/model_step_100000.pt",
        )
        self.declare_parameter("grasp_thresh", 0.4)

        # --- frames / topics
        # NOTE: camera_frame param is kept only for info / fallback, but we use cloud.header.frame_id.
        self.declare_parameter("camera_frame", "panda_mounted_husky/camera_pan_tilt_link/camera_pan_tilt")
        self.declare_parameter("base_frame", "panda_link0")
        self.declare_parameter("pc_topic", "/camera/depth/color/points")

        # --- inference params
        self.declare_parameter("device", "cuda:0")
        self.declare_parameter("num_points", 4096)
        self.declare_parameter("unit_scale", 1.0)
        self.declare_parameter("keep_ratio", 0.995)
        self.declare_parameter("timeout_s", 120.0)
        self.declare_parameter("place_lift", 0.15)

        # --- workspace crop (base-frame axis-aligned box)
        self.declare_parameter("use_workspace_crop", True)
        self.declare_parameter("crop_x_min", 0.20)
        self.declare_parameter("crop_x_max", 0.90)
        self.declare_parameter("crop_y_min", -0.60)
        self.declare_parameter("crop_y_max", 0.60)
        self.declare_parameter("crop_z_min", 0.00)
        self.declare_parameter("crop_z_max", 1.20)

        # --- pointcloud read limits (avoid huge CPU)
        self.declare_parameter("max_read_points", 200_000)  # raw points max to read from cloud
        self.declare_parameter("random_seed", 0)

        # --- debug dumps
        self.declare_parameter("debug_dump", True)
        self.declare_parameter("debug_dir", "/tmp/sugar_debug")
        self.declare_parameter("debug_write_ply", True)

        # --- state
        self._latest_pc: Optional[PointCloud2] = None
        self._latest_pc_stamp_wall: Optional[float] = None

        # --- TF
        self.tf_buffer = Buffer(cache_time=Duration(seconds=10.0))
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # --- subs / pubs
        qos_pc = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self.pc_sub = self.create_subscription(
            PointCloud2,
            self.get_parameter("pc_topic").value,
            self._pc_cb,
            qos_pc,
        )
        self.instr_sub = self.create_subscription(String, "/task_instruction", self._instr_cb, 10)

        latched = QoSProfile(
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            reliability=ReliabilityPolicy.RELIABLE,
            depth=1,
        )
        self.pub_pick = self.create_publisher(PoseStamped, "/pick_pose", latched)
        self.pub_place = self.create_publisher(PoseStamped, "/place_pose", latched)

        # seed
        seed = int(self.get_parameter("random_seed").value)
        if seed != 0:
            np.random.seed(seed)

        # ensure debug dir exists
        if bool(self.get_parameter("debug_dump").value):
            os.makedirs(self.get_parameter("debug_dir").value, exist_ok=True)

        self.get_logger().info("SugarBridge ready. Publish to /task_instruction to run inference.")

    def _pc_cb(self, msg: PointCloud2):
        self._latest_pc = msg
        self._latest_pc_stamp_wall = time.time()

    # ---------------- TF helpers ----------------
    def _quat_to_R(self, qx, qy, qz, qw):
        n = (qx*qx + qy*qy + qz*qz + qw*qw) ** 0.5
        if n < 1e-12:
            return np.eye(3, dtype=np.float64)
        qx, qy, qz, qw = qx/n, qy/n, qz/n, qw/n

        xx, yy, zz = qx*qx, qy*qy, qz*qz
        xy, xz, yz = qx*qy, qx*qz, qy*qz
        wx, wy, wz = qw*qx, qw*qy, qw*qz
        return np.array([
            [1-2*(yy+zz), 2*(xy-wz),   2*(xz+wy)],
            [2*(xy+wz),   1-2*(xx+zz), 2*(yz-wx)],
            [2*(xz-wy),   2*(yz+wx),   1-2*(xx+yy)]
        ], dtype=np.float64)

    def _log_cloud_stats(self, tag: str, xyz: np.ndarray, rgb: np.ndarray):
        if xyz.size == 0:
            self.get_logger().warn(f"{tag}: empty xyz")
            return
        mfin = np.isfinite(xyz).all(axis=1)
        fin_ratio = float(mfin.mean())
        self.get_logger().info(
            f"{tag}: N={xyz.shape[0]} finite%={fin_ratio:.3f} "
            f"xyz_min={xyz.min(axis=0)} xyz_max={xyz.max(axis=0)} xyz_mean={xyz.mean(axis=0)}"
        )
        if rgb.size:
            redness = rgb[:, 0] - 0.5*(rgb[:, 1] + rgb[:, 2])
            self.get_logger().info(
                f"{tag}: rgb_mean={rgb.mean(axis=0)} redness(min/mean/max)={(float(redness.min()), float(redness.mean()), float(redness.max()))}"
            )

    def _dump_ply(self, path: str, xyz: np.ndarray, rgb: np.ndarray):
        # Simple ASCII PLY for debugging in CloudCompare/MeshLab
        if xyz.shape[0] == 0:
            return
        rgb255 = np.clip(rgb * 255.0, 0, 255).astype(np.uint8)
        with open(path, "w") as f:
            f.write("ply\nformat ascii 1.0\n")
            f.write(f"element vertex {xyz.shape[0]}\n")
            f.write("property float x\nproperty float y\nproperty float z\n")
            f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
            f.write("end_header\n")
            for p, c in zip(xyz, rgb255):
                f.write(f"{p[0]} {p[1]} {p[2]} {int(c[0])} {int(c[1])} {int(c[2])}\n")

    def _crop_workspace_box(
        self,
        xyz_cam: np.ndarray,
        rgb: np.ndarray,
        cam_frame: str,
        base_frame: str,
        stamp
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        from tf2_ros import TransformException

        try:
            tf = self.tf_buffer.lookup_transform(
                base_frame, cam_frame, stamp,
                timeout=Duration(seconds=0.5)
            )
        except TransformException as ex:
            self.get_logger().warn(f"workspace crop: TF failed {cam_frame}->{base_frame}: {ex}")
            return xyz_cam, rgb, 1.0

        tx = tf.transform.translation.x
        ty = tf.transform.translation.y
        tz = tf.transform.translation.z
        qx = tf.transform.rotation.x
        qy = tf.transform.rotation.y
        qz = tf.transform.rotation.z
        qw = tf.transform.rotation.w

        vals = np.array([tx, ty, tz, qx, qy, qz, qw], dtype=np.float64)
        if not np.isfinite(vals).all():
            self.get_logger().error(f"workspace crop: TF has non-finite values: {vals}")
            return xyz_cam, rgb, 1.0

        R = self._quat_to_R(qx, qy, qz, qw)
        t = np.array([tx, ty, tz], dtype=np.float64)

        # filter non-finite points (inf can exist even with skip_nans=True)
        mfin = np.isfinite(xyz_cam).all(axis=1)
        if mfin.mean() < 1.0:
            self.get_logger().warn(f"workspace crop: dropping non-finite xyz_cam: {(~mfin).sum()} / {xyz_cam.shape[0]}")
        xyz_cam = xyz_cam[mfin]
        rgb = rgb[mfin]

        xyz_base = (xyz_cam.astype(np.float64) @ R.T) + t[None, :]

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
        ratio = float(m.mean()) if m.size else 1.0

        # Debug: where are reddest points in cam/base?
        redness = rgb[:, 0] - 0.5*(rgb[:, 1] + rgb[:, 2])
        top = np.argsort(-redness)[:50]
        if top.size:
            self.get_logger().info(f"Top-red mean xyz_cam: {xyz_cam[top].mean(axis=0)}")
            self.get_logger().info(f"Top-red mean xyz_base: {xyz_base[top].mean(axis=0)}")

        return xyz_cam[m], rgb[m], ratio

    def _transform_pose(self, pose_in: PoseStamped, target_frame: str) -> PoseStamped:
        from tf2_ros import TransformException

        try:
            tf = self.tf_buffer.lookup_transform(
                target_frame,
                pose_in.header.frame_id,
                pose_in.header.stamp,  # sync to the pose time
                timeout=Duration(seconds=0.5),
            )
        except TransformException as ex:
            raise RuntimeError(f"TF lookup failed {pose_in.header.frame_id} -> {target_frame}: {ex}")

        px = pose_in.pose.position.x
        py = pose_in.pose.position.y
        pz = pose_in.pose.position.z
        qx = pose_in.pose.orientation.x
        qy = pose_in.pose.orientation.y
        qz = pose_in.pose.orientation.z
        qw = pose_in.pose.orientation.w

        tx = tf.transform.translation.x
        ty = tf.transform.translation.y
        tz = tf.transform.translation.z
        rqx = tf.transform.rotation.x
        rqy = tf.transform.rotation.y
        rqz = tf.transform.rotation.z
        rqw = tf.transform.rotation.w

        def quat_mul(a, b):
            ax, ay, az, aw = a
            bx, by, bz, bw = b
            return np.array([
                aw*bx + ax*bw + ay*bz - az*by,
                aw*by - ax*bz + ay*bw + az*bx,
                aw*bz + ax*by - ay*bx + az*bw,
                aw*bw - ax*bx - ay*by - az*bz
            ], dtype=np.float64)

        def quat_conj(q):
            x, y, z, w = q
            return np.array([-x, -y, -z, w], dtype=np.float64)

        def quat_rotate(q, v):
            vq = np.array([v[0], v[1], v[2], 0.0], dtype=np.float64)
            return quat_mul(quat_mul(q, vq), quat_conj(q))[:3]

        q_tf = np.array([rqx, rqy, rqz, rqw], dtype=np.float64)
        p_in = np.array([px, py, pz], dtype=np.float64)

        p_out = quat_rotate(q_tf, p_in) + np.array([tx, ty, tz], dtype=np.float64)

        q_in = np.array([qx, qy, qz, qw], dtype=np.float64)
        q_out = quat_mul(q_tf, q_in)
        q_out = q_out / (np.linalg.norm(q_out) + 1e-12)

        out = PoseStamped()
        out.header.frame_id = target_frame
        out.header.stamp = pose_in.header.stamp
        out.pose.position.x = float(p_out[0])
        out.pose.position.y = float(p_out[1])
        out.pose.position.z = float(p_out[2])
        out.pose.orientation.x = float(q_out[0])
        out.pose.orientation.y = float(q_out[1])
        out.pose.orientation.z = float(q_out[2])
        out.pose.orientation.w = float(q_out[3])
        return out

    # ---------------- Main callback ----------------
    def _instr_cb(self, msg: String):
        instr = msg.data.strip()
        if not instr:
            self.get_logger().warn("Empty instruction; ignoring.")
            return

        if self._latest_pc is None or (time.time() - (self._latest_pc_stamp_wall or 0.0)) > 5.0:
            self.get_logger().warn("No recent point cloud yet; waiting for camera cloud.")
            return

        cloud = self._latest_pc
        self.get_logger().info(f"Instruction received: {instr!r} -> running inference.")

        self.get_logger().info(
            f"PC frame_id={cloud.header.frame_id} width={cloud.width} height={cloud.height} "
            f"fields={[f.name for f in cloud.fields]}"
        )
        self.get_logger().info(
            f"params: base_frame={self.get_parameter('base_frame').value} "
            f"use_workspace_crop={self.get_parameter('use_workspace_crop').value}"
        )

        try:
            npy_path = self._write_cloud_to_npy(cloud)
        except Exception as e:
            self.get_logger().error(f"Failed to serialize point cloud: {e}")
            return

        try:
            result = self._run_sugar(npy_path, instr)
        except Exception as e:
            self.get_logger().error(f"Inference failed: {e}")
            return

        if "error" in result:
            self.get_logger().error(f"Returned error: {result['error']}")
            return

        try:
            pick_cam = PoseStamped()
            # IMPORTANT: use the actual cloud frame + stamp for TF consistency
            pick_cam.header.frame_id = cloud.header.frame_id
            pick_cam.header.stamp = cloud.header.stamp
            pick_cam.pose.position.x, pick_cam.pose.position.y, pick_cam.pose.position.z = result["pos"]
            pick_cam.pose.orientation.x, pick_cam.pose.orientation.y, pick_cam.pose.orientation.z, pick_cam.pose.orientation.w = result["quat"]

            base_frame = self.get_parameter("base_frame").value
            pick_base = self._transform_pose(pick_cam, base_frame)

            place_base = PoseStamped()
            place_base.header.frame_id = base_frame
            place_base.header.stamp = pick_base.header.stamp
            place_base.pose = pick_base.pose
            place_base.pose.position.z += float(self.get_parameter("place_lift").value)

            self.pub_pick.publish(pick_base)
            self.pub_place.publish(place_base)

            mp = int(result.get("mask_points", -1))
            gs = float(result.get("best_grasp_score", -1.0))
            self.get_logger().info(
                f"Published /pick_pose and /place_pose in {base_frame}. mask_points={mp}, grasp_score={gs:.3f}"
            )

        except Exception as e:
            self.get_logger().error(f"Failed to transform/publish pose: {e}")

    def _write_cloud_to_npy(self, cloud: PointCloud2) -> str:
        fields = [f.name for f in cloud.fields]
        total = int(cloud.width * cloud.height)

        use_packed_rgb = ("rgb" in fields) or ("rgba" in fields)
        have_split_rgb = all(c in fields for c in ("r", "g", "b"))

        if use_packed_rgb:
            read_names = ["x", "y", "z", "rgb"]
        elif have_split_rgb:
            read_names = ["x", "y", "z", "r", "g", "b"]
        else:
            read_names = ["x", "y", "z"]

        pts = []
        cols = []

        # Limit how many points we read (CPU protection)
        max_read = int(self.get_parameter("max_read_points").value)
        # We’ll do a simple reservoir-like decimation by skipping, but deterministically
        stride = max(1, total // max_read)

        for i, p in enumerate(point_cloud2.read_points(cloud, field_names=read_names, skip_nans=True)):
            if (i % stride) != 0:
                continue
            if len(p) < 3:
                continue

            x, y, z = float(p[0]), float(p[1]), float(p[2])
            pts.append([x, y, z])

            if use_packed_rgb and len(p) >= 4:
                rgb_val = float(p[3])
                s = struct.pack("f", rgb_val)
                b = struct.unpack("I", s)[0]
                r = ((b >> 16) & 0xFF) / 255.0
                g = ((b >> 8) & 0xFF) / 255.0
                bl = (b & 0xFF) / 255.0
                cols.append([r, g, bl])
            elif have_split_rgb and len(p) >= 6:
                r, g, bl = float(p[3]), float(p[4]), float(p[5])
                if max(r, g, bl) > 1.5:
                    r, g, bl = r / 255.0, g / 255.0, bl / 255.0
                cols.append([r, g, bl])
            else:
                cols.append([0.5, 0.5, 0.5])

        if not pts:
            raise RuntimeError("No finite points read from PointCloud2.")

        xyz = np.asarray(pts, dtype=np.float32)
        rgb = np.asarray(cols, dtype=np.float32)

        # Remove inf (skip_nans removes NaNs only)
        mfin = np.isfinite(xyz).all(axis=1)
        if mfin.mean() < 1.0:
            self.get_logger().warn(f"Dropping non-finite points: {(~mfin).sum()} / {xyz.shape[0]}")
        xyz = xyz[mfin]
        rgb = rgb[mfin]

        self._log_cloud_stats("RAW(cam)", xyz, rgb)

        # Debug dump (raw)
        debug_dump = bool(self.get_parameter("debug_dump").value)
        debug_dir = self.get_parameter("debug_dir").value
        if debug_dump:
            os.makedirs(debug_dir, exist_ok=True)
            raw_path = os.path.join(debug_dir, "raw_sample.npy")
            np.save(raw_path, np.concatenate([xyz, rgb], axis=1).astype(np.float32))
            if bool(self.get_parameter("debug_write_ply").value):
                self._dump_ply(os.path.join(debug_dir, "raw_sample.ply"), xyz, rgb)

        # ---------- optional workspace crop ----------
        if bool(self.get_parameter("use_workspace_crop").value):
            cam_frame = cloud.header.frame_id  # IMPORTANT: real frame
            base_frame = self.get_parameter("base_frame").value
            stamp = cloud.header.stamp         # IMPORTANT: sync TF to cloud time

            xyz2, rgb2, ratio = self._crop_workspace_box(xyz, rgb, cam_frame, base_frame, stamp)
            self.get_logger().info(f"Workspace crop kept {xyz2.shape[0]} pts (ratio {ratio:.3f})")

            if xyz2.shape[0] > 300:
                xyz, rgb = xyz2, rgb2
                self._log_cloud_stats("CROP(cam)", xyz, rgb)

                if debug_dump:
                    crop_path = os.path.join(debug_dir, "crop_sample.npy")
                    np.save(crop_path, np.concatenate([xyz, rgb], axis=1).astype(np.float32))
                    if bool(self.get_parameter("debug_write_ply").value):
                        self._dump_ply(os.path.join(debug_dir, "crop_sample.ply"), xyz, rgb)
            else:
                self.get_logger().warn(f"Workspace crop too small ({xyz2.shape[0]} pts); keeping original cloud.")
        # -------------------------------------------

        # Final sample to N points
        N = int(self.get_parameter("num_points").value)
        if xyz.shape[0] >= N:
            idx = np.random.choice(xyz.shape[0], N, replace=False)
        else:
            pad = np.random.choice(xyz.shape[0], N - xyz.shape[0], replace=True)
            idx = np.concatenate([np.arange(xyz.shape[0]), pad])

        arr = np.concatenate([xyz[idx], rgb[idx]], axis=1).astype(np.float32)

        # Save npy in debug dir (stable) instead of random temp, plus also return a temp file path
        if debug_dump:
            npy_debug = os.path.join(debug_dir, "sugar_input.npy")
            np.save(npy_debug, arr)
            self.get_logger().info(f"Debug saved npy: {npy_debug} shape={arr.shape}")

        tmp = tempfile.NamedTemporaryFile(prefix="sugar_pc_", suffix=".npy", delete=False)
        np.save(tmp.name, arr)
        tmp.close()
        return tmp.name

    def _run_sugar(self, npy_path: str, instruction: str) -> dict:
        conda_exe = self.get_parameter("conda_exe").value
        conda_env = self.get_parameter("conda_env").value
        sugar_script = self.get_parameter("sugar_script").value
        device = self.get_parameter("device").value

        reg_config = self.get_parameter("reg_config").value
        reg_checkpoint = self.get_parameter("reg_checkpoint").value
        gps_config = self.get_parameter("gps_config").value
        gps_checkpoint = self.get_parameter("gps_checkpoint").value

        mask_thresh = float(self.get_parameter("mask_thresh").value)
        grasp_thresh = float(self.get_parameter("grasp_thresh").value)

        debug_dump = bool(self.get_parameter("debug_dump").value)
        debug_dir = self.get_parameter("debug_dir").value
        os.makedirs(debug_dir, exist_ok=True)

        # Use deterministic paths in debug_dir if debug enabled
        if debug_dump:
            out_json = os.path.join(debug_dir, "sugar_out.json")
            out_mask_ply = os.path.join(debug_dir, "sugar_mask.ply")
        else:
            out_json = tempfile.NamedTemporaryFile(prefix="sugar_out_", suffix=".json", delete=False).name
            out_mask_ply = tempfile.NamedTemporaryFile(prefix="sugar_mask_", suffix=".ply", delete=False).name

        cmd = [
            conda_exe, "run", "-n", conda_env, "python", sugar_script,
            "--pc", npy_path,
            "--text", instruction,
            "--reg_config", reg_config,
            "--reg_checkpoint", reg_checkpoint,
            "--gps_config", gps_config,
            "--gps_checkpoint", gps_checkpoint,
            "--device", device,
            "--num_points", str(int(self.get_parameter("num_points").value)),
            "--unit_scale", str(float(self.get_parameter("unit_scale").value)),
            "--keep_ratio", str(float(self.get_parameter("keep_ratio").value)),
            "--mask_thresh", str(mask_thresh),
            "--grasp_thresh", str(grasp_thresh),
            "--out_json", out_json,
            "--out_mask_ply", out_mask_ply,
        ]

        self.get_logger().info(" ".join(cmd))
        timeout_s = float(self.get_parameter("timeout_s").value)

        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout_s
        )
        self.get_logger().info(proc.stdout)

        if proc.returncode != 0:
            raise RuntimeError(f"Standalone script failed (rc={proc.returncode}). See log above.")

        with open(out_json, "r") as f:
            data = json.load(f)

        self.get_logger().info(f"Debug mask ply: {out_mask_ply}")
        return data


def main():
    rclpy.init()
    node = SugarBridge()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
