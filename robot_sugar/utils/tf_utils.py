# robot_sugar/utils/tf_utils.py
from __future__ import annotations

import numpy as np
from typing import Tuple
from tf2_ros import Buffer
import rclpy
from rclpy.duration import Duration
from tf2_ros import TransformException


def quat_to_R(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
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


def lookup_T_base_cam(
    tf_buffer: Buffer,
    base_frame: str,
    cam_frame: str,
    timeout_s: float = 0.5
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return R, t such that:
      p_base = p_cam @ R.T + t
    """
    try:
        tf = tf_buffer.lookup_transform(
            base_frame, cam_frame, rclpy.time.Time(),
            timeout=Duration(seconds=float(timeout_s))
        )
    except TransformException as ex:
        raise RuntimeError(f"TF lookup failed {cam_frame} -> {base_frame}: {ex}")

    tx = tf.transform.translation.x
    ty = tf.transform.translation.y
    tz = tf.transform.translation.z
    qx = tf.transform.rotation.x
    qy = tf.transform.rotation.y
    qz = tf.transform.rotation.z
    qw = tf.transform.rotation.w

    R = quat_to_R(qx, qy, qz, qw)
    t = np.array([tx, ty, tz], dtype=np.float64)
    return R, t


def transform_points_cam_to_base(
    xyz_cam: np.ndarray,
    R: np.ndarray,
    t: np.ndarray
) -> np.ndarray:
    """
    xyz_base = xyz_cam @ R.T + t
    """
    return (xyz_cam.astype(np.float64) @ R.T) + t[None, :]
