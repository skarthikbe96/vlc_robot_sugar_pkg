# robot_sugar/utils/pc_utils.py
from __future__ import annotations

import struct
import numpy as np
from typing import Tuple, Optional

from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2


def decode_pointcloud2_xyzrgb(
    cloud: PointCloud2,
    prefer_rgb_field: bool = True,
    skip_nans: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Decode PointCloud2 into xyz (Nx3 float32) and rgb (Nx3 float32, 0..1).

    Supports:
      - packed 'rgb' or 'rgba' float field
      - split 'r','g','b' fields
      - otherwise rgb default 0.5 gray
    """
    fields = [f.name for f in cloud.fields]

    use_packed_rgb = prefer_rgb_field and (("rgb" in fields) or ("rgba" in fields))
    have_split_rgb = all(c in fields for c in ("r", "g", "b"))

    if use_packed_rgb:
        read_names = ["x", "y", "z", "rgb"] if "rgb" in fields else ["x", "y", "z", "rgba"]
        rgb_idx = 3
        packed_name = read_names[rgb_idx]
    elif have_split_rgb:
        read_names = ["x", "y", "z", "r", "g", "b"]
    else:
        read_names = ["x", "y", "z"]

    pts = []
    cols = []

    for p in point_cloud2.read_points(cloud, field_names=read_names, skip_nans=skip_nans):
        x, y, z = float(p[0]), float(p[1]), float(p[2])
        pts.append((x, y, z))

        if use_packed_rgb and len(p) >= 4:
            rgb_val = float(p[rgb_idx])
            # rgb packed as float32 -> uint32 bits
            s = struct.pack("f", rgb_val)
            b = struct.unpack("I", s)[0]
            r = ((b >> 16) & 0xFF) / 255.0
            g = ((b >> 8) & 0xFF) / 255.0
            bl = (b & 0xFF) / 255.0
            cols.append((r, g, bl))
        elif have_split_rgb and len(p) >= 6:
            r, g, bl = float(p[3]), float(p[4]), float(p[5])
            # normalize if 0..255
            if max(r, g, bl) > 1.5:
                r, g, bl = r / 255.0, g / 255.0, bl / 255.0
            cols.append((r, g, bl))
        else:
            cols.append((0.5, 0.5, 0.5))

    if not pts:
        raise RuntimeError("No points decoded from PointCloud2 (empty after skip_nans?).")

    xyz = np.asarray(pts, dtype=np.float32)
    rgb = np.asarray(cols, dtype=np.float32)
    return xyz, rgb


def pack_rgb_float(rgb01: np.ndarray) -> np.ndarray:
    """
    Pack Nx3 rgb [0..1] into Nx float32 suitable for PointCloud2 'rgb' field (PCL convention).
    """
    rgb255 = np.clip(rgb01 * 255.0, 0, 255).astype(np.uint32)
    packed = (rgb255[:, 0] << 16) | (rgb255[:, 1] << 8) | (rgb255[:, 2])
    return packed.view(np.float32)


def voxel_downsample_xyzrgb(
    xyz: np.ndarray,
    rgb: np.ndarray,
    voxel_size: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fast voxel downsample (keep one point per voxel).
    - xyz: Nx3 in meters
    - voxel_size: e.g. 0.005 (paper 0.5 cm)
    Returns reduced xyz, rgb.

    Note: keeps the first point encountered per voxel (fast). Optionally you can average colors later.
    """
    if xyz.shape[0] == 0:
        return xyz, rgb
    if voxel_size <= 0:
        return xyz, rgb

    # Compute voxel indices
    v = np.floor(xyz / voxel_size).astype(np.int64)

    # Hash 3D integer coords into one int64 key (use large primes)
    key = v[:, 0] * 73856093 + v[:, 1] * 19349663 + v[:, 2] * 83492791

    # Unique keys -> first index
    _, idx = np.unique(key, return_index=True)
    idx = np.sort(idx)

    return xyz[idx], rgb[idx]


def sample_points(
    xyz: np.ndarray,
    rgb: np.ndarray,
    num_points: int,
    method: str = "random",
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sample to exactly num_points.
    method:
      - "random": fast random sample (recommended for real-time)
      - "fps": farthest-point sampling (more paper-like but slower)
    """
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()

    n = xyz.shape[0]
    if n == 0:
        raise RuntimeError("Cannot sample from empty point cloud.")

    if n == num_points:
        return xyz, rgb

    if method.lower() == "random":
        if n >= num_points:
            idx = rng.choice(n, num_points, replace=False)
        else:
            pad = rng.choice(n, num_points - n, replace=True)
            idx = np.concatenate([np.arange(n), pad])
        return xyz[idx], rgb[idx]

    if method.lower() == "fps":
        # Simple FPS: O(n * num_points). Use only if voxel cloud is not huge.
        if n <= num_points:
            # pad if smaller
            pad = rng.choice(n, num_points - n, replace=True)
            idx = np.concatenate([np.arange(n), pad])
            return xyz[idx], rgb[idx]

        # initialize with random point
        start = int(rng.integers(0, n))
        selected = np.empty((num_points,), dtype=np.int64)
        selected[0] = start

        # distances to closest selected point
        d2 = np.sum((xyz - xyz[start]) ** 2, axis=1)

        for i in range(1, num_points):
            farthest = int(np.argmax(d2))
            selected[i] = farthest
            d2 = np.minimum(d2, np.sum((xyz - xyz[farthest]) ** 2, axis=1))

        return xyz[selected], rgb[selected]

    raise ValueError(f"Unknown sampling method: {method}")
