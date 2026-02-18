# robot_sugar/utils/io_utils.py
from __future__ import annotations

import os
import json
import shutil
from datetime import datetime
from typing import Dict, Any, Optional
import numpy as np


def ensure_empty_dir(path: str, recreate: bool = True) -> None:
    if os.path.exists(path):
        shutil.rmtree(path, ignore_errors=True)
    if recreate:
        os.makedirs(path, exist_ok=True)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def timestamped_dir(root: str, prefix: str = "episode") -> str:
    ensure_dir(root)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    out = os.path.join(root, f"{prefix}_{ts}")
    os.makedirs(out, exist_ok=True)
    return out


def save_npy(path: str, arr: np.ndarray) -> None:
    ensure_dir(os.path.dirname(path))
    np.save(path, arr.astype(np.float32))


def save_json(path: str, obj: Dict[str, Any]) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def save_text(path: str, s: str) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w") as f:
        f.write(s)
