#!/usr/bin/env python3
"""
sugar_infer_standalone.py  (OPTION B)
REG (mask) -> GPS (grasp) -> choose best grasp inside mask

Fix:
- Uses PCPretrainModel (repo class name) instead of PCTPretrainModel.
"""

import argparse
import json
import math
import numpy as np
import torch


# -------------------- Point cloud helpers --------------------
def clean_pc(pc: np.ndarray, unit_scale: float = 1.0, keep_ratio: float = 0.995) -> np.ndarray:
    if pc.size == 0:
        return pc
    pc = pc[np.isfinite(pc).all(axis=1)]
    if pc.size == 0:
        return pc
    pc = pc.astype(np.float32, copy=False)
    pc[:, :3] *= float(unit_scale)

    if pc.shape[0] >= 512:
        med = np.median(pc[:, :3], axis=0)
        d = np.linalg.norm(pc[:, :3] - med[None, :], axis=1)
        thr = np.percentile(d, float(keep_ratio) * 100.0)
        thr = max(float(thr), 1e-3)
        pc = pc[d <= thr]
    return pc


def load_pc(path: str, N: int = 4096, unit_scale: float = 1.0, keep_ratio: float = 0.995) -> np.ndarray:
    if path.endswith(".npy"):
        arr = np.load(path)
    elif path.endswith(".csv"):
        arr = np.loadtxt(path, delimiter=",")
    elif path.endswith(".pcd"):
        import open3d as o3d
        pcd = o3d.io.read_point_cloud(path)
        xyz = np.asarray(pcd.points, dtype=np.float32)
        rgb = np.asarray(pcd.colors, dtype=np.float32) if len(pcd.colors) > 0 else np.full_like(xyz, 0.5)
        arr = np.concatenate([xyz, rgb], axis=1)
    else:
        arr = np.loadtxt(path)

    if arr.ndim != 2 or arr.shape[1] not in (3, 6):
        raise ValueError(f"Point cloud must be (M,3) or (M,6); got {arr.shape}")

    if arr.shape[1] == 3:
        rgb = np.full((arr.shape[0], 3), 0.5, dtype=np.float32)
        pc = np.concatenate([arr.astype(np.float32), rgb], axis=1)
    else:
        pc = arr.astype(np.float32)

    pc = clean_pc(pc, unit_scale=unit_scale, keep_ratio=keep_ratio)
    if pc.shape[0] == 0:
        raise ValueError("No valid points after cleaning.")

    M = pc.shape[0]
    if M >= N:
        idx = np.random.choice(M, N, replace=False)
    else:
        pad = np.random.choice(M, N - M, replace=True)
        idx = np.concatenate([np.arange(M), pad])
    return pc[idx].astype(np.float32)


# -------------------- Math helpers --------------------
def rot_to_quat(R: np.ndarray) -> np.ndarray:
    R = R.astype(np.float64, copy=False)
    tr = float(np.trace(R))
    if tr > 0.0:
        S = math.sqrt(tr + 1.0) * 2.0
        qw = 0.25 * S
        qx = (R[2, 1] - R[1, 2]) / S
        qy = (R[0, 2] - R[2, 0]) / S
        qz = (R[1, 0] - R[0, 1]) / S
    else:
        if (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
            S = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
            qw = (R[2, 1] - R[1, 2]) / S
            qx = 0.25 * S
            qy = (R[0, 1] + R[1, 0]) / S
            qz = (R[0, 2] + R[2, 0]) / S
        elif R[1, 1] > R[2, 2]:
            S = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
            qw = (R[0, 2] - R[2, 0]) / S
            qx = (R[0, 1] + R[1, 0]) / S
            qy = 0.25 * S
            qz = (R[1, 2] + R[2, 1]) / S
        else:
            S = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
            qw = (R[1, 0] - R[0, 1]) / S
            qx = (R[0, 2] + R[2, 0]) / S
            qy = (R[1, 2] + R[2, 1]) / S
            qz = 0.25 * S

    q = np.array([qx, qy, qz, qw], dtype=np.float32)
    n = float(np.linalg.norm(q))
    if n < 1e-8:
        return np.array([0, 0, 0, 1], dtype=np.float32)
    return q / n


def pad_tokens(tokens_list, max_len: int):
    B = len(tokens_list)
    D = int(tokens_list[0].shape[-1])
    dev = tokens_list[0].device
    dt = tokens_list[0].dtype
    out = torch.zeros(B, max_len, D, device=dev, dtype=dt)
    lens = []
    for i, t in enumerate(tokens_list):
        Ti = min(int(t.shape[0]), max_len)
        out[i, :Ti] = t[:Ti]
        lens.append(Ti)
    return out, torch.tensor(lens, device=dev, dtype=torch.long)


def save_mask_ply(xyzrgb: np.ndarray, mask: np.ndarray, out_ply: str):
    try:
        import open3d as o3d
        pts = xyzrgb[mask]
        if pts.shape[0] == 0:
            return
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts[:, :3].astype(np.float64))
        pcd.colors = o3d.utility.Vector3dVector(pts[:, 3:6].astype(np.float64))
        o3d.io.write_point_cloud(out_ply, pcd)
    except Exception:
        pass


def save_full_ply(xyzrgb: np.ndarray, out_ply: str):
    import open3d as o3d
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyzrgb[:, :3].astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(np.clip(xyzrgb[:, 3:6], 0, 1).astype(np.float64))
    o3d.io.write_point_cloud(out_ply, pcd)


# -------------------- main --------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pc", required=True)
    ap.add_argument("--text", required=True)
    ap.add_argument("--ref_phrase", default=None)

    ap.add_argument("--reg_config", required=True)
    ap.add_argument("--reg_checkpoint", required=True)
    ap.add_argument("--mask_thresh", type=float, default=0.1)

    ap.add_argument("--gps_config", required=True)
    ap.add_argument("--gps_checkpoint", required=True)
    ap.add_argument("--grasp_thresh", type=float, default=None)

    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--num_points", type=int, default=10000)
    ap.add_argument("--unit_scale", type=float, default=1.0)
    ap.add_argument("--keep_ratio", type=float, default=0.995)

    ap.add_argument("--out_json", required=True)
    ap.add_argument("--out_mask_ply", default=None)
    ap.add_argument("--pc_raw", default=None, help="Optional raw full cloud .npy to interpolate mask onto")
    ap.add_argument("--out_raw_mask_ply", default="/tmp/raw_mask.ply")

    args = ap.parse_args()
    np.random.seed(0)
    torch.manual_seed(0)
    device = torch.device(args.device)
    ref_phrase = (args.ref_phrase.strip() if args.ref_phrase else args.text.strip())

    # ---- robo3d imports
    from robo3d.configs.default import get_config
    from robo3d.models.pct_ref_models import PCTRefModel
    from robo3d.models.pct_pretrain import PCPretrainModel  
    from robo3d.models.clip_encoder import OpenClipEncoder

    # ---- point cloud
    pc_np = load_pc(args.pc, N=args.num_points, unit_scale=args.unit_scale, keep_ratio=args.keep_ratio)
    save_full_ply(pc_np, "/tmp/scene_sampled_{}.ply".format(args.num_points))

    xyz = pc_np[:, :3].astype(np.float32)
    rgb = pc_np[:, 3:6].astype(np.float32)

    # normalize xyz (same idea as normalize_pc)
    centroid = xyz.mean(axis=0)
    xyz0 = xyz - centroid
    radius = np.max(np.sqrt(np.sum(xyz0 ** 2, axis=1)))
    radius = max(float(radius), 1e-6)
    xyz_norm = xyz0 / radius

    # normalize rgb to [-1, 1]
    rgb = np.clip(rgb, 0.0, 1.0)
    rgb_norm = rgb * 2.0 - 1.0

    pc_np_norm = np.concatenate([xyz_norm, rgb_norm], axis=1).astype(np.float32)
    pc_b = torch.from_numpy(pc_np_norm).to(device).unsqueeze(0)

    # ===================== REG =====================
    reg_cfg = get_config(args.reg_config, opts=[
        "checkpoint_strict_load", "False",
        "MODEL.freeze_all_except_head", "False",
    ])
    reg_model = PCTRefModel(reg_cfg.MODEL).to(device).eval()
    reg_sd_raw = torch.load(args.reg_checkpoint, map_location="cpu")
    reg_sd = reg_sd_raw.get("model", reg_sd_raw)
    reg_model.load_state_dict(reg_sd, strict=False)

    clip_device = torch.device("cpu")
    txt_encoder = OpenClipEncoder(
        model_name="ViT-bigG-14",
        pretrained="laion2b_s39b_b160k",
        # model_name="ViT-L-14",
        # pretrained="laion2b_s32b_b82k",
        device=clip_device
    )

    


    with torch.no_grad():
        toks_list = txt_encoder.forward_text(ref_phrase, use_prompt=True, output_hidden_states=True)

        max_len = 77
        if hasattr(reg_cfg.MODEL, "ref_decoder_config") and hasattr(reg_cfg.MODEL.ref_decoder_config, "max_txt_len"):
            max_len = int(reg_cfg.MODEL.ref_decoder_config.max_txt_len)
        elif hasattr(reg_cfg.MODEL, "max_txt_len"):
            max_len = int(reg_cfg.MODEL.max_txt_len)

        txt_embeds, txt_lens = pad_tokens(toks_list, max_len=max_len)
        if txt_embeds.shape[0] > 1:
            # Prompt ensemble: average token embeddings, keep a single batch entry.
            txt_embeds = txt_embeds.mean(dim=0, keepdim=True)
            txt_lens = torch.tensor([int(txt_lens.max().item())], device=txt_lens.device, dtype=txt_lens.dtype)

        print("\n===== CLIP TEXT DEBUG =====")
        print("ref_phrase:", ref_phrase)
        print("txt_embeds shape:", txt_embeds.shape)
        print("txt_lens:", txt_lens)
        print("txt_embeds min/max/mean:",
            float(txt_embeds.min()),
            float(txt_embeds.max()),
            float(txt_embeds.mean()))
        print("first token sample:", txt_embeds[0, 0, :8])
        print("===========================\n")


        mask_logits = reg_model({"pc_fts": pc_b, "txt_embeds": txt_embeds, "txt_lens": txt_lens}, compute_loss=False)
        if isinstance(mask_logits, (list, tuple)):
            mask_logits = mask_logits[0]

        mask_prob = torch.sigmoid(mask_logits)[0]

        p = mask_prob.detach().cpu().numpy()

        print("\n===== REG MASK DEBUG =====")
        print("mask_prob stats:",
            "min", float(p.min()),
            "mean", float(p.mean()),
            "max", float(p.max()),
            "p90", float(np.quantile(p, 0.90)),
            "p95", float(np.quantile(p, 0.95)),
            "p99", float(np.quantile(p, 0.99)))
        print("mask_thresh:", args.mask_thresh)

        p = mask_prob.detach().cpu().numpy()

        # keep top-K points by probability (more stable than absolute threshold)
        top_ratio = 0.03
        k = max(80, int(top_ratio * p.size))
        thr = np.partition(p, -k)[-k]
        obj_mask_np = (p >= thr)

        # keep largest spatial cluster inside those points (remove scattered outliers)
        try:
            import open3d as o3d
            pts = pc_np[obj_mask_np, :3].astype(np.float64)
            if pts.shape[0] > 0:
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(pts)
                labels = np.array(pcd.cluster_dbscan(eps=0.02, min_points=20, print_progress=False))
                if labels.size > 0 and labels.max() >= 0:
                    best = max(range(labels.max() + 1), key=lambda c: int((labels == c).sum()))
                    keep = (labels == best)
                    idx_all = np.where(obj_mask_np)[0]
                    obj_mask_np2 = np.zeros_like(obj_mask_np, dtype=bool)
                    obj_mask_np2[idx_all[keep]] = True
                    obj_mask_np = obj_mask_np2
        except Exception:
            pass

        obj_mask = torch.from_numpy(obj_mask_np).to(mask_prob.device)
        obj_idx = torch.where(obj_mask)[0].long()

        if obj_idx.numel() > 0:
            idx = obj_idx.cpu().numpy()
            mask_rgb = rgb[idx]
            mask_xyz = pc_np[idx, :3]
            print("MASK rgb mean:", mask_rgb.mean(0), "min:", mask_rgb.min(0), "max:", mask_rgb.max(0))
            print("MASK bbox size (m):", (mask_xyz.max(0) - mask_xyz.min(0)))
            print("MASK z-range:", mask_xyz[:, 2].min(), mask_xyz[:, 2].max())
        print("mask_points after threshold:", int(obj_idx.numel()))
        print("============================\n")
        print("txt_embeds:", tuple(txt_embeds.shape), "device:", txt_embeds.device)

        # ----- Author-style: interpolate logits to raw cloud for clean visualization -----
        if args.pc_raw is not None:
            if not args.pc_raw.endswith(".npy"):
                raise RuntimeError("pc_raw only supports .npy in this debug block")
            raw_arr = np.load(args.pc_raw)
            raw_arr = clean_pc(raw_arr.astype(np.float32), unit_scale=args.unit_scale, keep_ratio=args.keep_ratio)
            raw_xyz = raw_arr[:, :3].astype(np.float32)
            raw_rgb = np.clip(raw_arr[:, 3:6].astype(np.float32), 0.0, 1.0)

            raw_xyz_norm = (raw_xyz - centroid) / radius
            raw_rgb_norm = raw_rgb * 2.0 - 1.0

            from robo3d.utils.pc_utils import three_interpolate_feature
            with torch.no_grad():
                sampled_xyz_norm_t = torch.from_numpy(xyz_norm).to(device).unsqueeze(0)
                raw_xyz_norm_t = torch.from_numpy(raw_xyz_norm).to(device).unsqueeze(0)
                logits_t = mask_logits.unsqueeze(-1)

                raw_logits = three_interpolate_feature(raw_xyz_norm_t, sampled_xyz_norm_t, logits_t)
                raw_prob = torch.sigmoid(raw_logits[0, :, 0]).detach().cpu().numpy()

            raw_mask = raw_prob > float(args.mask_thresh)
            save_mask_ply(
                np.concatenate([raw_xyz, raw_rgb], axis=1).astype(np.float32),
                raw_mask,
                args.out_raw_mask_ply,
            )
            print("Wrote author-style raw mask:", args.out_raw_mask_ply, "mask_points:", int(raw_mask.sum()))
        # -------------------------------------------------------------------------------

    if obj_idx.numel() == 0:
        out = {
            "text": args.text,
            "ref_phrase": ref_phrase,
            "error": "REG produced empty mask",
            "mask_points": 0,
            "pos": [0.0, 0.0, 0.0],
            "quat": [0.0, 0.0, 0.0, 1.0],
            "gripper_open_prob": 1.0,
        }
        with open(args.out_json, "w") as f:
            json.dump(out, f, indent=2)
        print("REG empty mask. Wrote:", args.out_json)
        return

    if args.out_mask_ply:
        save_mask_ply(pc_np, obj_mask.detach().cpu().numpy(), args.out_mask_ply)


    if obj_idx.numel() > 0:
        pts = pc_np[obj_idx.cpu().numpy(), :3]
        print("mask centroid:", pts.mean(0))
        print("mask bbox min:", pts.min(0))
        print("mask bbox max:", pts.max(0))
    print("REG done. mask_points:", int(obj_idx.numel()))


    # ===================== GPS =====================
    gps_cfg = get_config(args.gps_config, opts=["checkpoint_strict_load", "False"])
    gps_model = PCPretrainModel(gps_cfg.MODEL).to(device).eval()
    gps_sd_raw = torch.load(args.gps_checkpoint, map_location="cpu")
    gps_sd = gps_sd_raw.get("model", gps_sd_raw)
    gps_model.load_state_dict(gps_sd, strict=False)

    with torch.no_grad():
        grasp_out = gps_model.forward_grasp({"pc_fts": pc_b}, compute_loss=False)
        g_logits = grasp_out["grasp_logits"][0]
        g_offsets = grasp_out["grasp_offsets"][0]
        g_rots = grasp_out["grasp_rotations"][0]

        g_obj = g_logits[obj_idx]
        best_local = torch.argmax(g_obj)
        best_i = int(obj_idx[best_local].item())

        best_score = float(torch.sigmoid(g_logits[best_i]).item())
        if args.grasp_thresh is not None and best_score < float(args.grasp_thresh):
            out = {
                "text": args.text,
                "ref_phrase": ref_phrase,
                "error": f"Best grasp score {best_score:.3f} < grasp_thresh",
                "mask_points": int(obj_idx.numel()),
                "best_grasp_score": best_score,
                "pos": [0.0, 0.0, 0.0],
                "quat": [0.0, 0.0, 0.0, 1.0],
                "gripper_open_prob": 1.0,
            }
            with open(args.out_json, "w") as f:
                json.dump(out, f, indent=2)
            print("Low grasp score. Wrote:", args.out_json)
            return

        p = pc_b[0, best_i, :3]

        T_norm = (p + g_offsets[best_i]).detach().cpu().numpy()
        T_world = centroid + radius * T_norm
        R = g_rots[best_i].detach().cpu().numpy()
        quat = rot_to_quat(R)

    out = {
        "text": args.text,
        "ref_phrase": ref_phrase,
        "mask_points": int(obj_idx.numel()),
        "best_grasp_score": float(best_score),
        "pos": [float(T_world[0]), float(T_world[1]), float(T_world[2])],
        "quat": [float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])],
        "gripper_open_prob": 0.0,
    }
    with open(args.out_json, "w") as f:
        json.dump(out, f, indent=2)

    print("Wrote:", args.out_json)
    print("mask_points:", out["mask_points"], "best_grasp_score:", out["best_grasp_score"])


if __name__ == "__main__":
    main()
