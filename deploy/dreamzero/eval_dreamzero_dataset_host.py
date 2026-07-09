#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import cv2
import msgpack
import numpy as np
import pandas as pd
import requests


CAMERA_KEYS = ("middle_zed", "left_camera", "right_camera")
STATE_NAMES = (
    *[f"left_joint_positions_{i}" for i in range(8)],
    *[f"right_joint_positions_{i}" for i in range(8)],
)


def _pack_array(obj: Any):
    if isinstance(obj, np.ndarray):
        return {b"__ndarray__": True, b"data": obj.tobytes(), b"dtype": obj.dtype.str, b"shape": obj.shape}
    if isinstance(obj, np.generic):
        return {b"__npgeneric__": True, b"data": obj.item(), b"dtype": obj.dtype.str}
    return obj


def _unpack_array(obj: dict):
    if b"__ndarray__" in obj:
        return np.ndarray(buffer=obj[b"data"], dtype=np.dtype(obj[b"dtype"]), shape=obj[b"shape"])
    if "__ndarray__" in obj:
        return np.ndarray(buffer=obj["data"], dtype=np.dtype(obj["dtype"]), shape=obj["shape"])
    if b"__npgeneric__" in obj:
        return np.dtype(obj[b"dtype"]).type(obj[b"data"])
    if "__npgeneric__" in obj:
        return np.dtype(obj["dtype"]).type(obj["data"])
    return obj


def packb(data: Any) -> bytes:
    return msgpack.packb(data, default=_pack_array, use_bin_type=True)


def unpackb(data: bytes) -> Any:
    return msgpack.unpackb(data, object_hook=_unpack_array, raw=False)


def load_stats(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    obj = json.loads(path.read_text())
    stats = obj.get("norm_stats", obj)
    action_stats = stats.get("action", stats.get("actions"))
    state_stats = stats["state"]
    if action_stats is None:
        raise KeyError(f"Cannot find action/actions stats in {path}")
    return (
        np.asarray(state_stats["q01"], dtype=np.float32),
        np.asarray(state_stats["q99"], dtype=np.float32),
        np.asarray(action_stats["q01"], dtype=np.float32),
        np.asarray(action_stats["q99"], dtype=np.float32),
    )


def normalize_q01_q99(x: np.ndarray, q01: np.ndarray, q99: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    q01 = np.asarray(q01, dtype=np.float32)
    q99 = np.asarray(q99, dtype=np.float32)
    mask = q01 != q99
    y = np.zeros_like(x, dtype=np.float32)
    y[..., mask] = 2.0 * ((x[..., mask] - q01[mask]) / (q99[mask] - q01[mask])) - 1.0
    y[..., ~mask] = x[..., ~mask]
    return np.ascontiguousarray(np.clip(y, -1.0, 1.0), dtype=np.float32)


def pad_with_mask(x: np.ndarray, dim: int, dtype=np.float32) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=dtype).reshape(-1)
    out = np.zeros((dim,), dtype=dtype)
    n = min(dim, x.shape[0])
    out[:n] = x[:n]
    mask = np.zeros((dim,), dtype=bool)
    mask[:n] = True
    return np.ascontiguousarray(out), np.ascontiguousarray(mask)


def read_rgb_frame(video_path: Path, frame_index: int) -> np.ndarray:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
    ok, frame_bgr = cap.read()
    cap.release()
    if not ok or frame_bgr is None:
        raise RuntimeError(f"Failed to read frame {frame_index} from {video_path}")
    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)


def tile_views(frames: dict[str, np.ndarray]) -> np.ndarray:
    middle = cv2.resize(frames["middle_zed"], (256, 128), interpolation=cv2.INTER_LINEAR)
    left = cv2.resize(frames["left_camera"], (128, 128), interpolation=cv2.INTER_LINEAR)
    right = cv2.resize(frames["right_camera"], (128, 128), interpolation=cv2.INTER_LINEAR)
    return np.ascontiguousarray(np.concatenate([middle, np.concatenate([left, right], axis=1)], axis=0), dtype=np.uint8)


def parquet_file_index(parquet: Path) -> str:
    stem = parquet.stem
    if not stem.startswith("file-"):
        raise ValueError(f"Expected parquet named file-XXX.parquet, got {parquet}")
    return stem.split("-", 1)[1]


def video_paths_for_parquet(task_root: Path, parquet: Path) -> dict[str, Path]:
    file_index = parquet_file_index(parquet)
    return {
        key: task_root / "videos" / f"observation.images.{key}" / "chunk-000" / f"file-{file_index}.mp4"
        for key in CAMERA_KEYS
    }


def prompt_for_row(task_root: Path, row: pd.Series, default_prompt: str) -> str:
    tasks_path = task_root / "meta" / "tasks.parquet"
    if tasks_path.is_file() and "task_index" in row:
        tasks = pd.read_parquet(tasks_path)
        task_index = int(row["task_index"])
        match = tasks[tasks["task_index"] == task_index]
        if len(match):
            return str(match.iloc[0]["task"])
    return default_prompt


def build_payload(
    *,
    row: pd.Series,
    tiled_image: np.ndarray,
    state_q01: np.ndarray,
    state_q99: np.ndarray,
    action_q01: np.ndarray,
    action_q99: np.ndarray,
    prompt: str,
    payload_key: str,
    executed_tiled_images: np.ndarray | None = None,
) -> dict[str, Any]:
    raw_state = np.asarray(row["observation.state"], dtype=np.float32).reshape(-1)[:16]
    state_pad = np.zeros((64,), dtype=np.float32)
    state_mask = np.zeros((64,), dtype=bool)
    action_q01_pad, _ = pad_with_mask(action_q01, 32)
    action_q99_pad, _ = pad_with_mask(action_q99, 32)
    action_mask = np.ones((32,), dtype=bool)

    block = {
        "tiled_image": tiled_image,
        "embodiment_id": 49,
        "state_order": STATE_NAMES,
        "action_order": STATE_NAMES,
        "preprocessed": True,
        "format": "dreamzero_franka_dual_client_preprocessed_v1",
        "task_id": "arrange_vegetables",
        "stats_source": "global_lerobot_q01_q99",
        "state": state_pad,
        "state_raw": raw_state,
        "state_mask": state_mask,
        "state_q01": state_q01,
        "state_q99": state_q99,
        "action_q01": action_q01_pad,
        "action_q99": action_q99_pad,
        "action_mask": action_mask,
        "normalization": {
            "preprocessed": True,
            "state": state_pad,
            "state_raw": raw_state,
            "state_mask": state_mask,
            "action_q01": action_q01_pad,
            "action_q99": action_q99_pad,
            "action_mask": action_mask,
        },
    }
    if executed_tiled_images is not None:
        block["executed_tiled_images"] = np.ascontiguousarray(executed_tiled_images, dtype=np.uint8)
    return {
        "model_type": "dreamzero_client",
        "state": raw_state,
        "prompt": prompt,
        "action_dim": 16,
        "horizon": 48,
        "n_action_steps": 48,
        "tiled_image": tiled_image,
        "stats_task_id": "arrange_vegetables",
        payload_key: block,
    }


def post_actions(host: str, payload: dict[str, Any], timeout_s: float) -> tuple[np.ndarray, dict[str, Any]]:
    url = host.rstrip("/") + "/infer"
    response = requests.post(
        url,
        data=packb(payload),
        headers={"Content-Type": "application/msgpack"},
        timeout=timeout_s,
    )
    if not response.ok:
        raise RuntimeError(f"{response.status_code} {response.reason}: {response.text[:2000]}")
    data = unpackb(response.content)
    actions = np.asarray(data["actions"] if isinstance(data, dict) else data, dtype=np.float32)
    info = data.get("info", {}) if isinstance(data, dict) else {}
    return actions, info


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", required=True)
    parser.add_argument("--task-root", type=Path, default=Path("/inspire/qb-ilm2/project/robot-body/public/bokai/franka_dual_test/arrange_vegetables"))
    parser.add_argument("--parquet", type=Path, default=None)
    parser.add_argument("--norm-stats", type=Path, default=Path("/inspire/qb-ilm2/project/robot-body/public/bokai/franka_dual_test/norm_stats.json"))
    parser.add_argument("--payload-key", default="dreamzero")
    parser.add_argument("--default-prompt", default="arrange the vegetables")
    parser.add_argument("--indices", default="0")
    parser.add_argument("--num-samples", type=int, default=1)
    parser.add_argument("--stride", type=int, default=48)
    parser.add_argument("--timeout-s", type=float, default=180.0)
    parser.add_argument("--unset-proxy", action="store_true", default=True)
    args = parser.parse_args()

    if args.unset_proxy:
        for key in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
            os.environ.pop(key, None)

    parquet = args.parquet or (args.task_root / "data" / "chunk-000" / "file-000.parquet")
    df = pd.read_parquet(parquet)
    video_paths = video_paths_for_parquet(args.task_root, parquet)
    state_q01, state_q99, action_q01, action_q99 = load_stats(args.norm_stats)

    base_indices = [int(x) for x in args.indices.split(",") if x.strip()]
    indices: list[int] = []
    for base in base_indices:
        indices.extend([base + i * args.stride for i in range(args.num_samples)])

    rows = []
    for idx in indices:
        if idx < 0 or idx + 47 >= len(df):
            print(f"[skip] idx={idx} does not have 48 future actions in {parquet.name}")
            continue
        row = df.iloc[idx]
        episode = int(row["episode_index"])
        future = df.iloc[idx : idx + 48]
        if not np.all(future["episode_index"].to_numpy() == episode):
            print(f"[skip] idx={idx} crosses episode boundary")
            continue
        frames = {key: read_rgb_frame(path, idx) for key, path in video_paths.items()}
        tiled = tile_views(frames)
        prompt = prompt_for_row(args.task_root, row, args.default_prompt).lower()
        payload = build_payload(
            row=row,
            tiled_image=tiled,
            state_q01=state_q01,
            state_q99=state_q99,
            action_q01=action_q01,
            action_q99=action_q99,
            prompt=prompt,
            payload_key=args.payload_key,
        )
        started = time.perf_counter()
        pred, info = post_actions(args.host, payload, args.timeout_s)
        elapsed = time.perf_counter() - started
        pred16 = pred[:48, :16]
        gt_raw = np.stack(future["action"].to_numpy()).astype(np.float32)
        gt = normalize_q01_q99(gt_raw, action_q01, action_q99)
        diff = pred16 - gt
        mse = float(np.mean(diff * diff))
        mae = float(np.mean(np.abs(diff)))
        row_out = {
            "idx": idx,
            "episode": episode,
            "frame_index": int(row["frame_index"]),
            "prompt": prompt,
            "mse": mse,
            "mae": mae,
            "pred_abs_gt_095": float(np.mean(np.abs(pred16) > 0.95)),
            "gt_abs_gt_095": float(np.mean(np.abs(gt) > 0.95)),
            "pred_mean": float(np.mean(pred16)),
            "pred_std": float(np.std(pred16)),
            "gt_mean": float(np.mean(gt)),
            "gt_std": float(np.std(gt)),
            "elapsed_s": elapsed,
            "artifact_dir": info.get("artifact_dir") if isinstance(info, dict) else None,
            "action_inference_mode": info.get("action_inference_mode") if isinstance(info, dict) else None,
        }
        rows.append(row_out)
        print(json.dumps(row_out, ensure_ascii=False))

    if rows:
        print(
            json.dumps(
                {
                    "summary": {
                        "count": len(rows),
                        "mse_mean": float(np.mean([r["mse"] for r in rows])),
                        "mae_mean": float(np.mean([r["mae"] for r in rows])),
                        "pred_abs_gt_095_mean": float(np.mean([r["pred_abs_gt_095"] for r in rows])),
                        "gt_abs_gt_095_mean": float(np.mean([r["gt_abs_gt_095"] for r in rows])),
                    }
                },
                ensure_ascii=False,
            )
        )


if __name__ == "__main__":
    main()
