#!/usr/bin/env python3
"""Filter task-level Franka dual LeRobot v3 datasets by TCP motion.

The source is expected to be the merged layout:

    franka_dual_v2/<task>/{data,meta,videos}

The destination keeps the same LeRobot v3 task-level layout. Data parquet files
are rewritten with fewer rows and fresh numeric stats. Videos are re-encoded
from the kept frames so the filtered dataset has a self-consistent frame index
and timestamp timeline.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.spatial.transform import Rotation as R

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover - tqdm is optional
    tqdm = None

try:
    from numba import njit
except Exception:  # pragma: no cover - optional speedup
    njit = None


STATE_KEY = "observation.state"
ACTION_KEY = "action"
VIDEO_KEYS = (
    "observation.images.left_camera",
    "observation.images.middle_zed",
    "observation.images.right_camera",
)
NUMERIC_STATS_KEYS = (
    ACTION_KEY,
    STATE_KEY,
    "timestamp",
    "frame_index",
    "episode_index",
    "index",
    "task_index",
)
QUANTILES = {
    "q01": 0.01,
    "q10": 0.10,
    "q50": 0.50,
    "q90": 0.90,
    "q99": 0.99,
}
SIM_ROOT = Path("/inspire/qb-ilm2/project/robot-body/public/hairuoliu/franka_dual/sim")

sys.path.insert(0, str(SIM_ROOT))
from fr3_fk import Fr3FK  # noqa: E402


if njit is not None:

    @njit(cache=True)
    def _greedy_keep_indices_numba(
        pos: np.ndarray,
        quat_xyzw: np.ndarray,
        grip: np.ndarray,
        trans_delta: float,
        rot_delta: float,
        gripper_delta: float,
    ) -> np.ndarray:
        n = pos.shape[0]
        keep = np.zeros(n, dtype=np.bool_)
        if n <= 0:
            return np.empty((0,), dtype=np.int64)
        keep[0] = True
        last = 0
        for i in range(1, n):
            trans = 0.0
            for arm in range(2):
                for xyz_i in range(3):
                    d = abs(pos[i, arm, xyz_i] - pos[last, arm, xyz_i])
                    if d > trans:
                        trans = d

            rot = 0.0
            for arm in range(2):
                dot = 0.0
                for k in range(4):
                    dot += quat_xyzw[i, arm, k] * quat_xyzw[last, arm, k]
                if dot < 0:
                    dot = -dot
                if dot > 1:
                    dot = 1.0
                d = 2.0 * np.arccos(dot)
                if d > rot:
                    rot = d

            grip_delta = 0.0
            for arm in range(2):
                d = abs(grip[i, arm] - grip[last, arm])
                if d > grip_delta:
                    grip_delta = d

            if trans > trans_delta or rot > rot_delta or grip_delta > gripper_delta:
                keep[i] = True
                last = i

        count = 0
        for i in range(n):
            if keep[i]:
                count += 1
        out = np.empty((count,), dtype=np.int64)
        j = 0
        for i in range(n):
            if keep[i]:
                out[j] = i
                j += 1
        return out

else:
    _greedy_keep_indices_numba = None


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def format_lerobot_path(
    task_dir: Path,
    template: str,
    *,
    chunk: int,
    file: int,
    video_key: str = "",
) -> Path:
    return (
        task_dir
        / template.format(
            chunk_index=int(chunk),
            file_index=int(file),
            episode_chunk=int(chunk),
            episode_index=int(file),
            video_key=video_key,
        )
    ).resolve()


def discover_tasks(root: Path) -> list[Path]:
    if (root / "meta" / "info.json").is_file():
        return [root]
    return sorted(path for path in root.iterdir() if (path / "meta" / "info.json").is_file())


def load_episode_rows(task_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted((task_dir / "meta" / "episodes").glob("chunk-*/file-*.parquet")):
        rows.extend(pq.read_table(str(path)).to_pylist())
    return rows


def scalar_sequence_stats(values: np.ndarray) -> dict[str, list[float] | list[int]]:
    arr = np.asarray(values, dtype=np.float64).reshape(-1, 1)
    out: dict[str, list[float] | list[int]] = {
        "min": arr.min(axis=0).astype(float).tolist(),
        "max": arr.max(axis=0).astype(float).tolist(),
        "mean": arr.mean(axis=0).astype(float).tolist(),
        "std": arr.std(axis=0).astype(float).tolist(),
        "count": [int(arr.shape[0])],
    }
    for name, q in QUANTILES.items():
        out[name] = np.quantile(arr, q, axis=0).astype(float).tolist()
    return out


def array_stats(arr: np.ndarray) -> dict[str, list[float] | list[int]]:
    values = np.asarray(arr)
    if values.ndim == 1:
        values = values.reshape(-1, 1)
    values = values.astype(np.float64, copy=False)
    out: dict[str, list[float] | list[int]] = {
        "min": values.min(axis=0).astype(float).tolist(),
        "max": values.max(axis=0).astype(float).tolist(),
        "mean": values.mean(axis=0).astype(float).tolist(),
        "std": values.std(axis=0).astype(float).tolist(),
        "count": [int(values.shape[0])],
    }
    for name, q in QUANTILES.items():
        out[name] = np.quantile(values, q, axis=0).astype(float).tolist()
    return out


def write_stat_columns(row: dict[str, Any], key: str, values: np.ndarray) -> None:
    for stat_name, stat_value in array_stats(values).items():
        row[f"stats/{key}/{stat_name}"] = stat_value


def write_scalar_stat_columns(row: dict[str, Any], key: str, values: np.ndarray) -> None:
    for stat_name, stat_value in scalar_sequence_stats(values).items():
        row[f"stats/{key}/{stat_name}"] = stat_value


def collect_numeric_stats(data_files: list[Path]) -> dict[str, dict[str, Any]]:
    arrays: dict[str, list[np.ndarray]] = {key: [] for key in NUMERIC_STATS_KEYS}
    for data_path in data_files:
        schema = set(pq.read_schema(str(data_path)).names)
        columns = [key for key in NUMERIC_STATS_KEYS if key in schema]
        table = pq.read_table(str(data_path), columns=columns)
        for key in columns:
            values = table.column(key).to_pylist()
            if key in (ACTION_KEY, STATE_KEY):
                dtype = np.float32
            elif key == "timestamp":
                dtype = np.float64
            else:
                dtype = np.int64
            arrays[key].append(np.asarray(values, dtype=dtype))
    stats: dict[str, dict[str, Any]] = {}
    for key, parts in arrays.items():
        if parts:
            stats[key] = array_stats(np.concatenate(parts, axis=0))
    return stats


def aggregate_image_stats_from_episodes(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    for key in VIDEO_KEYS:
        count_key = f"stats/{key}/count"
        valid = [row for row in rows if count_key in row]
        if not valid:
            continue
        counts = np.asarray([float((row[count_key] or [0])[0]) for row in valid], dtype=np.float64)
        total = float(counts.sum())
        if total <= 0:
            continue

        def arr(stat_name: str) -> np.ndarray:
            return np.asarray([row[f"stats/{key}/{stat_name}"] for row in valid], dtype=np.float64)

        mean = (arr("mean") * counts[:, None, None, None]).sum(axis=0) / total
        second = ((arr("std") ** 2 + arr("mean") ** 2) * counts[:, None, None, None]).sum(axis=0) / total
        std = np.sqrt(np.maximum(second - mean**2, 0.0))
        out = {
            "min": arr("min").min(axis=0).astype(float).tolist(),
            "max": arr("max").max(axis=0).astype(float).tolist(),
            "mean": mean.astype(float).tolist(),
            "std": std.astype(float).tolist(),
            "count": [int(total)],
        }
        for stat_name in QUANTILES:
            out[stat_name] = (
                (arr(stat_name) * counts[:, None, None, None]).sum(axis=0) / total
            ).astype(float).tolist()
        merged[key] = out
    return merged


def to_thwc_uint8(frames: Any) -> np.ndarray:
    try:
        import torch
    except Exception:  # pragma: no cover - torch is available in training envs
        torch = None
    if torch is not None and torch.is_tensor(frames):
        arr = frames.detach().cpu().numpy()
    else:
        arr = np.asarray(frames)
    if arr.ndim == 3:
        arr = arr[None]
    if arr.ndim != 4:
        raise ValueError(f"Expected decoded video frames with 4 dims, got {arr.shape}")
    if arr.shape[1] in (1, 3, 4) and arr.shape[-1] not in (1, 3, 4):
        arr = np.transpose(arr, (0, 2, 3, 1))
    if arr.dtype != np.uint8:
        arr_f = arr.astype(np.float32, copy=False)
        if arr_f.size and float(arr_f.max()) <= 1.0 + 1e-3:
            arr = np.clip(arr_f * 255.0, 0, 255).astype(np.uint8)
        else:
            arr = np.clip(arr_f, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(arr[..., :3])


def read_video_frames_by_index(video_path: Path, frame_indices: np.ndarray) -> np.ndarray:
    import cv2

    indices = np.asarray(frame_indices, dtype=np.int64)
    if indices.ndim != 1 or indices.size <= 0:
        raise ValueError(f"Expected non-empty 1D frame indices for {video_path}")
    if np.any(indices < 0):
        raise ValueError(f"Negative frame index requested from {video_path}: {indices.min()}")
    if np.any(indices[1:] < indices[:-1]):
        raise ValueError("Frame indices must be sorted for sequential video decode")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open source video: {video_path}")
    frames: list[np.ndarray] = []
    current = -1
    try:
        for target in indices.tolist():
            frame = None
            while current < int(target):
                ok, bgr = cap.read()
                current += 1
                if not ok:
                    raise RuntimeError(
                        f"Failed to read frame {target} from {video_path}; "
                        f"stopped at frame {current}"
                    )
                frame = bgr
            if frame is None:
                raise RuntimeError(f"Duplicate frame index {target} is not supported")
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    finally:
        cap.release()
    return np.ascontiguousarray(np.stack(frames, axis=0))


class VideoFrameWriter:
    def __init__(
        self,
        path: Path,
        *,
        fps: float,
        codec: str,
        crf: int,
        preset: str,
        gop_size: int,
        keyint_min: int,
        sc_threshold: int,
    ):
        import av

        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.container = av.open(str(self.path), "w")
        self.codec = str(codec)
        self.fps = float(fps)
        self.crf = int(crf)
        self.preset = str(preset)
        self.gop_size = int(gop_size)
        self.keyint_min = int(keyint_min)
        self.sc_threshold = int(sc_threshold)
        self.stream = None
        self.frames = 0

    def _init_stream(self, frame: np.ndarray) -> None:
        if abs(self.fps - round(self.fps)) < 1e-6:
            rate = int(round(self.fps))
        else:
            rate = Fraction(int(round(self.fps * 1000)), 1000)
        self.stream = self.container.add_stream(self.codec, rate=rate)
        self.stream.width = int(frame.shape[1])
        self.stream.height = int(frame.shape[0])
        self.stream.pix_fmt = "yuv420p"
        if self.codec in {"libx264", "h264"}:
            self.stream.options = {
                "preset": self.preset,
                "crf": str(self.crf),
                "g": str(self.gop_size),
                "keyint_min": str(self.keyint_min),
                "sc_threshold": str(self.sc_threshold),
            }

    def append(self, frames: np.ndarray) -> None:
        import av

        frames = to_thwc_uint8(frames)
        if frames.shape[0] <= 0:
            return
        if self.stream is None:
            self._init_stream(frames[0])
        for arr in frames:
            frame = av.VideoFrame.from_ndarray(arr, format="rgb24")
            for packet in self.stream.encode(frame):
                self.container.mux(packet)
            self.frames += 1

    def close(self) -> None:
        if self.stream is not None:
            for packet in self.stream.encode():
                self.container.mux(packet)
        self.container.close()


def image_stats(frames: np.ndarray, *, quantile_max_frames: int) -> dict[str, Any]:
    frames = to_thwc_uint8(frames)
    out: dict[str, Any] = {
        "min": frames.min(axis=0).astype(float).tolist(),
        "max": frames.max(axis=0).astype(float).tolist(),
        "mean": frames.mean(axis=0, dtype=np.float64).astype(float).tolist(),
        "std": frames.std(axis=0, dtype=np.float64).astype(float).tolist(),
        "count": [int(frames.shape[0])],
    }
    q_values = frames
    if quantile_max_frames > 0 and frames.shape[0] > int(quantile_max_frames):
        idx = np.linspace(0, frames.shape[0] - 1, int(quantile_max_frames), dtype=np.int64)
        q_values = frames[idx]
    for name, q in QUANTILES.items():
        out[name] = np.quantile(q_values, q, axis=0).astype(float).tolist()
    return out


def openpi_norm_stats_from_lerobot_stats(stats: dict[str, Any]) -> dict[str, Any]:
    return {
        "norm_stats": {
            "state": {
                "mean": stats[STATE_KEY]["mean"],
                "std": stats[STATE_KEY]["std"],
                "q01": stats[STATE_KEY]["q01"],
                "q99": stats[STATE_KEY]["q99"],
            },
            "actions": {
                "mean": stats[ACTION_KEY]["mean"],
                "std": stats[ACTION_KEY]["std"],
                "q01": stats[ACTION_KEY]["q01"],
                "q99": stats[ACTION_KEY]["q99"],
            },
        }
    }


def replace_column(table: pa.Table, name: str, values: np.ndarray) -> pa.Table:
    if name not in table.column_names:
        return table
    idx = table.column_names.index(name)
    return table.set_column(idx, name, pa.array(values, type=table.schema.field(name).type))


def state_to_tcp(states: np.ndarray, fk: Fr3FK) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    states = np.asarray(states, dtype=np.float32)
    if states.ndim != 2 or states.shape[1] < 16:
        raise ValueError(f"Expected state shape [N, >=16], got {states.shape}")
    poses_l = fk.flange_pose(states[:, 0:7])
    poses_r = fk.flange_pose(states[:, 8:15])
    pos = np.stack([poses_l[:, :3, 3], poses_r[:, :3, 3]], axis=1).astype(np.float32)
    quat = np.stack(
        [
            R.from_matrix(poses_l[:, :3, :3]).as_quat(),
            R.from_matrix(poses_r[:, :3, :3]).as_quat(),
        ],
        axis=1,
    ).astype(np.float32)
    grip = np.stack([states[:, 7], states[:, 15]], axis=1).astype(np.float32)
    return pos, quat, grip


def greedy_keep_indices(
    states: np.ndarray,
    fk: Fr3FK,
    *,
    trans_delta: float,
    rot_delta: float,
    gripper_delta: float,
) -> np.ndarray:
    if len(states) <= 0:
        return np.empty((0,), dtype=np.int64)
    if len(states) == 1:
        return np.asarray([0], dtype=np.int64)
    pos, quat, grip = state_to_tcp(states, fk)
    if _greedy_keep_indices_numba is not None:
        return _greedy_keep_indices_numba(pos, quat, grip, trans_delta, rot_delta, gripper_delta)

    keep = [0]
    last = 0
    for i in range(1, len(states)):
        trans = float(np.max(np.abs(pos[i] - pos[last])))
        dot = np.abs(np.sum(quat[i] * quat[last], axis=-1)).clip(0.0, 1.0)
        rot = float(np.max(2.0 * np.arccos(dot)))
        grip_delta = float(np.max(np.abs(grip[i] - grip[last])))
        if trans > trans_delta or rot > rot_delta or grip_delta > gripper_delta:
            keep.append(i)
            last = i
    return np.asarray(keep, dtype=np.int64)


def write_episodes(rows: list[dict[str, Any]], meta_dir: Path, *, chunks_size: int) -> None:
    episodes_root = meta_dir / "episodes"
    episodes_root.mkdir(parents=True, exist_ok=True)
    for chunk_index in sorted({i // chunks_size for i in range(len(rows))}):
        file_index = chunk_index
        start = chunk_index * chunks_size
        chunk_rows = rows[start : start + chunks_size]
        for row in chunk_rows:
            row["meta/episodes/chunk_index"] = int(chunk_index)
            row["meta/episodes/file_index"] = int(file_index)
        out = episodes_root / f"chunk-{chunk_index:03d}" / f"file-{file_index:03d}.parquet"
        out.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.Table.from_pylist(chunk_rows), str(out))


def filter_task(
    src_task: Path,
    dst_task: Path,
    *,
    fk: Fr3FK,
    trans_delta: float,
    rot_delta: float,
    gripper_delta: float,
    video_codec: str,
    video_crf: int,
    video_preset: str,
    video_gop_size: int,
    video_keyint_min: int,
    video_sc_threshold: int,
    image_stats_mode: str,
    image_stats_max_frames: int,
    max_episodes_per_task: int | None,
) -> tuple[list[Path], list[dict[str, Any]], dict[str, Any]]:
    info = read_json(src_task / "meta" / "info.json")
    chunks_size = int(info.get("chunks_size") or 1000)
    fps = float(info.get("fps") or 30.0)
    data_tmpl = str(info["data_path"])
    video_tmpl = str(info["video_path"])
    rows = load_episode_rows(src_task)
    if max_episodes_per_task is not None:
        rows = rows[: int(max_episodes_per_task)]
    if not rows:
        raise RuntimeError(f"No episodes found under {src_task}")

    if (src_task / "meta" / "tasks.parquet").is_file():
        (dst_task / "meta").mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_task / "meta" / "tasks.parquet", dst_task / "meta" / "tasks.parquet")

    src_table_cache: dict[tuple[int, int], pa.Table] = {}
    filtered_tables_by_file: dict[tuple[int, int], list[pa.Table]] = {}
    output_offsets_by_file: dict[tuple[int, int], int] = {}
    video_writers: dict[tuple[str, int, int], VideoFrameWriter] = {}
    video_offsets_by_file: dict[tuple[str, int, int], int] = {}
    episode_rows: list[dict[str, Any]] = []
    frame_start = 0
    original_frames = 0

    progress = None
    row_iter = rows
    if tqdm is not None:
        progress = tqdm(
            rows,
            desc=f"[filter] {src_task.name}",
            unit="ep",
            mininterval=5.0,
            dynamic_ncols=True,
            file=sys.stdout,
        )
        row_iter = progress

    for row_idx, row in enumerate(row_iter):
        if progress is None and (
            row_idx == 0 or (row_idx + 1) % 20 == 0 or row_idx + 1 == len(rows)
        ):
            print(
                f"[filter]   episode {row_idx + 1}/{len(rows)} "
                f"current_keep={frame_start}/{original_frames}",
                flush=True,
            )
        length = int(row.get("length") or row.get("episode_length") or row.get("num_frames") or 0)
        if length <= 0:
            continue
        data_chunk = int(row.get("data/chunk_index", 0))
        data_file = int(row.get("data/file_index", 0))
        data_from = int(row.get("dataset_from_index", 0))
        data_to = int(row.get("dataset_to_index", data_from + length))
        table_key = (data_chunk, data_file)
        if table_key not in src_table_cache:
            src_data = format_lerobot_path(src_task, data_tmpl, chunk=data_chunk, file=data_file)
            src_table_cache[table_key] = pq.read_table(str(src_data))
        segment = src_table_cache[table_key].slice(data_from, max(0, data_to - data_from))
        if segment.num_rows != length:
            raise ValueError(
                f"{src_task.name} episode row {row_idx} length mismatch: "
                f"metadata={length} parquet_slice={segment.num_rows}"
            )

        states = np.asarray(segment.column(STATE_KEY).to_pylist(), dtype=np.float32)
        keep = greedy_keep_indices(
            states,
            fk,
            trans_delta=trans_delta,
            rot_delta=rot_delta,
            gripper_delta=gripper_delta,
        )
        if keep.size <= 0:
            keep = np.asarray([0], dtype=np.int64)
        if "timestamp" in segment.column_names:
            source_relative_timestamps = np.asarray(
                segment.column("timestamp").to_pylist(),
                dtype=np.float64,
            )[keep]
        else:
            source_relative_timestamps = np.asarray(keep, dtype=np.float64) / fps
        filtered = segment.take(pa.array(keep, type=pa.int64()))
        new_length = int(filtered.num_rows)
        episode_index = int(row.get("episode_index", len(episode_rows)))
        task_index_values = (
            np.asarray(filtered.column("task_index").to_pylist(), dtype=np.int64)
            if "task_index" in filtered.column_names
            else np.zeros((new_length,), dtype=np.int64)
        )
        task_index = int(task_index_values[0]) if len(task_index_values) else 0

        filtered = replace_column(
            filtered,
            "episode_index",
            np.full((new_length,), episode_index, dtype=np.int64),
        )
        filtered = replace_column(
            filtered,
            "index",
            np.arange(frame_start, frame_start + new_length, dtype=np.int64),
        )
        filtered = replace_column(
            filtered,
            "frame_index",
            np.arange(new_length, dtype=np.int64),
        )
        filtered = replace_column(
            filtered,
            "task_index",
            np.full((new_length,), task_index, dtype=np.int64),
        )
        new_timestamps = np.arange(new_length, dtype=np.float64) / fps
        filtered = replace_column(filtered, "timestamp", new_timestamps)
        output_file_key = (data_chunk, data_file)
        local_frame_start = int(output_offsets_by_file.get(output_file_key, 0))
        filtered_tables_by_file.setdefault(output_file_key, []).append(filtered)
        output_offsets_by_file[output_file_key] = local_frame_start + new_length

        new_row = dict(row)
        new_row["length"] = new_length
        if "episode_length" in new_row:
            new_row["episode_length"] = new_length
        if "num_frames" in new_row:
            new_row["num_frames"] = new_length
        new_row["data/chunk_index"] = int(data_chunk)
        new_row["data/file_index"] = int(data_file)
        new_row["dataset_from_index"] = int(local_frame_start)
        new_row["dataset_to_index"] = int(local_frame_start + new_length)

        for video_key in VIDEO_KEYS:
            prefix = f"videos/{video_key}"
            src_v_chunk = int(row.get(f"{prefix}/chunk_index", data_chunk))
            src_v_file = int(row.get(f"{prefix}/file_index", data_file))
            src_video = format_lerobot_path(
                src_task,
                video_tmpl,
                chunk=src_v_chunk,
                file=src_v_file,
                video_key=video_key,
            )
            dst_v_chunk = src_v_chunk
            dst_v_file = src_v_file
            dst_video = format_lerobot_path(
                dst_task,
                video_tmpl,
                chunk=dst_v_chunk,
                file=dst_v_file,
                video_key=video_key,
            )
            src_from_timestamp = float(row.get(f"{prefix}/from_timestamp") or 0.0)
            source_frame_indices = np.rint(
                (src_from_timestamp + source_relative_timestamps) * fps
            ).astype(np.int64)
            frames = read_video_frames_by_index(src_video, source_frame_indices)
            if int(frames.shape[0]) != new_length:
                raise ValueError(
                    f"{src_task.name} episode={episode_index} video={video_key} "
                    f"decoded {frames.shape[0]} frames, expected {new_length}"
                )
            writer_key = (video_key, dst_v_chunk, dst_v_file)
            if writer_key not in video_writers:
                video_writers[writer_key] = VideoFrameWriter(
                    dst_video,
                    fps=fps,
                    codec=video_codec,
                    crf=video_crf,
                    preset=video_preset,
                    gop_size=video_gop_size,
                    keyint_min=video_keyint_min,
                    sc_threshold=video_sc_threshold,
                )
            dst_start_frame = int(video_offsets_by_file.get(writer_key, 0))
            video_writers[writer_key].append(frames)
            video_offsets_by_file[writer_key] = dst_start_frame + new_length
            new_row[f"{prefix}/chunk_index"] = dst_v_chunk
            new_row[f"{prefix}/file_index"] = dst_v_file
            new_row[f"{prefix}/from_timestamp"] = float(dst_start_frame) / fps
            new_row[f"{prefix}/to_timestamp"] = float(dst_start_frame + new_length) / fps
            if image_stats_mode == "compute":
                for stat_name, stat_value in image_stats(
                    frames,
                    quantile_max_frames=image_stats_max_frames,
                ).items():
                    new_row[f"stats/{video_key}/{stat_name}"] = stat_value
            elif f"stats/{video_key}/count" in new_row:
                new_row[f"stats/{video_key}/count"] = [int(new_length)]

        actions = np.asarray(filtered.column(ACTION_KEY).to_pylist(), dtype=np.float32)
        states_filtered = np.asarray(filtered.column(STATE_KEY).to_pylist(), dtype=np.float32)
        write_stat_columns(new_row, ACTION_KEY, actions)
        write_stat_columns(new_row, STATE_KEY, states_filtered)
        if "timestamp" in filtered.column_names:
            timestamps = np.asarray(filtered.column("timestamp").to_pylist(), dtype=np.float64)
            write_scalar_stat_columns(new_row, "timestamp", timestamps)
        write_scalar_stat_columns(new_row, "frame_index", np.arange(new_length, dtype=np.int64))
        write_scalar_stat_columns(new_row, "episode_index", np.full((new_length,), episode_index, dtype=np.int64))
        write_scalar_stat_columns(new_row, "index", np.arange(frame_start, frame_start + new_length, dtype=np.int64))
        write_scalar_stat_columns(new_row, "task_index", np.full((new_length,), task_index, dtype=np.int64))
        episode_rows.append(new_row)

        frame_start += new_length
        original_frames += length
        if progress is not None:
            keep_ratio = 100.0 * frame_start / original_frames if original_frames else 0.0
            progress.set_postfix(
                keep=f"{keep_ratio:.2f}%",
                frames=f"{frame_start}/{original_frames}",
                refresh=False,
            )

    if not filtered_tables_by_file:
        raise RuntimeError(f"Filtering removed every episode under {src_task}")

    for writer in video_writers.values():
        writer.close()

    target_data_files: list[Path] = []
    for (chunk_index, file_index), parts in sorted(filtered_tables_by_file.items()):
        dst_data = format_lerobot_path(dst_task, data_tmpl, chunk=chunk_index, file=file_index)
        dst_data.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.concat_tables(parts, promote_options="default"), str(dst_data))
        target_data_files.append(dst_data)

    meta_dir = dst_task / "meta"
    write_episodes(episode_rows, meta_dir, chunks_size=chunks_size)
    stats = collect_numeric_stats(target_data_files)
    stats.update(aggregate_image_stats_from_episodes(episode_rows))
    write_json(meta_dir / "stats.json", stats)
    write_json(dst_task / "norm_stats.json", openpi_norm_stats_from_lerobot_stats(stats))

    out_info = dict(info)
    out_info["total_episodes"] = int(len(episode_rows))
    out_info["total_frames"] = int(frame_start)
    out_info["total_videos"] = int(len(video_writers))
    data_chunks = [chunk for chunk, _ in filtered_tables_by_file]
    video_chunks = [chunk for _, chunk, _ in video_writers]
    out_info["total_chunks"] = int(max([0, *data_chunks, *video_chunks]) + 1)
    out_info["splits"] = {"train": f"0:{len(episode_rows)}"}
    write_json(meta_dir / "info.json", out_info)

    summary = {
        "episodes": int(len(episode_rows)),
        "frames_before": int(original_frames),
        "frames_after": int(frame_start),
        "keep_ratio": float(frame_start / original_frames) if original_frames else 0.0,
        "video_files": int(len(video_writers)),
        "data_files": int(len(target_data_files)),
    }
    return target_data_files, episode_rows, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", "--src-root", dest="src_root", type=Path, required=True)
    parser.add_argument("--dst", "--dst-root", dest="dst_root", type=Path, required=True)
    parser.add_argument("--trans-delta", type=float, default=0.001)
    parser.add_argument("--rot-delta", type=float, default=0.0003)
    parser.add_argument("--gripper-delta", type=float, default=0.001)
    parser.add_argument("--video-codec", default="libx264")
    parser.add_argument("--video-crf", type=int, default=23)
    parser.add_argument("--video-preset", default="veryfast")
    parser.add_argument(
        "--video-gop-size",
        type=int,
        default=2,
        help="FFmpeg/libx264 GOP size. Default 2 matches the original Franka dataset's short keyframe interval.",
    )
    parser.add_argument(
        "--video-keyint-min",
        type=int,
        default=2,
        help="FFmpeg/libx264 minimum keyframe interval. Default 2 keeps random single-frame decoding fast.",
    )
    parser.add_argument(
        "--video-sc-threshold",
        type=int,
        default=0,
        help="FFmpeg/libx264 scene-cut threshold. Default 0 disables scene-cut keyframe decisions.",
    )
    parser.add_argument(
        "--image-stats-mode",
        choices=("reuse", "compute"),
        default="reuse",
        help="reuse keeps source image stat tensors and updates counts; compute recomputes image stats from re-encoded frames.",
    )
    parser.add_argument(
        "--image-stats-max-frames",
        type=int,
        default=256,
        help="Temporal samples for per-pixel image quantiles. min/max/mean/std use every kept frame.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--include-task", action="append", default=None)
    parser.add_argument("--max-episodes-per-task", type=int, default=None)
    args = parser.parse_args()

    src_root = args.src_root.resolve()
    dst_root = args.dst_root.resolve()
    if not src_root.is_dir():
        raise FileNotFoundError(src_root)
    if dst_root.exists():
        if not args.overwrite:
            raise FileExistsError(f"{dst_root} exists; pass --overwrite to replace it")
        shutil.rmtree(dst_root)
    dst_root.mkdir(parents=True, exist_ok=True)

    include = set(args.include_task or [])
    task_dirs = discover_tasks(src_root)
    if include:
        task_dirs = [task for task in task_dirs if task.name in include]
    if not task_dirs:
        raise RuntimeError(f"No task-level LeRobot v3 datasets found under {src_root}")

    fk = Fr3FK(urdf_path=SIM_ROOT / "fr3v2_1.urdf")
    root_data_files: list[Path] = []
    summary: dict[str, Any] = {}
    total_before = 0
    total_after = 0
    total_episodes = 0

    for task_dir in task_dirs:
        print(f"[filter] task={task_dir.name}", flush=True)
        data_files, episode_rows, task_summary = filter_task(
            task_dir,
            dst_root / task_dir.name,
            fk=fk,
            trans_delta=float(args.trans_delta),
            rot_delta=float(args.rot_delta),
            gripper_delta=float(args.gripper_delta),
            video_codec=str(args.video_codec),
            video_crf=int(args.video_crf),
            video_preset=str(args.video_preset),
            video_gop_size=int(args.video_gop_size),
            video_keyint_min=int(args.video_keyint_min),
            video_sc_threshold=int(args.video_sc_threshold),
            image_stats_mode=str(args.image_stats_mode),
            image_stats_max_frames=int(args.image_stats_max_frames),
            max_episodes_per_task=args.max_episodes_per_task,
        )
        root_data_files.extend(data_files)
        summary[task_dir.name] = task_summary
        total_before += int(task_summary["frames_before"])
        total_after += int(task_summary["frames_after"])
        total_episodes += len(episode_rows)
        print(
            "[filter]   frames "
            f"{task_summary['frames_after']}/{task_summary['frames_before']} "
            f"keep={100.0 * task_summary['keep_ratio']:.2f}%",
            flush=True,
        )

    root_stats = collect_numeric_stats(root_data_files)
    write_json(dst_root / "stats.json", root_stats)
    write_json(dst_root / "norm_stats.json", openpi_norm_stats_from_lerobot_stats(root_stats))
    write_json(
        dst_root / "filter_summary.json",
        {
            "src_root": str(src_root),
            "dst_root": str(dst_root),
            "thresholds": {
                "trans_delta": float(args.trans_delta),
                "rot_delta": float(args.rot_delta),
                "gripper_delta": float(args.gripper_delta),
            },
            "video_mode": "reencode",
            "video_reader": "cv2_frame_index",
            "video_codec": str(args.video_codec),
            "video_crf": int(args.video_crf),
            "video_preset": str(args.video_preset),
            "video_gop_size": int(args.video_gop_size),
            "video_keyint_min": int(args.video_keyint_min),
            "video_sc_threshold": int(args.video_sc_threshold),
            "image_stats_mode": str(args.image_stats_mode),
            "image_stats_max_frames": int(args.image_stats_max_frames),
            "max_episodes_per_task": args.max_episodes_per_task,
            "tasks": summary,
            "total_tasks": len(summary),
            "total_episodes": int(total_episodes),
            "frames_before": int(total_before),
            "frames_after": int(total_after),
            "keep_ratio": float(total_after / total_before) if total_before else 0.0,
            "delete_ratio": float(1.0 - total_after / total_before) if total_before else 0.0,
        },
    )
    print(f"[filter] wrote {dst_root}", flush=True)
    print(
        "[filter] total "
        f"frames_after={total_after} frames_before={total_before} "
        f"keep={100.0 * total_after / total_before:.2f}%",
        flush=True,
    )
    print(f"[filter] root stats: {dst_root / 'stats.json'}", flush=True)
    print(f"[filter] OpenPI norm stats: {dst_root / 'norm_stats.json'}", flush=True)


if __name__ == "__main__":
    main()
