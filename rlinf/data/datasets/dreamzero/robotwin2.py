# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import bisect
import json
import os
import random
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from groot.vla.data.schema import DatasetMetadata
from torch.utils.data import Dataset
from torchdata.stateful_dataloader import StatefulDataLoader

from rlinf.data.datasets.dreamzero.sampling_strategy import MultiAnchorTemporalConfig
from rlinf.utils.logging import get_logger

logger = get_logger()

VIDEO_KEYS = ["video.cam_high", "video.cam_left_wrist", "video.cam_right_wrist"]
STATE_KEYS = ["state.left_arm", "state.left_gripper", "state.right_arm", "state.right_gripper"]
ACTION_KEYS = ["action.left_arm", "action.left_gripper", "action.right_arm", "action.right_gripper"]
LANGUAGE_KEYS = ["annotation.language.action_text"]
DEFAULT_ROBOTWIN2_TAG_MAPPING = {"robotwin2": 33}


def _drop_file_cache(path: Path) -> None:
    if not hasattr(os, "posix_fadvise") or not hasattr(os, "POSIX_FADV_DONTNEED"):
        return
    fd = -1
    try:
        fd = os.open(str(path), os.O_RDONLY)
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    except OSError:
        return
    finally:
        if fd >= 0:
            os.close(fd)


def _as_optional_int(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, str) and value.lower() in {"", "none", "null"}:
        return None
    out = int(value)
    return out if out > 0 else None


def _as_tokens(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [part.strip() for part in value.split() if part.strip()]
    return [str(part).strip() for part in value if str(part).strip()]


def _as_path_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(v) for v in value if str(v)]
    text = str(value).strip()
    if not text or text.lower() in {"none", "null"}:
        return []
    if text.startswith("[") and text.endswith("]"):
        text = text[1:-1]
    sep = "," if "," in text else None
    return [part.strip().strip("'\"") for part in text.split(sep) if part.strip()]


def discover_robotwin2_task_paths(
    root: str | Path,
    data_cfg: Any,
    *,
    require_motion: bool = False,
) -> list[str]:
    explicit = _as_path_list(data_cfg.get("robotwin2_task_paths", None))
    if explicit:
        return explicit

    root = Path(root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"RoboTwin2 data root not found: {root}")

    subset_tokens = _as_tokens(data_cfg.get("robotwin2_subsets", None))
    name_contains = data_cfg.get("robotwin2_task_name_contains", None)
    if isinstance(name_contains, str) and name_contains.lower() in {"", "none", "null"}:
        name_contains = None
    max_tasks = _as_optional_int(data_cfg.get("robotwin2_max_tasks", None))

    paths: list[str] = []
    for task_dir in sorted(root.iterdir()):
        if not task_dir.is_dir():
            continue
        if not (task_dir / "meta" / "info.json").is_file():
            continue
        if subset_tokens and not any(token in task_dir.name for token in subset_tokens):
            continue
        if name_contains and str(name_contains) not in task_dir.name and str(name_contains) not in str(task_dir):
            continue
        if require_motion and not ((task_dir / "motions").is_dir() or (task_dir / "motion").is_dir()):
            logger.warning("Skipping RoboTwin2 task without motion dir: %s", task_dir)
            continue
        paths.append(str(task_dir))
        if max_tasks is not None and len(paths) >= max_tasks:
            break
    if not paths:
        raise RuntimeError(
            f"No RoboTwin2 LeRobot tasks discovered under {root} "
            f"with subsets={subset_tokens!r} require_motion={require_motion}."
        )
    return paths


def _slice_stats(stats: dict[str, Any], source_key: str, sl: slice) -> dict[str, list[float]]:
    source = stats[source_key]
    return {
        name: np.asarray(source[name], dtype=np.float32)[sl].tolist()
        for name in ("max", "min", "mean", "std", "q01", "q99")
    }


def _metadata_for_task(task_dir: Path) -> DatasetMetadata:
    meta_dir = task_dir / "meta"
    stats = json.loads((meta_dir / "stats.json").read_text(encoding="utf-8"))
    modality = json.loads((meta_dir / "modality.json").read_text(encoding="utf-8"))
    info = json.loads((meta_dir / "info.json").read_text(encoding="utf-8"))
    fps = float(info.get("fps", 30))

    state_stats: dict[str, dict[str, list[float]]] = {}
    action_stats: dict[str, dict[str, list[float]]] = {}
    state_modalities: dict[str, dict[str, Any]] = {}
    action_modalities: dict[str, dict[str, Any]] = {}

    for key in ("left_arm", "left_gripper", "right_arm", "right_gripper"):
        entry = modality["state"][key]
        sl = slice(int(entry.get("start", 0)), int(entry.get("end")))
        state_stats[key] = _slice_stats(stats, str(entry.get("original_key", "observation.state")), sl)
        width = sl.stop - sl.start
        state_modalities[key] = {
            "absolute": True,
            "shape": [width],
            "continuous": "gripper" not in key,
        }

        entry = modality["action"][key]
        sl = slice(int(entry.get("start", 0)), int(entry.get("end")))
        action_stats[key] = _slice_stats(stats, str(entry.get("original_key", "action")), sl)
        width = sl.stop - sl.start
        action_modalities[key] = {
            "absolute": True,
            "shape": [width],
            "continuous": "gripper" not in key,
        }

    video_modalities: dict[str, dict[str, Any]] = {}
    for key in ("cam_high", "cam_left_wrist", "cam_right_wrist"):
        original_key = str(modality["video"][key].get("original_key", f"observation.images.{key}"))
        feature = (info.get("features") or {}).get(original_key, {})
        shape = feature.get("shape") or [3, 128, 160]
        channels = int(shape[0] if len(shape) == 3 else 3)
        height = int(shape[-2] if len(shape) >= 2 else 128)
        width = int(shape[-1] if len(shape) >= 1 else 160)
        video_modalities[key] = {
            "resolution": [width, height],
            "channels": channels,
            "fps": fps,
        }

    blob = {
        "statistics": {"state": state_stats, "action": action_stats},
        "modalities": {
            "video": video_modalities,
            "state": state_modalities,
            "action": action_modalities,
        },
        "embodiment_tag": "robotwin2",
    }
    return DatasetMetadata.model_validate(blob)


def _metadata_to_json(metadata: DatasetMetadata) -> dict[str, Any]:
    if hasattr(metadata, "model_dump"):
        return metadata.model_dump(mode="json")
    return json.loads(metadata.json())


def _merge_stats(
    per_task_stats: list[dict[str, dict[str, Any]]],
    *,
    dataset_weights: list[float] | np.ndarray | None = None,
    percentile_mixing_method: str = "min_max",
) -> dict[str, dict[str, list[float]]]:
    if not per_task_stats:
        raise ValueError("Cannot merge empty RoboTwin2 metadata list")
    if dataset_weights is None:
        weights = np.ones(len(per_task_stats), dtype=np.float64)
    else:
        weights = np.asarray(dataset_weights, dtype=np.float64)
        if weights.shape[0] != len(per_task_stats):
            raise ValueError(
                f"RoboTwin2 metadata merge weight count {weights.shape[0]} "
                f"does not match metadata count {len(per_task_stats)}"
            )
        if float(weights.sum()) <= 0:
            raise ValueError("RoboTwin2 metadata merge weights must have positive sum")
    weights /= weights.sum()
    merged: dict[str, dict[str, list[float]]] = {}
    for key in per_task_stats[0]:
        means = []
        stds = []
        mins = []
        maxs = []
        q01s = []
        q99s = []
        for stats in per_task_stats:
            if key not in stats:
                raise KeyError(f"RoboTwin2 metadata key {key!r} missing while merging")
            value = stats[key]
            means.append(np.asarray(value["mean"], dtype=np.float64))
            stds.append(np.asarray(value["std"], dtype=np.float64))
            mins.append(np.asarray(value["min"], dtype=np.float64))
            maxs.append(np.asarray(value["max"], dtype=np.float64))
            q01s.append(np.asarray(value["q01"], dtype=np.float64))
            q99s.append(np.asarray(value["q99"], dtype=np.float64))

        mean_arr = np.stack(means, axis=0)
        std_arr = np.stack(stds, axis=0)
        min_arr = np.stack(mins, axis=0)
        max_arr = np.stack(maxs, axis=0)
        q01_arr = np.stack(q01s, axis=0)
        q99_arr = np.stack(q99s, axis=0)
        weighted_mean = np.average(mean_arr, axis=0, weights=weights)
        weighted_square = np.average(std_arr**2 + mean_arr**2, axis=0, weights=weights)
        weighted_std = np.sqrt(np.maximum(weighted_square - weighted_mean**2, 0.0))
        if percentile_mixing_method == "weighted_average":
            q01 = np.average(q01_arr, axis=0, weights=weights)
            q99 = np.average(q99_arr, axis=0, weights=weights)
        elif percentile_mixing_method == "min_max":
            q01 = q01_arr.min(axis=0)
            q99 = q99_arr.max(axis=0)
        else:
            raise ValueError(
                f"Invalid RoboTwin2 percentile_mixing_method: {percentile_mixing_method}"
            )

        merged[key] = {
            "min": min_arr.min(axis=0).astype(np.float32).tolist(),
            "max": max_arr.max(axis=0).astype(np.float32).tolist(),
            "mean": weighted_mean.astype(np.float32).tolist(),
            "std": weighted_std.astype(np.float32).tolist(),
            "q01": q01.astype(np.float32).tolist(),
            "q99": q99.astype(np.float32).tolist(),
        }
    return merged


def _merge_robotwin2_metadata(
    metadatas: list[DatasetMetadata],
    *,
    dataset_weights: list[float] | np.ndarray | None = None,
    percentile_mixing_method: str = "min_max",
) -> DatasetMetadata:
    blobs = [_metadata_to_json(metadata) for metadata in metadatas]
    if not blobs:
        raise ValueError("Cannot merge empty RoboTwin2 metadata list")
    first = blobs[0]
    for blob in blobs[1:]:
        if blob["embodiment_tag"] != first["embodiment_tag"]:
            raise ValueError(
                "All RoboTwin2 task metadata must use the same embodiment_tag: "
                f"{first['embodiment_tag']!r} vs {blob['embodiment_tag']!r}"
            )
        for modality in ("video", "state", "action"):
            if set(blob["modalities"][modality]) != set(first["modalities"][modality]):
                raise ValueError(
                    f"RoboTwin2 {modality} modality keys differ while merging metadata"
                )
    merged = {
        "statistics": {
            "state": _merge_stats(
                [blob["statistics"]["state"] for blob in blobs],
                dataset_weights=dataset_weights,
                percentile_mixing_method=percentile_mixing_method,
            ),
            "action": _merge_stats(
                [blob["statistics"]["action"] for blob in blobs],
                dataset_weights=dataset_weights,
                percentile_mixing_method=percentile_mixing_method,
            ),
        },
        "modalities": {
            "video": first["modalities"]["video"],
            "state": first["modalities"]["state"],
            "action": first["modalities"]["action"],
        },
        "embodiment_tag": first["embodiment_tag"],
    }
    return DatasetMetadata.model_validate(merged)


def _robotwin2_metadata_path(cfg, data_cfg: Any) -> Path:
    explicit = data_cfg.get("robotwin2_metadata_json_path", None)
    if explicit:
        return Path(str(explicit)).expanduser()
    logger_cfg = cfg.runner.get("logger", {})
    log_path = Path(str(logger_cfg.get("log_path", "results"))).expanduser()
    experiment_name = str(logger_cfg.get("experiment_name", "robotwin2_sft_dreamzero_5b"))
    return log_path / experiment_name / "metadata.json"


def _write_robotwin2_metadata_json(path: Path, metadata: DatasetMetadata) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"robotwin2": _metadata_to_json(metadata)}
    tmp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp_path, path)


def _wait_for_robotwin2_metadata_json(path: Path, *, timeout_s: float = 600.0) -> None:
    deadline = time.monotonic() + float(timeout_s)
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        if path.is_file():
            try:
                with path.open("r", encoding="utf-8") as f:
                    json.load(f)
                return
            except Exception as exc:
                last_error = exc
        time.sleep(1.0)
    detail = f" last_error={last_error!r}" if last_error is not None else ""
    raise TimeoutError(f"Timed out waiting for RoboTwin2 metadata.json: {path}{detail}")


class _RobotWin2MotionMixin:
    def __init__(
        self,
        *args,
        include_motion: bool = False,
        motion_downsample_ratio: int = 6,
        motion_cache_size: int = 0,
        drop_motion_npy_file_cache: bool = True,
        robotwin2_bad_sample_resample_attempts: int = 32,
        robotwin2_macro_stride: int = 48,
        robotwin2_video_frame_stride: int = 6,
        fast_local_index: bool = True,
        **kwargs,
    ):
        self.include_motion = bool(include_motion)
        self.motion_downsample_ratio = max(1, int(motion_downsample_ratio))
        self._motion_cache: OrderedDict[int, dict[str, np.ndarray]] = OrderedDict()
        self._motion_cache_size = max(0, int(motion_cache_size))
        self._drop_motion_npy_file_cache = bool(drop_motion_npy_file_cache)
        self._bad_sample_resample_attempts = max(1, int(robotwin2_bad_sample_resample_attempts))
        self._robotwin2_macro_stride = max(1, int(robotwin2_macro_stride))
        self._robotwin2_video_frame_stride = max(1, int(robotwin2_video_frame_stride))
        self.fast_local_index = bool(fast_local_index)
        super().__init__(*args, **kwargs)
        if getattr(self, "sampling_mode", None) == "multi_anchor":
            self._multi_anchor_cfg = MultiAnchorTemporalConfig(
                max_chunk_size=int(self.max_chunk_size),
                macro_stride=self._robotwin2_macro_stride,
                action_horizon=int(self.action_horizon),
                video_in_chunk_offsets=tuple(
                    range(
                        0,
                        self._robotwin2_video_frame_stride * 8,
                        self._robotwin2_video_frame_stride,
                    )
                ),
            )

    @staticmethod
    def _to_channel_first_3ch(arr: np.ndarray, *, name: str, path: Path) -> np.ndarray:
        arr = np.asarray(arr)
        if arr.ndim != 4:
            raise ValueError(f"Expected 4-D motion tensor for {name}, got {arr.shape} from {path}")
        if arr.shape[1] == 3:
            return np.ascontiguousarray(arr, dtype=np.float32)
        axes = [i for i in range(1, arr.ndim) if arr.shape[i] == 3]
        if len(axes) != 1:
            raise ValueError(f"Cannot locate 3-channel axis for {name}: shape={arr.shape} path={path}")
        return np.ascontiguousarray(np.moveaxis(arr, axes[0], 1), dtype=np.float32)

    @staticmethod
    def _broadcast_mask(target: np.ndarray, mask: np.ndarray, *, path: Path) -> np.ndarray:
        mask = np.asarray(mask)
        if mask.shape[0] < target.shape[0]:
            pad = np.zeros((target.shape[0] - mask.shape[0], *mask.shape[1:]), dtype=mask.dtype)
            mask = np.concatenate([mask, pad], axis=0)
        elif mask.shape[0] > target.shape[0]:
            mask = mask[: target.shape[0]]
        if mask.ndim == 1:
            mask = mask.reshape((mask.shape[0],) + (1,) * (target.ndim - 1))
        elif mask.ndim == 3:
            mask = mask[:, None, :, :]
        elif mask.ndim == 4 and mask.shape[-1] == 1 and target.shape[1] == 3:
            mask = np.moveaxis(mask, -1, 1)
        if mask.ndim != target.ndim:
            raise ValueError(f"Bad motion mask shape={mask.shape} target={target.shape} path={path}")
        return mask.astype(bool).astype(target.dtype)

    def _motion_path(self, episode_index: int) -> Path:
        chunk = int(episode_index) // int(self._chunks_size)
        filename = f"episode_{int(episode_index):06d}.npz"
        for dirname in ("motions", "motion"):
            path = self._root / dirname / f"chunk-{chunk:03d}" / filename
            if path.is_file():
                return path
        return self._root / "motions" / f"chunk-{chunk:03d}" / filename

    def _motion_npy_path(self, episode_index: int) -> Path:
        return self._motion_path(episode_index).with_suffix(".npy")

    def _load_motion_episode(self, episode_index: int) -> dict[str, np.ndarray]:
        episode_index = int(episode_index)
        if self._motion_cache_size > 0 and episode_index in self._motion_cache:
            item = self._motion_cache.pop(episode_index)
            self._motion_cache[episode_index] = item
            return item
        path = self._motion_path(episode_index)
        if not path.is_file():
            raise FileNotFoundError(f"Motion npz not found for episode {episode_index}: {path}")
        with np.load(path) as f:
            point_map = np.asarray(f["point_map"])
            scene_flow = np.asarray(f["scene_flow"])
            valid_mask = np.asarray(f["valid_mask"]) if "valid_mask" in f.files else None
            valid_flow_mask = np.asarray(f["valid_flow_mask"]) if "valid_flow_mask" in f.files else None
        point_map = self._to_channel_first_3ch(point_map, name="point_map", path=path)
        scene_flow = self._to_channel_first_3ch(scene_flow, name="scene_flow", path=path)
        if scene_flow.shape[0] < point_map.shape[0]:
            pad = np.zeros((point_map.shape[0] - scene_flow.shape[0], *scene_flow.shape[1:]), dtype=scene_flow.dtype)
            scene_flow = np.concatenate([scene_flow, pad], axis=0)
        elif scene_flow.shape[0] > point_map.shape[0]:
            scene_flow = scene_flow[: point_map.shape[0]]
        if valid_mask is not None:
            point_map = point_map * self._broadcast_mask(point_map, valid_mask, path=path)
        if valid_flow_mask is not None:
            scene_flow = scene_flow * self._broadcast_mask(scene_flow, valid_flow_mask, path=path)
        item = {
            "point_map": np.ascontiguousarray(point_map, dtype=np.float32),
            "scene_flow": np.ascontiguousarray(scene_flow, dtype=np.float32),
        }
        if self._motion_cache_size > 0:
            self._motion_cache[episode_index] = item
            if len(self._motion_cache) > self._motion_cache_size:
                self._motion_cache.popitem(last=False)
        return item

    def _load_motion_frames_from_npy(self, indices: np.ndarray, path: Path) -> tuple[np.ndarray, np.ndarray]:
        arr = None
        frame_idx = np.asarray(indices, dtype=np.int64)
        try:
            arr = np.load(path, mmap_mode="r", allow_pickle=False)
            if frame_idx.size == 0:
                return np.zeros((0, 3, 1, 1), dtype=np.float32), np.zeros((0, 3, 1, 1), dtype=np.float32)
            if int(frame_idx.min()) < 0 or int(frame_idx.max()) >= int(arr.shape[0]):
                raise IndexError(
                    f"motion frame index out of range for npy length={arr.shape[0]}: "
                    f"min={int(frame_idx.min())} max={int(frame_idx.max())} path={path}"
                )
            selected = np.asarray(arr[frame_idx])
        finally:
            del arr
            if self._drop_motion_npy_file_cache:
                _drop_file_cache(path)
        if selected.ndim != 4:
            raise ValueError(f"Expected selected motion npy frames to be 4-D, got {selected.shape} from {path}")
        if selected.shape[-1] >= 12:
            point_map = np.moveaxis(selected[..., 0:3], -1, 1)
            scene_flow = np.moveaxis(selected[..., 3:6], -1, 1)
            point_mask = np.moveaxis((selected[..., 6:7] >= 0.5).astype(point_map.dtype), -1, 1)
            flow_mask = np.moveaxis((selected[..., 7:8] >= 0.5).astype(scene_flow.dtype), -1, 1)
            point_map = point_map * point_mask
            scene_flow = scene_flow * flow_mask
        elif selected.shape[-1] >= 11:
            point_map = np.moveaxis(selected[..., 0:3], -1, 1)
            scene_flow = np.moveaxis(selected[..., 3:6], -1, 1)
            scene_flow = scene_flow * np.moveaxis((selected[..., 6:7] >= 0.5).astype(scene_flow.dtype), -1, 1)
        elif selected.shape[-1] >= 6:
            point_map = np.moveaxis(selected[..., 0:3], -1, 1)
            scene_flow = np.moveaxis(selected[..., 3:6], -1, 1)
            if selected.shape[-1] >= 7:
                scene_flow = scene_flow * np.moveaxis((selected[..., 6:7] >= 0.5).astype(scene_flow.dtype), -1, 1)
        elif selected.shape[1] >= 12:
            point_map = selected[:, 0:3]
            scene_flow = selected[:, 3:6]
            point_map = point_map * (selected[:, 6:7] >= 0.5).astype(point_map.dtype)
            scene_flow = scene_flow * (selected[:, 7:8] >= 0.5).astype(scene_flow.dtype)
        elif selected.shape[1] >= 11:
            point_map = selected[:, 0:3]
            scene_flow = selected[:, 3:6]
            scene_flow = scene_flow * (selected[:, 6:7] >= 0.5).astype(scene_flow.dtype)
        elif selected.shape[1] >= 6:
            point_map = selected[:, 0:3]
            scene_flow = selected[:, 3:6]
            if selected.shape[1] >= 7:
                scene_flow = scene_flow * (selected[:, 6:7] >= 0.5).astype(scene_flow.dtype)
        else:
            raise ValueError(f"Unsupported motion npy shape {selected.shape} from {path}")
        return (
            np.ascontiguousarray(point_map, dtype=np.float32),
            np.ascontiguousarray(scene_flow, dtype=np.float32),
        )

    def _load_motion_frames(self, episode_index: int, indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        npy_path = self._motion_npy_path(episode_index)
        if npy_path.is_file():
            return self._load_motion_frames_from_npy(indices, npy_path)
        motion = self._load_motion_episode(episode_index)
        return motion["point_map"][indices], motion["scene_flow"][indices]

    def _downsample_motion_indices(self, indices: np.ndarray) -> np.ndarray:
        if self.motion_downsample_ratio <= 1 or len(indices) == 0:
            return indices
        parts = []
        chunk = int(self.action_horizon)
        for start in range(0, len(indices), chunk):
            parts.append(indices[start : start + chunk : self.motion_downsample_ratio])
        return np.concatenate(parts) if parts else np.array([], dtype=np.int64)

    def _materialize_parquet_sample(
        self,
        frame_in_ep: int,
        episode_index: int,
        ep_len: int,
        table: Any,
        *,
        decode_video: bool,
    ) -> dict[str, Any]:
        sample = super()._materialize_parquet_sample(
            frame_in_ep,
            episode_index,
            ep_len,
            table,
            decode_video=decode_video,
        )
        if not self.include_motion:
            return sample
        _, _, action_offsets = self._temporal_offsets_for_frame(frame_in_ep, episode_index, ep_len)
        action_idx = self._clip_indices(frame_in_ep + action_offsets, ep_len)
        motion_idx = self._downsample_motion_indices(action_idx)
        point_map, scene_flow = self._load_motion_frames(episode_index, motion_idx)
        sample["motion.point_map"] = point_map
        sample["motion.scene_flow"] = scene_flow
        return sample

    def _get_v2_image_sample(self, idx: int) -> dict[str, Any]:
        sample = super()._get_v2_image_sample(idx)
        if not self.include_motion:
            return sample
        frame_in_ep, episode_index, ep_len = self._resolve_index_context(idx)
        _, _, action_offsets = self._temporal_offsets_for_frame(frame_in_ep, episode_index, ep_len)
        action_idx = self._clip_indices(frame_in_ep + action_offsets, ep_len)
        motion_idx = self._downsample_motion_indices(action_idx)
        point_map, scene_flow = self._load_motion_frames(episode_index, motion_idx)
        sample["motion.point_map"] = point_map
        sample["motion.scene_flow"] = scene_flow
        return sample

    def _build_modality_dict(self, sample: dict[str, Any]) -> dict[str, Any]:
        out = super()._build_modality_dict(sample)
        if self.include_motion:
            out["motion.point_map"] = np.asarray(sample["motion.point_map"], dtype=np.float32)
            out["motion.scene_flow"] = np.asarray(sample["motion.scene_flow"], dtype=np.float32)
        return out

    def __getitem__(self, idx: int) -> dict[str, Any]:
        last_error: Exception | None = None
        current_idx = int(idx)
        for _ in range(self._bad_sample_resample_attempts):
            try:
                return super().__getitem__(current_idx)
            except Exception as exc:
                last_error = exc
                if not self.include_motion:
                    raise
                logger.warning("RoboTwin2 sample failed at idx=%s under %s: %s", current_idx, self._root, exc)
                current_idx = random.randint(0, max(0, len(self) - 1))
        raise RuntimeError(
            f"Failed to sample a valid RoboTwin2 item after {self._bad_sample_resample_attempts} "
            f"attempts from {self._root}: {last_error}"
        ) from last_error


class RobotWin2DreamZeroCollator:
    def __init__(
        self,
        *,
        tokenizer_path: str,
        max_seq_len: int,
        embodiment_tag_mapping: dict[str, int],
    ):
        from rlinf.data.datasets.dreamzero.dreamzero import DreamZeroCollator

        self._base = DreamZeroCollator(
            tokenizer_path=tokenizer_path,
            max_seq_len=max_seq_len,
            embodiment_tag_mapping=embodiment_tag_mapping,
        )

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        batch = self._base(features)
        if "motion.point_map" in batch and "motion.scene_flow" in batch:
            batch["motion"] = {
                "point_map": batch.pop("motion.point_map"),
                "scene_flow": batch.pop("motion.scene_flow"),
            }
        return batch


class RobotWin2ConcatDataset(Dataset):
    def __init__(self, datasets: list[Dataset], paths: list[str]):
        if not datasets:
            raise ValueError("RobotWin2ConcatDataset requires at least one dataset")
        self.datasets = datasets
        self.paths = paths
        self.cumulative: list[int] = []
        total = 0
        for ds in datasets:
            total += len(ds)
            self.cumulative.append(total)

    def __len__(self) -> int:
        return self.cumulative[-1]

    def __getitem__(self, idx: int) -> dict[str, Any]:
        ds_idx = bisect.bisect_right(self.cumulative, int(idx))
        prev = 0 if ds_idx == 0 else self.cumulative[ds_idx - 1]
        return self.datasets[ds_idx][int(idx) - prev]


def build_robotwin2_sft_dataloader(
    cfg,
    world_size: int,
    rank: int,
    data_paths: str | list[str],
    eval_dataset: bool = False,
):
    from rlinf.data.datasets.dreamzero.dreamzero import (
        DreamZeroCollator,
        DreamZeroLeRobotDataset,
    )

    class RobotWin2LeRobotDataset(_RobotWin2MotionMixin, DreamZeroLeRobotDataset):
        pass

    data_cfg = cfg.data
    model_cfg = cfg.actor.model
    include_motion = bool(model_cfg.action_head_cfg.config.get("use_motion_modality", False))
    root = data_paths[0] if isinstance(data_paths, (list, tuple)) else data_paths
    if data_cfg.get("robotwin2_root", None):
        root = data_cfg.robotwin2_root
    task_paths = discover_robotwin2_task_paths(root, data_cfg, require_motion=include_motion)

    tokenizer_path = model_cfg.get("tokenizer_path", "google/umt5-xxl")
    max_seq_len = int(model_cfg.get("max_seq_len", 512))
    embodiment_tag_mapping = dict(
        model_cfg.get("embodiment_tag_mapping") or DEFAULT_ROBOTWIN2_TAG_MAPPING
    )
    sampling_mode = data_cfg.get("sampling_mode", "multi_anchor")
    max_chunk_size = int(model_cfg.action_head_cfg.config.diffusion_model_cfg.max_chunk_size)
    num_frames = int(model_cfg.action_head_cfg.config.num_frames)
    state_horizon = int(model_cfg.get("state_horizon", 1))
    action_horizon = int(model_cfg.get("action_horizon", 48))

    logger.info(
        "Building RoboTwin2 DreamZero dataset: motion=%s tasks=%s root=%s subsets=%s",
        include_motion,
        len(task_paths),
        root,
        data_cfg.get("robotwin2_subsets", None),
    )

    datasets: list[Dataset] = []
    paths: list[str] = []
    task_metadatas: list[DatasetMetadata] = []
    dataset_lengths: list[int] = []
    for task_i, task_path in enumerate(task_paths, start=1):
        task_dir = Path(task_path)
        if task_i == 1 or task_i % 25 == 0:
            logger.info("Building RoboTwin2 task %s/%s: %s", task_i, len(task_paths), task_dir)
        from rlinf.data.datasets.dreamzero.data_transforms.robotwin2 import (
            RobotWin2DataTransform,
        )

        transform = RobotWin2DataTransform.get_transform(
            tokenizer_path=tokenizer_path,
            cfg=model_cfg,
            embodiment_tag_mapping=embodiment_tag_mapping,
        )
        task_metadata = _metadata_for_task(task_dir)
        transform.set_metadata(task_metadata)
        if eval_dataset:
            transform.eval()
        else:
            transform.train()
        ds = RobotWin2LeRobotDataset(
            data_path=str(task_dir),
            video_keys=list(VIDEO_KEYS),
            state_keys=list(STATE_KEYS),
            action_keys=list(ACTION_KEYS),
            language_keys=list(LANGUAGE_KEYS),
            data_transform=transform,
            lazy_load=bool(data_cfg.get("lazy_load", True)),
            num_frames=num_frames,
            state_horizon=state_horizon,
            action_horizon=action_horizon,
            relative_action=bool(model_cfg.get("relative_action", False)),
            relative_action_keys=list(model_cfg.get("relative_action_keys", [])),
            pq_cache_max_episodes=int(data_cfg.get("parquet_cache_size", 128)),
            video_tolerance_s=float(data_cfg.get("video_tolerance_s", 0.1)),
            video_backend=data_cfg.get("video_backend", "pyav"),
            max_chunk_size=max_chunk_size,
            sampling_mode=sampling_mode,
            multi_anchor_resample_attempts=int(data_cfg.get("multi_anchor_resample_attempts", 8)),
            include_motion=include_motion,
            motion_downsample_ratio=int(data_cfg.get("motion_downsample_ratio", 6)),
            motion_cache_size=int(data_cfg.get("motion_cache_size", 0)),
            drop_motion_npy_file_cache=bool(data_cfg.get("drop_motion_npy_file_cache", True)),
            robotwin2_bad_sample_resample_attempts=int(data_cfg.get("robotwin2_bad_sample_resample_attempts", 32)),
            robotwin2_macro_stride=int(data_cfg.get("robotwin2_macro_stride", action_horizon)),
            robotwin2_video_frame_stride=int(data_cfg.get("robotwin2_video_frame_stride", 6)),
            fast_local_index=bool(data_cfg.get("robotwin2_fast_local_index", True)),
        )
        if len(ds) == 0:
            logger.warning("Skipping empty RoboTwin2 task: %s", task_dir)
            continue
        datasets.append(ds)
        paths.append(str(task_dir))
        task_metadatas.append(task_metadata)
        dataset_lengths.append(len(ds))

    merged_metadata = _merge_robotwin2_metadata(
        task_metadatas,
        dataset_weights=dataset_lengths,
        percentile_mixing_method=str(
            data_cfg.get("robotwin2_metadata_percentile_mixing_method", "min_max")
        ),
    )
    metadata_json_path = _robotwin2_metadata_path(cfg, data_cfg)
    if int(rank) == 0:
        _write_robotwin2_metadata_json(metadata_json_path, merged_metadata)
    else:
        _wait_for_robotwin2_metadata_json(metadata_json_path)
    model_cfg["metadata_json_path"] = str(metadata_json_path)
    for ds in datasets:
        ds.data_transform.set_metadata(merged_metadata)
    logger.info(
        "Using merged RoboTwin2 metadata for all tasks: path=%s tasks=%s samples=%s",
        metadata_json_path,
        len(datasets),
        int(sum(dataset_lengths)),
    )
    dataset = RobotWin2ConcatDataset(datasets, paths)
    sampler = torch.utils.data.distributed.DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=not eval_dataset,
        drop_last=not eval_dataset,
    )
    num_workers = int(data_cfg.get("num_workers", 4))
    prefetch_factor = int(data_cfg.get("prefetch_factor", 4))
    loader = StatefulDataLoader(
        dataset,
        batch_size=int(cfg.actor.micro_batch_size),
        sampler=sampler,
        drop_last=not eval_dataset,
        num_workers=num_workers,
        pin_memory=bool(data_cfg.get("pin_memory", True)),
        persistent_workers=num_workers > 0,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        collate_fn=RobotWin2DreamZeroCollator(
            tokenizer_path=tokenizer_path,
            max_seq_len=max_seq_len,
            embodiment_tag_mapping=dict(embodiment_tag_mapping),
        ),
    )
    return loader, {"num_samples": len(dataset), "num_tasks": len(datasets)}
