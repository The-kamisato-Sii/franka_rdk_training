# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0

"""OpenPI SFT dataloader for local LeRobot v3 franka_dual datasets.

This adapter intentionally does not depend on RLinf/DreamZero real_world_joint
readers.  It mirrors OpenPI's own training loader shape:

    local LeRobot-style dataset -> OpenPI repack/data/normalize/model transforms
    -> (Observation, actions)

The input root is a directory of LeRobot v3 task/session datasets such as:

    franka_dual/<task>/<session>/{meta,data,videos}

Only standard v3 metadata/parquet/video files are used.  Motion folders are not
read.
"""

from __future__ import annotations

import bisect
import fcntl
import json
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch
from openpi import transforms as _transforms
from openpi.shared.normalize import NormStats

from rlinf.data.datasets.recap.common import BaseDataLoaderImpl
from rlinf.utils.logging import get_logger

logger = get_logger()


DEFAULT_VIDEO_KEYS = (
    "observation.images.middle_zed",
    "observation.images.left_camera",
    "observation.images.right_camera",
)
STATE_KEY = "observation.state"
ACTION_KEY = "action"


@dataclass(frozen=True)
class _EpisodeInfo:
    episode_index: int
    length: int
    data_chunk: int
    data_file: int
    data_from: int
    data_to: int
    tasks: tuple[str, ...]
    video_files: dict[str, tuple[int, int]]
    video_start_ts: dict[str, float]


class _OpenPIDataLoaderImpl(BaseDataLoaderImpl):
    """Yield OpenPI SFT tuples from a torch dataloader of transformed dicts."""

    def __init__(
        self,
        data_config: Any,
        data_loader: torch.utils.data.DataLoader,
        *,
        infinite: bool = False,
    ):
        super().__init__(data_config, data_loader)
        self._infinite = bool(infinite)

    @property
    def sampler(self) -> Any:
        return getattr(self._data_loader, "sampler", None)

    def __iter__(self) -> Iterator[tuple[Any, torch.Tensor]]:
        from openpi.models import model as _model

        while True:
            for batch in self._data_loader:
                actions = batch["actions"]
                obs_dict = {k: v for k, v in batch.items() if k != "actions"}
                yield _model.Observation.from_dict(obs_dict), actions
            if not self._infinite:
                return


class _OpenPITransformedDataset(torch.utils.data.Dataset):
    """Small equivalent of openpi.training.data_loader.TransformedDataset."""

    def __init__(self, dataset: torch.utils.data.Dataset, transforms: list[Any]):
        self._dataset = dataset
        self._transform = _transforms.compose(transforms)

    def __getitem__(self, index: Any) -> Any:
        return self._transform(self._dataset[index])

    def __len__(self) -> int:
        return len(self._dataset)


class _RemoveStrings:
    def __call__(self, x: dict[str, Any]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for key, value in x.items():
            if isinstance(value, str):
                continue
            arr = np.asarray(value)
            if arr.dtype.kind in ("U", "S", "O"):
                continue
            out[key] = value
        return out


class _LeRobotV3FrankaDualOpenPIDataset(torch.utils.data.Dataset):
    """Map-style local LeRobot v3 reader for one franka_dual session.

    The returned sample already uses the raw keys consumed by
    LeRobotRealWorldJointDataConfig:

        image, left_wrist_image, right_wrist_image, image masks, state, actions,
        prompt
    """

    def __init__(
        self,
        task_dir: Path,
        *,
        action_horizon: int,
        num_frames: int,
        use_state: bool,
        video_backend: str,
        video_tolerance_s: float,
        parquet_cache_size: int,
        default_prompt: str,
    ):
        self.task_dir = Path(task_dir)
        self.meta_dir = self.task_dir / "meta"
        self.action_horizon = int(action_horizon)
        self.num_frames = int(num_frames)
        self.use_state = bool(use_state)
        self.video_backend = str(video_backend)
        self.video_tolerance_s = float(video_tolerance_s)
        self.default_prompt = str(default_prompt)
        self.info = _read_json(self.meta_dir / "info.json")
        self.features = dict(self.info.get("features") or {})
        self.fps = float(self.info.get("fps") or 30)
        self.chunks_size = int(self.info.get("chunks_size") or 1000)
        self.data_tmpl = str(self.info.get("data_path") or "")
        self.video_tmpl = str(self.info.get("video_path") or "")
        if not self.data_tmpl or not self.video_tmpl:
            raise ValueError(f"{self.task_dir} is missing LeRobot v3 data/video path templates")

        self.video_keys = self._resolve_video_keys()
        self.state_dim = int((self.features.get(STATE_KEY, {}).get("shape") or [0])[0])
        self.action_dim = int((self.features.get(ACTION_KEY, {}).get("shape") or [0])[0])
        if self.state_dim <= 0 or self.action_dim <= 0:
            raise ValueError(f"{self.task_dir} missing valid state/action feature shapes")

        self.episodes = self._load_episodes()
        self.episode_starts = [0]
        total = 0
        for ep in self.episodes:
            total += ep.length
            self.episode_starts.append(total)
        self.total_frames = int(total)
        self._pq_cache: OrderedDict[tuple[int, int, int, int], Any] = OrderedDict()
        self._pq_cache_size = max(1, int(parquet_cache_size))

    def _resolve_video_keys(self) -> tuple[str, str, str]:
        available = set(self.features)
        preferred = [key for key in DEFAULT_VIDEO_KEYS if key in available]
        if len(preferred) == 3:
            return tuple(preferred)
        video_keys = sorted(
            key for key, feature in self.features.items() if feature.get("dtype") == "video"
        )
        if len(video_keys) < 1:
            raise ValueError(f"{self.task_dir} has no video features")
        while len(video_keys) < 3:
            video_keys.append(video_keys[0])
        return tuple(video_keys[:3])

    def _format_path(self, template: str, *, chunk: int, file: int, video_key: str | None = None) -> Path:
        rel = template.format(
            chunk_index=int(chunk),
            file_index=int(file),
            episode_chunk=int(chunk),
            episode_index=int(file),
            video_key=video_key or "",
        )
        return (self.task_dir / rel).resolve()

    def _load_episodes(self) -> list[_EpisodeInfo]:
        import pyarrow.parquet as pq

        episodes_root = self.meta_dir / "episodes"
        if not episodes_root.is_dir():
            raise FileNotFoundError(f"{episodes_root} does not exist; expected LeRobot v3 metadata")

        episodes: list[_EpisodeInfo] = []
        for meta_path in sorted(episodes_root.glob("chunk-*/file-*.parquet")):
            schema = set(pq.read_schema(str(meta_path)).names)
            columns = [
                "episode_index",
                "tasks",
                "length",
                "episode_length",
                "num_frames",
                "data/chunk_index",
                "data/file_index",
                "dataset_from_index",
                "dataset_to_index",
            ]
            for video_key in self.video_keys:
                prefix = f"videos/{video_key}"
                columns.extend(
                    [
                        f"{prefix}/chunk_index",
                        f"{prefix}/file_index",
                        f"{prefix}/from_timestamp",
                    ]
                )
            table = pq.read_table(
                str(meta_path),
                columns=[col for col in dict.fromkeys(columns) if col in schema],
            )
            for row in table.to_pylist():
                ep_idx = int(row.get("episode_index", len(episodes)))
                length = int(row.get("length") or row.get("episode_length") or row.get("num_frames") or 0)
                if length <= 0:
                    continue
                data_chunk = int(row.get("data/chunk_index", ep_idx // self.chunks_size))
                data_file = int(row.get("data/file_index", ep_idx))
                data_from = int(row.get("dataset_from_index", 0))
                data_to = int(row.get("dataset_to_index", data_from + length))
                data_path = self._format_path(self.data_tmpl, chunk=data_chunk, file=data_file)
                if not data_path.is_file():
                    logger.warning("Skipping episode %s because data parquet is missing: %s", ep_idx, data_path)
                    continue

                video_files: dict[str, tuple[int, int]] = {}
                video_start_ts: dict[str, float] = {}
                missing_video = False
                for video_key in self.video_keys:
                    prefix = f"videos/{video_key}"
                    v_chunk = int(row.get(f"{prefix}/chunk_index", data_chunk))
                    v_file = int(row.get(f"{prefix}/file_index", data_file))
                    video_path = self._format_path(
                        self.video_tmpl,
                        chunk=v_chunk,
                        file=v_file,
                        video_key=video_key,
                    )
                    if not video_path.is_file():
                        logger.warning("Skipping episode %s because video is missing: %s", ep_idx, video_path)
                        missing_video = True
                        break
                    video_files[video_key] = (v_chunk, v_file)
                    video_start_ts[video_key] = float(row.get(f"{prefix}/from_timestamp") or 0.0)
                if missing_video:
                    continue
                tasks = row.get("tasks") or ()
                episodes.append(
                    _EpisodeInfo(
                        episode_index=ep_idx,
                        length=length,
                        data_chunk=data_chunk,
                        data_file=data_file,
                        data_from=data_from,
                        data_to=data_to,
                        tasks=tuple(str(x) for x in tasks if str(x)),
                        video_files=video_files,
                        video_start_ts=video_start_ts,
                    )
                )
        if not episodes:
            raise RuntimeError(f"No usable LeRobot v3 episodes under {self.task_dir}")
        return sorted(episodes, key=lambda ep: ep.episode_index)

    def __len__(self) -> int:
        return self.total_frames

    def _resolve_index(self, idx: int) -> tuple[_EpisodeInfo, int]:
        if idx < 0 or idx >= self.total_frames:
            raise IndexError(f"Index {idx} out of range for dataset of len {self.total_frames}")
        ep_pos = bisect.bisect_right(self.episode_starts, int(idx)) - 1
        ep_pos = max(0, min(ep_pos, len(self.episodes) - 1))
        frame_in_ep = int(idx) - int(self.episode_starts[ep_pos])
        return self.episodes[ep_pos], frame_in_ep

    def _episode_table(self, ep: _EpisodeInfo):
        key = (ep.data_chunk, ep.data_file, ep.data_from, ep.data_to)
        if key in self._pq_cache:
            table = self._pq_cache.pop(key)
            self._pq_cache[key] = table
            return table
        import pyarrow.parquet as pq

        data_path = self._format_path(self.data_tmpl, chunk=ep.data_chunk, file=ep.data_file)
        schema = set(pq.read_schema(str(data_path)).names)
        columns = [
            col
            for col in (
                STATE_KEY,
                ACTION_KEY,
                "task_index",
                "frame_index",
                "timestamp",
            )
            if col in schema
        ]
        table = pq.read_table(str(data_path), columns=columns)
        table = table.slice(ep.data_from, max(0, ep.data_to - ep.data_from))
        self._pq_cache[key] = table
        if len(self._pq_cache) > self._pq_cache_size:
            self._pq_cache.popitem(last=False)
        return table

    @staticmethod
    def _read_vector_rows(table: Any, column: str, rows: np.ndarray, dim: int) -> np.ndarray:
        if column not in table.column_names:
            raise KeyError(f"episode parquet missing column {column!r}")
        col = table.column(column)
        out = np.asarray([col[int(i)].as_py() for i in rows.tolist()], dtype=np.float32)
        if out.ndim == 1:
            out = out.reshape(-1, dim)
        return out

    @staticmethod
    def _read_timestamp(table: Any, row: int, default: float) -> float:
        if "timestamp" not in table.column_names:
            return float(default)
        value = table.column("timestamp")[int(row)].as_py()
        return float(default if value is None else value)

    @staticmethod
    def _to_hwc_uint8(frames: Any) -> np.ndarray:
        if torch.is_tensor(frames):
            arr = frames.detach().cpu().numpy()
        else:
            arr = np.asarray(frames)
        if arr.ndim == 5:
            arr = arr[0]
        if arr.ndim == 4:
            arr = arr[0]
        if arr.ndim == 3 and arr.shape[0] in (1, 3, 4) and arr.shape[-1] not in (1, 3, 4):
            arr = np.transpose(arr, (1, 2, 0))
        if arr.dtype != np.uint8:
            arr_f = arr.astype(np.float32, copy=False)
            if arr_f.size and float(arr_f.max()) <= 1.0 + 1e-3:
                arr = np.clip(arr_f * 255.0, 0, 255).astype(np.uint8)
            else:
                arr = np.clip(arr_f, 0, 255).astype(np.uint8)
        return np.ascontiguousarray(arr[..., :3])

    def _decode_frame(
        self,
        ep: _EpisodeInfo,
        video_key: str,
        frame_in_ep: int,
        *,
        timestamp: float | None = None,
    ) -> np.ndarray:
        try:
            from lerobot.datasets.video_utils import decode_video_frames
        except ModuleNotFoundError:
            from lerobot.common.datasets.video_utils import decode_video_frames

        chunk, file = ep.video_files[video_key]
        video_path = self._format_path(self.video_tmpl, chunk=chunk, file=file, video_key=video_key)
        if timestamp is None:
            timestamp = ep.video_start_ts.get(video_key, 0.0) + float(frame_in_ep) / self.fps
        frames = decode_video_frames(
            video_path,
            [timestamp],
            tolerance_s=self.video_tolerance_s,
            backend=self.video_backend,
        )
        return self._to_hwc_uint8(frames)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        ep, frame_in_ep = self._resolve_index(int(idx))
        table = self._episode_table(ep)
        state_rows = np.asarray([frame_in_ep], dtype=np.int64)
        action_rows = np.clip(
            frame_in_ep + np.arange(self.action_horizon, dtype=np.int64),
            0,
            max(0, ep.length - 1),
        )
        state = self._read_vector_rows(table, STATE_KEY, state_rows, self.state_dim)[0]
        actions = self._read_vector_rows(table, ACTION_KEY, action_rows, self.action_dim)
        base_key, left_key, right_key = self.video_keys
        frame_timestamp = self._read_timestamp(
            table,
            frame_in_ep,
            float(frame_in_ep) / self.fps,
        )
        prompt = ep.tasks[0] if ep.tasks else self.default_prompt

        return {
            "image": self._decode_frame(
                ep,
                base_key,
                frame_in_ep,
                timestamp=ep.video_start_ts.get(base_key, 0.0) + frame_timestamp,
            ),
            "left_wrist_image": self._decode_frame(
                ep,
                left_key,
                frame_in_ep,
                timestamp=ep.video_start_ts.get(left_key, 0.0) + frame_timestamp,
            ),
            "right_wrist_image": self._decode_frame(
                ep,
                right_key,
                frame_in_ep,
                timestamp=ep.video_start_ts.get(right_key, 0.0) + frame_timestamp,
            ),
            "image_mask": np.asarray(True),
            "left_wrist_image_mask": np.asarray(left_key != base_key),
            "right_wrist_image_mask": np.asarray(right_key != base_key),
            "state": state.astype(np.float32, copy=False) if self.use_state else np.zeros((self.state_dim,), dtype=np.float32),
            "actions": actions.astype(np.float32, copy=False),
            "prompt": str(prompt).lower(),
        }


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _looks_like_lerobot_v3(task_dir: Path) -> bool:
    meta = task_dir / "meta"
    info_path = meta / "info.json"
    if not info_path.is_file():
        return False
    try:
        info = _read_json(info_path)
    except json.JSONDecodeError:
        return False
    return (
        str(info.get("codebase_version", "")).startswith("v3")
        and (meta / "episodes").is_dir()
        and not (meta / "episodes.jsonl").exists()
    )


def _discover_lerobot_v3_tasks(root: str | Path, data_cfg: Any) -> list[Path]:
    root = Path(root)
    if _looks_like_lerobot_v3(root):
        candidates = [root]
    else:
        candidates = sorted(
            {
                *(path.parent.parent for path in root.glob("*/meta/info.json")),
                *(path.parent.parent for path in root.glob("*/*/meta/info.json")),
            }
        )
    contains = data_cfg.get("real_world_joint_task_name_contains", None)
    if contains:
        needles = [str(x).lower() for x in (contains if isinstance(contains, list) else [contains])]
        candidates = [p for p in candidates if any(n in str(p).lower() for n in needles)]
    candidates = [p for p in candidates if _looks_like_lerobot_v3(p)]
    max_tasks = data_cfg.get("real_world_joint_max_tasks", None)
    if max_tasks is not None:
        candidates = candidates[: int(max_tasks)]
    return candidates


def _norm_stats_from_v3_stats(task_dir: Path, *, use_state: bool) -> dict[str, NormStats]:
    stats = _read_json(task_dir / "meta" / "stats.json")

    def make(key: str, *, identity: bool = False) -> NormStats:
        q01 = np.asarray(stats[key]["q01"], dtype=np.float32)
        q99 = np.asarray(stats[key]["q99"], dtype=np.float32)
        dim = int(q01.shape[-1])
        if identity:
            q01 = np.full((dim,), -1.0, dtype=np.float32)
            q99 = np.full((dim,), 1.0, dtype=np.float32)
        return NormStats(
            mean=np.zeros((dim,), dtype=np.float32),
            std=np.ones((dim,), dtype=np.float32),
            q01=q01,
            q99=q99,
        )

    return {
        "state": make(STATE_KEY, identity=not use_state),
        "actions": make(ACTION_KEY),
    }


def _norm_stats_from_lerobot_stats_file(
    stats_path: Path,
    *,
    use_state: bool,
) -> dict[str, NormStats]:
    stats = _read_json(stats_path)

    def make(key: str, *, identity: bool = False) -> NormStats:
        q01 = np.asarray(stats[key]["q01"], dtype=np.float32)
        q99 = np.asarray(stats[key]["q99"], dtype=np.float32)
        dim = int(q01.shape[-1])
        mean = np.asarray(stats[key].get("mean", np.zeros((dim,))), dtype=np.float32)
        std = np.asarray(stats[key].get("std", np.ones((dim,))), dtype=np.float32)
        if identity:
            mean = np.zeros((dim,), dtype=np.float32)
            std = np.ones((dim,), dtype=np.float32)
            q01 = np.full((dim,), -1.0, dtype=np.float32)
            q99 = np.full((dim,), 1.0, dtype=np.float32)
        return NormStats(mean=mean, std=std, q01=q01, q99=q99)

    return {
        "state": make(STATE_KEY, identity=not use_state),
        "actions": make(ACTION_KEY),
    }


def _identity_norm_stats(dim: int) -> NormStats:
    return NormStats(
        mean=np.zeros((dim,), dtype=np.float32),
        std=np.ones((dim,), dtype=np.float32),
        q01=np.full((dim,), -1.0, dtype=np.float32),
        q99=np.full((dim,), 1.0, dtype=np.float32),
    )


def _norm_stats_to_float32(norm_stats: dict[str, NormStats]) -> dict[str, NormStats]:
    return {
        key: NormStats(
            mean=np.asarray(value.mean, dtype=np.float32),
            std=np.asarray(value.std, dtype=np.float32),
            q01=None if value.q01 is None else np.asarray(value.q01, dtype=np.float32),
            q99=None if value.q99 is None else np.asarray(value.q99, dtype=np.float32),
        )
        for key, value in norm_stats.items()
    }


def _norm_stats_dir_from_cfg(cfg: Any) -> Path:
    data_cfg = cfg.data
    explicit_dir = data_cfg.get("openpi_real_world_joint_norm_stats_dir", None)
    if explicit_dir:
        return Path(str(explicit_dir))
    model_path = Path(str(cfg.actor.model.model_path))
    norm_stats_key = str(
        cfg.actor.model.openpi.get("norm_stats_key", "real_world_joint")
    )
    return model_path / norm_stats_key


def _compute_global_norm_stats_from_v3_parquets(
    task_dirs: list[Path],
    *,
    use_state: bool,
    max_frames: int | None = None,
) -> dict[str, NormStats]:
    """Compute one shared OpenPI norm-stats dict for the whole franka_dual mix."""

    import pyarrow.parquet as pq
    from openpi.shared import normalize

    running: dict[str, Any] = {"actions": normalize.RunningStats()}
    if use_state:
        running["state"] = normalize.RunningStats()

    frames_seen = 0
    action_dim = 0
    state_dim = 0
    for task_dir in task_dirs:
        for data_path in sorted((Path(task_dir) / "data").glob("chunk-*/file-*.parquet")):
            schema = set(pq.read_schema(str(data_path)).names)
            missing = [key for key in (ACTION_KEY, STATE_KEY) if key not in schema]
            if missing:
                raise KeyError(f"{data_path} is missing columns: {missing}")
            table = pq.read_table(str(data_path), columns=[STATE_KEY, ACTION_KEY])
            if table.num_rows <= 0:
                continue

            if max_frames is not None:
                remaining = int(max_frames) - frames_seen
                if remaining <= 0:
                    break
                table = table.slice(0, min(int(table.num_rows), remaining))

            actions = np.asarray(table.column(ACTION_KEY).to_pylist(), dtype=np.float32)
            action_dim = int(actions.shape[-1])
            running["actions"].update(actions)
            if use_state:
                state = np.asarray(table.column(STATE_KEY).to_pylist(), dtype=np.float32)
                state_dim = int(state.shape[-1])
                running["state"].update(state)
            frames_seen += int(table.num_rows)
        if max_frames is not None and frames_seen >= int(max_frames):
            break

    if frames_seen < 2:
        raise ValueError(
            f"Cannot compute global OpenPI norm stats from only {frames_seen} frames"
        )

    stats = {"actions": running["actions"].get_statistics()}
    if use_state:
        stats["state"] = running["state"].get_statistics()
    else:
        stats["state"] = _identity_norm_stats(max(action_dim, state_dim))
    logger.info(
        "Computed global OpenPI LeRobot v3 norm stats from %s frames across %s sessions",
        frames_seen,
        len(task_dirs),
    )
    return _norm_stats_to_float32(stats)


def _load_or_create_global_norm_stats(
    task_dirs: list[Path],
    cfg: Any,
    *,
    rank: int,
    dataset_root: str | Path | None = None,
) -> dict[str, NormStats]:
    """Load or create the shared norm stats used by both training and serving."""

    from openpi.shared import normalize

    if dataset_root is not None:
        root_stats = Path(dataset_root) / "stats.json"
        if root_stats.is_file():
            logger.info("Loading global OpenPI norm stats from dataset root %s", root_stats)
            return _norm_stats_to_float32(
                _norm_stats_from_lerobot_stats_file(
                    root_stats,
                    use_state=bool(cfg.data.get("openpi_real_world_joint_use_state", True)),
                )
            )

    stats_dir = _norm_stats_dir_from_cfg(cfg)
    stats_file = stats_dir / "norm_stats.json"
    if stats_file.is_file():
        logger.info("Loading global OpenPI norm stats from %s", stats_dir)
        return _norm_stats_to_float32(normalize.load(stats_dir))

    stats_dir.mkdir(parents=True, exist_ok=True)
    lock_path = stats_dir / ".norm_stats.lock"
    if int(rank) == 0:
        with open(lock_path, "w", encoding="utf-8") as lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            try:
                if not stats_file.is_file():
                    max_frames = cfg.data.get(
                        "openpi_real_world_joint_norm_stats_max_frames", None
                    )
                    stats = _compute_global_norm_stats_from_v3_parquets(
                        task_dirs,
                        use_state=bool(
                            cfg.data.get("openpi_real_world_joint_use_state", True)
                        ),
                        max_frames=None if max_frames is None else int(max_frames),
                    )
                    normalize.save(stats_dir, stats)
                    logger.info("Saved global OpenPI norm stats to %s", stats_file)
            finally:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
    else:
        deadline = time.time() + 3600.0
        while not stats_file.is_file():
            if time.time() > deadline:
                raise TimeoutError(f"Timed out waiting for global norm stats: {stats_file}")
            time.sleep(5.0)

    logger.info("Loading global OpenPI norm stats from %s", stats_dir)
    return _norm_stats_to_float32(normalize.load(stats_dir))


def _compute_adapter_norm_stats(
    base_dataset: torch.utils.data.Dataset,
    data_config: Any,
    *,
    max_frames: int | None = None,
) -> dict[str, NormStats]:
    from openpi.shared import normalize

    stats_ds = _OpenPITransformedDataset(
        base_dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            _RemoveStrings(),
        ],
    )
    running = {"state": normalize.RunningStats(), "actions": normalize.RunningStats()}
    n = len(stats_ds) if max_frames is None else min(len(stats_ds), int(max_frames))
    if n <= 0:
        raise ValueError("Cannot compute OpenPI norm stats from an empty dataset")
    for i in range(n):
        sample = stats_ds[i]
        for key, stat in running.items():
            stat.update(np.asarray(sample[key], dtype=np.float32))
    return _norm_stats_to_float32(
        {key: stat.get_statistics() for key, stat in running.items()}
    )


def _expected_specs_from_model(config: Any) -> tuple[set[str], tuple[int, int]]:
    obs_spec, action_spec = config.model.inputs_spec(batch_size=1)
    action_shape = tuple(int(x) for x in action_spec.shape[1:])
    return set(obs_spec.images.keys()), action_shape


def _validate_openpi_sample(sample: dict[str, Any], *, config: Any) -> None:
    required_image_keys, expected_action_shape = _expected_specs_from_model(config)
    if "actions" not in sample or "state" not in sample:
        raise KeyError(f"sample missing state/actions; keys={list(sample.keys())}")
    if not isinstance(sample.get("image"), dict) or not isinstance(sample.get("image_mask"), dict):
        raise TypeError(f"sample image/image_mask must be dicts; keys={list(sample.keys())}")
    missing_images = required_image_keys - set(sample["image"])
    missing_masks = required_image_keys - set(sample["image_mask"])
    if missing_images or missing_masks:
        raise KeyError(f"missing image keys={missing_images}, missing mask keys={missing_masks}")
    actions_shape = tuple(np.asarray(sample["actions"]).shape[-2:])
    if actions_shape != expected_action_shape:
        raise ValueError(
            f"actions shape mismatch: got {np.asarray(sample['actions']).shape}, "
            f"expected trailing {expected_action_shape}"
        )


def _build_single_task_dataset(
    *,
    task_dir: Path,
    cfg: Any,
    data_config: Any,
    openpi_config: Any,
    action_horizon: int,
    eval_dataset: bool,
    openpi_norm_stats: dict[str, NormStats] | None = None,
):
    data_cfg = cfg.data
    if bool(data_cfg.get("openpi_real_world_joint_require_lerobot_v3", True)) and not _looks_like_lerobot_v3(task_dir):
        raise ValueError(f"{task_dir} does not look like a LeRobot v3 dataset")

    base_dataset = _LeRobotV3FrankaDualOpenPIDataset(
        task_dir,
        action_horizon=action_horizon,
        num_frames=int(data_cfg.get("openpi_real_world_joint_num_frames", 1)),
        use_state=bool(data_cfg.get("openpi_real_world_joint_use_state", True)),
        video_backend=data_cfg.get("video_backend", "pyav"),
        video_tolerance_s=float(data_cfg.get("video_tolerance_s", 0.1)),
        parquet_cache_size=int(data_cfg.get("parquet_cache_size", 16)),
        default_prompt=str(data_cfg.get("default_instruction", "perform the default behavior.")),
    )

    if openpi_norm_stats is None:
        norm_stats_mode = str(
            data_cfg.get("openpi_real_world_joint_norm_stats_mode", "lerobot_q01_q99")
        )
        if norm_stats_mode == "compute":
            max_frames = data_cfg.get("openpi_real_world_joint_norm_stats_max_frames", None)
            openpi_norm_stats = _compute_adapter_norm_stats(
                base_dataset,
                data_config,
                max_frames=None if max_frames is None else int(max_frames),
            )
        elif norm_stats_mode == "lerobot_q01_q99":
            openpi_norm_stats = _norm_stats_from_v3_stats(
                task_dir,
                use_state=bool(data_cfg.get("openpi_real_world_joint_use_state", True)),
            )
        else:
            raise ValueError(
                "Unsupported per-task openpi_real_world_joint_norm_stats_mode="
                f"{norm_stats_mode!r}; expected 'lerobot_q01_q99' or 'compute'"
            )

    transforms = [
        *data_config.repack_transforms.inputs,
        *data_config.data_transforms.inputs,
        _transforms.Normalize(openpi_norm_stats, use_quantiles=True, strict=True),
        *data_config.model_transforms.inputs,
    ]
    ds = _OpenPITransformedDataset(base_dataset, transforms)
    if len(ds) > 0 and bool(data_cfg.get("openpi_real_world_joint_validate_first_sample", True)):
        _validate_openpi_sample(ds[0], config=openpi_config)
    return ds


def build_openpi_real_world_joint_sft_dataloader(
    cfg: Any,
    world_size: int,
    rank: int,
    data_paths: str | list[str],
    eval_dataset: bool = False,
):
    """Build an OpenPI SFT dataloader for local franka_dual LeRobot v3 data."""

    from rlinf.models.embodiment.openpi.dataconfig import get_openpi_config

    root = data_paths[0] if isinstance(data_paths, (list, tuple)) else data_paths
    if cfg.data.get("real_world_joint_root", None):
        root = cfg.data.real_world_joint_root
    task_dirs = _discover_lerobot_v3_tasks(root, cfg.data)
    if not task_dirs:
        raise RuntimeError(f"No LeRobot v3 task/session datasets discovered under {root}")

    config = get_openpi_config(
        cfg.actor.model.openpi.config_name,
        model_path=cfg.actor.model.model_path,
        batch_size=cfg.actor.micro_batch_size * world_size,
        data_kwargs=getattr(cfg.actor, "openpi_data", None),
    )
    data_config = config.data.create(config.assets_dirs, config.model)

    norm_stats_mode = str(
        cfg.data.get("openpi_real_world_joint_norm_stats_mode", "lerobot_q01_q99")
    )
    global_norm_stats = None
    if norm_stats_mode in ("global_lerobot_q01_q99", "global_compute"):
        global_norm_stats = _load_or_create_global_norm_stats(
            task_dirs,
            cfg,
            rank=rank,
            dataset_root=root,
        )

    datasets = []
    for i, task_dir in enumerate(task_dirs, start=1):
        if i == 1 or i % int(cfg.data.get("real_world_joint_discover_log_every", 20)) == 0:
            logger.info("Building OpenPI LeRobot v3 task %s/%s: %s", i, len(task_dirs), task_dir)
        ds = _build_single_task_dataset(
            task_dir=task_dir,
            cfg=cfg,
            data_config=data_config,
            openpi_config=config,
            action_horizon=int(config.model.action_horizon),
            eval_dataset=eval_dataset,
            openpi_norm_stats=global_norm_stats,
        )
        if len(ds) == 0:
            logger.warning("Skipping empty OpenPI LeRobot v3 task: %s", task_dir)
            continue
        datasets.append(ds)

    if not datasets:
        raise RuntimeError(f"No non-empty LeRobot v3 datasets under {root}")
    dataset = torch.utils.data.ConcatDataset(datasets)
    logger.info("OpenPI LeRobot v3 mixture: tasks=%s samples=%s", len(datasets), len(dataset))

    sampler = torch.utils.data.distributed.DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=not eval_dataset,
        drop_last=not eval_dataset,
    )
    num_workers = int(cfg.data.get("num_workers", config.num_workers))
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=int(cfg.actor.micro_batch_size),
        sampler=sampler,
        drop_last=not eval_dataset,
        num_workers=num_workers,
        pin_memory=bool(cfg.data.get("pin_memory", True)),
        persistent_workers=num_workers > 0,
        prefetch_factor=int(cfg.data.get("prefetch_factor", 4)) if num_workers > 0 else None,
    )
    return (
        _OpenPIDataLoaderImpl(
            data_config,
            loader,
            infinite=bool(cfg.data.get("openpi_real_world_joint_infinite_loader", False)),
        ),
        data_config,
    )
