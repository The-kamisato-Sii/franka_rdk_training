# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0

from __future__ import annotations

import bisect
import json
import os
import random
import subprocess
import time
from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from rlinf.data.datasets.dreamzero.sampling_strategy import EmptyTemporalSampleError
from rlinf.data.datasets.dreamzero.utils import collate_ready_sample
from rlinf.utils.logging import get_logger

logger = get_logger()


_REAL_WORLD_PROMPT_IDS = set(range(35, 50))
_REAL_WORLD_LAYOUT_TEXT = (
    " The video is split into three views: the top view is the agent view, "
    "the bottom-left view is the left wrist camera, and the bottom-right view is the right wrist camera. The robot "
)


def _normalize_instruction_text(raw: Any) -> str:
    if not isinstance(raw, str):
        return str(raw).lower()
    return raw.lower()


def _format_real_world_training_prompt(
    instruction: str,
    embodiment_id: int,
    embodiment_tag_mapping: dict[str, int],
) -> str:
    if int(embodiment_id) in _REAL_WORLD_PROMPT_IDS:
        return (
            "A multi-view video shows that a robot "
            + instruction
            + _REAL_WORLD_LAYOUT_TEXT
            + instruction
        )
    from rlinf.data.datasets.dreamzero.data_transforms import (
        format_training_prompt,
    )

    return format_training_prompt(instruction, embodiment_id, embodiment_tag_mapping)


class RealWorldJointCollator:
    """Real-world joint collator; keep base DreamZero collator untouched."""

    def __init__(
        self,
        tokenizer_path: str,
        max_seq_len: int,
        embodiment_tag_mapping: dict[str, int],
    ):
        from groot.vla.model.dreamzero.transform.dreamzero_cotrain import (
            HuggingfaceTokenizer,
        )

        self.tokenizer = HuggingfaceTokenizer(
            name=tokenizer_path,
            seq_len=max_seq_len,
            clean="whitespace",
            local_files_only=True,
        )
        self.embodiment_tag_mapping = embodiment_tag_mapping

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        batch: dict[str, Any] = {}
        for key in features[0]:
            if key == "sample_metadata":
                batch[key] = [elem[key] for elem in features]
                continue
            if key == "text":
                texts = [
                    _format_real_world_training_prompt(
                        _normalize_instruction_text(elem[key]),
                        int(elem["embodiment_id"]),
                        self.embodiment_tag_mapping,
                    )
                    for elem in features
                ]
                ids, mask = self.tokenizer(
                    texts, return_mask=True, add_special_tokens=True
                )
                batch[key] = ids
                batch["text_attention_mask"] = mask
                continue
            if key == "text_negative":
                values = [elem[key] for elem in features]
                ids, mask = self.tokenizer(
                    values, return_mask=True, add_special_tokens=True
                )
                batch[key] = ids
                batch["text_attention_mask_negative"] = mask
                continue

            values = [elem[key] for elem in features]
            values = [
                v.detach().cpu().numpy() if isinstance(v, torch.Tensor) else v
                for v in values
            ]
            try:
                batch[key] = torch.from_numpy(np.stack(values))
            except ValueError as exc:
                shapes = [np.asarray(v).shape for v in values]
                raise ValueError(
                    f"Shape mismatch in collate for key='{key}': shapes={shapes}"
                ) from exc
        if "motion.point_map" in batch and "motion.scene_flow" in batch:
            batch["motion"] = {
                "point_map": batch.pop("motion.point_map"),
                "scene_flow": batch.pop("motion.scene_flow"),
            }
        return batch


def _drop_file_cache(path: Path) -> None:
    """Tell Linux that a just-read motion NPY should not stay in page cache."""

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


class RealWorldJointBadEpisodeError(RuntimeError):
    """real_world_joint 数据校验失败时使用的异常。

    这个异常会被数据集的 ``__getitem__`` 捕获，然后打印 warning 并重新采样，
    因此单条坏轨迹不会让整个训练中断。
    """

    def __init__(self, message: str, episode_index: int | None = None):
        super().__init__(message)
        self.episode_index = episode_index

REAL_WORLD_JOINT_TAGS: tuple[str, ...] = (
    "allenai_real_world_joint",
    "robomind2_ark_joint",
    "robomind2_franka_joint",
    "robomind2_ur_joint",
    "robocoin_galaxea_r1_lite",
    "robocoin_galaxea_r1_lite_dual",
    "robocoin_agilex_cobot_magic_14d",
    "robocoin_agilex_cobot_magic_26d",
    "robocoin_ruantong_a2d_17d",
    "robocoin_ruantong_a2d_34d",
    "robocoin_realman_rmc_aidal",
    "robocoin_yinhe",
    "robocoin_alpha_bot_2",
    "real_world_franka_dual",
)

DEFAULT_REAL_WORLD_JOINT_TAG_MAPPING: dict[str, int] = {
    "allenai_real_world_joint": 36,
    "robomind2_ark_joint": 37,
    "robomind2_franka_joint": 38,
    "robomind2_ur_joint": 39,
    "robocoin_galaxea_r1_lite": 40,
    "robocoin_galaxea_r1_lite_dual": 41,
    "robocoin_agilex_cobot_magic_14d": 42,
    "robocoin_agilex_cobot_magic_26d": 43,
    "robocoin_ruantong_a2d_17d": 44,
    "robocoin_ruantong_a2d_34d": 45,
    "robocoin_realman_rmc_aidal": 46,
    "robocoin_yinhe": 47,
    "robocoin_alpha_bot_2": 48,
    "real_world_franka_dual": 49,
}

LEGACY_DREAMZERO_TAG_MAPPING: dict[str, int] = {
    "real_gr1_arms_only": 0,
    "real_gr1_arms_only_annotated": 1,
    "real_gr1_arms_waist": 2,
    "real_gr1_arms_waist_annotated": 3,
    "dexmg_gr1_arms_only_inspire": 4,
    "dexmg_gr1_arms_only_fourier": 5,
    "dexmg_gr1_arms_waist_fourier": 6,
    "robocasa_single_arm": 7,
    "onex_eve_gripper": 8,
    "robocasa_gr1_arms_only_inspire_hands": 9,
    "robocasa_gr1_arms_only_fourier_hands": 10,
    "robocasa_gr1_fixed_lower_body_inspire_hands": 11,
    "robocasa_gr1_fixed_lower_body_fourier_hands": 12,
    "robocasa_panda_omron": 13,
    "gr1_unified_segmentation": 14,
    "robocasa_bimanual_panda_parallel_gripper": 15,
    "robocasa_bimanual_panda_inspire_hand": 16,
    "oxe_droid": 17,
    "oxe_fractal": 18,
    "oxe_language_table": 19,
    "oxe_bridge": 20,
    "real_panda_single_arm": 21,
    "xdof": 22,
    "hot3d_hands_only": 23,
    "gr1_unified": 24,
    "robocasa_gr1_arms_waist_fourier_hands": 25,
    "agibot": 26,
    "lapa": 27,
    "oxe_mutex": 28,
    "oxe_roboset": 29,
    "oxe_plex": 30,
    "dream": 31,
    "yam": 32,
    "robotwin": 33,
    "robotwin2": 33,
    "libero": 34,
    "language_table_sim": 7,
    "gr1_isaac": 0,
    "sim_behavior_r1_pro": 31,
    "mecka_hands": 27,
    "real_r1_pro_sharpa": 28,
}

ROBOMIND_SRC = "robomind2_lerobot_v21"
ROBOMIND_ROBOT_TO_TAG = {
    "ark": "robomind2_ark_joint",
    "franka": "robomind2_franka_joint",
    "ur": "robomind2_ur_joint",
}
FLAT_SRCS = {"allenai_lerobot_v3_filtered", "robocoin_filtered_v1"}
SKIP_NAMES = {"filter_plan", "logs", "motion_vis"}
VIDEO_KEYS = ["video.agent_view", "video.left_wrsit_view", "video.right_wrist_view"]
FRANKA_DUAL_VIDEO_KEYS = ["video.middle_zed", "video.left_camera", "video.right_camera"]
FRANKA_DUAL_STATE_KEYS = [
    "state.left_joint_angle",
    "state.left_joint_gripper",
    "state.right_joint_angle",
    "state.right_joint_gripper",
]
FRANKA_DUAL_ACTION_KEYS = [
    "action.left_joint_angle",
    "action.left_joint_gripper",
    "action.right_joint_angle",
    "action.right_joint_gripper",
]
FRANKA_DUAL_COMPONENT_SLICES: dict[str, slice] = {
    "left_joint_angle": slice(0, 7),
    "left_joint_gripper": slice(7, 8),
    "right_joint_angle": slice(8, 15),
    "right_joint_gripper": slice(15, 16),
}
LANGUAGE_KEYS = ["annotation.task_index"]
RUANTONG_34D_TAG = "robocoin_ruantong_a2d_34d"
FRANKA_DUAL_TAG = "real_world_franka_dual"


def _embodiment_tag_mapping(model_cfg: Any) -> dict[str, int]:
    """合并 DreamZero 旧 embodiment id 和 real-world joint 新 id。"""

    mapping = dict(LEGACY_DREAMZERO_TAG_MAPPING)
    mapping.update(DEFAULT_REAL_WORLD_JOINT_TAG_MAPPING)
    mapping.update(dict(model_cfg.get("embodiment_tag_mapping") or {}))
    return mapping


def _read_embodiment_tag(task_dir: Path) -> str | None:
    """从单个任务目录读取 ``meta/embodiment.json`` 里的 embodiment_tag。"""

    path = task_dir / "meta" / "embodiment.json"
    if not path.is_file():
        return None
    try:
        tag = json.loads(path.read_text(encoding="utf-8")).get("embodiment_tag")
    except Exception:
        return None
    return str(tag) if tag else None


def discover_real_world_joint_task_paths(root: str | Path, data_cfg: Any | None = None) -> dict[str, list[str]]:
    """扫描 real-world 数据根目录，返回 ``{embodiment_tag: [task_dir, ...]}``。

    这里做的是“任务发现”，不是读取样本。比如 root 下面有
    ``allenai_lerobot_v3_filtered/01122025-box-01``，且该目录的
    ``meta/embodiment.json`` 写着 ``allenai_real_world_joint``，返回值里就会有：
    ``{"allenai_real_world_joint": [".../01122025-box-01"]}``。
    """

    root = Path(root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"real_world_joint data root not found: {root}")

    allowed_tags = _as_optional_tag_set(data_cfg.get("real_world_joint_tags", None)) if data_cfg is not None else None
    max_per_tag = _as_optional_int(data_cfg.get("real_world_joint_max_tasks_per_tag", None)) if data_cfg is not None else None
    max_total = _as_optional_int(data_cfg.get("real_world_joint_max_tasks", None)) if data_cfg is not None else None
    name_contains = data_cfg.get("real_world_joint_task_name_contains", None) if data_cfg is not None else None
    if isinstance(name_contains, str) and name_contains.lower() in {"", "none", "null"}:
        name_contains = None

    tag_to_paths: dict[str, list[str]] = {tag: [] for tag in REAL_WORLD_JOINT_TAGS}
    total = 0
    scanned = 0

    def emit_progress(src_name: str, *, force: bool = False) -> None:
        return None
    def accept(tag: str, path: Path) -> bool:
        nonlocal total
        if tag not in tag_to_paths:
            return False
        if allowed_tags is not None and tag not in allowed_tags:
            return False
        if name_contains:
            needle = str(name_contains)
            if needle not in path.name and needle not in str(path):
                return False
        if max_per_tag is not None and len(tag_to_paths[tag]) >= max_per_tag:
            return False
        if max_total is not None and total >= max_total:
            return False
        tag_to_paths[tag].append(str(path))
        total += 1
        return True

    def done() -> bool:
        return max_total is not None and total >= max_total

    for src_dir in sorted(root.iterdir()):
        if done():
            break
        if not src_dir.is_dir() or src_dir.name in SKIP_NAMES:
            continue
        src_scanned_before = scanned
        src_accepted_before = total
        # Some Franka dual debug datasets are already task-level roots:
        #   root/<task_name>/{meta,data,videos}
        # Older mixed roots may still use:
        #   root/<source_or_group>/<task_name>/{meta,data,videos}
        # Accept the direct task directory first so both layouts work.
        if (src_dir / "meta" / "info.json").is_file():
            scanned += 1
            tag = _read_embodiment_tag(src_dir) or FRANKA_DUAL_TAG
            accept(tag, src_dir)
            emit_progress(root.name)
            continue
        if src_dir.name == ROBOMIND_SRC:
            for robot_dir in sorted(src_dir.iterdir()):
                if done():
                    break
                if not robot_dir.is_dir() or robot_dir.name not in ROBOMIND_ROBOT_TO_TAG:
                    continue
                scanned += 1
                tag = _read_embodiment_tag(robot_dir) or ROBOMIND_ROBOT_TO_TAG[robot_dir.name]
                accept(tag, robot_dir)
                emit_progress(src_dir.name)
            emit_progress(src_dir.name, force=True)
            continue
        if src_dir.name in FLAT_SRCS:
            for task_dir in sorted(src_dir.iterdir()):
                if done():
                    break
                if not task_dir.is_dir() or task_dir.name in SKIP_NAMES:
                    continue
                scanned += 1
                tag = _read_embodiment_tag(task_dir)
                if tag is not None:
                    accept(tag, task_dir)
                emit_progress(src_dir.name)
            emit_progress(src_dir.name, force=True)
            continue
        # Franka dual exports are organized as:
        #   root/<task_name>/<timestamp>/{meta,data,videos}
        # They are plain LeRobot v3 datasets and do not carry embodiment.json.
        for task_dir in sorted(src_dir.iterdir()):
            if done():
                break
            if not task_dir.is_dir() or task_dir.name in SKIP_NAMES:
                continue
            if not (task_dir / "meta" / "info.json").is_file():
                continue
            scanned += 1
            tag = _read_embodiment_tag(task_dir) or FRANKA_DUAL_TAG
            accept(tag, task_dir)
            emit_progress(src_dir.name)
        emit_progress(src_dir.name, force=True)
    return {tag: paths for tag, paths in tag_to_paths.items() if paths}


def _as_optional_int(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, str) and value.lower() in {"", "none", "null"}:
        return None
    out = int(value)
    return out if out > 0 else None


def _as_optional_tag_set(value: Any) -> set[str] | None:
    if value is None:
        return None
    if isinstance(value, str):
        parts = [part.strip() for part in value.split(",") if part.strip()]
    else:
        parts = [str(part).strip() for part in value if str(part).strip()]
    return set(parts) if parts else None


def _filter_discovered_real_world_tasks(
    discovered: dict[str, list[str]], data_cfg: Any
) -> dict[str, list[str]]:
    """根据 debug 配置过滤已发现的任务，并保持确定性的 tag/path 顺序。

    支持按 tag、任务名包含字符串、最多任务数、每个 tag 最多任务数过滤。
    这些过滤只影响“用哪些任务建数据集”，不会改变单个任务内部的数据内容。
    """

    allowed_tags = _as_optional_tag_set(data_cfg.get("real_world_joint_tags", None))
    max_per_tag = _as_optional_int(data_cfg.get("real_world_joint_max_tasks_per_tag", None))
    max_total = _as_optional_int(data_cfg.get("real_world_joint_max_tasks", None))
    name_contains = data_cfg.get("real_world_joint_task_name_contains", None)
    if isinstance(name_contains, str) and name_contains.lower() in {"", "none", "null"}:
        name_contains = None

    filtered: dict[str, list[str]] = {}
    remaining = max_total
    for tag in REAL_WORLD_JOINT_TAGS:
        if allowed_tags is not None and tag not in allowed_tags:
            continue
        paths = list(discovered.get(tag, []))
        if name_contains:
            needle = str(name_contains)
            paths = [p for p in paths if needle in Path(p).name or needle in p]
        if max_per_tag is not None:
            paths = paths[:max_per_tag]
        if remaining is not None:
            paths = paths[:remaining]
            remaining -= len(paths)
        if paths:
            filtered[tag] = paths
        if remaining is not None and remaining <= 0:
            break
    return filtered


def _state_action_keys_for_tag(tag: str) -> tuple[list[str], list[str]]:
    """返回某个 embodiment tag 对应的 state/action key 列表。"""

    if tag == FRANKA_DUAL_TAG:
        return list(FRANKA_DUAL_STATE_KEYS), list(FRANKA_DUAL_ACTION_KEYS)
    if tag == RUANTONG_34D_TAG:
        return ["state.arm_joints", "state.body_gripper_pose"], ["action.arm_joints", "action.body_gripper"]
    return ["state.joint"], ["action.joint"]


def _video_keys_for_tag(tag: str) -> list[str]:
    """返回某个 embodiment tag 对应的三视角 video key 顺序。"""

    if tag == FRANKA_DUAL_TAG:
        return list(FRANKA_DUAL_VIDEO_KEYS)
    return list(VIDEO_KEYS)


def _ensure_legacy_tokenizer_path(tokenizer_path: str) -> None:
    """确保旧路径 ``checkpoints/umt5-xxl`` 可用，避免离线机器运行时去联网。"""

    legacy = Path("checkpoints/umt5-xxl")
    if legacy.exists() or legacy.is_symlink():
        return
    target = Path(str(tokenizer_path)).expanduser()
    if not target.exists():
        return
    legacy.parent.mkdir(parents=True, exist_ok=True)
    try:
        legacy.symlink_to(target, target_is_directory=True)
    except FileExistsError:
        pass


class _RealWorldJointVideoResizeTile:
    """把多视角视频 resize 后拼成 DreamZero 期望的 agentview 图像。

    real-world 数据一般有 agent/left/right 三个视角。这里把 agent 视角放上方，
    左右腕部视角拼在下方，使后续 DreamZero transform 仍然看到一个统一的
    ``video.agentview``。
    """

    def __init__(self, apply_to: list[str], agent_height: int = 128, agent_width: int = 256, wrist_height: int = 128, wrist_width: int = 128, interpolation: str = "linear"):
        self.apply_to = list(apply_to)
        self.agent_height = int(agent_height)
        self.agent_width = int(agent_width)
        self.wrist_height = int(wrist_height)
        self.wrist_width = int(wrist_width)
        self.interpolation = interpolation
        self.training = True

    def set_metadata(self, dataset_metadata):
        if len(self.apply_to) != 3:
            raise ValueError(f"RealWorldJointVideoResizeTile expects 3 views, got {self.apply_to}")

    def train(self):
        self.training = True

    def eval(self):
        self.training = False

    @staticmethod
    def _resize(frames: np.ndarray, height: int, width: int, interpolation: str) -> np.ndarray:
        import cv2

        arr = np.asarray(frames)
        if arr.ndim != 4:
            raise ValueError(f"Expected [T,H,W,C], got {arr.shape}")
        if arr.dtype != np.uint8:
            arr_f = arr.astype(np.float32, copy=False)
            if arr_f.size and float(arr_f.max()) <= 1.0 + 1e-3:
                arr = np.clip(arr_f * 255.0, 0, 255).astype(np.uint8)
            else:
                arr = np.clip(arr_f, 0, 255).astype(np.uint8)
        interp = cv2.INTER_LINEAR if interpolation == "linear" else cv2.INTER_NEAREST
        return np.stack([cv2.resize(frame, (width, height), interpolation=interp) for frame in arr], axis=0)

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        for key in self.apply_to:
            if key not in data:
                raise KeyError(f"Missing required video key {key}; available keys: {list(data.keys())}")
        agent_key, left_key, right_key = self.apply_to
        agent = self._resize(data[agent_key], self.agent_height, self.agent_width, self.interpolation)
        left = self._resize(data[left_key], self.wrist_height, self.wrist_width, self.interpolation)
        right = self._resize(data[right_key], self.wrist_height, self.wrist_width, self.interpolation)
        if not (agent.shape[0] == left.shape[0] == right.shape[0]):
            raise ValueError(f"View time dimensions differ: agent={agent.shape}, left={left.shape}, right={right.shape}")
        bottom = np.concatenate([left, right], axis=2)
        composite = np.concatenate([agent, bottom], axis=1)
        data["video"] = np.expand_dims(composite, axis=1)
        for key in self.apply_to:
            data.pop(key, None)
        return data


class _LocalStateActionToTensor:
    """把 state/action 的 numpy 数组转成 torch Tensor。"""

    def __init__(self, apply_to: list[str]):
        self.apply_to = list(apply_to)
        self.training = True

    def set_metadata(self, dataset_metadata):
        pass

    def train(self):
        self.training = True

    def eval(self):
        self.training = False

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        for key in self.apply_to:
            if key in data and not torch.is_tensor(data[key]):
                data[key] = torch.from_numpy(np.asarray(data[key]))
        return data


class _LocalStateActionTransform:
    """根据 metadata 里的统计量对 state/action 做归一化。"""

    def __init__(self, apply_to: list[str], normalization_modes: dict[str, str]):
        self.apply_to = list(apply_to)
        self.normalization_modes = dict(normalization_modes)
        self._stats: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        self.training = True

    def set_metadata(self, dataset_metadata):
        for key, mode in self.normalization_modes.items():
            if mode != "q99":
                raise ValueError(f"Only q99 normalization is supported in real-world joint fast path, got {mode}")
            modality, short = key.split(".", 1)
            values = getattr(dataset_metadata.statistics, modality)[short]
            self._stats[key] = (
                torch.as_tensor(values.q01, dtype=torch.float32),
                torch.as_tensor(values.q99, dtype=torch.float32),
            )

    def train(self):
        self.training = True

    def eval(self):
        self.training = False

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        for key in self.apply_to:
            if key not in data:
                continue
            x = data[key]
            if not torch.is_tensor(x):
                x = torch.from_numpy(np.asarray(x))
            if key in self._stats:
                q01, q99 = self._stats[key]
                q01 = q01.to(dtype=x.dtype, device=x.device)
                q99 = q99.to(dtype=x.dtype, device=x.device)
                mask = q01 != q99
                y = torch.zeros_like(x)
                y[..., mask] = 2 * ((x[..., mask] - q01[..., mask]) / (q99[..., mask] - q01[..., mask])) - 1
                y[..., ~mask] = x[..., ~mask]
                x = torch.clamp(y, -1, 1)
            data[key] = x
        return data


class _LocalConcatTransform:
    """按固定顺序拼接多路 video/state/action。

    例如 ``state.joint`` 和 ``state.eef`` 会被拼成一个 ``state``，
    action 也是同理。这样模型侧仍然接收 DreamZero 原来的统一字段。
    """

    def __init__(self, video_concat_order: list[str], state_concat_order: list[str], action_concat_order: list[str]):
        self.video_concat_order = list(video_concat_order)
        self.state_concat_order = list(state_concat_order)
        self.action_concat_order = list(action_concat_order)
        self.training = True

    def set_metadata(self, dataset_metadata):
        pass

    def set_transform_pipeline(self, transforms):
        pass

    def train(self):
        self.training = True

    def eval(self):
        self.training = False

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        if self.video_concat_order:
            views = [np.expand_dims(data.pop(key), axis=-4) for key in self.video_concat_order]
            data["video"] = np.concatenate(views, axis=-4)
        if self.state_concat_order:
            data["state"] = torch.cat([data.pop(key) for key in self.state_concat_order], dim=-1)
        if self.action_concat_order:
            data["action"] = torch.cat([data.pop(key) for key in self.action_concat_order], dim=-1)
        return data


class _LocalComposedTransform:
    """轻量版 transform pipeline，按顺序调用多个 transform。"""

    def __init__(self, transforms: list[Any]):
        self.transforms = list(transforms)
        self.training = True

    def set_metadata(self, dataset_metadata):
        for transform in self.transforms:
            transform.set_metadata(dataset_metadata)
            if hasattr(transform, "set_transform_pipeline"):
                transform.set_transform_pipeline(self.transforms)

    def train(self):
        for transform in self.transforms:
            transform.train()
        self.training = True

    def eval(self):
        for transform in self.transforms:
            transform.eval()
        self.training = False

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        for i, transform in enumerate(self.transforms):
            try:
                data = transform(data)
            except Exception as exc:
                raise ValueError(f"Error applying transform {i} to data: {exc}") from exc
        return data


class _RealWorldJointDreamTransform:
    """把 real-world joint 原始字段整理成 DreamZero 训练样本字段。"""

    def __init__(
        self,
        *,
        default_instruction: str,
        language_dropout_prob: float,
        always_use_default_instruction: bool,
        max_state_dim: int,
        max_action_dim: int,
        use_motion_modality: bool,
        motion_representation: str,
        motion_horizon: int,
        max_length: int,
        state_horizon: int,
        action_horizon: int,
        embodiment_tag_mapping: dict[str, int],
        tokenizer_path: str,
        use_proprioception: bool,
    ):
        self.default_instruction = default_instruction
        self.language_dropout_prob = float(language_dropout_prob)
        self.always_use_default_instruction = bool(always_use_default_instruction)
        self.max_state_dim = int(max_state_dim)
        self.max_action_dim = int(max_action_dim)
        self.use_motion_modality = bool(use_motion_modality)
        self.motion_representation = motion_representation
        self.motion_horizon = int(motion_horizon)
        self.max_length = int(max_length)
        self.state_horizon = int(state_horizon)
        self.action_horizon = int(action_horizon)
        self.embodiment_tag_mapping = dict(embodiment_tag_mapping)
        self.tokenizer_path = tokenizer_path
        self.use_proprioception = bool(use_proprioception)
        self.embodiment_tag = None
        self.training = True

    def set_metadata(self, dataset_metadata):
        tag = dataset_metadata.embodiment_tag
        self.embodiment_tag = str(tag.value if hasattr(tag, "value") else tag)

    def train(self):
        self.training = True

    def eval(self):
        self.training = False

    def get_embodiment_tag(self) -> int:
        if self.embodiment_tag is None:
            raise RuntimeError("embodiment_tag is not set; call set_metadata first")
        return int(self.embodiment_tag_mapping[self.embodiment_tag])

    def _prepare_video(self, data: dict[str, Any]) -> np.ndarray:
        video = np.asarray(data["video"])
        if video.ndim != 5:
            raise ValueError(f"Expected video [T,V,H,W,C], got {video.shape}")
        if video.shape[1] != 1:
            raise ValueError(f"Expected tiled single view video, got {video.shape}")
        return video[:, 0].astype(np.uint8, copy=False)

    def _prepare_language(self, data: dict[str, Any]) -> str:
        if self.always_use_default_instruction:
            return self.default_instruction
        text = data.get("annotation.task_index", None)
        if text is None:
            text = data.get("language", None)
        if text is None or str(text) == "":
            text = self.default_instruction
        return str(text).lower()

    def _prepare_state(self, data: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
        if not self.use_proprioception:
            state = np.zeros((0, self.max_state_dim), dtype=np.float32)
            return state, np.zeros_like(state, dtype=bool)

        state = data.get("state")
        if state is None:
            out = np.zeros((self.state_horizon, self.max_state_dim), dtype=np.float32)
            return out, np.zeros_like(out, dtype=bool)
        if torch.is_tensor(state):
            state = state.detach().cpu().numpy()
        state = np.asarray(state, dtype=np.float32)
        if state.ndim == 1:
            state = state[None, :]
        n_dims = min(int(state.shape[1]), self.max_state_dim)
        out = np.zeros((state.shape[0], self.max_state_dim), dtype=np.float32)
        out[:, :n_dims] = state[:, :n_dims]
        mask = np.zeros_like(out, dtype=bool)
        mask[:, :n_dims] = True
        return out, mask

    def _prepare_action(self, data: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
        action = data.get("action")
        if action is None:
            out = np.zeros((self.action_horizon, self.max_action_dim), dtype=np.float32)
            return out, np.zeros_like(out, dtype=bool)
        if torch.is_tensor(action):
            action = action.detach().cpu().numpy()
        action = np.asarray(action, dtype=np.float32)
        if action.ndim == 1:
            action = action[None, :]
        if action.shape[0] % self.action_horizon != 0:
            raise ValueError(f"action length must be divisible by action_horizon: {action.shape=}, {self.action_horizon=}")
        n_dims = min(int(action.shape[1]), self.max_action_dim)
        out = np.zeros((action.shape[0], self.max_action_dim), dtype=np.float32)
        out[:, :n_dims] = action[:, :n_dims]
        # Padded action dimensions are real training targets too: their target
        # value is zero after padding, and the action head should learn that
        # full 32-D target/noise distribution instead of masking dimensions out.
        mask = np.ones_like(out, dtype=bool)
        return out, mask

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        images = self._prepare_video(data)
        state, state_mask = self._prepare_state(data)
        action, action_mask = self._prepare_action(data)
        out: dict[str, Any] = {
            "images": images,
            "text": self._prepare_language(data),
            "state": state,
            "state_mask": state_mask,
            "action": action,
            "action_mask": action_mask,
            "segmentation_target": np.zeros((2,), dtype=np.float32),
            "segmentation_target_mask": np.zeros((1,), dtype=np.float32),
            "has_real_action": np.ones((), dtype=bool),
            "lapa_action": np.zeros_like(action),
            "lapa_action_mask": np.zeros_like(action_mask),
            "text_negative": "Vibrant colors, overexposed, static, blurry details, text, subtitles, style, artwork, painting, image, still, grayscale, dull, worst quality, low quality, JPEG artifacts, ugly, mutilated, extra fingers, bad hands, bad face, deformed, disfigured, mutated limbs, fused fingers, stagnant image, cluttered background, three legs, many people in the background, walking backwards.",
            "embodiment_id": self.get_embodiment_tag(),
            "has_lapa_action": np.zeros((), dtype=bool),
            "is_cotrain_instance": np.zeros((), dtype=bool),
        }
        if "motion.point_map" in data and "motion.scene_flow" in data:
            out["motion.point_map"] = data["motion.point_map"]
            out["motion.scene_flow"] = data["motion.scene_flow"]
        return out


def _stats_for_component(stats: dict[str, Any], source: str, sl: slice, width: int) -> dict[str, list[float]]:
    """从完整统计量里切出某个 component 对应的 mean/std/q01/q99。"""

    source_stats = stats.get(source) or {}
    out: dict[str, list[float]] = {}
    for key in ("mean", "std", "min", "max", "q01", "q99"):
        values = source_stats.get(key)
        if values is None:
            if key == "std":
                out[key] = [1.0] * width
            elif key == "q01":
                out[key] = [-1.0] * width
            elif key == "q99":
                out[key] = [1.0] * width
            else:
                out[key] = [0.0] * width
            continue
        sliced = list(values[sl])
        if len(sliced) < width:
            fill = 1.0 if key in ("std", "q99") else (-1.0 if key == "q01" else 0.0)
            sliced.extend([fill] * (width - len(sliced)))
        out[key] = [float(x) for x in sliced[:width]]
    return out


def _modality_entry(meta: dict[str, Any], modality: str, key: str) -> dict[str, Any] | None:
    """在 LeRobot metadata 的 modalities 中查找某个 key 的描述。"""

    short = key.split(".", 1)[1] if "." in key else key
    entries = meta.get(modality)
    if isinstance(entries, dict):
        entry = entries.get(short)
        if isinstance(entry, dict):
            return entry
    return None


def _feature_dim(info: dict[str, Any], source: str) -> int:
    """从 features metadata 推断某个源字段的最后一维宽度。"""

    feature = (info.get("features") or {}).get(source) or {}
    shape = feature.get("shape") or [0]
    return int(shape[0] or 0)


def _component_source_slice_width(task_dir: Path, info: dict[str, Any], modality_meta: dict[str, Any], modality: str, key: str) -> tuple[str, slice, int]:
    """确定一个 state/action component 在 parquet 源字段中的切片范围。

    例如 ``action.joint`` 可能来自 parquet 里的 ``action`` 列，并对应
    ``action`` 向量的 ``slice(0, 14)``。返回值就是：
    ``(source_column, slice_range, width)``。
    """

    default_source = "observation.state" if modality == "state" else "action"
    entry = _modality_entry(modality_meta, modality, key)
    source = str((entry or {}).get("original_key") or default_source)
    source_dim = _feature_dim(info, source)
    short = key.split(".", 1)[1] if "." in key else key
    if entry is None and short in FRANKA_DUAL_COMPONENT_SLICES and source_dim >= 16:
        sl = FRANKA_DUAL_COMPONENT_SLICES[short]
        start = int(sl.start or 0)
        end = int(sl.stop if sl.stop is not None else source_dim)
        return source, slice(start, end), max(0, end - start)
    start = int((entry or {}).get("start", 0))
    end_val = (entry or {}).get("end")
    end = source_dim if end_val is None else int(end_val)
    if end <= start:
        end = source_dim
    width = max(0, end - start)
    if width <= 0:
        raise ValueError(f"Cannot infer width for {key} under {task_dir}; source={source} start={start} end={end}")
    return source, slice(start, end), width


def _local_metadata_object(blob: dict[str, Any]):
    """Return a DatasetMetadata-like object for the franka-dual local transform.

    Groot's DatasetMetadata enum may not know the new tag yet, but its nested
    statistics/modalities classes are still useful and keep interfaces such as
    ``model_dump()`` and ``StateActionMetadata`` type checks intact.
    """

    from groot.vla.data.schema import DatasetModalities, DatasetStatistics

    return SimpleNamespace(
        statistics=DatasetStatistics.model_validate(blob.get("statistics") or {}),
        modalities=DatasetModalities.model_validate(blob.get("modalities") or {}),
        embodiment_tag=str(blob.get("embodiment_tag", "")),
    )


def _metadata_for_task(
    task_dir: Path,
    tag: str,
    state_keys: list[str],
    action_keys: list[str],
    *,
    stats_root: Path | None = None,
    norm_stats_mode: str = "global_lerobot_q01_q99",
):
    """为单个任务构建 transform 需要的 metadata 和归一化统计量。

    ``global_lerobot_q01_q99`` is the real-world default so every task shares
    one dataset-root q01/q99 normalization convention, matching OpenPI and
    deployment.
    """

    from groot.vla.data.schema import DatasetMetadata

    meta_dir = task_dir / "meta"
    info = json.loads((meta_dir / "info.json").read_text(encoding="utf-8"))
    norm_stats_mode = str(norm_stats_mode)
    if norm_stats_mode == "task_lerobot_q01_q99":
        stats_path = meta_dir / "stats.json"
    elif norm_stats_mode == "global_lerobot_q01_q99":
        if stats_root is None:
            raise ValueError("stats_root is required when norm_stats_mode=global_lerobot_q01_q99")
        stats_path = Path(stats_root) / "stats.json"
        if not stats_path.is_file():
            raise FileNotFoundError(
                f"Global real-world joint stats not found: {stats_path}. "
                "Run dataset filtering/stat generation first or use task_lerobot_q01_q99."
            )
    else:
        raise ValueError(
            "Unsupported real_world_joint_norm_stats_mode="
            f"{norm_stats_mode!r}; expected 'task_lerobot_q01_q99' or 'global_lerobot_q01_q99'."
        )
    stats = json.loads(stats_path.read_text(encoding="utf-8")) if stats_path.is_file() else {}
    modality_path = meta_dir / "modality.json"
    modality_meta = json.loads(modality_path.read_text(encoding="utf-8")) if modality_path.is_file() else {}

    state_stats = {}
    state_modalities = {}
    for key in state_keys:
        source, sl, width = _component_source_slice_width(task_dir, info, modality_meta, "state", key)
        short = key.split(".", 1)[1]
        state_stats[short] = _stats_for_component(stats, source, sl, width)
        state_modalities[short] = {"absolute": True, "shape": [width], "continuous": True}

    action_stats = {}
    action_modalities = {}
    for key in action_keys:
        source, sl, width = _component_source_slice_width(task_dir, info, modality_meta, "action", key)
        short = key.split(".", 1)[1]
        action_stats[short] = _stats_for_component(stats, source, sl, width)
        action_modalities[short] = {"absolute": True, "shape": [width], "continuous": True}

    blob = {
        "statistics": {"state": state_stats, "action": action_stats},
        "modalities": {
            "video": {
                "agent_view": {"resolution": [256, 128], "channels": 3, "fps": 30},
                "left_wrsit_view": {"resolution": [128, 128], "channels": 3, "fps": 30},
                "right_wrist_view": {"resolution": [128, 128], "channels": 3, "fps": 30},
            },
            "state": state_modalities,
            "action": action_modalities,
        },
        "embodiment_tag": tag,
    }
    if tag == FRANKA_DUAL_TAG:
        return _local_metadata_object(blob)
    return DatasetMetadata.model_validate(blob)


def _make_transform(*, tokenizer_path: str, max_seq_len: int, max_state_dim: int, max_action_dim: int, embodiment_tag_mapping: dict[str, int], include_motion: bool, use_proprioception: bool, video_keys: list[str], state_keys: list[str], action_keys: list[str]):
    """构建 real-world joint 使用的完整 transform pipeline。"""

    _ensure_legacy_tokenizer_path(tokenizer_path)
    transforms: list[Any] = [
        _RealWorldJointVideoResizeTile(
            apply_to=video_keys,
            agent_height=128,
            agent_width=256,
            wrist_height=128,
            wrist_width=128,
            interpolation="linear",
        ),
        _LocalStateActionToTensor(apply_to=state_keys),
        _LocalStateActionTransform(apply_to=state_keys, normalization_modes={key: "q99" for key in state_keys}),
        _LocalStateActionToTensor(apply_to=action_keys),
        _LocalStateActionTransform(apply_to=action_keys, normalization_modes={key: "q99" for key in action_keys}),
        _LocalConcatTransform(video_concat_order=[], state_concat_order=state_keys, action_concat_order=action_keys),
        _RealWorldJointDreamTransform(
            default_instruction="Perform the default behavior.",
            language_dropout_prob=0.0,
            always_use_default_instruction=False,
            max_state_dim=max_state_dim,
            max_action_dim=max_action_dim,
            use_motion_modality=include_motion,
            motion_representation="point_map",
            motion_horizon=8 if include_motion else 24,
            max_length=max_seq_len,
            state_horizon=1,
            action_horizon=48,
            embodiment_tag_mapping=embodiment_tag_mapping,
            tokenizer_path=tokenizer_path,
            use_proprioception=use_proprioception,
        ),
    ]
    return _LocalComposedTransform(transforms=transforms)


class _RealWorldJointMotionMixin:
    """给 DreamZeroLeRobotDataset 增加 real-world joint 的校验、motion 和重采样逻辑。

    这个 mixin 不改变模型输入接口，只是在样本读取阶段保证：
    1. 坏 episode 会被 warning 后跳过；
    2. action parquet 只覆盖完整视频的一段时，视频和 motion 会按 parquet 里的
       ``frame_index`` 对齐到真实帧号。
    """

    def __init__(
        self,
        *args,
        include_motion: bool = False,
        motion_downsample_ratio: int = 6,
        motion_cache_size: int = 0,
        scene_flow_training: bool = True,
        use_sam_scene_flow: bool = False,
        motion_dir_name: str | None = None,
        scene_flow_visconf_threshold: float = 0.5,
        drop_motion_npy_file_cache: bool = True,
        fast_local_index: bool = True,
        bad_sample_resample_attempts: int = 64,
        **kwargs,
    ):
        self.fast_local_index = bool(fast_local_index)
        super().__init__(*args, **kwargs)
        self.include_motion = bool(include_motion)
        self.motion_downsample_ratio = max(1, int(motion_downsample_ratio))
        self.scene_flow_training = bool(scene_flow_training)
        self.use_sam_scene_flow = bool(use_sam_scene_flow)
        self.scene_flow_visconf_threshold = float(scene_flow_visconf_threshold)
        self._motion_dir_candidates = self._build_motion_dir_candidates(motion_dir_name)
        self._motion_dir_name = self._motion_dir_candidates[0]
        self._motion_cache: OrderedDict[int, dict[str, np.ndarray]] = OrderedDict()
        # motion npz 是整条 episode 的 point_map/scene_flow，单条就可能很大。
        # 多机多卡时每个 rank、每个 DataLoader worker 都会各自持有 Dataset 实例；
        # 默认关闭 cache，避免 900+ task 训练时 CPU 内存被整条 motion episode 放大占满。
        self._motion_cache_size = max(0, int(motion_cache_size))
        self._drop_motion_npy_file_cache = bool(drop_motion_npy_file_cache)
        self._bad_sample_resample_attempts = max(
            int(getattr(self, "multi_anchor_resample_attempts", 8)),
            int(bad_sample_resample_attempts),
        )
        self._bad_sample_warning_keys: set[tuple[Any, ...]] = set()
        self._bad_sample_warning_keys_max = 4096
        self._runtime_bad_episodes: set[int] = set()
        self._frame_index_cache: OrderedDict[int, np.ndarray] = OrderedDict()
        self._motion_shard_by_episode: dict[int, dict[str, Any]] = {}
        if self.include_motion:
            self._init_motion_shard_index()
            self._filter_episodes_without_motion()
        # 数据已经由离线过滤脚本整理到 *_motion 目录；训练启动时不要再逐个
        # episode 读取 parquet frame_index、motion npz header 或视频 frame count。
        # 这些检查会在 900+ task 上非常耗时，真正读取失败时由 __getitem__
        # 的 warning + resample 兜底。

    def _build_motion_dir_candidates(self, motion_dir_name: str | None) -> list[str]:
        """Return the single configured motion directory name.

        When ``motion_dir_name`` is provided, motion loading is strict: only that
        directory is read. The old v2 defaults remain unchanged for configs that
        do not set it explicitly.
        """

        if motion_dir_name is not None and str(motion_dir_name).strip():
            return [str(motion_dir_name).strip()]
        if self.use_sam_scene_flow:
            return ["motions_sam"]
        return ["motions"]

    def _motion_dirs(self) -> list[Path]:
        return [self._root / name for name in self._motion_dir_candidates]

    def _init_motion_shard_index(self) -> None:
        """Load LeRobot v3 merged motion shard indices when present."""

        index_suffixes = (".motion_index.json", ".sam_motion_index.json")
        for motion_dir in self._motion_dirs():
            if not motion_dir.is_dir():
                continue
            index_paths: list[Path] = []
            for suffix in index_suffixes:
                index_paths.extend(sorted(motion_dir.glob(f"chunk-*/file-*{suffix}")))
            for index_path in sorted(set(index_paths)):
                npy_name = None
                for suffix in index_suffixes:
                    if index_path.name.endswith(suffix):
                        npy_name = index_path.name[: -len(suffix)] + ".npy"
                        break
                if npy_name is None:
                    continue
                npy_path = index_path.with_name(npy_name)
                if not npy_path.is_file():
                    logger.warning("Skipping motion index without npy: index=%s npy=%s", index_path, npy_path)
                    continue
                try:
                    with open(index_path) as f:
                        index = json.load(f)
                except Exception as exc:
                    logger.warning("Skipping unreadable motion index %s: %s", index_path, exc)
                    continue
                for ep in index.get("episodes") or []:
                    try:
                        ep_idx = int(ep["episode_index"])
                        merged_start = int(ep["merged_start"])
                        merged_end = int(ep["merged_end"])
                    except Exception:
                        continue
                    if merged_end <= merged_start:
                        continue
                    self._motion_shard_by_episode[ep_idx] = {
                        "path": npy_path,
                        "index_path": index_path,
                        "merged_start": merged_start,
                        "merged_end": merged_end,
                        "dir_name": motion_dir.name,
                    }

    def _has_motion_for_episode(self, episode_index: int) -> bool:
        episode_index = int(episode_index)
        if episode_index in self._motion_shard_by_episode:
            return True
        if self._motion_npy_path(episode_index).is_file():
            return True
        return (not self.use_sam_scene_flow) and self._motion_path(episode_index).is_file()

    def _filter_episodes_without_motion(self) -> None:
        if self._use_lazy_video_tree:
            kept = [
                (int(ep), int(length))
                for ep, length in zip(self._episodes, self._episode_lengths)
                if self._has_motion_for_episode(int(ep))
            ]
            missing = len(self._episodes) - len(kept)
            if missing:
                logger.warning(
                    "Filtering %s episodes without motion under %s/%s",
                    missing,
                    self._root,
                    self._motion_dir_name,
                )
            if not kept:
                raise FileNotFoundError(
                    f"No trainable motion episodes found under {self._root / self._motion_dir_name}"
                )
            self._episodes = [ep for ep, _ in kept]
            self._episode_lengths = [length for _, length in kept]
            self._episode_starts = [0]
            total = 0
            for length in self._episode_lengths:
                total += int(length)
                self._episode_starts.append(total)
            self._total_frames = int(total)
            return

        if getattr(self, "_use_v2_image_parquet", False):
            kept_meta = []
            kept_frames = []
            kept_paths = []
            for meta, frames, pq_path in zip(self._episodes_meta, self._ep_frames, self._ep_parquet_paths):
                ep_idx = int(meta.get("episode_index", len(kept_meta)))
                if self._has_motion_for_episode(ep_idx):
                    kept_meta.append(meta)
                    kept_frames.append(int(frames))
                    kept_paths.append(pq_path)
            missing = len(self._episodes_meta) - len(kept_meta)
            if missing:
                logger.warning(
                    "Filtering %s v2 episodes without motion under %s/%s",
                    missing,
                    self._root,
                    self._motion_dir_name,
                )
            if not kept_meta:
                raise FileNotFoundError(
                    f"No trainable motion episodes found under {self._root / self._motion_dir_name}"
                )
            self._episodes_meta = kept_meta
            self._ep_frames = kept_frames
            self._ep_parquet_paths = kept_paths
            self._cumulative = np.cumsum(self._ep_frames)
            self._total_frames = int(self._cumulative[-1])

    def _temporal_offsets_for_frame(
        self, frame_in_ep: int, episode_index: int, ep_len: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        video, state, action = super()._temporal_offsets_for_frame(
            frame_in_ep, episode_index, ep_len
        )
        if getattr(self, "sampling_mode", None) == "multi_anchor":
            anchor = int(frame_in_ep)
            return video - anchor, state - anchor, action - anchor
        return video, state, action

    def _warn_bad_episode_once(self, episode_index: int | None, exc: Exception) -> None:
        """同一个坏 episode 只打印一次 warning，避免日志刷屏。"""

        key = ("episode", episode_index, type(exc).__name__, str(exc)[:240])
        if key in self._bad_sample_warning_keys:
            return
        if len(self._bad_sample_warning_keys) >= self._bad_sample_warning_keys_max:
            self._bad_sample_warning_keys.clear()
        self._bad_sample_warning_keys.add(key)
        logger.warning("Skipping bad real-world episode: root=%s episode=%s reason=%s", self._root, episode_index, exc)

    def _warn_bad_sample_once(self, idx: int, exc: Exception) -> None:
        """同一种坏样本错误只打印一次 warning。"""

        episode_index = getattr(exc, "episode_index", None)
        if episode_index is None:
            try:
                _, episode_index, _ = self._resolve_index_context(int(idx))
            except Exception:
                episode_index = None
        key = ("sample", episode_index, type(exc).__name__, str(exc)[:240])
        if key in self._bad_sample_warning_keys:
            return
        if len(self._bad_sample_warning_keys) >= self._bad_sample_warning_keys_max:
            self._bad_sample_warning_keys.clear()
        self._bad_sample_warning_keys.add(key)
        logger.warning(
            "Skipping bad real-world sample and resampling: root=%s idx=%s episode=%s reason=%s",
            self._root, idx, episode_index, exc,
        )

    def _sample_replacement_index(self) -> int:
        """当前样本坏掉时，随机挑一个新的全局 idx 继续读取。"""

        if len(self) <= 0:
            raise RuntimeError(f"No valid samples left in real-world dataset {self._root}")
        for _ in range(32):
            candidate = random.randint(0, max(0, len(self) - 1))
            try:
                _, ep_idx, _ = self._resolve_index_context(candidate)
            except Exception:
                return candidate
            if int(ep_idx) not in self._runtime_bad_episodes:
                return candidate
        return random.randint(0, max(0, len(self) - 1))

    @staticmethod
    def _looks_like_video_decode_error(exc: Exception) -> bool:
        """粗略判断异常是否来自视频解码。"""

        text = str(exc).lower()
        return any(
            token in text
            for token in (
                "decode_video",
                "video",
                "torchcodec",
                "pyav",
                "ffmpeg",
                "averror",
                "cannot decode",
            )
        )

    def _get_episode_frame_indices(
        self,
        episode_index: int,
        expected_len: int,
        table: Any | None = None,
        parquet_path: Path | None = None,
    ) -> np.ndarray:
        """读取 action parquet 行号到真实视频帧号的映射。

        返回数组长度等于 action 行数。例子：
        - parquet/action 有 301 行；
        - ``frame_index = [2567, 2568, ..., 2867]``；
        - 训练采样 action 行 ``[0, 1, 2]`` 时，视频/motion 应该取真实帧
          ``[2567, 2568, 2569]``，而不是 ``[0, 1, 2]``。
        """

        episode_index = int(episode_index)
        expected_len = int(expected_len)
        if episode_index in self._frame_index_cache:
            frame_index = self._frame_index_cache.pop(episode_index)
            self._frame_index_cache[episode_index] = frame_index
            return frame_index

        if table is not None and hasattr(table, "column_names") and "frame_index" in table.column_names:
            frame_index = np.asarray(table.column("frame_index")).astype(np.int64)
        else:
            import pyarrow.parquet as pq

            path = parquet_path if parquet_path is not None else self._get_parquet_path(episode_index)
            schema = pq.read_schema(str(path))
            if "frame_index" in schema.names:
                frame_index = np.asarray(pq.read_table(str(path), columns=["frame_index"]).column("frame_index")).astype(np.int64)
            else:
                frame_index = np.arange(expected_len, dtype=np.int64)

        if frame_index.shape[0] != expected_len:
            raise RealWorldJointBadEpisodeError(
                f"frame_index rows={frame_index.shape[0]} != episode/action length {expected_len}",
                episode_index,
            )
        self._frame_index_cache[episode_index] = frame_index
        if len(self._frame_index_cache) > self._pq_cache_max_episodes:
            self._frame_index_cache.popitem(last=False)
        return frame_index

    def _validate_episode_for_training(self, episode_index: int, expected_len: int, *, check_video: bool) -> None:
        """训练读取样本前只跳过运行时已确认坏掉的 episode。

        离线过滤后的数据默认可信；这里不再读取 frame_index/motion header/
        video frame count，避免每个 rank 在训练启动或首次采样时重复做慢检查。
        """

        episode_index = int(episode_index)
        if episode_index in self._runtime_bad_episodes:
            raise RealWorldJointBadEpisodeError("episode was marked bad after a previous read/decode failure", episode_index)
        return

    @staticmethod
    def _to_channel_first_3ch(arr: np.ndarray, *, name: str, path: Path) -> np.ndarray:
        """把 motion 数组统一成 ``(T, 3, H, W)``。"""

        if arr.ndim != 4:
            raise ValueError(f"Expected 4-D {name}, got {arr.shape} from {path}")
        if arr.shape[1] == 3:
            return np.ascontiguousarray(arr)
        axes = [i for i in range(1, arr.ndim) if arr.shape[i] == 3]
        if len(axes) != 1:
            raise ValueError(f"Cannot locate 3-channel axis for {name} shape={arr.shape} path={path}")
        return np.ascontiguousarray(np.moveaxis(arr, axes[0], 1))

    @staticmethod
    def _broadcast_mask(
        target: np.ndarray,
        mask: np.ndarray,
        mask_key: str | None,
        visconf_threshold: float = 0.5,
    ) -> np.ndarray:
        """把 visconf 等 mask broadcast 到 scene_flow 的形状。"""

        if mask.shape[0] < target.shape[0]:
            pad = np.zeros((target.shape[0] - mask.shape[0], *mask.shape[1:]), dtype=mask.dtype)
            mask = np.concatenate([mask, pad], axis=0)
        elif mask.shape[0] > target.shape[0]:
            mask = mask[: target.shape[0]]
        channel_last = target.ndim == 4 and target.shape[-1] in (1, 2, 3)
        if mask.ndim == 1:
            mask = mask.reshape((mask.shape[0],) + (1,) * (target.ndim - 1))
        elif mask.ndim == 3:
            mask = mask[..., None] if channel_last else mask[:, None, :, :]
        elif mask.ndim == 4:
            mask_channel_last = mask.shape[-1] in (1, 2, 3)
            if channel_last and not mask_channel_last:
                mask = np.moveaxis(mask, 1, -1)
            elif (not channel_last) and mask_channel_last:
                mask = np.moveaxis(mask, -1, 1)
        if mask_key == "visconf":
            return (mask > float(visconf_threshold)).astype(target.dtype)
        return mask.astype(bool).astype(target.dtype)

    def _motion_episode_path_for_dir(self, episode_index: int, dir_name: str, suffix: str) -> Path:
        return self._root / dir_name / f"chunk-{episode_index // self._chunks_size:03d}" / f"episode_{episode_index:06d}{suffix}"

    def _motion_path(self, episode_index: int) -> Path:
        """Return the first existing per-episode motion NPZ path, or the default path."""

        episode_index = int(episode_index)
        default = self._motion_episode_path_for_dir(episode_index, self._motion_dir_name, ".npz")
        for dir_name in self._motion_dir_candidates:
            path = self._motion_episode_path_for_dir(episode_index, dir_name, ".npz")
            if path.is_file():
                return path
        return default

    def _motion_npy_path(self, episode_index: int) -> Path:
        """Return the first existing per-episode motion NPY path, or the default path."""

        episode_index = int(episode_index)
        default = self._motion_episode_path_for_dir(episode_index, self._motion_dir_name, ".npy")
        for dir_name in self._motion_dir_candidates:
            path = self._motion_episode_path_for_dir(episode_index, dir_name, ".npy")
            if path.is_file():
                return path
        return default

    def _load_motion_episode(self, episode_index: int) -> dict[str, np.ndarray]:
        """读取一个 episode 的 point_map/scene_flow，并做 cache。

        scene_flow 如果比 point_map 短，会在末尾补 0；如果有 visconf，则会把低置信度
        scene flow 置零。
        """

        episode_index = int(episode_index)
        if self._motion_cache_size > 0 and episode_index in self._motion_cache:
            item = self._motion_cache.pop(episode_index)
            self._motion_cache[episode_index] = item
            return item
        path = self._motion_path(episode_index)
        if not path.is_file():
            raise FileNotFoundError(f"Motion npz not found for episode {episode_index}: {path}")
        with np.load(path) as f:
            point_map = np.array(f["point_map"])
            if self.scene_flow_training:
                scene_flow = np.array(f["scene_flow"])
                visconf = np.array(f["visconf"]) if "visconf" in f.files else None
            else:
                scene_flow = None
                visconf = None
        point_map = self._to_channel_first_3ch(point_map, name="point_map", path=path)
        if self.scene_flow_training:
            scene_flow = self._to_channel_first_3ch(scene_flow, name="scene_flow", path=path)
            if scene_flow.shape[0] < point_map.shape[0]:
                pad = np.zeros((point_map.shape[0] - scene_flow.shape[0], *scene_flow.shape[1:]), dtype=scene_flow.dtype)
                scene_flow = np.concatenate([scene_flow, pad], axis=0)
            elif scene_flow.shape[0] > point_map.shape[0]:
                scene_flow = scene_flow[: point_map.shape[0]]
            if visconf is not None:
                scene_flow = scene_flow * self._broadcast_mask(
                    scene_flow,
                    visconf,
                    "visconf",
                    self.scene_flow_visconf_threshold,
                )
        else:
            scene_flow = np.zeros_like(point_map)
        item = {"point_map": point_map.astype(np.float32, copy=False), "scene_flow": scene_flow.astype(np.float32, copy=False)}
        if self._motion_cache_size > 0:
            self._motion_cache[episode_index] = item
            if len(self._motion_cache) > self._motion_cache_size:
                self._motion_cache.popitem(last=False)
        return item

    def _load_motion_frames_from_npy(self, episode_index: int, indices: np.ndarray, path: Path) -> tuple[np.ndarray, np.ndarray]:
        """从转换后的 mmap npy 里只读取本 sample 需要的 motion 帧。

        ``transform_npz2npy.py`` 写出的格式是 ``(T, H, W, 11)``：
        - ``0:3`` 是 point_map；
        - ``3:6`` 是已经补齐到 T 的 scene_flow；
        - ``6:7`` 是已经补齐到 T 的 visconf；
        - ``7:10`` 和 ``10:11`` 是 mu/S，目前训练输入不使用。

        ``use_sam_scene_flow`` 对应的 ``motions_sam`` 会在最后追加一个
        SAM 前景 mask channel，训练时只用它把背景 scene flow 置零。
        """

        frame_idx = np.asarray(indices, dtype=np.int64)
        arr = None
        try:
            arr = np.load(path, mmap_mode="r", allow_pickle=False)
            if arr.ndim != 4 or arr.shape[-1] < 7:
                raise ValueError(f"Expected motion npy shape (T,H,W,>=7), got {arr.shape} from {path}")
            if self.scene_flow_training and self.use_sam_scene_flow and arr.shape[-1] < 12:
                raise ValueError(
                    f"use_sam_scene_flow expects motions_sam npy shape (T,H,W,>=12) with the SAM mask "
                    f"in the last channel, got {arr.shape} from {path}"
                )
            if frame_idx.size == 0:
                h, w = int(arr.shape[1]), int(arr.shape[2])
                empty = np.zeros((0, 3, h, w), dtype=np.float32)
                return empty, empty.copy()
            if int(frame_idx.min()) < 0 or int(frame_idx.max()) >= int(arr.shape[0]):
                raise RealWorldJointBadEpisodeError(
                    f"motion frame index out of range for npy length={arr.shape[0]}: "
                    f"min={int(frame_idx.min())} max={int(frame_idx.max())} path={path}",
                    episode_index,
                )

            # Fancy indexing on mmap only materializes the selected frames, not the full episode.
            if self.scene_flow_training:
                selected = np.asarray(arr[frame_idx])
            else:
                selected = np.asarray(arr[frame_idx, ..., 0:3])
        finally:
            del arr
            if self._drop_motion_npy_file_cache:
                _drop_file_cache(path)
        if self.scene_flow_training:
            point_map = np.moveaxis(selected[..., 0:3], -1, 1)
            scene_flow = selected[..., 3:6]
            visconf = selected[..., 6:7]
            valid_scene_flow = visconf > self.scene_flow_visconf_threshold
            if self.use_sam_scene_flow:
                sam_mask = selected[..., -1:]
                valid_scene_flow = np.logical_and(valid_scene_flow, sam_mask == 1)
            scene_flow = scene_flow * valid_scene_flow.astype(scene_flow.dtype)
            scene_flow = np.moveaxis(scene_flow, -1, 1)
        else:
            point_map = np.moveaxis(selected, -1, 1)
            scene_flow = np.zeros_like(point_map)
        return (
            np.ascontiguousarray(point_map, dtype=np.float32),
            np.ascontiguousarray(scene_flow, dtype=np.float32),
        )

    def _load_motion_frames_from_v3_shard(self, episode_index: int, indices: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
        """Read episode-local motion indices from a merged LeRobot v3 motion shard."""

        entry = self._motion_shard_by_episode.get(int(episode_index))
        if entry is None:
            return None
        frame_idx = np.asarray(indices, dtype=np.int64)
        if frame_idx.size == 0:
            return self._load_motion_frames_from_npy(episode_index, frame_idx, entry["path"])
        local_len = int(entry["merged_end"]) - int(entry["merged_start"])
        if int(frame_idx.min()) < 0 or int(frame_idx.max()) >= local_len:
            raise RealWorldJointBadEpisodeError(
                f"motion episode-local index out of range for merged shard: len={local_len} "
                f"min={int(frame_idx.min())} max={int(frame_idx.max())} path={entry['path']}",
                int(episode_index),
            )
        merged_idx = frame_idx + int(entry["merged_start"])
        return self._load_motion_frames_from_npy(episode_index, merged_idx, entry["path"])

    def _load_motion_frames(self, episode_index: int, indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Read motion from the configured directory without directory fallback."""

        merged = self._load_motion_frames_from_v3_shard(episode_index, indices)
        if merged is not None:
            return merged
        npy_path = self._motion_npy_path(episode_index)
        if npy_path.is_file():
            return self._load_motion_frames_from_npy(episode_index, indices, npy_path)
        if not self.use_sam_scene_flow and self._motion_path(episode_index).is_file():
            motion = self._load_motion_episode(episode_index)
            return motion["point_map"][indices], motion["scene_flow"][indices]
        raise FileNotFoundError(
            f"Motion file not found for episode {episode_index} under configured dir "
            f"{self._root / self._motion_dir_name}; expected npy={npy_path}"
        )

    def _downsample_motion_indices(self, indices: np.ndarray) -> np.ndarray:
        """按 motion_downsample_ratio 对 action 对应的真实帧号下采样。

        例子：action horizon=48，ratio=6，则每个 action chunk 只取
        ``[0, 6, 12, ..., 42]`` 这些位置对应的 motion 帧。
        """

        ratio = self.motion_downsample_ratio
        if ratio <= 1 or len(indices) == 0:
            return indices
        chunk = int(self.action_horizon)
        parts = []
        for start in range(0, len(indices), chunk):
            parts.append(indices[start : start + chunk : ratio])
        return np.concatenate(parts) if parts else np.array([], dtype=np.int64)

    @staticmethod
    def _require_episode_local_indices(
        name: str,
        indices: np.ndarray,
        ep_len: int,
        *,
        episode_index: int,
        frame_in_ep: int,
    ) -> np.ndarray:
        indices = np.asarray(indices, dtype=np.int64)
        if indices.size == 0:
            raise EmptyTemporalSampleError(
                f"Empty {name} indices at frame {frame_in_ep} episode {episode_index}"
            )
        if int(indices.min()) < 0 or int(indices.max()) >= int(ep_len):
            raise EmptyTemporalSampleError(
                f"{name} indices out of episode bounds at frame {frame_in_ep} "
                f"episode {episode_index}: len={int(ep_len)} "
                f"min={int(indices.min())} max={int(indices.max())}"
            )
        return indices

    def _require_monotonic_motion_frames(
        self,
        motion_idx: np.ndarray,
        *,
        episode_index: int,
        frame_in_ep: int,
    ) -> np.ndarray:
        motion_idx = np.asarray(motion_idx, dtype=np.int64)
        if motion_idx.size == 0:
            raise EmptyTemporalSampleError(
                f"Empty motion frame indices at frame {frame_in_ep} episode {episode_index}"
            )
        if motion_idx.size > 1 and not np.all(np.diff(motion_idx) > 0):
            raise EmptyTemporalSampleError(
                f"Motion future horizon is not strictly increasing at frame {frame_in_ep} "
                f"episode {episode_index}: first={int(motion_idx[0])} "
                f"last={int(motion_idx[-1])} unique={int(np.unique(motion_idx).size)} "
                f"total={int(motion_idx.size)}"
            )
        return motion_idx

    def _materialize_parquet_sample(self, frame_in_ep: int, episode_index: int, ep_len: int, table: Any, *, decode_video: bool) -> dict[str, Any]:
        """从 lazy-video parquet 路径构造一个训练样本。

        motion 训练使用 action parquet 的 ``frame_index`` 对齐完整视频/motion；
        标准 LeRobot v3 baseline 没有 motion，按官方 v3 元数据使用
        ``videos/.../from_timestamp + frame_in_episode / fps`` 解码。

        motion 路径是 action/video/motion 对齐的核心：
        - ``frame_in_ep`` 是 action parquet 内部的行号；
        - ``frame_index[frame_in_ep]`` 是完整视频/motion 里的真实帧号；
        - video 用真实帧号解码；
        - action/state 仍然从 parquet 行号读取；
        - motion 用 action 行对应的真实帧号读取。
        """

        self._validate_episode_for_training(episode_index, ep_len, check_video=decode_video)
        video_offsets, state_offsets, action_offsets = self._temporal_offsets_for_frame(
            frame_in_ep, episode_index, ep_len
        )
        video_idx = np.asarray(frame_in_ep + video_offsets, dtype=np.int64)
        state_idx = np.asarray(frame_in_ep + state_offsets, dtype=np.int64)
        action_idx = np.asarray(frame_in_ep + action_offsets, dtype=np.int64)
        video_idx = self._require_episode_local_indices(
            "video", video_idx, ep_len, episode_index=episode_index, frame_in_ep=frame_in_ep
        )
        state_idx = self._require_episode_local_indices(
            "state", state_idx, ep_len, episode_index=episode_index, frame_in_ep=frame_in_ep
        )
        action_idx = self._require_episode_local_indices(
            "action", action_idx, ep_len, episode_index=episode_index, frame_in_ep=frame_in_ep
        )
        if self.include_motion:
            frame_index = self._get_episode_frame_indices(episode_index, ep_len, table)
            current_frame_index = int(frame_index[int(frame_in_ep)])
            video_frame_idx = frame_index[video_idx]
            action_frame_idx = frame_index[action_idx]
        else:
            current_frame_index = (
                int(table.column("frame_index")[int(frame_in_ep)].as_py())
                if self._col_exists(table, "frame_index")
                else int(frame_in_ep)
            )
            video_frame_idx = video_idx
            action_frame_idx = action_idx

        sample: dict[str, Any] = {
            "episode_index": episode_index,
            "frame_in_ep": int(frame_in_ep),
            "frame_index": current_frame_index,
            "video_row_indices": [int(i) for i in video_idx.tolist()],
            "state_row_indices": [int(i) for i in state_idx.tolist()],
            "action_row_indices": [int(i) for i in action_idx.tolist()],
            "video_frame_indices": [int(i) for i in video_frame_idx.tolist()],
            "action_frame_indices": [int(i) for i in action_frame_idx.tolist()],
        }
        if decode_video:
            try:
                from lerobot.datasets.video_utils import decode_video_frames
            except Exception as exc:
                raise RuntimeError(
                    "LeRobot's decode_video_frames is required for DreamZero video decoding. "
                    "Install the RLinf-compatible LeRobot package in the active environment."
                ) from exc

            for transform_key, source_key in self._source_video_key.items():
                video_path = self._get_video_path(episode_index, source_key)
                fps = self._decode_fps_for_video_file(video_path)
                start_ts = self._episode_video_start_ts.get((int(episode_index), source_key), 0.0)
                sample[transform_key] = decode_video_frames(
                    video_path,
                    [start_ts + float(int(i)) / fps for i in video_frame_idx.tolist()],
                    tolerance_s=self._video_tolerance_s,
                    backend=self._video_backend,
                )

        for key in ("task", "task_index"):
            if self._col_exists(table, key):
                sample[key] = table.column(key)[int(frame_in_ep)].as_py()
        for key, source in self._language_sources.items():
            if self._col_exists(table, source):
                sample[key] = table.column(source)[int(frame_in_ep)].as_py()

        for source, _ in self._state_components.values():
            if source in sample:
                continue
            if self._col_exists(table, source):
                sample[source] = self._read_list_column(table, source, state_idx)
            elif source == "observation.state" and self._col_exists(table, "observation"):
                sample[source] = self._read_struct_list_field(table, "observation", "state", state_idx)
            else:
                raise KeyError(f"episode parquet missing state source column {source!r}")
        for source, _ in self._action_components.values():
            if source in sample:
                continue
            if not self._col_exists(table, source):
                raise KeyError(f"episode parquet missing action source column {source!r}")
            sample[source] = self._read_list_column(table, source, action_idx)
        if not self.include_motion:
            return sample
        motion_idx = self._downsample_motion_indices(action_frame_idx)
        motion_idx = self._require_monotonic_motion_frames(
            motion_idx,
            episode_index=episode_index,
            frame_in_ep=frame_in_ep,
        )
        point_map, scene_flow = self._load_motion_frames(episode_index, motion_idx)
        sample["motion_frame_indices"] = [int(i) for i in motion_idx.tolist()]
        sample["motion.point_map"] = point_map
        sample["motion.scene_flow"] = scene_flow
        return sample

    def _get_v2_image_sample(self, idx: int) -> dict[str, Any]:
        """从 v2 parquet-image 路径读取样本，并给 motion 使用真实 frame_index。"""

        _, episode_index, ep_len = self._resolve_index_context(idx)
        self._validate_episode_for_training(episode_index, ep_len, check_video=False)
        sample = super()._get_v2_image_sample(idx)
        frame_in_ep = int(sample["frame_index"])
        sample["frame_in_ep"] = frame_in_ep
        if not self.include_motion:
            return sample
        episode_index = int(sample["episode_index"])
        ep_len = int(self._ep_frames[int(np.searchsorted(self._cumulative, idx, side="right"))])
        video_offsets, state_offsets, action_offsets = self._temporal_offsets_for_frame(frame_in_ep, episode_index, ep_len)
        video_idx = np.asarray(frame_in_ep + video_offsets, dtype=np.int64)
        state_idx = np.asarray(frame_in_ep + state_offsets, dtype=np.int64)
        action_idx = np.asarray(frame_in_ep + action_offsets, dtype=np.int64)
        frame_index = self._get_episode_frame_indices(episode_index, ep_len)
        video_frame_idx = frame_index[video_idx]
        action_frame_idx = frame_index[action_idx]
        motion_idx = self._downsample_motion_indices(frame_index[action_idx])
        motion_idx = self._require_monotonic_motion_frames(
            motion_idx,
            episode_index=episode_index,
            frame_in_ep=frame_in_ep,
        )
        point_map, scene_flow = self._load_motion_frames(episode_index, motion_idx)
        sample["frame_index"] = int(frame_index[frame_in_ep])
        sample["video_row_indices"] = [int(i) for i in video_idx.tolist()]
        sample["state_row_indices"] = [int(i) for i in state_idx.tolist()]
        sample["action_row_indices"] = [int(i) for i in action_idx.tolist()]
        sample["video_frame_indices"] = [int(i) for i in video_frame_idx.tolist()]
        sample["action_frame_indices"] = [int(i) for i in action_frame_idx.tolist()]
        sample["motion_frame_indices"] = [int(i) for i in motion_idx.tolist()]
        sample["motion.point_map"] = point_map
        sample["motion.scene_flow"] = scene_flow
        return sample

    @staticmethod
    def _json_scalar(value: Any) -> Any:
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, (list, tuple)):
            return [_RealWorldJointMotionMixin._json_scalar(v) for v in value]
        if isinstance(value, dict):
            return {str(k): _RealWorldJointMotionMixin._json_scalar(v) for k, v in value.items()}
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        try:
            return int(value)
        except (TypeError, ValueError):
            return str(value)

    def _sample_metadata(self, sample: dict[str, Any], idx: int) -> dict[str, Any]:
        metadata = {
            "dataset_root": str(self._root),
            "sample_index": int(idx),
            "episode_index": self._json_scalar(sample.get("episode_index")),
            "frame_in_ep": self._json_scalar(sample.get("frame_in_ep")),
            "frame_index": self._json_scalar(sample.get("frame_index")),
        }
        for key in (
            "task",
            "task_index",
            "annotation.task_index",
            "video_row_indices",
            "state_row_indices",
            "action_row_indices",
            "video_frame_indices",
            "action_frame_indices",
            "motion_frame_indices",
        ):
            if key in sample:
                metadata[key] = self._json_scalar(sample[key])
        return metadata

    def _build_modality_dict(self, sample: dict[str, Any]) -> dict[str, Any]:
        """在基础 DreamZero 字段上追加 motion 字段。"""

        out = super()._build_modality_dict(sample)
        if self.include_motion:
            out["motion.point_map"] = np.asarray(sample["motion.point_map"], dtype=np.float32)
            out["motion.scene_flow"] = np.asarray(sample["motion.scene_flow"], dtype=np.float32)
        return out

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """读取样本；遇到坏 episode 或视频 decode 错误时 warning 后重采样。"""

        last_error: Exception | None = None
        current_idx = int(idx)
        for _ in range(self._bad_sample_resample_attempts):
            try:
                raw_sample = self._load_raw_sample(current_idx)
                metadata = self._sample_metadata(raw_sample, current_idx)
                raw = self._build_modality_dict(raw_sample)
                transformed = self.data_transform(raw)
                item = collate_ready_sample(transformed)
                item["sample_metadata"] = metadata
                return item
            except Exception as exc:
                last_error = exc
                if isinstance(exc, EmptyTemporalSampleError) and getattr(self, "sampling_mode", None) == "multi_anchor":
                    current_idx = random.randint(0, max(0, len(self) - 1))
                    continue
                episode_index = getattr(exc, "episode_index", None)
                if episode_index is None:
                    try:
                        _, episode_index, _ = self._resolve_index_context(current_idx)
                    except Exception:
                        episode_index = None
                if isinstance(exc, RealWorldJointBadEpisodeError) or self._looks_like_video_decode_error(exc):
                    if episode_index is not None:
                        self._runtime_bad_episodes.add(int(episode_index))
                self._warn_bad_sample_once(current_idx, exc)
                current_idx = self._sample_replacement_index()
        raise RuntimeError(
            f"Failed to sample a valid real-world joint item after "
            f"{self._bad_sample_resample_attempts} attempts from {self._root}: {last_error}"
        ) from last_error


class RealWorldJointConcatDataset(Dataset):
    """把多个 real-world joint 任务拼成一个 Dataset。"""

    def __init__(self, datasets: list[Dataset], tags: list[str], paths: list[str]):
        if not datasets:
            raise ValueError("RealWorldJointConcatDataset requires at least one dataset")
        self.datasets = datasets
        self.tags = tags
        self.paths = paths
        self.cumulative = []
        total = 0
        for ds in datasets:
            total += len(ds)
            self.cumulative.append(total)

    def __len__(self) -> int:
        return int(self.cumulative[-1])

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """把全局 idx 映射到某个子数据集的局部 idx。"""

        ds_idx = bisect.bisect_right(self.cumulative, int(idx))
        prev = 0 if ds_idx == 0 else self.cumulative[ds_idx - 1]
        return self.datasets[ds_idx][int(idx) - prev]


def build_real_world_joint_sft_dataloader(cfg, world_size: int, rank: int, data_paths: str | list[str], eval_dataset: bool = False):
    """构建 real-world joint SFT dataloader。

    入口会先 discover 任务目录，再为每个任务构建单独 Dataset，最后 concat 后用
    DistributedSampler 切给每个 rank。
    """

    from rlinf.data.datasets.dreamzero.dreamzero import DreamZeroLeRobotDataset
    from torchdata.stateful_dataloader import StatefulDataLoader

    class RealWorldJointLeRobotDataset(_RealWorldJointMotionMixin, DreamZeroLeRobotDataset):
        pass

    data_cfg = cfg.data
    model_cfg = cfg.actor.model
    include_motion = bool(model_cfg.action_head_cfg.config.get("use_motion_modality", False))
    scene_flow_training = bool(data_cfg.get("scene_flow_training", True))
    use_sam_scene_flow = bool(data_cfg.get("use_sam_scene_flow", False))
    scene_flow_visconf_threshold = float(data_cfg.get("scene_flow_visconf_threshold", 0.5))
    root = data_paths[0] if isinstance(data_paths, (list, tuple)) else data_paths
    if data_cfg.get("real_world_joint_root", None):
        root = data_cfg.real_world_joint_root
    discovered = discover_real_world_joint_task_paths(root, data_cfg)
    explicit = data_cfg.get("real_world_joint_task_paths", None)
    if explicit:
        for tag, paths in explicit.items():
            if paths:
                discovered[str(tag)] = [str(p) for p in paths]
    discovered = _filter_discovered_real_world_tasks(discovered, data_cfg)
    if not discovered:
        raise RuntimeError(f"No real-world joint tasks discovered under {root}")

    tokenizer_path = model_cfg.get("tokenizer_path", "google/umt5-xxl")
    max_seq_len = int(model_cfg.get("max_seq_len", 512))
    max_state_dim = int(model_cfg.get("max_state_dim", 64))
    max_action_dim = int(model_cfg.get("max_action_dim", 32))
    embodiment_tag_mapping = _embodiment_tag_mapping(model_cfg)
    sampling_mode = data_cfg.get("sampling_mode", "multi_anchor")
    max_chunk_size = int(model_cfg.action_head_cfg.config.diffusion_model_cfg.max_chunk_size)
    num_frames = int(model_cfg.action_head_cfg.config.num_frames)
    state_horizon = int(model_cfg.get("state_horizon", 1))
    action_horizon = int(model_cfg.get("action_horizon", 48))
    macro_stride = data_cfg.get("macro_stride")
    if macro_stride is not None:
        macro_stride = int(macro_stride)
    video_in_chunk_offsets = data_cfg.get("video_in_chunk_offsets")
    if video_in_chunk_offsets is not None:
        video_in_chunk_offsets = tuple(int(x) for x in video_in_chunk_offsets)
    norm_stats_mode = str(data_cfg.get("real_world_joint_norm_stats_mode", "global_lerobot_q01_q99"))
    use_proprioception = bool(data_cfg.get("use_proprioception", True))

    logger.info(
        "Discovered real-world joint tasks: motion=%s scene_flow_training=%s use_sam_scene_flow=%s "
        "scene_flow_visconf_threshold=%s sampling_mode=%s max_chunk_size=%s action_horizon=%s macro_stride=%s "
        "video_in_chunk_offsets=%s norm_stats_mode=%s use_proprioception=%s total_tasks=%s tags=%s",
        include_motion,
        scene_flow_training,
        use_sam_scene_flow,
        scene_flow_visconf_threshold,
        sampling_mode,
        max_chunk_size,
        action_horizon,
        macro_stride if macro_stride is not None else action_horizon,
        video_in_chunk_offsets,
        norm_stats_mode,
        use_proprioception,
        sum(len(v) for v in discovered.values()),
        {tag: len(paths) for tag, paths in discovered.items() if paths},
    )

    datasets: list[Dataset] = []
    tags: list[str] = []
    paths: list[str] = []
    task_counter = 0
    total_tasks = sum(len(v) for v in discovered.values())
    for tag in REAL_WORLD_JOINT_TAGS:
        for path_str in discovered.get(tag, []):
            task_counter += 1
            task_dir = Path(path_str)
            if task_counter == 1 or task_counter % 25 == 0:
                logger.info(
                    "Building real-world joint task %s/%s: tag=%s path=%s",
                    task_counter,
                    total_tasks,
                    tag,
                    task_dir,
                )
            state_keys, action_keys = _state_action_keys_for_tag(tag)
            video_keys = _video_keys_for_tag(tag)
            transform = _make_transform(
                tokenizer_path=tokenizer_path,
                max_seq_len=max_seq_len,
                max_state_dim=max_state_dim,
                max_action_dim=max_action_dim,
                embodiment_tag_mapping=embodiment_tag_mapping,
                include_motion=include_motion,
                use_proprioception=use_proprioception,
                video_keys=video_keys,
                state_keys=state_keys,
                action_keys=action_keys,
            )
            transform.set_metadata(
                _metadata_for_task(
                    task_dir,
                    tag,
                    state_keys,
                    action_keys,
                    stats_root=Path(root),
                    norm_stats_mode=norm_stats_mode,
                )
            )
            if eval_dataset:
                transform.eval()
            else:
                transform.train()
            ds = RealWorldJointLeRobotDataset(
                data_path=str(task_dir),
                video_keys=video_keys,
                state_keys=state_keys,
                action_keys=action_keys,
                language_keys=LANGUAGE_KEYS,
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
                macro_stride=macro_stride,
                video_in_chunk_offsets=video_in_chunk_offsets,
                include_motion=include_motion,
                motion_downsample_ratio=int(data_cfg.get("motion_downsample_ratio", 6)),
                motion_cache_size=int(data_cfg.get("motion_cache_size", 0)),
                scene_flow_training=scene_flow_training,
                use_sam_scene_flow=use_sam_scene_flow,
                motion_dir_name=data_cfg.get("motion_dir_name", None),
                scene_flow_visconf_threshold=scene_flow_visconf_threshold,
                drop_motion_npy_file_cache=bool(data_cfg.get("drop_motion_npy_file_cache", True)),
                fast_local_index=bool(data_cfg.get("real_world_joint_fast_local_index", True)),
                bad_sample_resample_attempts=int(data_cfg.get("real_world_joint_bad_sample_resample_attempts", 64)),
            )
            if len(ds) == 0:
                logger.warning("Skipping empty real-world dataset after filtering: tag=%s path=%s", tag, task_dir)
                continue
            datasets.append(ds)
            tags.append(tag)
            paths.append(str(task_dir))

    if not datasets:
        raise RuntimeError(
            f"No real-world joint datasets with trainable samples discovered under {root}; "
            f"include_motion={include_motion} motion_dir_name={data_cfg.get('motion_dir_name', None)} "
            f"tags={list(discovered.keys())}"
        )

    dataset = RealWorldJointConcatDataset(datasets, tags, paths)
    logger.info(
        "Real-world joint RLinf mixture: motion=%s scene_flow_training=%s use_sam_scene_flow=%s tasks=%s samples=%s tags=%s",
        include_motion,
        scene_flow_training,
        use_sam_scene_flow,
        len(datasets),
        len(dataset),
        {tag: len(discovered.get(tag, [])) for tag in REAL_WORLD_JOINT_TAGS if discovered.get(tag)},
    )

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
        collate_fn=RealWorldJointCollator(
            tokenizer_path=tokenizer_path,
            max_seq_len=max_seq_len,
            embodiment_tag_mapping=embodiment_tag_mapping,
        ),
    )
    return loader, {"num_samples": len(dataset), "num_tasks": len(datasets)}
