# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0

from __future__ import annotations

from typing import Any
from pathlib import Path

import numpy as np
from groot.vla.data.dataset.lerobot import ModalityConfig
from groot.vla.data.transform.base import ComposedModalityTransform
from groot.vla.data.transform.concat import ConcatTransform
from groot.vla.data.transform.state_action import StateActionToTensor, StateActionTransform
from groot.vla.data.transform.video import RealWorldJointVideoResizeTile

from rlinf.data.datasets.dreamzero.data_transforms.base import RolloutObsLayout
from rlinf.data.datasets.dreamzero.data_transforms.dream_transform import DreamTransform
from rlinf.data.datasets.dreamzero.real_world_joint import DEFAULT_REAL_WORLD_JOINT_TAG_MAPPING

_VIDEO_KEYS = ["video.agent_view", "video.left_wrsit_view", "video.right_wrist_view"]
_STATE_KEYS = ["state.joint"]
_ACTION_KEYS = ["action.joint"]
_PROMPT_PREFIX = "A multi-view video shows that a robot "
_LAYOUT_TEXT = (
    " The video is split into three views: the top view is the agent view, "
    "the bottom-left view is the left wrist camera, and the bottom-right view is the right wrist camera. The robot "
)


def _ensure_legacy_tokenizer_path(tokenizer_path: str) -> None:
    # Upstream Groot DreamTransform currently constructs its tokenizer from the
    # historical relative path "checkpoints/umt5-xxl" inside __init__. Keep RLinf
    # runnable from its own repo root without modifying the source dreamzero tree.
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


class RealWorldJointDataTransform:
    TAG = "allenai_real_world_joint"
    DEFAULT_TAG_MAPPING = DEFAULT_REAL_WORLD_JOINT_TAG_MAPPING
    DEFAULT_ACTION_HORIZON = 48
    ROLLOUT_OBS_LAYOUT = RolloutObsLayout(
        video_fields=(
            ("main_images", "video.agent_view"),
            ("wrist_images", "video.left_wrsit_view"),
            ("extra_view_images", "video.right_wrist_view"),
        ),
        state_fields=(("states", "state.joint"),),
        fill_missing_video_keys=True,
    )

    @staticmethod
    def format_training_prompt(instruction: str) -> str:
        return _PROMPT_PREFIX + instruction + _LAYOUT_TEXT + instruction

    @staticmethod
    def concat_multiview_video(images: np.ndarray) -> np.ndarray:
        v, t, c, h, w = images.shape
        if v < 3:
            raise ValueError(f"real_world_joint expects 3 video views, got {images.shape}")
        out = np.zeros((1, t, c, 2 * h, 2 * w), dtype=images.dtype)
        out[0, :, :, :h, :] = np.repeat(images[0], 2, axis=-1)
        out[0, :, :, h:, :w] = images[1]
        out[0, :, :, h:, w:] = images[2]
        return out

    @staticmethod
    def get_modality_config() -> dict[str, ModalityConfig]:
        return {
            "video": ModalityConfig(delta_indices=list(range(49)), eval_delta_indices=[0], modality_keys=list(_VIDEO_KEYS)),
            "state": ModalityConfig(delta_indices=[0], modality_keys=list(_STATE_KEYS)),
            "action": ModalityConfig(delta_indices=list(range(48)), modality_keys=list(_ACTION_KEYS)),
            "language": ModalityConfig(delta_indices=[0], modality_keys=["annotation.task_index"]),
            "motion": ModalityConfig(delta_indices=list(range(48)), eval_delta_indices=[0], modality_keys=["motion.point_map", "motion.scene_flow"]),
        }

    @staticmethod
    def get_transform(*, tokenizer_path: str, cfg: Any, embodiment_tag_mapping: dict[str, int]) -> ComposedModalityTransform:
        _ensure_legacy_tokenizer_path(tokenizer_path)
        transforms: list[Any] = [
            RealWorldJointVideoResizeTile(
                apply_to=list(_VIDEO_KEYS),
                agent_height=128,
                agent_width=256,
                wrist_height=128,
                wrist_width=128,
                interpolation="linear",
            ),
            StateActionToTensor(apply_to=list(_STATE_KEYS)),
            StateActionTransform(apply_to=list(_STATE_KEYS), normalization_modes={"state.joint": "q99"}),
            StateActionToTensor(apply_to=list(_ACTION_KEYS)),
            StateActionTransform(apply_to=list(_ACTION_KEYS), normalization_modes={"action.joint": "q99"}),
            ConcatTransform(video_concat_order=[], state_concat_order=list(_STATE_KEYS), action_concat_order=list(_ACTION_KEYS)),
            DreamTransform(
                default_instruction=str(cfg.get("default_instruction", "Perform the default behavior.")),
                language_dropout_prob=float(cfg.get("language_dropout_prob", 0.0)),
                always_use_default_instruction=bool(cfg.get("always_use_default_instruction", False)),
                max_state_dim=int(cfg.get("max_state_dim", 64)),
                max_action_dim=int(cfg.get("max_action_dim", 32)),
                use_motion_modality=bool(cfg.get("action_head_cfg", {}).get("config", {}).get("use_motion_modality", False)),
                motion_representation="point_map",
                motion_horizon=8 if bool(cfg.get("action_head_cfg", {}).get("config", {}).get("use_motion_modality", False)) else 24,
                max_length=int(cfg.get("max_seq_len", 512)),
                state_horizon=int(cfg.get("state_horizon", 1)),
                action_horizon=int(cfg.get("action_horizon", 48)),
                tokenizer_path=tokenizer_path,
                embodiment_tag_mapping=dict(embodiment_tag_mapping),
            ),
        ]
        return ComposedModalityTransform(transforms=transforms)
