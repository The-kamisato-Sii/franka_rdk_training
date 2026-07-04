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

from typing import Any

import numpy as np
from groot.vla.data.dataset.lerobot import ModalityConfig
from groot.vla.data.transform.base import ComposedModalityTransform
from groot.vla.data.transform.concat import ConcatTransform
from groot.vla.data.transform.state_action import (
    StateActionToTensor,
    StateActionTransform,
)
from groot.vla.data.transform.video import (
    VideoColorJitter,
    VideoCrop,
    VideoResize,
    VideoToNumpy,
    VideoToTensor,
)

from rlinf.data.datasets.dreamzero.data_transforms.base import RolloutObsLayout
from rlinf.data.datasets.dreamzero.data_transforms.dream_transform import DreamTransform

VIDEO_KEYS = [
    "video.cam_high",
    "video.cam_left_wrist",
    "video.cam_right_wrist",
]
STATE_KEYS = [
    "state.left_arm",
    "state.left_gripper",
    "state.right_arm",
    "state.right_gripper",
]
ACTION_KEYS = [
    "action.left_arm",
    "action.left_gripper",
    "action.right_arm",
    "action.right_gripper",
]
MOTION_KEYS = ["motion.point_map", "motion.scene_flow"]
LANGUAGE_KEYS = ["annotation.language.action_text"]

_VIDEO_BACKEND = "torchvision"
_TRAINING_PROMPT_PREFIX = "A multi-view video shows that a dual-arm robot "
_MULTIVIEW_LAYOUT = (
    " The video is split into three views: The top view shows the overhead "
    "camera (cam_high), the bottom-left view shows the left wrist camera "
    "(cam_left_wrist), and the bottom-right view shows the right wrist "
    "camera (cam_right_wrist). The robot "
)


class RobotWin2DataTransform:
    """DreamZero RoboTwin2 transform matching the original Groot config."""

    TAG = "robotwin2"
    DEFAULT_TAG_MAPPING = {"robotwin2": 33}
    DEFAULT_ACTION_HORIZON = 48
    ROLLOUT_OBS_LAYOUT = RolloutObsLayout(
        video_fields=(
            ("cam_high", "video.cam_high"),
            ("cam_left_wrist", "video.cam_left_wrist"),
            ("cam_right_wrist", "video.cam_right_wrist"),
        ),
        state_fields=(
            ("left_arm", "state.left_arm"),
            ("left_gripper", "state.left_gripper"),
            ("right_arm", "state.right_arm"),
            ("right_gripper", "state.right_gripper"),
        ),
        binarize_gripper=False,
    )

    @staticmethod
    def format_training_prompt(instruction: str) -> str:
        text = str(instruction).lower()
        return _TRAINING_PROMPT_PREFIX + text + _MULTIVIEW_LAYOUT + text

    @staticmethod
    def concat_multiview_video(images: np.ndarray) -> np.ndarray:
        """RoboTwin2 layout: top head camera spans width, wrists on bottom row."""
        v, t, c, h, w = images.shape
        if v < 3:
            raise ValueError(
                f"robotwin2 expects 3 video views, got v={v} with shape {images.shape}"
            )
        head = images[0]
        left_wrist = images[1]
        right_wrist = images[2]
        concat_images = np.zeros((1, t, c, 2 * h, 2 * w), dtype=images.dtype)
        concat_images[0, :, :, :h, :] = np.repeat(head, 2, axis=-1)
        concat_images[0, :, :, h:, :w] = left_wrist
        concat_images[0, :, :, h:, w:] = right_wrist
        return concat_images

    @staticmethod
    def get_modality_config() -> dict[str, ModalityConfig]:
        return {
            "video": ModalityConfig(
                delta_indices=list(range(49)),
                eval_delta_indices=[0],
                modality_keys=list(VIDEO_KEYS),
            ),
            "state": ModalityConfig(
                delta_indices=[0],
                modality_keys=list(STATE_KEYS),
            ),
            "action": ModalityConfig(
                delta_indices=list(range(48)),
                modality_keys=list(ACTION_KEYS),
            ),
            "motion": ModalityConfig(
                delta_indices=list(range(48)),
                eval_delta_indices=[0],
                modality_keys=list(MOTION_KEYS),
            ),
            "language": ModalityConfig(
                delta_indices=[0],
                modality_keys=list(LANGUAGE_KEYS),
            ),
        }

    @staticmethod
    def get_transform(
        *,
        tokenizer_path: str,
        cfg: Any,
        embodiment_tag_mapping: dict[str, int],
    ) -> ComposedModalityTransform:
        include_motion = bool(
            cfg.action_head_cfg.config.get("use_motion_modality", False)
            if cfg.get("action_head_cfg", None) is not None
            else False
        )
        return RobotWin2DataTransform._build_composed_transform(
            tokenizer_path=tokenizer_path,
            state_horizon=int(cfg.get("state_horizon", 1)),
            action_horizon=int(
                cfg.get("action_horizon", RobotWin2DataTransform.DEFAULT_ACTION_HORIZON)
            ),
            max_state_dim=int(cfg.get("max_state_dim", 16)),
            max_action_dim=int(cfg.get("max_action_dim", 16)),
            max_length=int(cfg.get("max_seq_len", 512)),
            default_instruction=str(
                cfg.get("default_instruction", "Perform the default behavior.")
            ),
            language_dropout_prob=float(cfg.get("language_dropout_prob", 0.0)),
            always_use_default_instruction=bool(
                cfg.get("always_use_default_instruction", False)
            ),
            embodiment_tag_mapping=dict(embodiment_tag_mapping),
            include_motion=include_motion,
        )

    @staticmethod
    def _build_composed_transform(
        tokenizer_path: str,
        state_horizon: int,
        action_horizon: int,
        max_state_dim: int,
        max_action_dim: int,
        max_length: int,
        default_instruction: str,
        language_dropout_prob: float,
        always_use_default_instruction: bool,
        embodiment_tag_mapping: dict[str, int],
        include_motion: bool,
    ) -> ComposedModalityTransform:
        vk = list(VIDEO_KEYS)
        state_k = list(STATE_KEYS)
        action_k = list(ACTION_KEYS)
        transforms: list[Any] = [
            VideoToTensor(apply_to=vk, backend=_VIDEO_BACKEND),
            VideoCrop(apply_to=vk, backend=_VIDEO_BACKEND, scale=0.95),
            VideoResize(
                apply_to=vk,
                backend=_VIDEO_BACKEND,
                height=128,
                width=160,
                interpolation="linear",
            ),
            VideoColorJitter(
                apply_to=vk,
                backend=_VIDEO_BACKEND,
                brightness=0.3,
                contrast=0.4,
                saturation=0.5,
                hue=0.08,
            ),
            VideoToNumpy(apply_to=vk, backend=_VIDEO_BACKEND),
            StateActionToTensor(apply_to=state_k),
            StateActionTransform(
                apply_to=state_k,
                normalization_modes={
                    "state.left_arm": "min_max",
                    "state.left_gripper": "binary",
                    "state.right_arm": "min_max",
                    "state.right_gripper": "binary",
                },
            ),
            StateActionToTensor(apply_to=action_k),
            StateActionTransform(
                apply_to=action_k,
                normalization_modes={
                    "action.left_arm": "min_max",
                    "action.left_gripper": "binary",
                    "action.right_arm": "min_max",
                    "action.right_gripper": "binary",
                },
            ),
        ]
        if include_motion:
            transforms.append(StateActionToTensor(apply_to=list(MOTION_KEYS)))
        transforms.extend(
            [
                ConcatTransform(
                    apply_to=[],
                    video_concat_order=vk,
                    state_concat_order=state_k,
                    action_concat_order=action_k,
                ),
                DreamTransform(
                    default_instruction=default_instruction,
                    language_dropout_prob=language_dropout_prob,
                    always_use_default_instruction=always_use_default_instruction,
                    max_state_dim=max_state_dim,
                    max_action_dim=max_action_dim,
                    max_length=max_length,
                    state_horizon=state_horizon,
                    action_horizon=action_horizon,
                    tokenizer_path=tokenizer_path,
                    embodiment_tag_mapping=embodiment_tag_mapping,
                    num_views=3,
                ),
            ]
        )
        return ComposedModalityTransform(transforms=transforms)
