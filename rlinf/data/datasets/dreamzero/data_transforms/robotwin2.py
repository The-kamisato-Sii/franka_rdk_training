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
    RobotWinVideoResizeTile,
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
_LEGACY_MULTIVIEW_LAYOUT = "head_top_wrists_bottom"
_ASYMMETRIC_MULTIVIEW_LAYOUT = "head_left_wrists_right_vertical"
_ASYMMETRIC_COMPOSITE_KEY = "video.robotwin_composite"
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
    def get_modality_config(
        action_horizon: int = DEFAULT_ACTION_HORIZON,
        video_delta_indices: list[int] | tuple[int, ...] | None = None,
        motion_downsample_ratio: int = 1,
    ) -> dict[str, ModalityConfig]:
        """Build the modality contract without baking in a 48-step horizon.

        Registry callers use the legacy defaults. Export/debug callers can pass
        the resolved training schedule (including non-uniform video offsets).
        """
        action_horizon = int(action_horizon)
        motion_downsample_ratio = int(motion_downsample_ratio)
        if action_horizon <= 0 or motion_downsample_ratio <= 0:
            raise ValueError(
                "action_horizon and motion_downsample_ratio must be positive, "
                f"got {action_horizon}, {motion_downsample_ratio}"
            )
        if video_delta_indices is None:
            video_delta_indices = list(range(action_horizon + 1))
        else:
            video_delta_indices = [int(index) for index in video_delta_indices]
        return {
            "video": ModalityConfig(
                delta_indices=list(video_delta_indices),
                eval_delta_indices=[0],
                modality_keys=list(VIDEO_KEYS),
            ),
            "state": ModalityConfig(
                delta_indices=[0],
                modality_keys=list(STATE_KEYS),
            ),
            "action": ModalityConfig(
                delta_indices=list(range(action_horizon)),
                modality_keys=list(ACTION_KEYS),
            ),
            "motion": ModalityConfig(
                delta_indices=list(
                    range(0, action_horizon, motion_downsample_ratio)
                ),
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
        target_video_height = int(cfg.get("target_video_height", 256))
        target_video_width = int(cfg.get("target_video_width", 320))
        multiview_layout = str(
            cfg.get("robotwin_multiview_layout", _LEGACY_MULTIVIEW_LAYOUT)
        )
        if multiview_layout == _LEGACY_MULTIVIEW_LAYOUT:
            if target_video_height % 2 != 0 or target_video_width % 2 != 0:
                raise ValueError(
                    "RoboTwin2's legacy 2x2 layout requires even target video "
                    f"dimensions, got {target_video_height}x{target_video_width}."
                )
            head_view_height = target_video_height // 2
            head_view_width = target_video_width // 2
            wrist_view_height = target_video_height // 2
            wrist_view_width = target_video_width // 2
        elif multiview_layout == _ASYMMETRIC_MULTIVIEW_LAYOUT:
            head_view_height = int(cfg.get("robotwin_head_view_height", 480))
            head_view_width = int(cfg.get("robotwin_head_view_width", 512))
            wrist_view_height = int(cfg.get("robotwin_wrist_view_height", 240))
            wrist_view_width = int(cfg.get("robotwin_wrist_view_width", 320))
            expected_height = head_view_height
            expected_width = head_view_width + wrist_view_width
            if head_view_height != 2 * wrist_view_height:
                raise ValueError(
                    "RoboTwin asymmetric layout requires head height to equal "
                    "two wrist heights; got "
                    f"head={head_view_height}, wrist={wrist_view_height}."
                )
            if (target_video_height, target_video_width) != (
                expected_height,
                expected_width,
            ):
                raise ValueError(
                    "RoboTwin asymmetric view sizes do not match the configured "
                    "composite: target="
                    f"{target_video_height}x{target_video_width}, expected="
                    f"{expected_height}x{expected_width}."
                )
        else:
            raise ValueError(
                "Unsupported robotwin_multiview_layout "
                f"{multiview_layout!r}; expected {_LEGACY_MULTIVIEW_LAYOUT!r} "
                f"or {_ASYMMETRIC_MULTIVIEW_LAYOUT!r}."
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
            multiview_layout=multiview_layout,
            head_view_height=head_view_height,
            head_view_width=head_view_width,
            wrist_view_height=wrist_view_height,
            wrist_view_width=wrist_view_width,
            arm_normalization_mode=str(
                cfg.get("arm_normalization_mode", "min_max")
            ),
            gripper_normalization_mode=str(
                cfg.get("gripper_normalization_mode", "binary")
            ),
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
        multiview_layout: str,
        head_view_height: int,
        head_view_width: int,
        wrist_view_height: int,
        wrist_view_width: int,
        arm_normalization_mode: str = "min_max",
        gripper_normalization_mode: str = "binary",
    ) -> ComposedModalityTransform:
        vk = list(VIDEO_KEYS)
        state_k = list(STATE_KEYS)
        action_k = list(ACTION_KEYS)
        transforms: list[Any] = [
            VideoToTensor(apply_to=vk, backend=_VIDEO_BACKEND),
            VideoCrop(apply_to=vk, backend=_VIDEO_BACKEND, scale=0.95),
        ]
        if multiview_layout == _ASYMMETRIC_MULTIVIEW_LAYOUT:
            transforms.append(
                RobotWinVideoResizeTile(
                    apply_to=vk,
                    output_key=_ASYMMETRIC_COMPOSITE_KEY,
                    head_height=head_view_height,
                    head_width=head_view_width,
                    wrist_height=wrist_view_height,
                    wrist_width=wrist_view_width,
                    interpolation="linear",
                )
            )
            transformed_video_keys = [_ASYMMETRIC_COMPOSITE_KEY]
        else:
            transforms.append(
                VideoResize(
                    apply_to=vk,
                    backend=_VIDEO_BACKEND,
                    height=head_view_height,
                    width=head_view_width,
                    interpolation="linear",
                )
            )
            transformed_video_keys = vk
        transforms.extend(
            [
            VideoColorJitter(
                apply_to=transformed_video_keys,
                backend=_VIDEO_BACKEND,
                brightness=0.3,
                contrast=0.4,
                saturation=0.5,
                hue=0.08,
            ),
            VideoToNumpy(
                apply_to=transformed_video_keys,
                backend=_VIDEO_BACKEND,
            ),
            StateActionToTensor(apply_to=state_k),
            StateActionTransform(
                apply_to=state_k,
                normalization_modes={
                    "state.left_arm": arm_normalization_mode,
                    "state.left_gripper": gripper_normalization_mode,
                    "state.right_arm": arm_normalization_mode,
                    "state.right_gripper": gripper_normalization_mode,
                },
            ),
            StateActionToTensor(apply_to=action_k),
            StateActionTransform(
                apply_to=action_k,
                normalization_modes={
                    "action.left_arm": arm_normalization_mode,
                    "action.left_gripper": gripper_normalization_mode,
                    "action.right_arm": arm_normalization_mode,
                    "action.right_gripper": gripper_normalization_mode,
                },
            ),
            ]
        )
        if include_motion:
            transforms.append(StateActionToTensor(apply_to=list(MOTION_KEYS)))
        transforms.extend(
            [
                ConcatTransform(
                    apply_to=[],
                    video_concat_order=transformed_video_keys,
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
                    num_views=len(transformed_video_keys),
                ),
            ]
        )
        return ComposedModalityTransform(transforms=transforms)


class RobotWinDataTransform(RobotWin2DataTransform):
    """Native RoboTwin absolute-EEF transform.

    It shares the three-view/state layout with the legacy ``robotwin2`` QPOS
    dataset while selecting DreamZero's native RoboTwin embodiment projector.
    """

    TAG = "robotwin"
    DEFAULT_TAG_MAPPING = {"robotwin": 0}
