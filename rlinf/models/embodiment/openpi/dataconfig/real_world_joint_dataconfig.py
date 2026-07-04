# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0

import dataclasses
import pathlib

import openpi.models.model as _model
import openpi.transforms as _transforms
from openpi.training.config import DataConfig, DataConfigFactory, ModelTransformFactory
from typing_extensions import override

from rlinf.models.embodiment.openpi.policies import real_world_joint_policy


@dataclasses.dataclass(frozen=True)
class LeRobotRealWorldJointDataConfig(DataConfigFactory):
    """Data config for OpenPI SFT on RLinf real_world_joint LeRobot v3 tasks."""

    default_prompt: str | None = None
    action_env_dim: int = 32

    @override
    def create(
        self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig
    ) -> DataConfig:
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/image": "image",
                        "observation/left_wrist_image": "left_wrist_image",
                        "observation/right_wrist_image": "right_wrist_image",
                        "observation/image_mask": "image_mask",
                        "observation/left_wrist_image_mask": "left_wrist_image_mask",
                        "observation/right_wrist_image_mask": "right_wrist_image_mask",
                        "observation/state": "state",
                        "actions": "actions",
                        "prompt": "prompt",
                    }
                )
            ]
        )
        data_transforms = _transforms.Group(
            inputs=[
                real_world_joint_policy.RealWorldJointInputs(
                    action_dim=model_config.action_dim,
                    model_type=model_config.model_type,
                )
            ],
            outputs=[
                real_world_joint_policy.RealWorldJointOutputs(
                    action_env_dim=self.action_env_dim
                )
            ],
        )
        model_transforms = ModelTransformFactory(default_prompt=self.default_prompt)(
            model_config
        )
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )
