# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0

import dataclasses
from typing import Any

import einops
import numpy as np
import torch
from openpi import transforms
from openpi.models import model as _model


def _parse_image(image: Any) -> np.ndarray:
    image = np.asarray(image)
    image = np.squeeze(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.ndim == 3 and image.shape[0] in (1, 3, 4) and image.shape[-1] not in (1, 3, 4):
        image = einops.rearrange(image, "c h w -> h w c")
    return np.ascontiguousarray(image[..., :3])


@dataclasses.dataclass(frozen=True)
class RealWorldJointOutputs(transforms.DataTransformFn):
    action_env_dim: int = 32

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][:, : self.action_env_dim])}


@dataclasses.dataclass(frozen=True)
class RealWorldJointInputs(transforms.DataTransformFn):
    action_dim: int
    model_type: _model.ModelType = _model.ModelType.PI0

    @staticmethod
    def _mask(data: dict, key: str, default: bool = True) -> np.bool_:
        value = data.get(key, default)
        if torch.is_tensor(value):
            value = value.detach().cpu().numpy()
        return np.bool_(np.asarray(value).item())

    def __call__(self, data: dict) -> dict:
        state = np.asarray(data["observation/state"], dtype=np.float32)

        base_image = _parse_image(data["observation/image"])
        left_wrist_image = _parse_image(
            data.get("observation/left_wrist_image", np.zeros_like(base_image))
        )
        right_wrist_image = _parse_image(
            data.get("observation/right_wrist_image", np.zeros_like(base_image))
        )

        if self.model_type in (_model.ModelType.PI0, _model.ModelType.PI05):
            names = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
            image_masks = (
                self._mask(data, "observation/image_mask"),
                self._mask(data, "observation/left_wrist_image_mask"),
                self._mask(data, "observation/right_wrist_image_mask"),
            )
        elif self.model_type == _model.ModelType.PI0_FAST:
            names = ("base_0_rgb", "base_1_rgb", "wrist_0_rgb")
            image_masks = (
                self._mask(data, "observation/image_mask"),
                self._mask(data, "observation/left_wrist_image_mask"),
                self._mask(data, "observation/right_wrist_image_mask"),
            )
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

        inputs = {
            "state": state,
            "image": dict(zip(names, (base_image, left_wrist_image, right_wrist_image), strict=True)),
            "image_mask": dict(zip(names, image_masks, strict=True)),
        }

        if "actions" in data:
            actions = np.asarray(data["actions"], dtype=np.float32)
            if actions.ndim != 2:
                raise ValueError(f"Expected actions shape (T, D), got {actions.shape}")
            inputs["actions"] = actions

        if "prompt" in data:
            prompt = data["prompt"]
            if isinstance(prompt, bytes):
                prompt = prompt.decode("utf-8")
            inputs["prompt"] = prompt

        return inputs
