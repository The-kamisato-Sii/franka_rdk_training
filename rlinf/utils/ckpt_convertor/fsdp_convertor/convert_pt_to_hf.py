# Copyright 2025 The RLinf Authors.
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

"""Convert FSDP PT checkpoint to HuggingFace safetensors format.

Usage with an SFT training config:
    python -m rlinf.utils.ckpt_convertor.fsdp_convertor.convert_pt_to_hf \
        convertor.train_config_path=/path/to/franka_sft_dreamzero.yaml \
        convertor.ckpt_path=/path/to/model.pt \
        convertor.save_path=/path/to/hf_model

Usage with RLinf training ``config.yaml`` (auto-discovered near ``ckpt_path``):
    python -m rlinf.utils.ckpt_convertor.fsdp_convertor.convert_pt_to_hf \
        --config-name fsdp_dreamzero_convertor \
        convertor.train_config_path=/path/to/config.yaml \
        convertor.ckpt_path=/path/to/model.pt \
        convertor.save_path=/path/to/hf_model \
        convertor.torch_dtype=bf16

Usage with an explicit convertor config:
    python -m rlinf.utils.ckpt_convertor.fsdp_convertor.convert_pt_to_hf \
        --config-path /path/to/config \
        --config-name fsdp_model_convertor \
        convertor.ckpt_path=/path/to/model.pt \
        convertor.save_path=/path/to/hf_model

If the checkpoint uses a custom ``model_type`` registered only through
``RLINF_EXT_MODULE``, export that variable the same way as for Ray workers so
``get_model`` can resolve the builder.
"""

import json
import os
from collections.abc import Mapping
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

from rlinf.config import SupportedModel
from rlinf.models import get_model
from rlinf.scheduler.cluster import load_user_extension_module

from .utils import (
    copy_model_config_and_code,
    get_model_save_helper,
    resolve_save_torch_dtype,
    save_state_dict_sharded_safetensors,
    torch_dtype_to_hf_str,
)


def _has_actor_model_config(config_path: str) -> bool:
    try:
        loaded = OmegaConf.load(config_path)
    except Exception:
        return False
    return "actor" in loaded and "model" in loaded.actor


def _find_rlinf_train_config_near_ckpt(ckpt_path: str) -> str | None:
    """Locate RLinf ``config.yaml`` saved under ``{log_path}/tensorboard/``."""
    ckpt_dir = Path(os.path.abspath(ckpt_path)).parent
    search_dirs = [ckpt_dir, *ckpt_dir.parents]
    for directory in search_dirs[:10]:
        candidates = [
            directory / "config.yaml",
            directory / "tensorboard" / "config.yaml",
        ]
        for candidate in candidates:
            if candidate.is_file() and _has_actor_model_config(str(candidate)):
                return str(candidate)
    return None


def _resolve_train_config_path(cfg: DictConfig) -> str | None:
    if not cfg.convertor.get("auto_find_train_config", True):
        return None

    ckpt_path = cfg.convertor.get("ckpt_path", None)
    if not ckpt_path:
        return None

    found = _find_rlinf_train_config_near_ckpt(ckpt_path)
    if found:
        print(f"Auto-discovered RLinf training config: {found}")
    return found


def _model_cfg_from_train_file(train_config_path: str, cfg: DictConfig) -> DictConfig:
    train_cfg = OmegaConf.load(train_config_path)
    if "actor" not in train_cfg or "model" not in train_cfg.actor:
        raise KeyError(
            f"Could not find actor.model in training config: {train_config_path}"
        )

    model_cfg = OmegaConf.create(OmegaConf.to_container(train_cfg.actor.model))
    model_overrides = cfg.convertor.get("model_overrides", None)
    if model_overrides:
        model_cfg = OmegaConf.merge(model_cfg, model_overrides)
    return model_cfg


def _resolve_model_cfg(cfg: DictConfig) -> DictConfig:
    """Return the model config used to build the checkpointed actor."""
    train_config_path = cfg.convertor.get("train_config_path", None)
    if train_config_path:
        return _model_cfg_from_train_file(train_config_path, cfg)

    if "actor" in cfg and "model" in cfg.actor:
        return cfg.actor.model
    if "model" in cfg:
        return cfg.model

    auto_found = _resolve_train_config_path(cfg)
    if auto_found:
        return _model_cfg_from_train_file(auto_found, cfg)

    raise KeyError(
        "No model config found. Provide either cfg.model, cfg.actor.model, "
        "convertor.train_config_path, or place RLinf config.yaml near ckpt_path."
    )


def _prepare_model_cfg(model_cfg: DictConfig) -> DictConfig:
    """Apply model-type validation (e.g. DreamZero cold start without config.json)."""
    model_type = model_cfg.get("model_type", None)
    if model_type is None:
        return model_cfg

    if SupportedModel(model_type) == SupportedModel.DREAMZERO:
        from rlinf.models.embodiment.dreamzero.dreamzero_config import (
            validate_dreamzero_sft_model_cfg,
        )

        return validate_dreamzero_sft_model_cfg(model_cfg)
    return model_cfg


def _effective_train_config_path(cfg: DictConfig) -> str | None:
    train_config_path = cfg.convertor.get("train_config_path", None)
    if train_config_path:
        return str(train_config_path)
    return _resolve_train_config_path(cfg)


def _pad_stat_values(
    values: object,
    *,
    dim: int,
    fill_value: float,
) -> list[float]:
    arr = list(values or [])
    out = [float(fill_value)] * int(dim)
    for idx, value in enumerate(arr[: int(dim)]):
        out[idx] = float(value)
    return out


def _padded_joint_stats(stats: Mapping, *, dim: int) -> dict[str, list[float]]:
    """Convert LeRobot q01/q99 stats into Groot DatasetStatistics format."""

    defaults = {
        "mean": 0.0,
        "std": 1.0,
        "min": 0.0,
        "max": 0.0,
        "q01": -1.0,
        "q99": 1.0,
    }
    return {
        key: _pad_stat_values(stats.get(key, []), dim=dim, fill_value=fill)
        for key, fill in defaults.items()
    }


def _norm_stats_path_from_train_cfg(train_cfg: DictConfig) -> Path | None:
    data_cfg = train_cfg.get("data", None)
    if data_cfg is None:
        return None

    root = data_cfg.get("real_world_joint_root", None) or data_cfg.get(
        "train_data_paths", None
    )
    if isinstance(root, (list, tuple)):
        root = root[0] if root else None
    if root:
        candidate = Path(str(root)) / "norm_stats.json"
        if candidate.is_file():
            return candidate
    return None


def _metadata_from_norm_stats(
    norm_stats_path: Path,
    *,
    state_dim: int,
    action_dim: int,
    embodiment_tag: str,
) -> dict[str, object]:
    raw = json.loads(norm_stats_path.read_text(encoding="utf-8"))
    norm_stats = raw.get("norm_stats", raw)
    state_stats = norm_stats.get("state")
    action_stats = norm_stats.get("action", norm_stats.get("actions"))
    if not isinstance(state_stats, Mapping) or not isinstance(action_stats, Mapping):
        raise ValueError(
            f"{norm_stats_path} must contain norm_stats.state and norm_stats.actions/action"
        )

    return {
        embodiment_tag: {
            "statistics": {
                "state": {
                    "joint": _padded_joint_stats(state_stats, dim=state_dim),
                },
                "action": {
                    "joint": _padded_joint_stats(action_stats, dim=action_dim),
                },
            },
            "modalities": {
                "video": {
                    "agent_view": {"resolution": [256, 128], "channels": 3, "fps": 30},
                    "left_wrsit_view": {
                        "resolution": [128, 128],
                        "channels": 3,
                        "fps": 30,
                    },
                    "right_wrist_view": {
                        "resolution": [128, 128],
                        "channels": 3,
                        "fps": 30,
                    },
                },
                "state": {
                    "joint": {
                        "absolute": True,
                        "shape": [state_dim],
                        "continuous": True,
                    },
                },
                "action": {
                    "joint": {
                        "absolute": True,
                        "shape": [action_dim],
                        "continuous": True,
                    },
                },
            },
            "embodiment_tag": embodiment_tag,
        }
    }


def _dreamzero_embodiment_mapping(model_cfg: DictConfig) -> dict[str, int]:
    try:
        mapping = model_cfg.action_head_cfg.config.get("embodiment_tag_mapping", None)
    except Exception:
        mapping = None
    if mapping:
        return {str(k): int(v) for k, v in OmegaConf.to_container(mapping).items()}
    return {
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


def _build_dreamzero_groot_conf(model_cfg: DictConfig) -> DictConfig:
    embodiment_tag = str(model_cfg.get("embodiment_tag", "real_world_franka_dual"))
    tokenizer_path = str(model_cfg.get("tokenizer_path", "checkpoints/umt5-xxl"))
    max_state_dim = int(model_cfg.get("max_state_dim", 64))
    max_action_dim = int(model_cfg.get("max_action_dim", 32))
    state_horizon = int(model_cfg.get("state_horizon", 1))
    action_horizon = int(model_cfg.get("action_horizon", 48))
    max_seq_len = int(model_cfg.get("max_seq_len", 512))
    use_motion = bool(
        model_cfg.get("action_head_cfg", {})
        .get("config", {})
        .get("use_motion_modality", False)
    )

    cfg = {
        "model": {"_target_": "groot.vla.model.dreamzero.base_vla.VLA"},
        "eval_bf16": False,
        "relative_action": bool(model_cfg.get("relative_action", False)),
        "relative_action_per_horizon": bool(
            model_cfg.get("relative_action_per_horizon", False)
        ),
        "transforms": {
            embodiment_tag: {
                "_target_": "groot.vla.data.transform.ComposedModalityTransform",
                "transforms": [
                    {
                        "_target_": "groot.vla.data.transform.video.RealWorldJointVideoResizeTile",
                        "apply_to": [
                            "video.agent_view",
                            "video.left_wrsit_view",
                            "video.right_wrist_view",
                        ],
                        "agent_height": 128,
                        "agent_width": 256,
                        "wrist_height": 128,
                        "wrist_width": 128,
                        "interpolation": "linear",
                    },
                    {
                        "_target_": "groot.vla.data.transform.StateActionToTensor",
                        "apply_to": ["state.joint"],
                    },
                    {
                        "_target_": "groot.vla.data.transform.StateActionTransform",
                        "apply_to": ["state.joint"],
                        "normalization_modes": {"state.joint": "q99"},
                    },
                    {
                        "_target_": "groot.vla.data.transform.StateActionToTensor",
                        "apply_to": ["action.joint"],
                    },
                    {
                        "_target_": "groot.vla.data.transform.StateActionTransform",
                        "apply_to": ["action.joint"],
                        "normalization_modes": {"action.joint": "q99"},
                    },
                    {
                        "_target_": "groot.vla.data.transform.ConcatTransform",
                        "video_concat_order": [],
                        "state_concat_order": ["state.joint"],
                        "action_concat_order": ["action.joint"],
                    },
                    {
                        "_target_": "rlinf.data.datasets.dreamzero.data_transforms.dream_transform.DreamTransform",
                        "default_instruction": "Perform the default behavior.",
                        "language_dropout_prob": 0.0,
                        "always_use_default_instruction": False,
                        "max_state_dim": max_state_dim,
                        "max_action_dim": max_action_dim,
                        "use_motion_modality": use_motion,
                        "motion_representation": "point_map",
                        "motion_horizon": 8 if use_motion else 24,
                        "max_length": max_seq_len,
                        "state_horizon": state_horizon,
                        "action_horizon": action_horizon,
                        "tokenizer_path": tokenizer_path,
                        "embodiment_tag_mapping": _dreamzero_embodiment_mapping(
                            model_cfg
                        ),
                    },
                ],
            }
        },
        "modality_configs": {
            embodiment_tag: {
                "video": {
                    "_target_": "groot.vla.data.dataset.ModalityConfig",
                    "delta_indices": list(range(49)),
                    "modality_keys": [
                        "video.agent_view",
                        "video.left_wrsit_view",
                        "video.right_wrist_view",
                    ],
                    "eval_delta_indices": [0],
                },
                "state": {
                    "_target_": "groot.vla.data.dataset.ModalityConfig",
                    "delta_indices": [0],
                    "modality_keys": ["state.joint"],
                    "eval_delta_indices": [0],
                },
                "action": {
                    "_target_": "groot.vla.data.dataset.ModalityConfig",
                    "delta_indices": list(range(action_horizon)),
                    "modality_keys": ["action.joint"],
                },
                "language": {
                    "_target_": "groot.vla.data.dataset.ModalityConfig",
                    "delta_indices": [0],
                    "modality_keys": ["annotation.task_index"],
                },
                "motion": {
                    "_target_": "groot.vla.data.dataset.ModalityConfig",
                    "delta_indices": list(range(action_horizon)),
                    "eval_delta_indices": [0],
                    "modality_keys": ["motion.point_map", "motion.scene_flow"],
                },
            }
        },
    }
    return OmegaConf.create(cfg)


def _write_dreamzero_experiment_cfg(
    *,
    model_cfg: DictConfig,
    cfg: DictConfig,
    save_path: str,
    train_config_path: str | None,
) -> None:
    if SupportedModel(model_cfg.model_type) != SupportedModel.DREAMZERO:
        return
    if not train_config_path:
        print(
            "DreamZero experiment_cfg was not written because no RLinf training "
            "config was provided or auto-discovered."
        )
        return

    train_cfg = OmegaConf.load(train_config_path)
    exp_cfg_dir = Path(save_path) / "experiment_cfg"
    exp_cfg_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(
        config=_build_dreamzero_groot_conf(model_cfg),
        f=exp_cfg_dir / "conf.yaml",
        resolve=True,
    )
    OmegaConf.save(
        config=train_cfg,
        f=exp_cfg_dir / "rlinf_train_config.yaml",
        resolve=True,
    )

    norm_stats_path = _norm_stats_path_from_train_cfg(train_cfg)
    if norm_stats_path is None:
        raise FileNotFoundError(
            "Could not locate data.real_world_joint_root/norm_stats.json from "
            f"training config: {train_config_path}"
        )

    metadata = _metadata_from_norm_stats(
        norm_stats_path,
        state_dim=int(model_cfg.get("max_state_dim", 64)),
        action_dim=int(model_cfg.get("max_action_dim", 32)),
        embodiment_tag=str(model_cfg.get("embodiment_tag", "real_world_franka_dual")),
    )
    with (exp_cfg_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)

    print(
        "Saved DreamZero experiment_cfg to "
        f"{exp_cfg_dir} using norm stats {norm_stats_path}"
    )


def _extract_state_dict(checkpoint) -> Mapping:
    """Accept direct state_dict files and common nested checkpoint layouts."""
    if not isinstance(checkpoint, Mapping):
        raise TypeError(
            f"Expected a mapping checkpoint/state_dict, got {type(checkpoint).__name__}"
        )

    candidates = [
        ("fsdp_checkpoint", "model"),
        ("state_dict",),
        ("model_state_dict",),
        ("model",),
        ("module",),
    ]
    for path in candidates:
        current = checkpoint
        for key in path:
            if not isinstance(current, Mapping) or key not in current:
                break
            current = current[key]
        else:
            if isinstance(current, Mapping) and any(
                torch.is_tensor(v) for v in current.values()
            ):
                return current

    if any(torch.is_tensor(v) for v in checkpoint.values()):
        return checkpoint

    raise KeyError(
        "Could not locate tensor weights in checkpoint. Expected a direct state_dict "
        "or one of: fsdp_checkpoint.model, state_dict, model_state_dict, model, module."
    )


def _normalize_state_dict_keys(state_dict: Mapping) -> dict:
    normalized = {}
    for key, value in state_dict.items():
        if not torch.is_tensor(value):
            continue

        name = key
        for prefix in ("_orig_mod.", "module."):
            if name.startswith(prefix):
                name = name[len(prefix) :]
        name = name.replace(".base_layer.", ".")
        normalized[name] = value
    return normalized


def _save_hf_checkpoint(
    model,
    model_cfg: DictConfig,
    cfg: DictConfig,
    save_path: str,
) -> None:
    save_dtype = resolve_save_torch_dtype(cfg.convertor.get("torch_dtype", None))
    if save_dtype is not None:
        print(f"Saving weights with torch_dtype={save_dtype}")
        model = model.to(dtype=save_dtype)

    model_save_helper_func = get_model_save_helper(model_cfg.model_type)
    helper_kwargs: dict = {"model": model}
    if save_dtype is not None:
        helper_kwargs["save_torch_dtype"] = save_dtype
        helper_kwargs["save_torch_dtype_str"] = torch_dtype_to_hf_str(save_dtype)

    if model_cfg.get("is_lora", False):
        if cfg.convertor.merge_lora_weighs:
            copy_model_config_and_code(
                model_path=model_cfg.model_path, save_path=save_path
            )
            model = model.merge_and_unload()
            if save_dtype is not None:
                model = model.to(dtype=save_dtype)
            model.save_pretrained(save_path, safe_serialization=True)

            model_state_dict = model.state_dict()
            if model_save_helper_func is not None:
                model_save_helper_func(
                    model_state_dict, model_cfg, save_path, **helper_kwargs
                )

        else:
            copy_model_config_and_code(
                model_path=model_cfg.model_path, save_path=save_path
            )
            save_path = os.path.join(save_path, "lora_adapter")
            model.save_pretrained(save_path, safe_serialization=True)
    else:
        copy_model_config_and_code(model_path=model_cfg.model_path, save_path=save_path)
        model_state_dict = model.state_dict()
        save_state_dict_sharded_safetensors(
            state_dict=model_state_dict,
            out_dir=save_path,
            dtype=save_dtype,
        )

        if model_save_helper_func is not None:
            model_save_helper_func(
                model_state_dict, model_cfg, save_path, **helper_kwargs
            )


@hydra.main(
    version_base="1.1", config_path="config", config_name="fsdp_model_convertor"
)
def main(cfg) -> None:
    load_user_extension_module()
    train_config_path = _effective_train_config_path(cfg)
    model_cfg = _prepare_model_cfg(_resolve_model_cfg(cfg))
    model = get_model(model_cfg)

    checkpoint = torch.load(cfg.convertor.ckpt_path, map_location="cpu")
    model_dict = _normalize_state_dict_keys(_extract_state_dict(checkpoint))
    strict_load = cfg.convertor.get("strict_load", True)
    missing_keys, unexpected_keys = model.load_state_dict(
        model_dict, strict=strict_load
    )
    if missing_keys or unexpected_keys:
        print(
            "Loaded checkpoint with "
            f"{len(missing_keys)} missing keys and {len(unexpected_keys)} unexpected keys."
        )
        if missing_keys:
            print(f"First missing keys: {missing_keys[:20]}")
        if unexpected_keys:
            print(f"First unexpected keys: {unexpected_keys[:20]}")

    _save_hf_checkpoint(model, model_cfg, cfg, cfg.convertor.save_path)
    _write_dreamzero_experiment_cfg(
        model_cfg=model_cfg,
        cfg=cfg,
        save_path=cfg.convertor.save_path,
        train_config_path=train_config_path,
    )

    print(f"Saved checkpoint to {cfg.convertor.save_path}")


if __name__ == "__main__":
    main()
