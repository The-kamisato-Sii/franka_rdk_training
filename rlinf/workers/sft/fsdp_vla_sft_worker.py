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
import fcntl
import json
import os
import time
from collections import deque
from typing import Any

import torch
from omegaconf import DictConfig, open_dict
from torch.utils._pytree import tree_map
from torchdata.stateful_dataloader import StatefulDataLoader

from rlinf.config import SupportedModel
from rlinf.models.embodiment.base_policy import ForwardType
from rlinf.utils.pytree import register_pytree_dataclasses
from rlinf.utils.logging import get_logger
from rlinf.utils.utils import get_rng_state, set_rng_state
from rlinf.workers.sft.fsdp_sft_worker import FSDPSftWorker

logger = get_logger()


class FSDPVlaSftWorker(FSDPSftWorker):
    def __init__(self, cfg: DictConfig):
        self._sync_real_world_joint_proprioception_config(cfg)
        super().__init__(cfg)
        self._dreamzero_loss = None
        self._dreamzero_loss_sample_metadata: list[dict[str, Any]] = []
        self._dreamzero_motion_debug: dict[str, Any] = {}
        self._motion_loss_prev100: deque[float] = deque(maxlen=100)
        self._grad_norm_prev100: deque[float] = deque(maxlen=100)
        self._spike_jsonl_path = self._build_spike_jsonl_path()
        self._first_batch_jsonl_path = self._build_first_batch_jsonl_path()
        self._logged_dreamzero_start_samples = False

    @staticmethod
    def _sync_real_world_joint_proprioception_config(cfg: DictConfig) -> None:
        try:
            model_type = SupportedModel(cfg.actor.model.model_type)
        except Exception:
            return
        if model_type not in [SupportedModel.DREAMZERO, SupportedModel.WMAM]:
            return
        if cfg.data.get("dataset_type", None) != "real_world_joint":
            return
        if "use_proprioception" not in cfg.data:
            return

        use_proprioception = bool(cfg.data.get("use_proprioception"))
        target_num_state_per_block = 1 if use_proprioception else 0
        diffusion_model_cfg = cfg.actor.model.action_head_cfg.config.diffusion_model_cfg
        current_num_state_per_block = diffusion_model_cfg.get(
            "num_state_per_block", None
        )
        if current_num_state_per_block == target_num_state_per_block:
            return

        with open_dict(diffusion_model_cfg):
            diffusion_model_cfg.num_state_per_block = target_num_state_per_block
        logger.info(
            "Synced real_world_joint proprioception config: "
            "use_proprioception=%s, num_state_per_block=%s",
            use_proprioception,
            target_num_state_per_block,
        )

    def _is_openpi_real_world_joint(self) -> bool:
        return (
            SupportedModel(self.cfg.actor.model.model_type) == SupportedModel.OPENPI
            and self.cfg.data.get("dataset_type", None) == "openpi_real_world_joint"
        )

    def _build_spike_jsonl_path(self) -> str:
        logger_cfg = self.cfg.runner.get("logger", {})
        log_path = str(logger_cfg.get("log_path", "results"))
        experiment_name = str(logger_cfg.get("experiment_name", "default"))
        return os.path.join(log_path, experiment_name, "motion_grad_spike_samples.jsonl")

    def _build_first_batch_jsonl_path(self) -> str:
        return os.path.join(
            os.path.dirname(self._spike_jsonl_path),
            "first_batch_sample_metadata.jsonl",
        )

    @staticmethod
    def _metric_to_float(value: Any) -> float | None:
        if value is None:
            return None
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu()
            if value.numel() != 1:
                return None
            value = value.item()
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _history_summary(values: deque[float]) -> dict[str, Any]:
        if not values:
            return {"count": 0}
        vals = list(values)
        return {
            "count": len(vals),
            "min": min(vals),
            "max": max(vals),
            "last": vals[-1],
        }

    @staticmethod
    def _jsonable(value: Any) -> Any:
        if isinstance(value, torch.Tensor):
            if value.numel() == 1:
                return value.detach().cpu().item()
            return value.detach().cpu().tolist()
        if isinstance(value, dict):
            return {str(k): FSDPVlaSftWorker._jsonable(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [FSDPVlaSftWorker._jsonable(v) for v in value]
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        if hasattr(value, "item"):
            try:
                return value.item()
            except Exception:
                pass
        if hasattr(value, "tolist"):
            try:
                return value.tolist()
            except Exception:
                pass
        return str(value)

    @staticmethod
    def _compact_jsonable(value: Any, max_list_items: int) -> Any:
        if isinstance(value, torch.Tensor):
            if value.numel() == 1:
                return value.detach().cpu().item()
            value = value.detach().cpu().tolist()
        if isinstance(value, dict):
            return {
                str(k): FSDPVlaSftWorker._compact_jsonable(v, max_list_items)
                for k, v in value.items()
            }
        if isinstance(value, (list, tuple)):
            if len(value) <= max_list_items:
                return [
                    FSDPVlaSftWorker._compact_jsonable(v, max_list_items)
                    for v in value
                ]
            head_count = max(1, max_list_items // 2)
            tail_count = max(1, max_list_items - head_count)
            return {
                "len": len(value),
                "first": [
                    FSDPVlaSftWorker._compact_jsonable(v, max_list_items)
                    for v in value[:head_count]
                ],
                "last": [
                    FSDPVlaSftWorker._compact_jsonable(v, max_list_items)
                    for v in value[-tail_count:]
                ],
            }
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        if hasattr(value, "item"):
            try:
                return value.item()
            except Exception:
                pass
        if hasattr(value, "tolist"):
            try:
                return FSDPVlaSftWorker._compact_jsonable(
                    value.tolist(), max_list_items
                )
            except Exception:
                pass
        return str(value)

    def _samples_with_motion_debug(self) -> list[dict[str, Any]]:
        samples = [dict(sample) for sample in self._dreamzero_loss_sample_metadata]
        debug = self._dreamzero_motion_debug or {}
        for key in (
            "motion_loss_per_sample",
            "motion_loss_weighted_per_sample",
            "timestep_motion_id",
            "motion_valid_mask",
        ):
            value = debug.get(key)
            if value is None:
                continue
            values = self._jsonable(value)
            if not isinstance(values, list):
                continue
            for idx, sample in enumerate(samples):
                if idx < len(values):
                    sample[key] = values[idx]
        return samples

    def _motion_debug_summary(self) -> dict[str, Any]:
        value = self._dreamzero_motion_debug.get("motion_loss_per_sample")
        if value is None or not isinstance(value, torch.Tensor) or value.numel() == 0:
            return {}
        tensor = value.detach().float().cpu()
        flat_index = int(torch.argmax(tensor).item())
        if tensor.ndim >= 2:
            sample_index = flat_index // int(tensor.shape[1])
            time_index = flat_index % int(tensor.shape[1])
        else:
            sample_index = flat_index
            time_index = 0
        summary: dict[str, Any] = {
            "max_motion_loss_per_sample": float(tensor.reshape(-1)[flat_index].item()),
            "argmax_sample_index": int(sample_index),
            "argmax_time_index": int(time_index),
        }
        samples = self._dreamzero_loss_sample_metadata
        if 0 <= sample_index < len(samples):
            summary["argmax_sample"] = self._jsonable(samples[sample_index])
        timestep = self._dreamzero_motion_debug.get("timestep_motion_id")
        if isinstance(timestep, torch.Tensor) and timestep.numel() > 0:
            t_cpu = timestep.detach().cpu()
            try:
                summary["argmax_timestep_motion_id"] = int(t_cpu[sample_index, time_index].item())
            except Exception:
                pass
        weighted = self._dreamzero_motion_debug.get("motion_loss_weighted_per_sample")
        if isinstance(weighted, torch.Tensor) and weighted.numel() > 0:
            w_cpu = weighted.detach().float().cpu()
            try:
                summary["argmax_weighted_motion_loss"] = float(w_cpu[sample_index, time_index].item())
            except Exception:
                pass
        return summary

    def _write_spike_event(
        self,
        *,
        reasons: list[str],
        train_metrics: dict[str, Any],
        motion_loss: float | None,
        grad_norm: float | None,
    ) -> None:
        record = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "global_step": int(self.global_step),
            "rank": int(self._rank),
            "local_rank": int(os.environ.get("LOCAL_RANK", -1)),
            "reasons": reasons,
            "conditions": {
                "motion_loss_prev100_all_lt_1_then_gt_2": (
                    len(self._motion_loss_prev100) == 100
                    and all(v < 1.0 for v in self._motion_loss_prev100)
                    and motion_loss is not None
                    and motion_loss > 2.0
                ),
                "grad_norm_prev100_all_lt_2_then_gt_3": (
                    len(self._grad_norm_prev100) == 100
                    and all(v < 2.0 for v in self._grad_norm_prev100)
                    and grad_norm is not None
                    and grad_norm > 3.0
                ),
            },
            "metrics": self._jsonable(train_metrics),
            "history": {
                "motion_loss_prev100": self._history_summary(self._motion_loss_prev100),
                "grad_norm_prev100": self._history_summary(self._grad_norm_prev100),
            },
            "motion_debug_summary": self._motion_debug_summary(),
            "samples": self._jsonable(self._samples_with_motion_debug()),
        }

        os.makedirs(os.path.dirname(self._spike_jsonl_path), exist_ok=True)
        lock_path = f"{self._spike_jsonl_path}.lock"
        line = json.dumps(record, ensure_ascii=False, sort_keys=True)
        with open(lock_path, "w", encoding="utf-8") as lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            try:
                with open(self._spike_jsonl_path, "a", encoding="utf-8") as out:
                    out.write(line + "\n")
            finally:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

    def _maybe_log_spike_event(self, train_metrics: dict[str, Any]) -> None:
        if SupportedModel(self.cfg.actor.model.model_type) not in [SupportedModel.DREAMZERO, SupportedModel.WMAM]:
            return
        motion_loss = self._metric_to_float(train_metrics.get("motion_loss"))
        grad_norm = self._metric_to_float(train_metrics.get("grad_norm"))
        reasons: list[str] = []
        if (
            motion_loss is not None
            and motion_loss > 2.0
            and len(self._motion_loss_prev100) == 100
            and all(v < 1.0 for v in self._motion_loss_prev100)
        ):
            reasons.append("motion_loss_spike")
        if (
            grad_norm is not None
            and grad_norm > 3.0
            and len(self._grad_norm_prev100) == 100
            and all(v < 2.0 for v in self._grad_norm_prev100)
        ):
            reasons.append("grad_norm_spike")
        if reasons:
            self._write_spike_event(
                reasons=reasons,
                train_metrics=train_metrics,
                motion_loss=motion_loss,
                grad_norm=grad_norm,
            )
        if motion_loss is not None:
            self._motion_loss_prev100.append(motion_loss)
        if grad_norm is not None:
            self._grad_norm_prev100.append(grad_norm)

    def _write_first_batch_sample_metadata(
        self, sample_metadata: list[dict[str, Any]]
    ) -> None:
        record = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "global_step": int(self.global_step),
            "rank": int(self._rank),
            "local_rank": int(os.environ.get("LOCAL_RANK", -1)),
            "pid": int(os.getpid()),
            "host": getattr(os, "uname", lambda: None)().nodename
            if hasattr(os, "uname")
            else "",
            "batch_size": len(sample_metadata),
            "samples": self._jsonable(sample_metadata),
        }
        os.makedirs(os.path.dirname(self._first_batch_jsonl_path), exist_ok=True)
        lock_path = f"{self._first_batch_jsonl_path}.lock"
        line = json.dumps(record, ensure_ascii=False, sort_keys=True)
        with open(lock_path, "w", encoding="utf-8") as lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            try:
                with open(self._first_batch_jsonl_path, "a", encoding="utf-8") as out:
                    out.write(line + "\n")
            finally:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

    def _log_dreamzero_start_samples(self, sample_metadata: list[dict[str, Any]]) -> None:
        if self._logged_dreamzero_start_samples or not sample_metadata:
            return
        self._logged_dreamzero_start_samples = True
        self._write_first_batch_sample_metadata(sample_metadata)
        logger.info(
            "[rank %s] wrote DreamZero first training micro-batch sample metadata: batch_size=%s path=%s",
            int(self._rank),
            len(sample_metadata),
            self._first_batch_jsonl_path,
        )

    def build_dataloader(self, data_paths: list[str], eval_dataset: bool = False):
        if SupportedModel(self.cfg.actor.model.model_type) in [SupportedModel.OPENPI]:
            if self.cfg.data.get("dataset_type", None) == "openpi_real_world_joint":
                from rlinf.data.datasets.openpi import (
                    build_openpi_real_world_joint_sft_dataloader,
                )

                return build_openpi_real_world_joint_sft_dataloader(
                    self.cfg, self._world_size, self._rank, data_paths, eval_dataset
                )

            import openpi.training.data_loader as openpi_data_loader

            from rlinf.models.embodiment.openpi.dataconfig import get_openpi_config

            config = get_openpi_config(
                self.cfg.actor.model.openpi.config_name,
                model_path=self.cfg.actor.model.model_path,
                batch_size=self.cfg.actor.micro_batch_size * self._world_size,
                data_kwargs=getattr(self.cfg.actor, "openpi_data", None),
            )
            data_loader = openpi_data_loader.create_data_loader(
                config, framework="pytorch", shuffle=True
            )
            return data_loader, data_loader.data_config()
        elif SupportedModel(self.cfg.actor.model.model_type) in [
            SupportedModel.LINGBOTVLA
        ]:
            from rlinf.models.embodiment.lingbotvla.sft_builder import (
                build_lingbot_sft_dataloader,
            )

            return build_lingbot_sft_dataloader(
                self.cfg, self._world_size, self._rank, data_paths
            )
        elif SupportedModel(self.cfg.actor.model.model_type) in [
            SupportedModel.DREAMZERO,
            SupportedModel.WMAM,
        ]:
            self._dreamzero_loss = None
            if self.cfg.data.get("dataset_type", None) == "real_world_joint":
                from rlinf.data.datasets.dreamzero.real_world_joint import (
                    build_real_world_joint_sft_dataloader,
                )

                return build_real_world_joint_sft_dataloader(
                    self.cfg, self._world_size, self._rank, data_paths, eval_dataset
                )
            if self.cfg.data.get("dataset_type", None) == "robotwin2":
                from rlinf.data.datasets.dreamzero.robotwin2 import (
                    build_robotwin2_sft_dataloader,
                )

                return build_robotwin2_sft_dataloader(
                    self.cfg, self._world_size, self._rank, data_paths, eval_dataset
                )
            from rlinf.data.datasets.dreamzero import (
                build_dreamzero_sft_dataloader,
            )

            return build_dreamzero_sft_dataloader(
                self.cfg, self._world_size, self._rank, data_paths, eval_dataset
            )
        else:
            raise KeyError(
                f"not support such model type {self.cfg.actor.model.model_type} for SFT right now."
            )

    def get_eval_model_output(self, batch: dict[str, Any]):
        # now the eval is not supported for embodied sft
        raise NotImplementedError("eval is not supported for embodied sft right now.")

    def get_train_model_output(self, batch: dict[str, Any]):
        if SupportedModel(self.cfg.actor.model.model_type) in [
            SupportedModel.LINGBOTVLA,
            SupportedModel.DREAMZERO,
            SupportedModel.WMAM,
        ]:
            sample_metadata = []
            if isinstance(batch, dict):
                sample_metadata = batch.pop("sample_metadata", [])
            if isinstance(sample_metadata, dict):
                sample_metadata = [sample_metadata]
            self._log_dreamzero_start_samples(list(sample_metadata))
            with self.amp_context:
                losses_dict = self.model(forward_type=ForwardType.SFT, data=batch)
            if losses_dict.get("dynamics_loss", None) is not None:
                self._dreamzero_loss = {
                    "dynamics_loss": losses_dict["dynamics_loss"],
                    "action_loss": losses_dict["action_loss"],
                }
                self._dreamzero_loss_sample_metadata = list(sample_metadata)
                self._dreamzero_motion_debug = {}
                if (
                    self.cfg.actor.model.action_head_cfg.config.get(
                        "use_motion_modality", False
                    )
                    and losses_dict.get("motion_loss", None) is not None
                ):
                    self._dreamzero_loss["motion_loss"] = losses_dict["motion_loss"]
                    for debug_key in (
                        "motion_loss_per_sample",
                        "motion_loss_weighted_per_sample",
                        "timestep_motion_id",
                        "motion_valid_mask",
                    ):
                        debug_value = losses_dict.get(debug_key, None)
                        if debug_value is not None:
                            self._dreamzero_motion_debug[debug_key] = debug_value.detach()
            return losses_dict["loss"]
        observation, actions = batch

        register_pytree_dataclasses(observation)
        observation = tree_map(
            lambda x: (
                torch.as_tensor(x, device=self.device).contiguous().clone()
                if x is not None
                else x
            ),
            observation,
        )
        actions = actions.to(torch.float32)
        actions = actions.to(self.device)

        model_data = {"observation": observation, "actions": actions}

        with self.amp_context:
            losses = self.model(
                forward_type=ForwardType.SFT,
                data=model_data,
            )

        # train model return the loss
        return losses

    def run_training(self):
        self._dreamzero_loss_sample_metadata = []
        self._dreamzero_motion_debug = {}
        train_metrics = super().run_training()
        if (
            SupportedModel(self.cfg.actor.model.model_type)
            in [SupportedModel.DREAMZERO, SupportedModel.WMAM]
            and self._dreamzero_loss is not None
        ):
            train_metrics.update(
                {
                    "dynamics_loss": self._dreamzero_loss["dynamics_loss"]
                    .detach()
                    .cpu()
                    .item(),
                    "action_loss": self._dreamzero_loss["action_loss"]
                    .detach()
                    .cpu()
                    .item(),
                    **(
                        {
                            "motion_loss": self._dreamzero_loss["motion_loss"]
                            .detach()
                            .cpu()
                            .item()
                        }
                        if "motion_loss" in self._dreamzero_loss
                        else {}
                    ),
                }
            )
            self._dreamzero_loss = None
        self._maybe_log_spike_event(train_metrics)
        return train_metrics

    def save_checkpoint(self, save_path: str, step: int = 0) -> None:
        super().save_checkpoint(save_path, step)

        if isinstance(self.data_loader, StatefulDataLoader):
            state = self.data_loader.state_dict()
            data_state_dir = os.path.join(save_path, "data_shard")
            os.makedirs(data_state_dir, exist_ok=True)
            torch.save(
                state,
                os.path.join(data_state_dir, f"rank_{self._rank:05d}.pt"),
            )
            torch.distributed.barrier()

        rng_state = get_rng_state()
        rng_state_dir = os.path.join(save_path, "rng_shard")
        os.makedirs(rng_state_dir, exist_ok=True)
        torch.save(
            rng_state,
            os.path.join(rng_state_dir, f"rank_{self._rank:05d}.pt"),
        )

        torch.distributed.barrier()

    def load_checkpoint(self, load_path: str) -> None:
        super().load_checkpoint(load_path)

        if isinstance(self.data_loader, StatefulDataLoader):
            data_shard_path = os.path.join(
                load_path, "data_shard", f"rank_{self._rank:05d}.pt"
            )
            data_all_path = os.path.join(load_path, "data.pt")
            if os.path.exists(data_shard_path):
                state = torch.load(data_shard_path, weights_only=False)
                self.data_loader.load_state_dict(state)
                self.data_iter = iter(self.data_loader)
            elif os.path.exists(data_all_path):
                all_states = torch.load(data_all_path, weights_only=False)
                state = all_states[self._rank]
                self.data_loader.load_state_dict(state)
                self.data_iter = iter(self.data_loader)

        rng_shard_path = os.path.join(
            load_path, "rng_shard", f"rank_{self._rank:05d}.pt"
        )
        rng_path = os.path.join(load_path, "rng.pt")
        if os.path.exists(rng_shard_path):
            rng_state = torch.load(rng_shard_path, weights_only=False)
            set_rng_state(rng_state)
        elif os.path.exists(rng_path):
            all_rng_states = torch.load(rng_path, weights_only=False)
            set_rng_state(all_rng_states[self._rank])

        torch.distributed.barrier()

    def get_max_steps_per_epoch(self):
        if self.data_loader is None:
            return 0
        if SupportedModel(self.cfg.actor.model.model_type) == SupportedModel.OPENPI:
            try:
                num_batches = len(self._openpi_pytorch_dataloader(self.data_loader))
            except TypeError:
                num_batches = len(self.data_loader)
            return max(1, num_batches // self.gradient_accumulation)
        return super().get_max_steps_per_epoch()

    @staticmethod
    def _openpi_pytorch_dataloader(openpi_dataloader: Any):
        """Unwrap OpenPI `DataLoaderImpl` to the inner PyTorch DataLoader.

        OpenPI torch path:
          DataLoaderImpl._data_loader -> TorchDataLoader
          TorchDataLoader._data_loader / .torch_loader -> torch.utils.data.DataLoader

        """
        torch_data_loader = getattr(openpi_dataloader, "_data_loader", None)
        pytorch_dl = getattr(torch_data_loader, "_data_loader", None) or getattr(
            torch_data_loader, "torch_loader", None
        )
        if pytorch_dl is None:
            raise TypeError(
                "OpenPI dataloader does not expose an inner torch DataLoader; cannot infer steps per epoch from len()."
            )
        return pytorch_dl
