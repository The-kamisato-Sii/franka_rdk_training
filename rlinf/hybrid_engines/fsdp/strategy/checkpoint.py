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

from collections.abc import Iterable
from contextlib import nullcontext
from typing import Union

import torch
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_state_dict,
    set_state_dict,
)
from torch.distributed.checkpoint.stateful import Stateful
from torch.distributed.fsdp import LocalStateDictConfig, StateDictType
from torch.distributed.tensor import DTensor
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from rlinf.hybrid_engines.fsdp import FSDP, FSDPModule
from rlinf.hybrid_engines.fsdp.utils import FSDPVersion, to_local_if_dtensor
from rlinf.utils.utils import get_rng_state, set_rng_state


class Checkpoint(Stateful):
    def __init__(
        self,
        model: Union[FSDP, FSDPModule],
        optimizers: Union[Optimizer, Iterable[Optimizer]],
        lr_schedulers: Union[LRScheduler, Iterable[LRScheduler]],
        opts: StateDictOptions,
        fsdp_version: FSDPVersion,
        checkpoint_format: str = "dcp",
    ):
        self.model = model
        self.optimizers = optimizers
        self.lr_schedulers = (
            (lr_schedulers,)
            if isinstance(lr_schedulers, LRScheduler)
            else tuple(lr_schedulers)
        )
        self.opts = opts
        self.fsdp_version = fsdp_version
        self.checkpoint_format = checkpoint_format

    def _get_local_optim_state_dicts(self):
        if isinstance(self.optimizers, Optimizer):
            return self.optimizers.state_dict()
        return [opt.state_dict() for opt in self.optimizers]

    def _load_local_optim_state_dicts(self, optim_state_dicts):
        if isinstance(self.optimizers, Optimizer):
            self.optimizers.load_state_dict(optim_state_dicts)
        else:
            for opt, opt_sd in zip(self.optimizers, optim_state_dicts):
                opt.load_state_dict(opt_sd)

    def _local_shard_state_dict_context(self):
        if self.fsdp_version == FSDPVersion.FSDP:
            return FSDP.state_dict_type(
                self.model,
                StateDictType.LOCAL_STATE_DICT,
                LocalStateDictConfig(offload_to_cpu=True),
            )
        return nullcontext()

    @staticmethod
    def _clean_state_key(key: str) -> str:
        return key.replace("_fsdp_wrapped_module.", "")

    @staticmethod
    def _is_recoverable_fsdp_state_dict_assertion(exc: AssertionError) -> bool:
        msg = str(exc)
        return (
            ("FSDP assumes" in msg and "is in the state_dict" in msg)
            or "All FSDP modules should have the same state_dict_type" in msg
        )

    @staticmethod
    def _is_recoverable_fsdp_state_dict_runtime_error(exc: RuntimeError) -> bool:
        msg = str(exc)
        return (
            "Error(s) in loading state_dict" in msg
            and "size mismatch for" in msg
            and "copying a param with shape" in msg
        )

    def _manual_local_model_state_dict(self):
        model_sd = {}
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                key = self._clean_state_key(name)
                value = to_local_if_dtensor(param.detach())
                model_sd[key] = value.cpu().clone()

            for name, buffer in self.model.named_buffers():
                key = self._clean_state_key(name)
                if key in model_sd:
                    continue
                value = to_local_if_dtensor(buffer.detach())
                model_sd[key] = value.cpu().clone()
        return model_sd

    def _local_model_state_dict(self):
        try:
            with self._local_shard_state_dict_context():
                model_sd = self.model.state_dict()
        except AssertionError as exc:
            if not self._is_recoverable_fsdp_state_dict_assertion(exc):
                raise
            model_sd = self._manual_local_model_state_dict()

        return {
            self._clean_state_key(key): to_local_if_dtensor(value).cpu()
            if isinstance(value, torch.Tensor)
            else value
            for key, value in model_sd.items()
        }

    def _load_manual_local_model_state_dict(self, model_sd):
        named_tensors = {}
        for name, param in self.model.named_parameters():
            named_tensors[self._clean_state_key(name)] = param
        for name, buffer in self.model.named_buffers():
            named_tensors.setdefault(self._clean_state_key(name), buffer)

        with torch.no_grad():
            for key, value in model_sd.items():
                target = named_tensors.get(self._clean_state_key(key))
                if target is None or not isinstance(value, torch.Tensor):
                    continue
                local_target = (
                    target.to_local() if isinstance(target, DTensor) else target
                )
                local_value = value.to(
                    device=local_target.device,
                    dtype=local_target.dtype,
                    non_blocking=True,
                )
                local_target.copy_(local_value)

    def _restore_local_model_state_dict(self, model_sd):
        current_sd = self._local_model_state_dict()
        restored_sd = {}

        for key, value in model_sd.items():
            clean_key = self._clean_state_key(key)
            template = current_sd.get(clean_key)
            if isinstance(template, DTensor) and not isinstance(value, DTensor):
                local_template = template.to_local()
                if value.shape != local_template.shape:
                    raise ValueError(
                        f"Local shard shape mismatch for {key}: checkpoint has "
                        f"{tuple(value.shape)}, current rank expects "
                        f"{tuple(local_template.shape)} for global shape "
                        f"{tuple(template.shape)}."
                    )

                local_value = value.to(
                    device=local_template.device,
                    dtype=local_template.dtype,
                    non_blocking=True,
                )
                restored_sd[key] = DTensor.from_local(
                    local_value.contiguous(),
                    device_mesh=template.device_mesh,
                    placements=template.placements,
                    run_check=False,
                    shape=template.shape,
                    stride=template.stride(),
                )
            else:
                restored_sd[clean_key] = value

        return restored_sd

    def state_dict(self):
        if self.checkpoint_format == "local_shard":
            model_sd = self._local_model_state_dict()
            optim_sd = self._get_local_optim_state_dicts()

            lr_sched_sd = [lr.state_dict() for lr in self.lr_schedulers]

            out = {
                "model": model_sd,
                "optimizers": optim_sd,
                "lr_schedulers": lr_sched_sd,
                "fsdp_version": self.fsdp_version.value,
                "rng": get_rng_state(),
            }
        else:
            model_sd, optim_sd = get_state_dict(
                model=self.model,
                optimizers=self.optimizers,
                options=self.opts,
            )

            lr_sched_sd = [lr.state_dict() for lr in self.lr_schedulers]

            out = {
                "model": model_sd,
                "optimizers": optim_sd,
                "lr_schedulers": lr_sched_sd,
                "fsdp_version": self.fsdp_version.value,
                "rng": get_rng_state(),
            }
        return out

    def load_state_dict(self, state):
        assert "fsdp_version" in state, "Checkpoint is missing FSDP version info."
        ckpt_fsdp_version = FSDPVersion(state["fsdp_version"])
        if ckpt_fsdp_version != self.fsdp_version:
            raise ValueError(
                f"FSDP version mismatch: {ckpt_fsdp_version} != {self.fsdp_version}"
            )

        if self.checkpoint_format == "local_shard":
            try:
                with self._local_shard_state_dict_context():
                    self.model.load_state_dict(
                        self._restore_local_model_state_dict(state["model"])
                    )
            except AssertionError as exc:
                if not self._is_recoverable_fsdp_state_dict_assertion(exc):
                    raise
                self._load_manual_local_model_state_dict(state["model"])
            except RuntimeError as exc:
                if not self._is_recoverable_fsdp_state_dict_runtime_error(exc):
                    raise
                self._load_manual_local_model_state_dict(state["model"])

            self._load_local_optim_state_dicts(state["optimizers"])

        else:
            set_state_dict(
                model=self.model,
                optimizers=self.optimizers,
                model_state_dict=state["model"],
                optim_state_dict=state.get("optimizers", state.get("optim")),
                options=self.opts,
            )

        # lr schedulers
        if "lr_schedulers" in state:
            for lr, lr_sd in zip(self.lr_schedulers, state["lr_schedulers"]):
                lr.load_state_dict(lr_sd)

        if "rng" in state:
            set_rng_state(state["rng"])
