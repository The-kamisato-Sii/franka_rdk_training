#!/usr/bin/env python
"""HTTP inference server for RLinf OpenPI pi0.5 on dual Franka.

The training-matched wire protocol is franka_rdk ``openpi_pi05_client``.
The client sends raw compatibility fields plus ``payload["openpi"]``:

    POST /infer
    Content-Type: application/msgpack

Current request payload shape:

    {
        "state": np.ndarray[16],                 # raw compatibility field
        "images": {...},                         # raw compatibility field
        "openpi": {
            "state": np.ndarray[32],             # q01/q99 normalized + padded
            "state_mask": np.ndarray[32],        # true entries are real state dims
            "images": {
                "base_0_rgb": np.ndarray[3,224,224] float32 in [-1,1],
                "left_wrist_0_rgb": np.ndarray[3,224,224] float32 in [-1,1],
                "right_wrist_0_rgb": np.ndarray[3,224,224] float32 in [-1,1],
            },
            "action_q01": np.ndarray[32],
            "action_q99": np.ndarray[32],
        },
        "prompt": str,
    }

Response payload:

    {"actions": np.ndarray[n_action_steps, 16]}

The returned actions are normalized OpenPI model outputs. ``openpi_pi05_client``
defaults to ``server_returns_normalized_actions=true`` and applies q01/q99
inverse normalization before sending commands to the robot.

The model can either load an RLinf local-shard checkpoint on top of
``base_model_path`` or, when ``checkpoint_path`` is disabled, serve the complete
``model.safetensors`` stored in ``base_model_path`` directly.
"""


from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import msgpack
import numpy as np
import torch
from fastapi import FastAPI, HTTPException, Request, Response
from omegaconf import OmegaConf
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_BASE_MODEL = (
    REPO_ROOT / "checkpoints" / "pi05_base_pytorch_real_world_joint"
)
DEFAULT_CKPT = Path(
    "/inspire/hdd/project/robot-body/linbokai-CZXS24250037/results/"
    "real_world_franka_dual_openpi_pi05_sft_v2/checkpoints/global_step_10000"
)

OPENPI_CLIENT_PREPROCESSED_FORMAT = "openpi_pi05_client_preprocessed_v1"
OPENPI_IMAGE_KEYS = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")


def _pack_array(obj: Any):
    if isinstance(obj, np.ndarray):
        if obj.dtype.kind in ("V", "O", "c"):
            raise ValueError(f"Unsupported ndarray dtype: {obj.dtype}")
        return {
            b"__ndarray__": True,
            b"data": obj.tobytes(),
            b"dtype": obj.dtype.str,
            b"shape": obj.shape,
        }
    if isinstance(obj, np.generic):
        return {b"__npgeneric__": True, b"data": obj.item(), b"dtype": obj.dtype.str}
    return obj


def _unpack_array(obj: dict):
    if b"__ndarray__" in obj:
        return np.ndarray(
            buffer=obj[b"data"],
            dtype=np.dtype(obj[b"dtype"]),
            shape=obj[b"shape"],
        )
    if b"__npgeneric__" in obj:
        return np.dtype(obj[b"dtype"]).type(obj[b"data"])
    return obj


def packb(data: Any) -> bytes:
    return msgpack.packb(data, default=_pack_array, use_bin_type=True)


def unpackb(data: bytes) -> Any:
    return msgpack.unpackb(data, object_hook=_unpack_array, raw=False)


def _as_chw_float_minus_one_one(image: Any) -> np.ndarray:
    arr = np.asarray(image)
    arr = np.squeeze(arr)
    if arr.ndim == 4 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim != 3:
        raise ValueError(f"Expected preprocessed image with 3 dims, got shape={arr.shape}")
    if arr.shape[0] in (1, 3, 4) and arr.shape[-1] not in (1, 3, 4):
        arr = arr[:3]
    elif arr.shape[-1] in (1, 3, 4):
        arr = np.transpose(arr[..., :3], (2, 0, 1))
    else:
        raise ValueError(f"Expected CHW or HWC RGB image, got shape={arr.shape}")
    if arr.shape[0] != 3:
        raise ValueError(f"Expected 3-channel RGB image, got shape={arr.shape}")
    if arr.dtype == np.uint8:
        arr = arr.astype(np.float32) / 255.0 * 2.0 - 1.0
    else:
        arr = arr.astype(np.float32, copy=False)
        finite = arr[np.isfinite(arr)]
        min_v = float(finite.min()) if finite.size else -1.0
        max_v = float(finite.max()) if finite.size else 1.0
        if min_v >= -1e-3 and max_v <= 1.0 + 1e-3:
            arr = arr * 2.0 - 1.0
        elif min_v < -1.5 or max_v > 2.0:
            arr = np.clip(arr, 0, 255) / 255.0 * 2.0 - 1.0
        else:
            arr = np.clip(arr, -1.0, 1.0)
    return np.ascontiguousarray(arr, dtype=np.float32)


def _chw_minus_one_one_to_hwc_uint8(image: Any) -> np.ndarray:
    arr = _as_chw_float_minus_one_one(image)
    arr = np.transpose(arr, (1, 2, 0))
    arr = np.clip((arr + 1.0) * 0.5 * 255.0, 0, 255)
    return np.ascontiguousarray(np.rint(arr).astype(np.uint8))


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(sub_value) for key, sub_value in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(sub_value) for sub_value in value]
    return value


def _pad_or_trim_vector(values: Any, dim: int) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32).reshape(-1)
    out = np.zeros((int(dim),), dtype=np.float32)
    n = min(arr.shape[0], int(dim))
    out[:n] = arr[:n]
    return np.ascontiguousarray(out)


def _unpadded_preprocessed_state(openpi: dict[str, Any]) -> np.ndarray:
    state_payload = np.asarray(openpi.get("state"), dtype=np.float32).reshape(-1)
    if state_payload.size == 0:
        raise ValueError("payload[openpi][state] must contain normalized state values")

    mask_payload = openpi.get("state_mask")
    if mask_payload is None:
        raise ValueError("payload[openpi][state_mask] is required for padded OpenPI state")
    state_mask = np.asarray(mask_payload, dtype=bool).reshape(-1)
    if state_mask.shape != state_payload.shape:
        raise ValueError(
            "payload[openpi][state_mask] shape must match payload[openpi][state]: "
            f"{state_mask.shape} vs {state_payload.shape}"
        )
    if not bool(np.any(state_mask)):
        raise ValueError("payload[openpi][state_mask] must contain at least one true entry")
    return np.ascontiguousarray(state_payload[state_mask], dtype=np.float32)


def _torchify_tree(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return torch.from_numpy(np.asarray(value).copy())
    if isinstance(value, np.generic):
        return torch.as_tensor(value.item())
    if isinstance(value, dict):
        return {key: _torchify_tree(sub_value) for key, sub_value in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_torchify_tree(sub_value) for sub_value in value)
    return value


def _cast_floating_tree(value: Any, dtype: torch.dtype) -> Any:
    if torch.is_tensor(value):
        if value.is_floating_point():
            return value.to(dtype=dtype)
        return value
    if isinstance(value, dict):
        return {key: _cast_floating_tree(sub_value, dtype) for key, sub_value in value.items()}
    if isinstance(value, list):
        return [_cast_floating_tree(sub_value, dtype) for sub_value in value]
    if isinstance(value, tuple):
        return tuple(_cast_floating_tree(sub_value, dtype) for sub_value in value)
    return value


def _openpi_preprocessed_payload(payload: dict) -> dict[str, Any] | None:
    openpi = payload.get("openpi")
    if not isinstance(openpi, dict):
        return None
    if not bool(openpi.get("preprocessed", False)):
        return None
    fmt = str(openpi.get("format", ""))
    if fmt and fmt != OPENPI_CLIENT_PREPROCESSED_FORMAT:
        raise ValueError(
            f"Unsupported OpenPI preprocessed payload format {fmt!r}; "
            f"expected {OPENPI_CLIENT_PREPROCESSED_FORMAT!r}"
        )
    return openpi

def resolve_checkpoint_file(path: Path, rank: int) -> Path:
    path = Path(path)
    if path.is_file():
        return path
    candidates = [
        path / "actor" / "local_shard_checkpoint" / f"checkpoint_rank_{rank}.pt",
        path / "local_shard_checkpoint" / f"checkpoint_rank_{rank}.pt",
        path / f"checkpoint_rank_{rank}.pt",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"Could not find checkpoint_rank_{rank}.pt under {path}. Tried: "
        + ", ".join(str(x) for x in candidates)
    )


def checkpoint_path_is_disabled(path: Any) -> bool:
    text = "" if path is None else str(path).strip()
    return text.lower() in ("", "none", "null", "skip", "disabled")


class OpenPIFrankaDualRunner:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.device = torch.device(args.device)
        self.n_action_steps = int(args.n_action_steps)
        self.action_dim = int(args.return_action_dim)
        self.log_input_output = bool(args.log_input_output)
        self.log_dir: Path | None = None
        self._log_request_index = 0
        if self.log_input_output:
            self.log_dir = (
                Path(__file__).resolve().parent
                / "log"
                / f"openpi_{time.strftime('%Y%m%d_%H%M%S')}"
            )
            (self.log_dir / "input").mkdir(parents=True, exist_ok=True)
            (self.log_dir / "output").mkdir(parents=True, exist_ok=True)
            print(f"[open_pi] input/output logging enabled: {self.log_dir}", flush=True)

        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("high")

        self.model = self._load_model()
        self.model.eval()
        self._setup_preprocessed_tokenizer()

    def _base_cfg(self):
        return OmegaConf.create(
            {
                "model_path": str(self.args.base_model_path),
                "precision": self.args.precision,
                "openpi": {
                    "config_name": "pi05_real_world_joint",
                    # get_model requires a stats key to initialize OpenPI
                    # wrappers, but this server bypasses input_transform and
                    # output_transform on the strict preprocessed path.
                    "repo_id": "real_world_franka_dual",
                    "norm_stats_key": "real_world_franka_dual",
                    "detach_critic_input": True,
                    "num_images_in_input": 3,
                    "train_expert_only": True,
                    "action_chunk": 48,
                    "num_steps": int(self.args.num_steps),
                    "noise_method": str(self.args.noise_method),
                    "noise_level": float(self.args.noise_level),
                    "action_env_dim": 32,
                    "add_value_head": False,
                    "value_after_vlm": False,
                    "value_vlm_mode": "mean_token",
                },
            }
        )

    def _load_model(self):
        from rlinf.models.embodiment.openpi import get_model

        model = get_model(self._base_cfg(), torch_dtype=self._torch_dtype())
        if checkpoint_path_is_disabled(self.args.checkpoint_path):
            print(
                "[open_pi] CHECKPOINT_PATH disabled; using weights from "
                f"BASE_MODEL_PATH={self.args.base_model_path}",
                flush=True,
            )
            model.to(self.device)
            return model

        ckpt_file = resolve_checkpoint_file(
            Path(self.args.checkpoint_path), int(self.args.checkpoint_rank)
        )
        print(f"[open_pi] Loading local-shard model state: {ckpt_file}", flush=True)
        checkpoint = torch.load(ckpt_file, map_location="cpu", weights_only=False)
        state_dict = checkpoint.get("model", checkpoint)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        print(
            f"[open_pi] load_state_dict complete: missing={len(missing)} "
            f"unexpected={len(unexpected)}",
            flush=True,
        )
        if missing:
            print(f"[open_pi] first missing keys: {list(missing)[:8]}", flush=True)
        if unexpected:
            print(f"[open_pi] first unexpected keys: {list(unexpected)[:8]}", flush=True)
        model.to(self.device)
        return model

    def _torch_dtype(self):
        precision = str(self.args.precision).lower()
        if precision in ("bf16", "bfloat16"):
            return torch.bfloat16
        if precision in ("fp16", "float16", "half"):
            return torch.float16
        return torch.float32

    def _setup_preprocessed_tokenizer(self) -> None:
        from openpi import transforms as _transforms
        from openpi.models import tokenizer as _tokenizer

        self._tokenize_preprocessed_prompt = _transforms.TokenizePrompt(
            _tokenizer.PaligemmaTokenizer(self.model.config.max_token_len),
            discrete_state_input=bool(getattr(self.model.config, "discrete_state_input", False)),
        )

    def _observation_from_client_preprocessed(self, payload: dict, prompt: str):
        from openpi.models import model as _model

        openpi = _openpi_preprocessed_payload(payload)
        if openpi is None:
            return None
        images_payload = openpi.get("images")
        if not isinstance(images_payload, dict):
            raise ValueError("payload[openpi][images] must be a dict")
        missing = [key for key in OPENPI_IMAGE_KEYS if key not in images_payload]
        if missing:
            raise ValueError(f"OpenPI preprocessed payload missing image keys: {missing}")

        image = {
            key: _as_chw_float_minus_one_one(images_payload[key])[None]
            for key in OPENPI_IMAGE_KEYS
        }
        mask_payload = openpi.get("image_mask") or {}
        image_mask = {
            key: np.asarray([bool(mask_payload.get(key, True))], dtype=bool)
            for key in OPENPI_IMAGE_KEYS
        }
        state_dim = int(getattr(self.model.config, "action_dim", 32))
        token_state = _unpadded_preprocessed_state(openpi)
        model_state = _pad_or_trim_vector(token_state, state_dim)[None]
        processed = {
            "image": image,
            "image_mask": image_mask,
            "state": token_state,
            "prompt": prompt,
        }
        processed = self._tokenize_preprocessed_prompt(processed)
        # Training applies TokenizePrompt before PadStatesAndActions. Keep the
        # tokenizer state unpadded, then provide the padded state tensor to the
        # action expert to match the OpenPI pi0.5 SFT input transform exactly.
        processed["state"] = model_state
        for key in ("tokenized_prompt", "tokenized_prompt_mask", "token_ar_mask", "token_loss_mask"):
            value = processed.get(key)
            if value is not None and getattr(value, "ndim", None) == 1:
                processed[key] = value[None]
        processed = _torchify_tree(processed)
        processed = self.model.precision_processor(processed)
        processed = _cast_floating_tree(processed, self._torch_dtype())
        return _model.Observation.from_dict(processed)

    def log_request_response(self, payload: dict, actions: np.ndarray) -> None:
        if self.log_dir is None:
            return

        request_index = self._log_request_index
        self._log_request_index += 1
        prefix = f"{request_index:06d}"

        openpi = _openpi_preprocessed_payload(payload)
        if openpi is None:
            return

        input_dir = self.log_dir / "input"
        output_dir = self.log_dir / "output"
        images_payload = openpi.get("images") or {}
        for key in OPENPI_IMAGE_KEYS:
            if key not in images_payload:
                continue
            image = _chw_minus_one_one_to_hwc_uint8(images_payload[key])
            Image.fromarray(image).save(input_dir / f"{prefix}_{key}.png")

        prompt = str(payload.get("prompt") or payload.get("task") or self.args.default_prompt)
        prompt_record = {
            "request_index": request_index,
            "prompt": prompt,
            "task": payload.get("task"),
            "model_type": payload.get("model_type"),
            "stats_task_id": payload.get("stats_task_id"),
            "openpi_format": openpi.get("format"),
            "openpi_task_id": openpi.get("task_id"),
            "stats_source": openpi.get("stats_source"),
            "stats_file": openpi.get("stats_file"),
            "image_format": openpi.get("image_format"),
            "image_transport": openpi.get("image_transport"),
            "image_mask": openpi.get("image_mask"),
            "state_shape": tuple(np.asarray(openpi.get("state")).shape),
            "state_mask": openpi.get("state_mask"),
            "action_shape": tuple(np.asarray(actions).shape),
        }
        with (input_dir / f"{prefix}_prompt.json").open("w", encoding="utf-8") as f:
            json.dump(_json_safe(prompt_record), f, ensure_ascii=False, indent=2)

        if "state" in openpi:
            np.save(input_dir / f"{prefix}_state_normalized.npy", np.asarray(openpi["state"], dtype=np.float32))
        np.save(output_dir / f"{prefix}_normalized_actions.npy", np.asarray(actions, dtype=np.float32))


    def _masked_initial_noise(self, observation) -> torch.Tensor:
        batch_size = int(observation.state.shape[0])
        action_horizon = int(getattr(self.model.config, "action_horizon", self.n_action_steps))
        model_action_dim = int(getattr(self.model.config, "action_dim", 32))
        real_action_dim = max(0, min(int(self.action_dim), model_action_dim))
        noise = torch.randn(
            (batch_size, action_horizon, model_action_dim),
            device=observation.state.device,
            dtype=torch.float32,
        )
        if real_action_dim < model_action_dim:
            noise[..., real_action_dim:] = 0.0
        return noise

    @torch.inference_mode()
    def infer(self, payload: dict) -> np.ndarray:
        prompt = str(payload.get("prompt") or payload.get("task") or self.args.default_prompt)
        observation = self._observation_from_client_preprocessed(payload, prompt)
        if observation is None:
            raise ValueError(
                "Missing payload['openpi'] preprocessed block. The Franka-dual "
                "OpenPI server requires client-side q01/q99 state normalization "
                "and image preprocessing from openpi_pi05_client."
            )

        noise = self._masked_initial_noise(observation)
        outputs = self.model.sample_actions(
            observation,
            noise=noise,
            mode="eval",
            compute_values=False,
        )
        # sample_actions returns the normalized model-space action. Do not call
        # model.output_transform here: RLinf's output transform includes
        # q01/q99 Unnormalize, and the robot client owns that inverse step.
        actions_np = outputs["actions"][0].detach().float().cpu().numpy()
        return np.ascontiguousarray(
            actions_np[: self.n_action_steps, : self.action_dim],
            dtype=np.float32,
        )


def build_app(runner: OpenPIFrankaDualRunner) -> FastAPI:
    app = FastAPI(title="RLinf OpenPI Franka Dual Server", version="0.1")

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {
            "ok": True,
            "device": str(runner.device),
            "n_action_steps": runner.n_action_steps,
            "return_action_dim": runner.action_dim,
            "num_steps": int(runner.model.config.num_steps),
            "noise_method": str(runner.model.config.noise_method),
            "noise_level": float(runner.model.config.noise_level),
            "requires_openpi_preprocessed_payload": True,
            "returns_normalized_actions": True,
            "log_input_output": runner.log_input_output,
            "log_dir": None if runner.log_dir is None else str(runner.log_dir),
        }

    @app.post("/infer")
    async def infer(request: Request) -> Response:
        started = time.perf_counter()
        request_body = await request.body()
        try:
            print(f"[open_pi] /infer request_bytes={len(request_body)}", flush=True)
            payload = unpackb(request_body)
            if not isinstance(payload, dict):
                raise ValueError(f"request payload must be a dict, got {type(payload)}")
            print(f"[open_pi] /infer payload_keys={sorted(payload.keys())}", flush=True)
            actions = runner.infer(payload)
            try:
                runner.log_request_response(payload, actions)
            except Exception as log_exc:
                print(f"[open_pi] WARNING failed to log input/output: {log_exc}", flush=True)
            body = packb({"actions": actions})
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            headers = {"X-Inference-Time-Ms": f"{elapsed_ms:.2f}"}
            return Response(
                content=body,
                media_type="application/msgpack",
                headers=headers,
            )
        except Exception as exc:
            print(
                f"[open_pi] /infer failed after {(time.perf_counter() - started) * 1000.0:.2f} ms: {exc}",
                flush=True,
            )
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--precision", default="float32")
    parser.add_argument("--base-model-path", type=Path, default=DEFAULT_BASE_MODEL)
    parser.add_argument("--checkpoint-path", type=Path, default=DEFAULT_CKPT)
    parser.add_argument("--checkpoint-rank", type=int, default=0)
    parser.add_argument("--n-action-steps", type=int, default=48)
    parser.add_argument("--return-action-dim", type=int, default=16)
    parser.add_argument(
        "--num-steps",
        type=int,
        default=5,
        help="OpenPI flow/diffusion denoising steps. 5 matches the RLinf pi0.5 SFT training config.",
    )
    parser.add_argument(
        "--noise-method",
        default="flow_sde",
        choices=("flow_ode", "flow_sde", "flow_noise", "flow_cps"),
        help="OpenPI sampling method. flow_sde matches the RLinf pi0.5 SFT training config.",
    )
    parser.add_argument(
        "--noise-level",
        type=float,
        default=0.5,
        help="OpenPI flow_sde sampling noise level. 0.5 matches the RLinf pi0.5 SFT training config.",
    )
    parser.add_argument("--default-prompt", default="")
    parser.add_argument(
        "--log-input-output",
        action="store_true",
        help=(
            "Save each request's OpenPI input images/prompt and returned normalized "
            "actions under deploy/open_pi/log/openpi_<timestamp>."
        ),
    )
    return parser.parse_args()


def main() -> None:
    # Keep OpenPI/transformers from importing TensorFlow in the serving process.
    os.environ.setdefault("USE_TF", "0")
    os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    args = parse_args()
    runner = OpenPIFrankaDualRunner(args)
    app = build_app(runner)

    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    with contextlib.suppress(KeyboardInterrupt):
        main()
