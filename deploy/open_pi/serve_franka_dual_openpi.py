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

The model checkpoint is an RLinf local-shard checkpoint. For this OpenPI SFT
run, rank 0 contains a loadable model state dict, so inference does not need
Ray/FSDP; it loads ``actor/local_shard_checkpoint/checkpoint_rank_0.pt``.
"""


from __future__ import annotations

import argparse
import contextlib
import dataclasses
import os
import sys
import time
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
    "real_world_franka_dual_openpi_pi05_sft/checkpoints/global_step_20000"
)

RAW_HWC_IMAGE_FORMAT = "raw_hwc"
LEGACY_224_CHW_IMAGE_FORMAT = "legacy_224_chw"
OPENPI_CLIENT_PREPROCESSED_FORMAT = "openpi_pi05_client_preprocessed_v1"
OPENPI_IMAGE_KEYS = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
SUPPORTED_IMAGE_FORMATS = {
    RAW_HWC_IMAGE_FORMAT,
    LEGACY_224_CHW_IMAGE_FORMAT,
    "auto",
}


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


def _squeezed_image_shape(image: Any) -> tuple[int, ...]:
    arr = np.asarray(image)
    arr = np.squeeze(arr)
    if arr.ndim == 4 and arr.shape[0] == 1:
        arr = arr[0]
    return tuple(int(x) for x in arr.shape)


def _looks_like_legacy_224_chw(image: Any) -> bool:
    shape = _squeezed_image_shape(image)
    return len(shape) == 3 and shape[0] in (1, 3, 4) and shape[1:] == (224, 224)


def _infer_image_payload_format(*images: Any) -> str:
    if any(_looks_like_legacy_224_chw(image) for image in images):
        return LEGACY_224_CHW_IMAGE_FORMAT
    return RAW_HWC_IMAGE_FORMAT


def _as_hwc_uint8(image: Any) -> np.ndarray:
    arr = np.asarray(image)
    arr = np.squeeze(arr)
    if arr.ndim == 4:
        arr = arr[0]
    if arr.ndim != 3:
        raise ValueError(f"Expected image with 3 dims, got shape={arr.shape}")
    if arr.shape[0] in (1, 3, 4) and arr.shape[-1] not in (1, 3, 4):
        arr = np.transpose(arr, (1, 2, 0))
    if arr.shape[-1] == 4:
        arr = arr[..., :3]
    if arr.shape[-1] != 3:
        raise ValueError(f"Expected RGB image with 3 channels, got shape={arr.shape}")
    if arr.dtype != np.uint8:
        arr_f = arr.astype(np.float32, copy=False)
        if arr_f.size and float(np.nanmax(arr_f)) <= 1.0 + 1e-3:
            arr = np.clip(arr_f * 255.0, 0, 255).astype(np.uint8)
        else:
            arr = np.clip(arr_f, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(arr)


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


def _pad_or_trim_vector(values: Any, dim: int) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32).reshape(-1)
    out = np.zeros((int(dim),), dtype=np.float32)
    n = min(arr.shape[0], int(dim))
    out[:n] = arr[:n]
    return np.ascontiguousarray(out)


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


def _resize_exact(image: np.ndarray, height: int, width: int) -> np.ndarray:
    pil = Image.fromarray(_as_hwc_uint8(image))
    return np.asarray(pil.resize((width, height), resample=Image.Resampling.BILINEAR))


@dataclasses.dataclass(frozen=True)
class CameraImages:
    middle_zed: np.ndarray
    left_camera: np.ndarray
    right_camera: np.ndarray
    stitched: np.ndarray
    payload_format: str


def _get_nested(payload: dict, *keys: str, default: Any = None) -> Any:
    for key in keys:
        if key in payload:
            return payload[key]
    return default


def _declared_image_payload_format(payload: dict) -> str | None:
    declared = _get_nested(
        payload,
        "image_payload_format",
        "image_format",
        "images_format",
        default=None,
    )
    if declared is None:
        return None
    declared_format = str(declared)
    if declared_format not in SUPPORTED_IMAGE_FORMATS - {"auto"}:
        raise ValueError(
            f"Unsupported image_payload_format={declared_format!r}; expected "
            f"{RAW_HWC_IMAGE_FORMAT!r} or {LEGACY_224_CHW_IMAGE_FORMAT!r}"
        )
    return declared_format


def preprocess_images(
    payload: dict,
    *,
    expected_format: str = RAW_HWC_IMAGE_FORMAT,
    allow_legacy_224_chw: bool = False,
) -> CameraImages:
    if expected_format not in SUPPORTED_IMAGE_FORMATS:
        raise ValueError(f"Unsupported server image payload format: {expected_format}")

    images = payload.get("images", {})
    if not isinstance(images, dict):
        raise ValueError("payload['images'] must be a dict")

    middle = _get_nested(
        images,
        "middle_zed",
        "image",
        "base_0_rgb",
        "observation.images.middle_zed",
        default=None,
    )
    left = _get_nested(
        images,
        "left_camera",
        "left_wrist_image",
        "left_wrist_0_rgb",
        "observation.images.left_camera",
        default=None,
    )
    right = _get_nested(
        images,
        "right_camera",
        "right_wrist_image",
        "right_wrist_0_rgb",
        "observation.images.right_camera",
        default=None,
    )
    if middle is None or left is None or right is None:
        raise ValueError(
            "Missing required images. Expected middle_zed, left_camera, right_camera"
        )

    inferred_format = _infer_image_payload_format(middle, left, right)
    declared_format = _declared_image_payload_format(payload)
    if declared_format is not None and declared_format != inferred_format:
        raise ValueError(
            f"Declared image_payload_format {declared_format!r} does not match "
            f"image shapes inferred as {inferred_format!r}. Shapes: "
            f"middle={_squeezed_image_shape(middle)}, "
            f"left={_squeezed_image_shape(left)}, right={_squeezed_image_shape(right)}"
        )

    payload_format = declared_format or inferred_format
    if expected_format != "auto" and payload_format != expected_format:
        if not (payload_format == LEGACY_224_CHW_IMAGE_FORMAT and allow_legacy_224_chw):
            raise ValueError(
                f"Received image payload format {payload_format!r}, but server "
                f"expects {expected_format!r}. Use policy.type=openpi_pi05_client "
                f"or set --policy.image_payload_format={RAW_HWC_IMAGE_FORMAT} on "
                "the robot client. To run the legacy path intentionally, restart "
                "this server with --allow-legacy-224-chw."
            )
    if payload_format == LEGACY_224_CHW_IMAGE_FORMAT and not allow_legacy_224_chw:
        raise ValueError(
            "Received legacy_224_chw images. This is the old client-side 224x224 "
            "resize/CHW transport and is not the training-matched OpenPI pi0.5 "
            "wire format. Use policy.type=openpi_pi05_client or set "
            "--policy.image_payload_format=raw_hwc on the robot client. To run "
            "the legacy path intentionally, restart this server with "
            "--allow-legacy-224-chw."
        )

    middle_hwc = _as_hwc_uint8(middle)
    left_hwc = _as_hwc_uint8(left)
    right_hwc = _as_hwc_uint8(right)

    # Optional/debug Franka deployment layout:
    # top: middle ZED 128x256; bottom: left/right wrist cameras 128x128 each.
    # The default model path below does not use this stitched image, because
    # RLinf training feeds OpenPI three separate views and lets OpenPI's model
    # transforms do their own resize/pad.
    middle_128x256 = _resize_exact(middle_hwc, 128, 256)
    left_128x128 = _resize_exact(left_hwc, 128, 128)
    right_128x128 = _resize_exact(right_hwc, 128, 128)
    wrist_row = np.concatenate([left_128x128, right_128x128], axis=1)
    stitched = np.concatenate([middle_128x256, wrist_row], axis=0)
    return CameraImages(
        middle_zed=middle_hwc,
        left_camera=left_hwc,
        right_camera=right_hwc,
        stitched=np.ascontiguousarray(stitched),
        payload_format=payload_format,
    )

def preprocess_state(payload: dict, action_dim: int = 16) -> np.ndarray:
    state = _get_nested(payload, "state", "observation.state", default=None)
    if state is None and "observation" in payload and isinstance(payload["observation"], dict):
        obs = payload["observation"]
        left = [obs.get(f"left_joint_positions_{i}", 0.0) for i in range(8)]
        right = [obs.get(f"right_joint_positions_{i}", 0.0) for i in range(8)]
        state = np.asarray(left + right, dtype=np.float32)
    if state is None:
        raise ValueError("Missing state. Expected payload['state'] with 16 values")
    arr = np.asarray(state, dtype=np.float32).reshape(-1)
    if arr.shape[0] < action_dim:
        arr = np.pad(arr, (0, action_dim - arr.shape[0]))
    return np.ascontiguousarray(arr[:action_dim], dtype=np.float32)


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


class OpenPIFrankaDualRunner:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.device = torch.device(args.device)
        self.n_action_steps = int(args.n_action_steps)
        self.action_dim = int(args.return_action_dim)
        self.feed_stitched_as_base = bool(args.feed_stitched_as_base)
        self.image_payload_format = str(args.image_payload_format)
        self.allow_legacy_224_chw = bool(args.allow_legacy_224_chw)
        self.use_client_preprocessed_payload = bool(args.use_client_preprocessed_payload)
        self._warned_legacy_image_payload = False

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
                    "repo_id": "real_world_franka_dual",
                    "norm_stats_key": "real_world_joint",
                    "detach_critic_input": True,
                    "num_images_in_input": 3,
                    "train_expert_only": True,
                    "action_chunk": 48,
                    "action_env_dim": 32,
                },
            }
        )

    def _load_model(self):
        from rlinf.models.embodiment.openpi import get_model

        model = get_model(self._base_cfg(), torch_dtype=self._torch_dtype())
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
        state = _pad_or_trim_vector(openpi.get("state"), state_dim)[None]
        processed = {
            "image": image,
            "image_mask": image_mask,
            "state": state,
            "prompt": prompt,
        }
        processed = self._tokenize_preprocessed_prompt(processed)
        processed = _torchify_tree(processed)
        processed = self.model.precision_processor(processed)
        return _model.Observation.from_dict(processed)

    @torch.inference_mode()
    def infer(self, payload: dict) -> np.ndarray:
        prompt = str(payload.get("prompt") or payload.get("task") or self.args.default_prompt)
        observation = None
        if self.use_client_preprocessed_payload:
            observation = self._observation_from_client_preprocessed(payload, prompt)

        if observation is None:
            state = preprocess_state(payload, action_dim=16)
            images = preprocess_images(
                payload,
                expected_format=self.image_payload_format,
                allow_legacy_224_chw=self.allow_legacy_224_chw,
            )
            if (
                images.payload_format == LEGACY_224_CHW_IMAGE_FORMAT
                and not self._warned_legacy_image_payload
            ):
                print(
                    "[open_pi] WARNING: serving legacy_224_chw images. This is a "
                    "compatibility path; raw_hwc matches OpenPI pi0.5 training better.",
                    flush=True,
                )
                self._warned_legacy_image_payload = True

            base_image = images.stitched if self.feed_stitched_as_base else images.middle_zed
            raw_obs = {
                "observation/image": base_image[None],
                "observation/left_wrist_image": images.left_camera[None],
                "observation/right_wrist_image": images.right_camera[None],
                "observation/image_mask": np.asarray([True]),
                "observation/left_wrist_image_mask": np.asarray([True]),
                "observation/right_wrist_image_mask": np.asarray([True]),
                "observation/state": state[None],
                "prompt": [prompt],
            }

            processed = self.model.input_transform(raw_obs, transpose=False)
            processed = self.model.precision_processor(processed)

            from openpi.models import model as _model

            observation = _model.Observation.from_dict(processed)

        outputs = self.model.sample_actions(
            observation,
            mode="eval",
            compute_values=False,
        )
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
            "feed_stitched_as_base": runner.feed_stitched_as_base,
            "image_payload_format": runner.image_payload_format,
            "allow_legacy_224_chw": runner.allow_legacy_224_chw,
            "use_client_preprocessed_payload": runner.use_client_preprocessed_payload,
            "returns_normalized_actions": True,
        }

    @app.post("/infer")
    async def infer(request: Request) -> Response:
        started = time.perf_counter()
        try:
            payload = unpackb(await request.body())
            if not isinstance(payload, dict):
                raise ValueError(f"request payload must be a dict, got {type(payload)}")
            actions = runner.infer(payload)
            body = packb({"actions": actions})
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            headers = {"X-Inference-Time-Ms": f"{elapsed_ms:.2f}"}
            return Response(
                content=body,
                media_type="application/msgpack",
                headers=headers,
            )
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--precision", default="bfloat16")
    parser.add_argument("--base-model-path", type=Path, default=DEFAULT_BASE_MODEL)
    parser.add_argument("--checkpoint-path", type=Path, default=DEFAULT_CKPT)
    parser.add_argument("--checkpoint-rank", type=int, default=0)
    parser.add_argument("--n-action-steps", type=int, default=48)
    parser.add_argument("--return-action-dim", type=int, default=16)
    parser.add_argument("--default-prompt", default="")
    parser.add_argument(
        "--image-payload-format",
        choices=sorted(SUPPORTED_IMAGE_FORMATS),
        default=RAW_HWC_IMAGE_FORMAT,
        help=(
            "Expected image transport. raw_hwc matches RLinf OpenPI pi0.5 "
            "training; legacy_224_chw is only for old franka_rdk clients."
        ),
    )
    parser.add_argument(
        "--allow-legacy-224-chw",
        action="store_true",
        help=(
            "Allow old client-side 224x224 CHW image payloads. Prefer raw_hwc "
            "for training-matched pi0.5 deployment."
        ),
    )
    parser.add_argument(
        "--ignore-client-preprocessed-payload",
        dest="use_client_preprocessed_payload",
        action="store_false",
        help=(
            "Ignore payload[openpi] and use the old raw image/state path. "
            "The default uses client-side q01/q99 state normalization and "
            "224x224 CHW image preprocessing from openpi_pi05_client."
        ),
    )
    parser.set_defaults(use_client_preprocessed_payload=True)
    parser.add_argument(
        "--feed-stitched-as-base",
        action="store_true",
        help=(
            "Use the 256x256 stitched image as observation/image. By default the "
            "model receives the three views separately, matching RLinf training."
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
