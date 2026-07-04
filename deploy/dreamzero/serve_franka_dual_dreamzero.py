#!/usr/bin/env python
"""HTTP deployment server for Franka-dual DreamZero/WMAM checkpoints.

The wire protocol matches the LeRobot clients under
``lerobot.policies.dreamzero_client`` and ``lerobot.policies.wmam_client``:

* the robot client performs camera tiling, state q01/q99 normalization, padding,
  and final action q01/q99 unnormalization;
* this server consumes only the preprocessed block, runs the model in normalized
  model space, and returns normalized actions ``[T, 16]``;
* there is no raw-image/raw-state fallback, so double preprocessing cannot hide.
"""

from __future__ import annotations

import argparse
import contextlib
import datetime as _dt
import json
import os
import pickle
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import msgpack
import numpy as np
import torch
import torch.distributed as dist
from einops import rearrange
from fastapi import FastAPI, HTTPException, Request, Response
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DREAMZERO_ROOT = Path("/inspire/hdd/project/robot-body/linbokai-CZXS24250037/dreamzero")
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(DEFAULT_DREAMZERO_ROOT) not in sys.path:
    sys.path.insert(0, str(DEFAULT_DREAMZERO_ROOT))

NEGATIVE_PROMPT = (
    "Vibrant colors, overexposed, static, blurry details, text, subtitles, style, artwork, painting, "
    "image, still, grayscale, dull, worst quality, low quality, JPEG artifacts, ugly, mutilated, "
    "extra fingers, bad hands, bad face, deformed, disfigured, mutated limbs, fused fingers, "
    "stagnant image, cluttered background, three legs, many people in the background, walking backwards."
)
LAYOUT_TEXT = (
    " The video is split into three views: the top view is the agent view, "
    "the bottom-left view is the left wrist camera, and the bottom-right view is the right wrist camera. The robot "
)
CLIENT_PREPROCESSED_FORMAT = "dreamzero_franka_dual_client_preprocessed_v1"


def _pack_array(obj: Any):
    if isinstance(obj, np.ndarray):
        if obj.dtype.kind in ("V", "O", "c"):
            raise ValueError(f"Unsupported ndarray dtype: {obj.dtype}")
        return {b"__ndarray__": True, b"data": obj.tobytes(), b"dtype": obj.dtype.str, b"shape": obj.shape}
    if isinstance(obj, np.generic):
        return {b"__npgeneric__": True, b"data": obj.item(), b"dtype": obj.dtype.str}
    return obj


def _unpack_array(obj: dict):
    if b"__ndarray__" in obj:
        return np.ndarray(buffer=obj[b"data"], dtype=np.dtype(obj[b"dtype"]), shape=obj[b"shape"])
    if b"__npgeneric__" in obj:
        return np.dtype(obj[b"dtype"]).type(obj[b"data"])
    return obj


def packb(data: Any) -> bytes:
    return msgpack.packb(data, default=_pack_array, use_bin_type=True)


def unpackb(data: bytes) -> Any:
    return msgpack.unpackb(data, object_hook=_unpack_array, raw=False)


def _format_prompt(prompt: str) -> str:
    instruction = str(prompt or "").lower()
    return "A multi-view video shows that a robot " + instruction + LAYOUT_TEXT + instruction


def _as_hwc_uint8_frames(value: Any, *, name: str) -> np.ndarray:
    arr = np.asarray(value)
    arr = np.squeeze(arr)
    if arr.ndim == 3:
        arr = arr[None]
    if arr.ndim != 4:
        raise ValueError(f"{name} must be [T,H,W,C] or [H,W,C], got shape={arr.shape}")
    if arr.shape[-1] not in (1, 3, 4):
        raise ValueError(f"{name} must be HWC RGB/RGBA, got shape={arr.shape}")
    arr = arr[..., :3]
    if arr.dtype != np.uint8:
        arr_f = arr.astype(np.float32, copy=False)
        if arr_f.size and float(np.nanmax(arr_f)) <= 1.0 + 1e-3:
            arr_f = arr_f * 255.0
        arr = np.clip(arr_f, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(arr)


def _pad_or_trim(values: Any, dim: int, *, dtype=np.float32) -> np.ndarray:
    arr = np.asarray(values, dtype=dtype).reshape(-1)
    out = np.zeros((int(dim),), dtype=dtype)
    n = min(arr.shape[0], int(dim))
    out[:n] = arr[:n]
    return np.ascontiguousarray(out)


def _require_preprocessed_block(payload: dict[str, Any], key: str) -> dict[str, Any]:
    block = payload.get(key)
    if not isinstance(block, dict):
        raise ValueError(f"Missing payload[{key!r}] preprocessed block")
    if not bool(block.get("preprocessed", False)):
        raise ValueError(f"payload[{key!r}] must be marked preprocessed=true")
    fmt = str(block.get("format", ""))
    if fmt and fmt != CLIENT_PREPROCESSED_FORMAT:
        raise ValueError(f"Unsupported payload[{key!r}] format {fmt!r}; expected {CLIENT_PREPROCESSED_FORMAT!r}")
    return block


def _save_mp4(path: Path, frames: np.ndarray, fps: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(path, list(frames), fps=fps, codec="libx264")


def _decode_video_pred(
    policy: Any,
    video_pred: Any,
    *,
    anchor_latent: torch.Tensor | None = None,
    future_frame_count: int = 8,
) -> tuple[np.ndarray | None, dict[str, Any]]:
    info: dict[str, Any] = {
        "requested_future_frame_count": int(future_frame_count),
        "anchor_latent_used": False,
        "dropped_first_anchor_frame": False,
    }
    if video_pred is None:
        info["error"] = "video_pred is None"
        return None, info
    try:
        vp = video_pred
        if isinstance(vp, (list, tuple)):
            vp = torch.cat(list(vp), dim=2)
        if not torch.is_tensor(vp):
            raise TypeError(f"video_pred must be a tensor/list of tensors, got {type(vp)!r}")
        info["video_pred_latent_shape"] = list(vp.shape)
        decode_latents = vp.detach().clone()
        if anchor_latent is not None:
            info["anchor_latent_shape"] = list(anchor_latent.shape)
        if vp.shape[2] == 2 and anchor_latent is not None:
            anchor = anchor_latent.detach().to(device=vp.device, dtype=vp.dtype).clone()
            decode_latents = torch.cat([anchor, decode_latents], dim=2)
            info["anchor_latent_used"] = True
        elif vp.shape[2] >= 3:
            info["video_pred_includes_anchor"] = True
        else:
            info["video_pred_includes_anchor"] = False
        info["decoded_latent_shape"] = list(decode_latents.shape)

        ah = policy.trained_model.action_head
        with torch.inference_mode():
            frames = ah.vae.decode(
                decode_latents,
                tiled=ah.tiled,
                tile_size=(ah.tile_size_height, ah.tile_size_width),
                tile_stride=(ah.tile_stride_height, ah.tile_stride_width),
            )
        frames = rearrange(frames, "B C T H W -> B T H W C")[0]
        frames_np = ((frames.float() + 1.0) * 127.5).clip(0, 255).cpu().numpy().astype(np.uint8)
        info["full_decoded_video_shape"] = list(frames_np.shape)
        if frames_np.shape[0] >= int(future_frame_count) + 1:
            frames_np = frames_np[1 : 1 + int(future_frame_count)]
            info["dropped_first_anchor_frame"] = True
        info["saved_future_video_shape"] = list(frames_np.shape)
        return frames_np, info
    except Exception as exc:
        info["error"] = str(exc)
        print(f"[dreamzero] failed to decode predicted video: {exc}", flush=True)
        traceback.print_exc()
        return None, info


def _motion_to_numpy(value: Any) -> Any:
    if value is None:
        return None
    if torch.is_tensor(value):
        arr = value.detach().float().cpu().numpy()
        if arr.ndim == 5 and arr.shape[0] == 1:
            arr = arr[0]
        return arr
    if isinstance(value, dict):
        return {key: _motion_to_numpy(sub) for key, sub in value.items()}
    return np.asarray(value)


def _save_motion_npz(path: Path, motion_pred: Any, motion_decoded: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    decoded = _motion_to_numpy(motion_decoded)
    if isinstance(decoded, dict) and "point_map" in decoded and "scene_flow" in decoded:
        np.savez(path, point_map=decoded["point_map"], scene_flow=decoded["scene_flow"])
        return
    latent = _motion_to_numpy(motion_pred)
    if latent is not None:
        np.savez(path, motion_latent=latent)


def _normalized_action_stats(actions: np.ndarray) -> dict[str, Any]:
    arr = np.asarray(actions, dtype=np.float32)
    if arr.size == 0:
        return {
            "min": 0.0,
            "max": 0.0,
            "mean": 0.0,
            "std": 0.0,
            "abs_gt_1_fraction": 0.0,
            "abs_gt_2_fraction": 0.0,
        }
    return {
        "min": float(arr.min()),
        "max": float(arr.max()),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "abs_gt_1_fraction": float((np.abs(arr) > 1.0).mean()),
        "abs_gt_2_fraction": float((np.abs(arr) > 2.0).mean()),
    }


def init_mesh() -> DeviceMesh:
    if not dist.is_initialized():
        dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank)
    print(f"[dreamzero] rank {rank}/{world_size} using cuda:{rank}", flush=True)
    return init_device_mesh(device_type="cuda", mesh_shape=(world_size,), mesh_dim_names=("ip",))


def _make_dit_step_mask(num_steps: int, num_dit_steps: int) -> list[bool]:
    num_steps = max(1, int(num_steps))
    num_dit_steps = max(1, min(int(num_dit_steps), num_steps))
    if num_dit_steps >= num_steps:
        return [True] * num_steps
    if num_dit_steps == 1:
        mask = [False] * num_steps
        mask[0] = True
        return mask

    positions = {round(i * (num_steps - 1) / (num_dit_steps - 1)) for i in range(num_dit_steps)}
    positions.add(0)
    positions.add(num_steps - 1)
    return [idx in positions for idx in range(num_steps)]


def _configure_action_head_sampling(action_head: Any, args: argparse.Namespace) -> dict[str, Any]:
    checkpoint_steps = int(getattr(action_head, "num_inference_timesteps", 0) or 0)
    previous_steps = int(getattr(action_head, "num_inference_steps", 0) or 0)
    requested_steps = int(args.num_inference_steps or checkpoint_steps or previous_steps or 16)
    if requested_steps <= 0:
        raise ValueError(f"Invalid DreamZero num_inference_steps={requested_steps}")

    requested_dit_steps = int(args.num_dit_steps or requested_steps)
    requested_dit_steps = max(1, min(requested_dit_steps, requested_steps))
    action_head.num_inference_steps = requested_steps
    action_head.dit_step_mask = _make_dit_step_mask(requested_steps, requested_dit_steps)

    metadata = {
        "checkpoint_num_inference_timesteps": checkpoint_steps,
        "previous_num_inference_steps": previous_steps,
        "num_inference_steps": requested_steps,
        "num_dit_steps": int(sum(action_head.dit_step_mask)),
        "dit_step_mask": list(action_head.dit_step_mask),
    }
    print(f"[dreamzero] action sampling: {metadata}", flush=True)
    return metadata


class FrankaDualDreamZeroRunner:
    def __init__(self, args: argparse.Namespace, *, model_kind: str, payload_key: str):
        self.args = args
        self.model_kind = model_kind
        self.payload_key = payload_key
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.output_root = Path(args.output_root)
        self.request_index = 0
        self.signal_group = dist.new_group(backend="gloo")

        if not str(args.model_path).strip():
            raise ValueError("--model-path is empty. Fill it with the converted DreamZero/WMAM checkpoint path.")

        from groot.vla.data.schema.embodiment_tags import EmbodimentTag
        from groot.vla.model.n1_5.sim_policy import GrootSimPolicy

        model_overrides = list(args.model_config_override or [])
        if args.decode_motion:
            model_overrides.append("action_head_cfg.config.decode_motion_on_inference=true")
        self.policy = GrootSimPolicy(
            embodiment_tag=EmbodimentTag(args.embodiment_tag),
            model_path=str(args.model_path),
            device="cuda" if torch.cuda.is_available() else "cpu",
            model_config_overrides=model_overrides,
            device_mesh=args.device_mesh,
        )
        self.sampling_metadata = _configure_action_head_sampling(self.policy.trained_model.action_head, args)
        self.tokenizer = getattr(self.policy.eval_transform, "tokenizer", None)
        if self.tokenizer is None:
            from groot.vla.model.dreamzero.transform.dreamzero_cotrain import HuggingfaceTokenizer

            self.tokenizer = HuggingfaceTokenizer(
                name=args.tokenizer_path,
                seq_len=args.max_seq_len,
                clean="whitespace",
                local_files_only=True,
            )

    def _broadcast_signal(self, signal: int) -> None:
        if self.world_size <= 1:
            return
        tensor = torch.tensor([int(signal)], dtype=torch.int32, device="cpu")
        dist.broadcast(tensor, src=0, group=self.signal_group)

    def _broadcast_object(self, obj: Any) -> None:
        if self.world_size <= 1:
            return
        data = pickle.dumps(obj)
        size_tensor = torch.tensor([len(data)], dtype=torch.int64, device="cuda")
        dist.broadcast(size_tensor, src=0)
        data_tensor = torch.frombuffer(memoryview(data), dtype=torch.uint8).to("cuda")
        dist.broadcast(data_tensor, src=0)

    def _receive_object(self) -> Any:
        size_tensor = torch.zeros(1, dtype=torch.int64, device="cuda")
        dist.broadcast(size_tensor, src=0)
        data_tensor = torch.zeros(int(size_tensor.item()), dtype=torch.uint8, device="cuda")
        dist.broadcast(data_tensor, src=0)
        return pickle.loads(data_tensor.cpu().numpy().tobytes())

    def _tokenize(self, text: str) -> tuple[torch.Tensor, torch.Tensor]:
        ids, mask = self.tokenizer([text], return_mask=True, add_special_tokens=True)
        return ids, mask

    def _select_model_frames(self, block: dict[str, Any]) -> tuple[np.ndarray, str]:
        if "executed_tiled_images" in block:
            frames = _as_hwc_uint8_frames(block["executed_tiled_images"], name="executed_tiled_images")
            if frames.shape[0] > 0:
                return frames, "executed_tiled_images"
        if "tiled_image" in block:
            return _as_hwc_uint8_frames(block["tiled_image"], name="tiled_image"), "tiled_image"
        if "images" in block:
            frames = _as_hwc_uint8_frames(block["images"], name="images")
            return frames[-1:], "images_last_frame"
        raise ValueError(f"payload[{self.payload_key!r}] must include tiled_image or images")

    def build_model_input(self, payload: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        block = _require_preprocessed_block(payload, self.payload_key)
        prompt = str(payload.get("prompt") or payload.get("task") or self.args.default_prompt)
        frames, frame_source = self._select_model_frames(block)

        raw_state = np.asarray(block.get("state"), dtype=np.float32).reshape(-1)
        if raw_state.size == 0:
            raise ValueError(f"payload[{self.payload_key!r}][state] is required")
        state = _pad_or_trim(raw_state, int(self.args.model_state_dim), dtype=np.float32)
        state_raw_payload = block.get("state_raw")
        state_raw = None
        if state_raw_payload is not None:
            state_raw = np.asarray(state_raw_payload, dtype=np.float32).reshape(-1)

        mask_payload = block.get("state_mask")
        if mask_payload is None:
            raise ValueError(f"payload[{self.payload_key!r}][state_mask] is required")
        state_mask = _pad_or_trim(mask_payload, int(self.args.model_state_dim), dtype=bool)
        if not bool(np.any(state_mask)):
            raise ValueError(f"payload[{self.payload_key!r}][state_mask] must contain at least one true entry")

        action_mask_payload = block.get("action_mask")
        if action_mask_payload is None:
            raise ValueError(f"payload[{self.payload_key!r}][action_mask] is required")
        action_mask_vec = _pad_or_trim(action_mask_payload, int(self.args.model_action_dim), dtype=bool)
        if not bool(np.any(action_mask_vec)):
            raise ValueError(f"payload[{self.payload_key!r}][action_mask] must contain at least one true entry")
        action_mask = np.broadcast_to(
            action_mask_vec[None, :],
            (int(self.args.n_action_steps), int(self.args.model_action_dim)),
        ).copy()

        positive, positive_mask = self._tokenize(_format_prompt(prompt))
        negative, negative_mask = self._tokenize(NEGATIVE_PROMPT)

        model_input = {
            "images": torch.from_numpy(frames[None]),
            "text": positive,
            "text_attention_mask": positive_mask,
            "text_negative": negative,
            "text_attention_mask_negative": negative_mask,
            "state": torch.from_numpy(state[None, None, :]),
            "state_mask": torch.from_numpy(state_mask[None, None, :]),
            "embodiment_id": torch.tensor([int(block.get("embodiment_id", self.args.embodiment_id))], dtype=torch.long),
            "has_lapa_action": torch.zeros((1,), dtype=torch.bool),
            "is_cotrain_instance": torch.zeros((1,), dtype=torch.bool),
        }
        model_input["action_mask"] = torch.from_numpy(action_mask[None])
        metadata = {
            "prompt": prompt,
            "formatted_prompt": _format_prompt(prompt),
            "payload_key": self.payload_key,
            "frame_source": frame_source,
            "frame_shape": list(frames.shape),
            "state_shape": list(state.shape),
            "state_true_dims": int(np.asarray(state_mask, dtype=bool).sum()),
            "state_normalized_first16": [float(x) for x in state[:16]],
            "state_normalized_stats": _normalized_action_stats(state),
            "state_raw_first16": None if state_raw is None else [float(x) for x in state_raw[:16]],
            "state_raw_stats": None if state_raw is None else _normalized_action_stats(state_raw),
            "action_true_dims": int(np.asarray(action_mask[0], dtype=bool).sum()),
            "returns_normalized_actions": True,
            "action_sampling": dict(self.sampling_metadata),
            "action_inference_mode": str(getattr(self.args, "action_inference_mode", "lazy")),
        }
        rollout_feedback = block.get("rollout_feedback") or payload.get("rollout_feedback")
        if isinstance(rollout_feedback, dict):
            metadata["rollout_feedback"] = rollout_feedback
        return model_input, metadata

    def _prepare_video_tensor_for_action_tf(self, frames: torch.Tensor) -> torch.Tensor:
        ah = self.policy.trained_model.action_head
        videos = rearrange(frames, "b t h w c -> b c t h w")
        if videos.dtype == torch.uint8:
            videos = videos.float() / 255.0
            b, c, t, h, w = videos.shape
            videos = videos.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
            videos = ah.normalize_video(videos)
            videos = videos.reshape(b, t, c, h, w).permute(0, 2, 1, 3, 4)
        else:
            videos = videos.to(dtype=torch.float32)
            if videos.numel() and float(videos.max()) > 1.5:
                videos = videos / 255.0 * 2.0 - 1.0
            elif videos.numel() and float(videos.min()) >= -1e-3:
                videos = videos * 2.0 - 1.0
        videos = videos.to(device=ah.device, dtype=torch.bfloat16)
        target_h = getattr(ah.config, "target_video_height", None)
        target_w = getattr(ah.config, "target_video_width", None)
        if target_h is not None and target_w is not None:
            _, _, _, h, w = videos.shape
            if (h, w) != (target_h, target_w):
                b, c, t, _, _ = videos.shape
                videos = torch.nn.functional.interpolate(
                    videos.reshape(b * t, c, h, w),
                    size=(target_h, target_w),
                    mode="bilinear",
                    align_corners=False,
                ).reshape(b, c, t, target_h, target_w)
        return videos

    def _encode_input_anchor_latent(self, action_input: Any) -> torch.Tensor | None:
        frames = action_input["images"]
        if frames.shape[1] < 2:
            return None
        ah = self.policy.trained_model.action_head
        videos = self._prepare_video_tensor_for_action_tf(frames).detach().clone()
        if videos.shape[2] == 8:
            videos = torch.cat([videos[:, :, :1], videos], dim=2)
        elif videos.shape[2] > 9:
            videos = videos[:, :, -9:]
        with torch.inference_mode():
            latents = ah.encode_video(
                videos,
                ah.tiled,
                (ah.tile_size_height, ah.tile_size_width),
                (ah.tile_stride_height, ah.tile_stride_width),
            )
        return latents[:, :, -1:].detach().clone().contiguous()

    def _clean_latents_for_action_tf(self, video_pred: Any, action_input: Any) -> torch.Tensor:
        if video_pred is None:
            raise RuntimeError("tf_video_unipc action mode requires video_pred from lazy DreamZero inference")
        if isinstance(video_pred, (list, tuple)):
            clean_latents = torch.cat(list(video_pred), dim=2)
        else:
            clean_latents = video_pred
        clean_latents = clean_latents.to(device=self.policy.trained_model.action_head.device, dtype=torch.bfloat16)
        if clean_latents.shape[2] >= 3:
            return clean_latents[:, :, :3].contiguous()
        if clean_latents.shape[2] == 2:
            anchor = self._encode_input_anchor_latent(action_input)
            if anchor is not None:
                return torch.cat([anchor, clean_latents], dim=2).contiguous()
        raise RuntimeError(f"tf_video_unipc expected 2 or 3 latent frames, got shape={tuple(clean_latents.shape)}")

    @torch.inference_mode()
    def _infer_actions_tf_video_unipc(self, model_input: dict[str, Any], video_pred: Any) -> torch.Tensor:
        """Sample actions with clean/generated video context using DreamZero's UniPC action scheduler.

        This is a diagnostic/deployment action path for the real-world setting where the video branch
        looks correct but the lazy joint action samples saturate. It keeps the checkpoint inference
        step count unchanged and mirrors the original lazy action scheduler, while feeding the action
        branch the clean generated video latents through the same clean_x argument used during training.
        """

        from groot.vla.model.dreamzero.modules.flow_unipc_multistep_scheduler import FlowUniPCMultistepScheduler

        ah = self.policy.trained_model.action_head
        _, action_input = self.policy.trained_model.prepare_input(model_input)
        clean_latents = self._clean_latents_for_action_tf(video_pred, action_input)
        batch_size, _, latent_frames, _, _ = clean_latents.shape
        if latent_frames != 3:
            raise RuntimeError(f"tf_video_unipc expects 3 latent frames for one 48-action block, got {latent_frames}")

        images = action_input["images"]
        first_image = self._prepare_video_tensor_for_action_tf(images[:, :1]).transpose(1, 2)
        _, _, _, height, width = first_image.shape
        clip_feas, ys, _ = ah.encode_image(first_image, ah.num_frames, height, width)
        prompt_embs = [
            ah.encode_prompt(text, attention_mask)
            for text, attention_mask in ah._prepare_text_inputs(action_input)
        ]
        prompt_emb = prompt_embs[0]

        device = clean_latents.device
        dtype = torch.bfloat16
        state = action_input.state.to(device=device, dtype=dtype)
        embodiment_id = action_input.embodiment_id.to(device=device)
        action_mask = action_input.action_mask.to(device=device, dtype=torch.bool)
        if action_mask.ndim == 2:
            action_mask = action_mask[None]
        padded_action_mask = torch.zeros((batch_size, ah.action_horizon, ah.model.action_dim), device=device, dtype=torch.bool)
        tdim = min(padded_action_mask.shape[1], action_mask.shape[1])
        ddim = min(padded_action_mask.shape[2], action_mask.shape[2])
        padded_action_mask[:, :tdim, :ddim] = action_mask[:, :tdim, :ddim]

        sample_scheduler_action = FlowUniPCMultistepScheduler(
            num_train_timesteps=ah.scheduler.num_train_timesteps,
            shift=1,
            use_dynamic_shifting=False,
        )
        sample_scheduler_action.set_timesteps(ah.num_inference_steps, device=device, shift=ah.sigma_shift)

        action = ah.generate_noise(
            (batch_size, ah.action_horizon, ah.model.action_dim),
            seed=ah.seed,
            device=str(device),
            dtype=dtype,
        ).masked_fill(~padded_action_mask, 0.0)
        video_noise = ah.generate_noise(tuple(clean_latents.shape), seed=ah.seed, device=str(device), dtype=dtype)
        clean_btchw = clean_latents.transpose(1, 2).contiguous()
        noise_btchw = video_noise.transpose(1, 2).contiguous()
        seq_len = int(latent_frames * ah.model.frame_seqlen)
        y_block = ys[:, :, :latent_frames] if ys is not None else None

        for step_index, action_timestep in enumerate(sample_scheduler_action.timesteps):
            timestep_video = torch.zeros((batch_size, latent_frames), device=device, dtype=action_timestep.dtype)
            timestep_video[:, 1:] = action_timestep
            timestep_action = torch.ones(
                (batch_size, ah.action_horizon), device=device, dtype=action_timestep.dtype
            ) * action_timestep
            noisy_btchw = ah.scheduler.add_noise(
                clean_btchw.flatten(0, 1),
                noise_btchw.flatten(0, 1),
                timestep_video.flatten(0, 1),
            ).unflatten(0, (batch_size, latent_frames))
            noisy_latents = noisy_btchw.transpose(1, 2).contiguous()
            with torch.amp.autocast(dtype=torch.bfloat16, device_type=torch.device(device).type):
                _, action_flow, _ = ah.model(
                    noisy_latents,
                    timestep=timestep_video,
                    clip_feature=clip_feas,
                    y=y_block,
                    context=prompt_emb,
                    seq_len=seq_len,
                    state=state,
                    embodiment_id=embodiment_id,
                    action=action,
                    timestep_action=timestep_action,
                    clean_x=clean_latents,
                )
            action = sample_scheduler_action.step(
                model_output=action_flow,
                timestep=action_timestep,
                sample=action,
                step_index=step_index,
                return_dict=False,
            )[0]
            action = action.masked_fill(~padded_action_mask, 0.0)
        return action

    @torch.inference_mode()
    def infer_model(self, model_input: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, Any, Any, Any, dict[str, Any]]:
        action_head = self.policy.trained_model.action_head
        cache_before = int(getattr(action_head, "current_start_frame", -1))
        local_attn_size = int(getattr(getattr(action_head, "model", None), "local_attn_size", -1))
        num_frame_per_block = int(getattr(getattr(action_head, "model", None), "num_frame_per_block", -1))
        input_num_frames = int(model_input["images"].shape[1])

        self._broadcast_signal(0)
        self._broadcast_object(model_input)
        dist.barrier()
        model_pred = self.policy.trained_model.lazy_joint_video_action_causal(model_input)
        dist.barrier()

        cache_after = int(getattr(action_head, "current_start_frame", -1))
        cache_reset_inferred = bool(
            cache_before == 0
            or input_num_frames == 1
            or (local_attn_size != -1 and cache_before >= local_attn_size)
            or (cache_after >= 0 and cache_before > cache_after)
        )
        video_cache_metadata = {
            "current_start_frame_before": cache_before,
            "current_start_frame_after": cache_after,
            "local_attn_size": local_attn_size,
            "num_frame_per_block": num_frame_per_block,
            "input_num_frames": input_num_frames,
            "reset_inferred": cache_reset_inferred,
        }
        video_pred = model_pred.get("video_pred")
        motion_pred = model_pred.get("motion_pred")
        motion_decoded = model_pred.get("motion_decoded")

        action_mode = str(getattr(self.args, "action_inference_mode", "lazy"))
        if action_mode == "tf_video_unipc":
            action_tensor = self._infer_actions_tf_video_unipc(model_input, video_pred)
            action_pred = action_tensor.detach().float().cpu().numpy()
        elif action_mode == "lazy":
            action_pred = model_pred["action_pred"].detach().float().cpu().numpy()
        else:
            raise ValueError(f"Unsupported action_inference_mode={action_mode!r}")

        raw_actions = np.ascontiguousarray(
            action_pred[0, : int(self.args.n_action_steps), : int(self.args.return_action_dim)],
            dtype=np.float32,
        )
        actions = np.ascontiguousarray(np.clip(raw_actions, -1.0, 1.0), dtype=np.float32)
        return actions, raw_actions, video_pred, motion_pred, motion_decoded, video_cache_metadata

    def _save_request_artifacts(
        self,
        *,
        model_input: dict[str, Any],
        metadata: dict[str, Any],
        actions: np.ndarray,
        raw_actions: np.ndarray,
        video_pred: Any,
        motion_pred: Any,
        motion_decoded: Any,
    ) -> Path:
        timestamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        save_dir = self.output_root / f"{self.model_kind}_{timestamp}_{self.request_index:06d}"
        self.request_index += 1
        save_dir.mkdir(parents=True, exist_ok=True)

        gt_frames = model_input["images"][0].detach().cpu().numpy().astype(np.uint8)
        _save_mp4(save_dir / "gt_video.mp4", gt_frames, int(self.args.save_fps))
        np.save(save_dir / "normalized_actions.npy", actions)
        np.save(save_dir / "normalized_actions_raw.npy", raw_actions)
        np.save(save_dir / "normalized_state.npy", model_input["state"].detach().cpu().numpy())
        np.save(save_dir / "state_mask.npy", model_input["state_mask"].detach().cpu().numpy())
        raw_state_values = metadata.get("state_raw_first16")
        if raw_state_values is not None:
            np.save(save_dir / "state_raw.npy", np.asarray(raw_state_values, dtype=np.float32))

        anchor_latent = None
        needs_anchor_latent = False
        if video_pred is not None:
            try:
                vp_for_shape = torch.cat(list(video_pred), dim=2) if isinstance(video_pred, (list, tuple)) else video_pred
                needs_anchor_latent = torch.is_tensor(vp_for_shape) and vp_for_shape.shape[2] == 2
            except Exception:
                needs_anchor_latent = False
        if needs_anchor_latent:
            try:
                anchor_latent = self._encode_input_anchor_latent(model_input)
            except Exception as exc:
                metadata.setdefault("pred_video_decode", {})["anchor_encode_error"] = str(exc)
                print(f"[dreamzero] failed to encode input anchor latent for pred_video logging: {exc}", flush=True)
                traceback.print_exc()
        pred_frames, pred_video_decode = _decode_video_pred(
            self.policy,
            video_pred,
            anchor_latent=anchor_latent,
            future_frame_count=8,
        )
        metadata["pred_video_decode"] = {**metadata.get("pred_video_decode", {}), **pred_video_decode}
        if pred_frames is not None:
            _save_mp4(save_dir / "pred_video.mp4", pred_frames, int(self.args.save_fps))
            metadata["pred_video_shape"] = list(pred_frames.shape)
        if self.model_kind == "wmam":
            _save_motion_npz(save_dir / "motion_pred.npz", motion_pred, motion_decoded)

        metadata["normalized_action_shape"] = list(actions.shape)
        metadata["normalized_action_raw_stats"] = _normalized_action_stats(raw_actions)
        metadata["normalized_action_returned_stats"] = _normalized_action_stats(actions)
        metadata["normalized_action_clipped"] = True
        (save_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        return save_dir

    def infer(self, payload: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
        model_input, metadata = self.build_model_input(payload)
        actions, raw_actions, video_pred, motion_pred, motion_decoded, video_cache_metadata = self.infer_model(model_input)
        metadata["video_cache"] = video_cache_metadata
        save_dir = self._save_request_artifacts(
            model_input=model_input,
            metadata=metadata,
            actions=actions,
            raw_actions=raw_actions,
            video_pred=video_pred,
            motion_pred=motion_pred,
            motion_decoded=motion_decoded,
        )
        return actions, {"artifact_dir": str(save_dir), **metadata}

    async def worker_loop(self) -> None:
        if self.world_size <= 1:
            return
        while True:
            signal_tensor = torch.zeros(1, dtype=torch.int32, device="cpu")
            dist.broadcast(signal_tensor, src=0, group=self.signal_group)
            signal = int(signal_tensor.item())
            if signal == 1:
                return
            if signal == 2:
                continue
            model_input = self._receive_object()
            dist.barrier()
            with torch.no_grad():
                self.policy.trained_model.lazy_joint_video_action_causal(model_input)
            dist.barrier()


def build_app(runner: FrankaDualDreamZeroRunner) -> FastAPI:
    app = FastAPI(title=f"RLinf {runner.model_kind} Franka Dual Server", version="0.1")

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {
            "ok": True,
            "model_kind": runner.model_kind,
            "payload_key": runner.payload_key,
            "n_action_steps": int(runner.args.n_action_steps),
            "return_action_dim": int(runner.args.return_action_dim),
            "model_state_dim": int(runner.args.model_state_dim),
            "model_action_dim": int(runner.args.model_action_dim),
            "requires_client_preprocessed_payload": True,
            "returns_normalized_actions": True,
            "clips_normalized_actions": True,
            "action_sampling": dict(runner.sampling_metadata),
            "action_inference_mode": str(getattr(runner.args, "action_inference_mode", "lazy")),
        }

    @app.post("/infer")
    async def infer(request: Request) -> Response:
        started = time.perf_counter()
        body = await request.body()
        try:
            print(f"[{runner.model_kind}] /infer request_bytes={len(body)}", flush=True)
            payload = unpackb(body)
            if not isinstance(payload, dict):
                raise ValueError(f"request payload must be a dict, got {type(payload)}")
            actions, info = runner.infer(payload)
            response = packb({"actions": actions, "info": info})
            headers = {"X-Inference-Time-Ms": f"{(time.perf_counter() - started) * 1000.0:.2f}"}
            return Response(content=response, media_type="application/msgpack", headers=headers)
        except Exception as exc:
            print(f"[{runner.model_kind}] /infer failed: {exc}", flush=True)
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    return app


def parse_args(default_kind: str = "dreamzero") -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-kind", default=default_kind, choices=("dreamzero", "wmam"))
    parser.add_argument("--payload-key", default="")
    parser.add_argument("--model-path", default="")
    parser.add_argument("--dreamzero-root", default=str(DEFAULT_DREAMZERO_ROOT))
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--n-action-steps", type=int, default=48)
    parser.add_argument("--return-action-dim", type=int, default=16)
    parser.add_argument("--model-state-dim", type=int, default=64)
    parser.add_argument("--model-action-dim", type=int, default=32)
    parser.add_argument("--num-inference-steps", type=int, default=0, help="0 means use checkpoint action_head num_inference_timesteps")
    parser.add_argument("--num-dit-steps", type=int, default=0, help="0 means run every configured inference step")
    parser.add_argument("--embodiment-tag", default="real_world_franka_dual")
    parser.add_argument("--embodiment-id", type=int, default=49)
    parser.add_argument("--tokenizer-path", default="/inspire/hdd/project/robot-body/linbokai-CZXS24250037/dreamzero/checkpoints/umt5-xxl")
    parser.add_argument("--max-seq-len", type=int, default=512)
    parser.add_argument("--default-prompt", default="")
    parser.add_argument(
        "--action-inference-mode",
        default="lazy",
        choices=("lazy", "tf_video_unipc"),
        help="lazy uses DreamZero joint action sampling; tf_video_unipc resamples actions through the training-style clean-video path.",
    )
    parser.add_argument("--output-root", default="/tmp/rlinf_dreamzero_wmam_infer")
    parser.add_argument("--save-fps", type=int, default=5)
    parser.add_argument("--decode-motion", action="store_true")
    parser.add_argument("--enable-dit-cache", action="store_true")
    parser.add_argument("--model-config-override", action="append", default=[])
    return parser.parse_args()


def run_server(default_kind: str = "dreamzero") -> None:
    args = parse_args(default_kind=default_kind)
    dreamzero_root = Path(args.dreamzero_root)
    if str(dreamzero_root) not in sys.path:
        sys.path.insert(0, str(dreamzero_root))

    os.environ["ENABLE_DIT_CACHE"] = "true" if args.enable_dit_cache else "false"
    os.environ.setdefault("ATTENTION_BACKEND", "TE")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    with contextlib.suppress(AttributeError):
        torch._dynamo.config.recompile_limit = 800

    args.device_mesh = init_mesh()
    model_kind = str(args.model_kind)
    payload_key = str(args.payload_key or ("wmam" if model_kind == "wmam" else "dreamzero"))
    runner = FrankaDualDreamZeroRunner(args, model_kind=model_kind, payload_key=payload_key)

    if dist.get_rank() == 0:
        app = build_app(runner)
        import uvicorn

        uvicorn.run(app, host=args.host, port=args.port, log_level="info")
        runner._broadcast_signal(1)
    else:
        import asyncio

        asyncio.run(runner.worker_loop())


if __name__ == "__main__":
    with contextlib.suppress(KeyboardInterrupt):
        run_server("dreamzero")
