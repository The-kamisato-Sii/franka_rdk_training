#!/usr/bin/env python
"""Offline OpenPI pi0.5 checkpoint fit check on franka_dual LeRobot v3 data."""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("USE_FLAX", "0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import ConcatDataset, DataLoader, Subset
from torch.utils._pytree import tree_map
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _move_to_device(value: Any, device: torch.device) -> Any:
    if torch.is_tensor(value):
        if value.is_floating_point():
            return value.to(device=device, dtype=torch.float32, non_blocking=True)
        return value.to(device=device, non_blocking=True)
    return value


def _resolve_checkpoint_file(checkpoint_dir: Path, rank: int = 0) -> Path:
    candidates = [
        checkpoint_dir / "actor" / "local_shard_checkpoint" / f"checkpoint_rank_{rank}.pt",
        checkpoint_dir / "local_shard_checkpoint" / f"checkpoint_rank_{rank}.pt",
        checkpoint_dir / f"checkpoint_rank_{rank}.pt",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"Cannot find checkpoint_rank_{rank}.pt under {checkpoint_dir}; tried {candidates}"
    )


def _load_model(
    *,
    base_model_path: Path,
    checkpoint_dir: Path,
    device: torch.device,
    checkpoint_rank: int,
):
    from rlinf.models.embodiment.openpi import get_model

    cfg = OmegaConf.create(
        {
            "model_path": str(base_model_path),
            "precision": "float32",
            "openpi": {
                "config_name": "pi05_real_world_joint",
                "repo_id": "real_world_franka_dual",
                "norm_stats_key": "real_world_franka_dual",
                "detach_critic_input": True,
                "num_images_in_input": 3,
                "train_expert_only": True,
                "action_chunk": 48,
                "num_steps": 5,
                "noise_method": "flow_sde",
                "noise_level": 0.5,
                "action_env_dim": 32,
                "add_value_head": False,
                "value_after_vlm": False,
                "value_vlm_mode": "mean_token",
            },
        }
    )
    model = get_model(cfg, torch_dtype=torch.float32)
    ckpt_file = _resolve_checkpoint_file(checkpoint_dir, checkpoint_rank)
    checkpoint = torch.load(ckpt_file, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("model", checkpoint)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    return model, {
        "checkpoint_file": str(ckpt_file),
        "missing_keys": len(missing),
        "unexpected_keys": len(unexpected),
        "first_missing_keys": list(missing)[:8],
        "first_unexpected_keys": list(unexpected)[:8],
    }


def _build_eval_dataset(args: argparse.Namespace):
    from rlinf.data.datasets.openpi.real_world_joint import (
        _build_single_task_dataset,
        _discover_lerobot_v3_tasks,
        _load_or_create_global_norm_stats,
    )
    from rlinf.models.embodiment.openpi.dataconfig import get_openpi_config

    cfg = OmegaConf.load(args.config)
    cfg.data.train_data_paths = str(args.dataset_root)
    cfg.data.real_world_joint_root = str(args.dataset_root)
    cfg.actor.model.model_path = str(args.base_model_path)
    cfg.actor.micro_batch_size = int(args.batch_size)
    cfg.data.num_workers = int(args.num_workers)

    task_dirs = _discover_lerobot_v3_tasks(args.dataset_root, cfg.data)
    if not task_dirs:
        raise RuntimeError(f"No LeRobot v3 tasks found under {args.dataset_root}")
    openpi_config = get_openpi_config(
        cfg.actor.model.openpi.config_name,
        model_path=cfg.actor.model.model_path,
        batch_size=int(args.batch_size),
        data_kwargs=None,
    )
    data_config = openpi_config.data.create(openpi_config.assets_dirs, openpi_config.model)
    norm_stats = _load_or_create_global_norm_stats(
        task_dirs,
        cfg,
        rank=0,
        dataset_root=args.dataset_root,
    )
    datasets = [
        _build_single_task_dataset(
            task_dir=task_dir,
            cfg=cfg,
            data_config=data_config,
            openpi_config=openpi_config,
            action_horizon=int(openpi_config.model.action_horizon),
            eval_dataset=True,
            openpi_norm_stats=norm_stats,
        )
        for task_dir in task_dirs
    ]
    dataset = ConcatDataset(datasets)
    return dataset, task_dirs, norm_stats


def _select_indices(dataset_len: int, num_samples: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    num_samples = min(int(num_samples), int(dataset_len))
    return np.sort(rng.choice(dataset_len, size=num_samples, replace=False))


def _unnormalize_actions(x: torch.Tensor, q01: torch.Tensor, q99: torch.Tensor) -> torch.Tensor:
    return (x + 1.0) * 0.5 * (q99 - q01 + 1e-6) + q01


def _masked_initial_noise(shape: tuple[int, ...], device: torch.device, real_dim: int) -> torch.Tensor:
    noise = torch.randn(shape, device=device, dtype=torch.float32)
    if real_dim < shape[-1]:
        noise[..., real_dim:] = 0.0
    return noise


def _evaluate_one_checkpoint(
    *,
    model,
    loader: DataLoader,
    device: torch.device,
    q01: torch.Tensor,
    q99: torch.Tensor,
    real_dim: int,
    step_name: str,
) -> dict[str, Any]:
    from openpi.models import model as _model

    sums = {
        "norm_sse": 0.0,
        "raw_sse": 0.0,
        "first_norm_sse": 0.0,
        "first_raw_sse": 0.0,
        "pred_raw_abs_sum": 0.0,
        "target_raw_abs_sum": 0.0,
        "pred_horizon_l2_sum": 0.0,
        "target_horizon_l2_sum": 0.0,
        "pred_step_delta_l2_sum": 0.0,
        "target_step_delta_l2_sum": 0.0,
        "pred_norm_std_sum": 0.0,
        "target_norm_std_sum": 0.0,
    }
    count_values = 0
    count_first_values = 0
    count_batches = 0
    count_samples = 0
    horizon_count = 0
    step_delta_count = 0

    progress = tqdm(loader, desc=f"eval {step_name}", dynamic_ncols=True)
    with torch.inference_mode():
        for batch in progress:
            actions = batch.pop("actions").to(device=device, dtype=torch.float32, non_blocking=True)
            obs_dict = tree_map(lambda x: _move_to_device(x, device), batch)
            observation = _model.Observation.from_dict(obs_dict)

            noise = _masked_initial_noise(tuple(actions.shape), device, real_dim)
            outputs = model.sample_actions(
                observation,
                noise=noise,
                mode="eval",
                compute_values=False,
            )
            pred = outputs["actions"].detach().float()[..., :real_dim]
            target = actions[..., :real_dim]

            pred_raw = _unnormalize_actions(pred, q01, q99)
            target_raw = _unnormalize_actions(target, q01, q99)
            diff_norm = pred - target
            diff_raw = pred_raw - target_raw

            sums["norm_sse"] += float((diff_norm * diff_norm).sum().item())
            sums["raw_sse"] += float((diff_raw * diff_raw).sum().item())
            sums["first_norm_sse"] += float((diff_norm[:, 0] * diff_norm[:, 0]).sum().item())
            sums["first_raw_sse"] += float((diff_raw[:, 0] * diff_raw[:, 0]).sum().item())
            sums["pred_raw_abs_sum"] += float(pred_raw.abs().sum().item())
            sums["target_raw_abs_sum"] += float(target_raw.abs().sum().item())
            count_values += int(np.prod(pred.shape))
            count_first_values += int(np.prod(pred[:, 0].shape))
            count_batches += 1
            count_samples += int(pred.shape[0])

            pred_horizon = torch.linalg.vector_norm(pred_raw - pred_raw[:, :1], dim=-1)
            target_horizon = torch.linalg.vector_norm(target_raw - target_raw[:, :1], dim=-1)
            sums["pred_horizon_l2_sum"] += float(pred_horizon.sum().item())
            sums["target_horizon_l2_sum"] += float(target_horizon.sum().item())
            horizon_count += int(np.prod(pred_horizon.shape))

            pred_delta = torch.linalg.vector_norm(pred_raw[:, 1:] - pred_raw[:, :-1], dim=-1)
            target_delta = torch.linalg.vector_norm(target_raw[:, 1:] - target_raw[:, :-1], dim=-1)
            sums["pred_step_delta_l2_sum"] += float(pred_delta.sum().item())
            sums["target_step_delta_l2_sum"] += float(target_delta.sum().item())
            step_delta_count += int(np.prod(pred_delta.shape))

            sums["pred_norm_std_sum"] += float(pred.std(dim=1).mean().item()) * int(pred.shape[0])
            sums["target_norm_std_sum"] += float(target.std(dim=1).mean().item()) * int(target.shape[0])

            progress.set_postfix(
                mse=f"{sums['norm_sse'] / max(1, count_values):.5f}",
                raw=f"{sums['raw_sse'] / max(1, count_values):.5g}",
            )

    return {
        "samples": count_samples,
        "batches": count_batches,
        "normalized_mse_all_48x16": sums["norm_sse"] / max(1, count_values),
        "raw_mse_all_48x16": sums["raw_sse"] / max(1, count_values),
        "normalized_mse_first_step_16": sums["first_norm_sse"] / max(1, count_first_values),
        "raw_mse_first_step_16": sums["first_raw_sse"] / max(1, count_first_values),
        "pred_raw_abs_mean": sums["pred_raw_abs_sum"] / max(1, count_values),
        "target_raw_abs_mean": sums["target_raw_abs_sum"] / max(1, count_values),
        "pred_horizon_l2_from_first_mean": sums["pred_horizon_l2_sum"] / max(1, horizon_count),
        "target_horizon_l2_from_first_mean": sums["target_horizon_l2_sum"] / max(1, horizon_count),
        "pred_step_delta_l2_mean": sums["pred_step_delta_l2_sum"] / max(1, step_delta_count),
        "target_step_delta_l2_mean": sums["target_step_delta_l2_sum"] / max(1, step_delta_count),
        "pred_normalized_horizon_std_mean": sums["pred_norm_std_sum"] / max(1, count_samples),
        "target_normalized_horizon_std_mean": sums["target_norm_std_sum"] / max(1, count_samples),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    project_root = Path("/inspire/hdd/project/robot-body/linbokai-CZXS24250037")
    parser.add_argument(
        "--checkpoint-root",
        type=Path,
        default=project_root / "results" / "real_world_franka_dual_openpi_pi05_sft_filtered_v2",
    )
    parser.add_argument("--steps", nargs="+", type=int, default=[40000, 60000, 80000, 100000])
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("/inspire/qb-ilm2/project/robot-body/public/bokai/franka_dual_filtered"),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "examples" / "sft" / "config" / "real_world_franka_dual_openpi_pi05_sft_filtered.yaml",
    )
    parser.add_argument(
        "--base-model-path",
        type=Path,
        default=REPO_ROOT / "checkpoints" / "pi05_base_pytorch_real_world_joint",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--num-samples", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20260630)
    parser.add_argument("--real-action-dim", type=int, default=16)
    parser.add_argument("--checkpoint-rank", type=int, default=0)
    parser.add_argument(
        "--output-json",
        type=Path,
        default=project_root
        / "results"
        / "real_world_franka_dual_openpi_pi05_sft_filtered_v2"
        / "offline_fit_eval_40k_100k.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

    device = torch.device(args.device)
    dataset, task_dirs, norm_stats = _build_eval_dataset(args)
    indices = _select_indices(len(dataset), args.num_samples, args.seed)
    subset = Subset(dataset, indices.tolist())
    loader = DataLoader(
        subset,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=True,
        persistent_workers=int(args.num_workers) > 0,
        drop_last=False,
    )

    q01_np = np.asarray(norm_stats["actions"].q01[: args.real_action_dim], dtype=np.float32)
    q99_np = np.asarray(norm_stats["actions"].q99[: args.real_action_dim], dtype=np.float32)
    q01 = torch.as_tensor(q01_np, dtype=torch.float32, device=device)
    q99 = torch.as_tensor(q99_np, dtype=torch.float32, device=device)

    results: dict[str, Any] = {
        "checkpoint_root": str(args.checkpoint_root),
        "dataset_root": str(args.dataset_root),
        "config": str(args.config),
        "base_model_path": str(args.base_model_path),
        "task_count": len(task_dirs),
        "dataset_len": len(dataset),
        "num_samples": int(len(indices)),
        "seed": int(args.seed),
        "indices_head": indices[:20].tolist(),
        "real_action_dim": int(args.real_action_dim),
        "noise_policy": "initial noise is N(0,1) for dims [0:16), zero for dims [16:32)",
        "metrics": {},
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    for step in args.steps:
        step_name = f"global_step_{int(step)}"
        ckpt_dir = args.checkpoint_root / "checkpoints" / step_name
        print(f"\n[eval] Loading {step_name}: {ckpt_dir}", flush=True)
        model, load_info = _load_model(
            base_model_path=args.base_model_path,
            checkpoint_dir=ckpt_dir,
            device=device,
            checkpoint_rank=int(args.checkpoint_rank),
        )
        metrics = _evaluate_one_checkpoint(
            model=model,
            loader=loader,
            device=device,
            q01=q01,
            q99=q99,
            real_dim=int(args.real_action_dim),
            step_name=step_name,
        )
        metrics.update(load_info)
        results["metrics"][step_name] = metrics
        with args.output_json.open("w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2, sort_keys=True)
        print(f"[eval] {step_name} metrics: {json.dumps(metrics, indent=2, sort_keys=True)}", flush=True)

        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"\n[eval] Wrote {args.output_json}", flush=True)


if __name__ == "__main__":
    main()
