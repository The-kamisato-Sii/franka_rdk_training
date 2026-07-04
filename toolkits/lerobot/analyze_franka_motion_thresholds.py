#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, math
from pathlib import Path
from collections import defaultdict
import numpy as np
import pyarrow.parquet as pq

try:
    from numba import njit
except Exception:  # pragma: no cover - optional speedup
    njit = None

STATE_KEY = "observation.state"
ACTION_KEY = "action"
ARM_DIMS = [0,1,2,3,4,5,6,8,9,10,11,12,13,14]
GRIPPER_DIMS = [7,15]


if njit is not None:
    @njit(cache=True)
    def _greedy_keep_counts_numba(states, joint_thresholds, gripper_thresholds, arm_dims, gripper_dims):
        out = np.zeros((len(joint_thresholds), len(gripper_thresholds)), dtype=np.int64)
        n = states.shape[0]
        if n <= 0:
            return out
        for a in range(len(joint_thresholds)):
            jd = joint_thresholds[a]
            for b in range(len(gripper_thresholds)):
                gd = gripper_thresholds[b]
                kept = 1
                last = 0
                for i in range(1, n):
                    arm_diff = 0.0
                    for k in range(len(arm_dims)):
                        d = abs(states[i, arm_dims[k]] - states[last, arm_dims[k]])
                        if d > arm_diff:
                            arm_diff = d
                    grip_diff = 0.0
                    for k in range(len(gripper_dims)):
                        d = abs(states[i, gripper_dims[k]] - states[last, gripper_dims[k]])
                        if d > grip_diff:
                            grip_diff = d
                    if arm_diff > jd or grip_diff > gd:
                        kept += 1
                        last = i
                out[a, b] = kept
        return out
else:
    _greedy_keep_counts_numba = None


def read_json(path: Path):
    return json.loads(path.read_text())


def discover_tasks(root: Path):
    if (root / "meta" / "info.json").is_file():
        return [root]
    return sorted([p for p in root.iterdir() if (p / "meta" / "info.json").is_file()])


def fmt_template(task_dir: Path, tmpl: str, chunk: int, file: int, video_key: str = "") -> Path:
    return task_dir / tmpl.format(
        chunk_index=int(chunk), file_index=int(file), episode_chunk=int(chunk), episode_index=int(file), video_key=video_key
    )


def load_episode_rows(task_dir: Path):
    info = read_json(task_dir / "meta" / "info.json")
    data_tmpl = info.get("data_path") or "data/chunk-{episode_chunk:03d}/file-{episode_index:03d}.parquet"
    chunks_size = int(info.get("chunks_size") or 1000)
    episodes_root = task_dir / "meta" / "episodes"
    if not episodes_root.is_dir():
        raise FileNotFoundError(f"missing {episodes_root}")
    rows = []
    for ep_meta in sorted(episodes_root.glob("chunk-*/file-*.parquet")):
        table = pq.read_table(str(ep_meta))
        for row in table.to_pylist():
            ep_idx = int(row.get("episode_index", len(rows)))
            length = int(row.get("length") or row.get("episode_length") or row.get("num_frames") or 0)
            if length <= 1:
                continue
            data_chunk = int(row.get("data/chunk_index", ep_idx // chunks_size))
            data_file = int(row.get("data/file_index", ep_idx))
            data_from = int(row.get("dataset_from_index", 0))
            data_to = int(row.get("dataset_to_index", data_from + length))
            data_path = fmt_template(task_dir, data_tmpl, data_chunk, data_file)
            rows.append((ep_idx, length, data_path, data_from, data_to))
    return sorted(rows, key=lambda x: x[0])


def read_state_episode(
    data_path: Path,
    data_from: int,
    data_to: int,
    state_cache: dict[Path, np.ndarray],
):
    data_path = Path(data_path)
    arr = state_cache.get(data_path)
    if arr is None:
        table = pq.read_table(str(data_path), columns=[STATE_KEY])
        arr = np.asarray(table.column(STATE_KEY).to_pylist(), dtype=np.float32)
        state_cache[data_path] = arr
    arr = arr[data_from:data_to]
    if arr.ndim != 2 or arr.shape[1] < 16:
        raise ValueError(f"bad state shape {arr.shape} in {data_path}")
    return arr[:, :16]


def greedy_keep_count(states: np.ndarray, joint_delta: float, gripper_delta: float):
    n = len(states)
    if n <= 0:
        return 0
    kept = 1
    last = 0
    for i in range(1, n):
        arm_diff = float(np.max(np.abs(states[i, ARM_DIMS] - states[last, ARM_DIMS]))) if ARM_DIMS else 0.0
        grip_diff = float(np.max(np.abs(states[i, GRIPPER_DIMS] - states[last, GRIPPER_DIMS]))) if GRIPPER_DIMS else 0.0
        if arm_diff > joint_delta or grip_diff > gripper_delta:
            kept += 1
            last = i
    return kept


def greedy_keep_counts(states: np.ndarray, joint_thresholds, gripper_thresholds) -> np.ndarray:
    joint_thresholds = np.asarray(joint_thresholds, dtype=np.float64)
    gripper_thresholds = np.asarray(gripper_thresholds, dtype=np.float64)
    if _greedy_keep_counts_numba is not None:
        return _greedy_keep_counts_numba(
            np.asarray(states, dtype=np.float32),
            joint_thresholds,
            gripper_thresholds,
            np.asarray(ARM_DIMS, dtype=np.int64),
            np.asarray(GRIPPER_DIMS, dtype=np.int64),
        )
    out = np.zeros((len(joint_thresholds), len(gripper_thresholds)), dtype=np.int64)
    for a, jd in enumerate(joint_thresholds):
        for b, gd in enumerate(gripper_thresholds):
            out[a, b] = greedy_keep_count(states, float(jd), float(gd))
    return out


def q(vals, ps=(0,1,5,10,25,50,75,90,95,99,100)):
    vals = np.asarray(vals, dtype=np.float64)
    if vals.size == 0:
        return {str(p): None for p in ps}
    out = np.percentile(vals, ps)
    return {str(p): float(v) for p,v in zip(ps,out)}


def analyze(root: Path, joint_thresholds, gripper_thresholds, max_episodes_per_task=None):
    tasks = discover_tasks(root)
    by_task = {}
    total_frames = 0
    total_episodes = 0
    global_adj_arm = []
    global_adj_grip = []
    global_adj_l2 = []
    global_abs_step = []
    global_ret = {(jd, gd): [0,0] for jd in joint_thresholds for gd in gripper_thresholds}

    for task_dir in tasks:
        eps = load_episode_rows(task_dir)
        if max_episodes_per_task:
            eps = eps[:max_episodes_per_task]
        task_frames = 0
        task_episodes = 0
        adj_arm=[]; adj_grip=[]; adj_l2=[]; abs_step=[]
        ret = {(jd, gd): [0,0] for jd in joint_thresholds for gd in gripper_thresholds}
        lengths=[]
        state_cache: dict[Path, np.ndarray] = {}
        for ep_idx, length, data_path, data_from, data_to in eps:
            states = read_state_episode(data_path, data_from, data_to, state_cache)
            n = len(states)
            if n <= 1:
                continue
            d = np.diff(states, axis=0)
            arm = np.max(np.abs(d[:, ARM_DIMS]), axis=1)
            grip = np.max(np.abs(d[:, GRIPPER_DIMS]), axis=1)
            l2 = np.linalg.norm(d, axis=1)
            absmax_since_start = np.maximum(
                np.max(np.abs(states[:, ARM_DIMS] - states[0:1, ARM_DIMS]), axis=1),
                np.max(np.abs(states[:, GRIPPER_DIMS] - states[0:1, GRIPPER_DIMS]), axis=1),
            )
            adj_arm.append(arm); adj_grip.append(grip); adj_l2.append(l2); abs_step.append(absmax_since_start)
            global_adj_arm.append(arm); global_adj_grip.append(grip); global_adj_l2.append(l2); global_abs_step.append(absmax_since_start)
            keep_counts = greedy_keep_counts(states, joint_thresholds, gripper_thresholds)
            for a, jd in enumerate(joint_thresholds):
                for b, gd in enumerate(gripper_thresholds):
                    kept = int(keep_counts[a, b])
                    ret[(jd, gd)][0] += kept
                    ret[(jd, gd)][1] += n
                    global_ret[(jd, gd)][0] += kept
                    global_ret[(jd, gd)][1] += n
            task_frames += n
            task_episodes += 1
            lengths.append(n)
        total_frames += task_frames
        total_episodes += task_episodes
        adj_arm_np = np.concatenate(adj_arm) if adj_arm else np.array([])
        adj_grip_np = np.concatenate(adj_grip) if adj_grip else np.array([])
        adj_l2_np = np.concatenate(adj_l2) if adj_l2 else np.array([])
        by_task[task_dir.name] = {
            "episodes": task_episodes,
            "frames": task_frames,
            "episode_length_quantiles": q(lengths),
            "adjacent_arm_max_abs_quantiles": q(adj_arm_np),
            "adjacent_gripper_max_abs_quantiles": q(adj_grip_np),
            "adjacent_l2_quantiles": q(adj_l2_np),
            "retention": [
                {"joint_delta": jd, "gripper_delta": gd, "kept": kept, "total": tot, "kept_ratio": kept/tot if tot else 0}
                for (jd, gd),(kept,tot) in sorted(ret.items())
            ],
        }
    g_arm = np.concatenate(global_adj_arm) if global_adj_arm else np.array([])
    g_grip = np.concatenate(global_adj_grip) if global_adj_grip else np.array([])
    g_l2 = np.concatenate(global_adj_l2) if global_adj_l2 else np.array([])
    return {
        "root": str(root),
        "tasks": len(tasks),
        "episodes": total_episodes,
        "frames": total_frames,
        "threshold_semantics": "greedy keep first frame, then keep frame i when max_abs_arm_joint_delta_since_last_kept > joint_delta OR max_abs_gripper_delta_since_last_kept > gripper_delta; arm dims=0:7,8:15; gripper dims=7,15",
        "global_adjacent_arm_max_abs_quantiles": q(g_arm),
        "global_adjacent_gripper_max_abs_quantiles": q(g_grip),
        "global_adjacent_l2_quantiles": q(g_l2),
        "global_retention": [
            {"joint_delta": jd, "gripper_delta": gd, "kept": kept, "total": tot, "kept_ratio": kept/tot if tot else 0, "delete_ratio": 1-(kept/tot if tot else 0)}
            for (jd, gd),(kept,tot) in sorted(global_ret.items())
        ],
        "by_task": by_task,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("roots", nargs="+")
    ap.add_argument("--joint-thresholds", default="0.002,0.003,0.004,0.005,0.0075,0.01,0.0125,0.015,0.02,0.025,0.03,0.04,0.05")
    ap.add_argument("--gripper-thresholds", default="0.001,0.002,0.003,0.004,0.005,0.0075,0.01,0.015,0.02")
    ap.add_argument("--output", default="/tmp/franka_motion_threshold_report.json")
    ap.add_argument("--max-episodes-per-task", type=int, default=None)
    args=ap.parse_args()
    jds=[float(x) for x in args.joint_thresholds.split(',') if x]
    gds=[float(x) for x in args.gripper_thresholds.split(',') if x]
    reports=[analyze(Path(r), jds, gds, args.max_episodes_per_task) for r in args.roots]
    Path(args.output).write_text(json.dumps(reports, indent=2), encoding='utf-8')
    for rep in reports:
        print('\nROOT', rep['root'])
        print('tasks', rep['tasks'], 'episodes', rep['episodes'], 'frames', rep['frames'])
        print('adj arm q', rep['global_adjacent_arm_max_abs_quantiles'])
        print('adj gripper q', rep['global_adjacent_gripper_max_abs_quantiles'])
        # Print candidates with global retention 60-80, sorted by closeness to 70%.
        candidates=[x for x in rep['global_retention'] if 0.60 <= x['kept_ratio'] <= 0.80]
        candidates=sorted(candidates, key=lambda x: abs(x['kept_ratio']-0.70))[:15]
        print('top global threshold candidates kept 60-80%:')
        for x in candidates:
            print(f"  joint={x['joint_delta']:.5g} grip={x['gripper_delta']:.5g} keep={x['kept_ratio']*100:.2f}% delete={x['delete_ratio']*100:.2f}% kept={x['kept']}/{x['total']}")
        print('per task retention for best candidate:')
        if candidates:
            best=candidates[0]
            for task, t in rep['by_task'].items():
                entry=next(y for y in t['retention'] if y['joint_delta']==best['joint_delta'] and y['gripper_delta']==best['gripper_delta'])
                print(f"  {task:24s} keep={entry['kept_ratio']*100:6.2f}% frames={entry['kept']}/{entry['total']}")
    print('\nwrote', args.output)

if __name__ == '__main__':
    main()
