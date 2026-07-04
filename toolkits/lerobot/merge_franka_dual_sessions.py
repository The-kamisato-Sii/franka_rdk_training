#!/usr/bin/env python
"""Merge Franka dual LeRobot v3 session datasets into task-level datasets.

Input layout:

    franka_dual/<task>/<session>/{data,meta,videos}

Output layout:

    franka_dual_v2/<task>/{data,meta,videos}

The converter rewrites data parquet index columns so the merged task dataset is
self-consistent. Video files are hard-linked by default because they dominate
storage size.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from collections import OrderedDict, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

STATE_KEY = "observation.state"
ACTION_KEY = "action"
VIDEO_KEYS = (
    "observation.images.left_camera",
    "observation.images.middle_zed",
    "observation.images.right_camera",
)
NUMERIC_STATS_KEYS = (
    ACTION_KEY,
    STATE_KEY,
    "timestamp",
    "frame_index",
    "episode_index",
    "index",
    "task_index",
)
QUANTILES = {
    "q01": 0.01,
    "q10": 0.10,
    "q50": 0.50,
    "q90": 0.90,
    "q99": 0.99,
}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def format_lerobot_path(
    task_dir: Path,
    template: str,
    *,
    chunk: int,
    file: int,
    video_key: str = "",
) -> Path:
    return (
        task_dir
        / template.format(
            chunk_index=int(chunk),
            file_index=int(file),
            episode_chunk=int(chunk),
            episode_index=int(file),
            video_key=video_key,
        )
    ).resolve()


def link_or_copy(src: Path, dst: Path, *, mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    if mode == "hardlink":
        os.link(src, dst)
    elif mode == "copy":
        shutil.copy2(src, dst)
    else:
        raise ValueError(f"Unsupported link mode: {mode}")


def list_session_dirs(src_root: Path) -> list[Path]:
    sessions = sorted(path.parent.parent for path in src_root.glob("*/*/meta/info.json"))
    return [session for session in sessions if session.name != ".cache"]


def load_episode_rows(session_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted((session_dir / "meta" / "episodes").glob("chunk-*/file-*.parquet")):
        rows.extend(pq.read_table(str(path)).to_pylist())
    return rows


def load_tasks(session_dir: Path) -> dict[int, str]:
    path = session_dir / "meta" / "tasks.parquet"
    if not path.is_file():
        return {}
    return {
        int(row["task_index"]): str(row["task"])
        for row in pq.read_table(str(path)).to_pylist()
    }


def first_task(row: dict[str, Any], task_map: dict[int, str], default: str) -> str:
    tasks = row.get("tasks") or []
    if tasks:
        return str(tasks[0])
    task_index = row.get("task_index")
    if task_index is not None and int(task_index) in task_map:
        return task_map[int(task_index)]
    return default


def scalar_sequence_stats(values: np.ndarray) -> dict[str, list[float] | list[int]]:
    arr = np.asarray(values, dtype=np.float64).reshape(-1, 1)
    out: dict[str, list[float] | list[int]] = {
        "min": arr.min(axis=0).astype(float).tolist(),
        "max": arr.max(axis=0).astype(float).tolist(),
        "mean": arr.mean(axis=0).astype(float).tolist(),
        "std": arr.std(axis=0).astype(float).tolist(),
        "count": [int(arr.shape[0])],
    }
    for name, q in QUANTILES.items():
        out[name] = np.quantile(arr, q, axis=0).astype(float).tolist()
    return out


def update_episode_index_stats(
    row: dict[str, Any],
    *,
    new_episode_index: int,
    global_frame_start: int,
    length: int,
    global_task_index: int,
) -> None:
    replacements = {
        "episode_index": np.full((length,), new_episode_index, dtype=np.int64),
        "index": np.arange(global_frame_start, global_frame_start + length, dtype=np.int64),
        "frame_index": np.arange(length, dtype=np.int64),
        "task_index": np.full((length,), global_task_index, dtype=np.int64),
    }
    for key, values in replacements.items():
        stats = scalar_sequence_stats(values)
        for stat_name, stat_value in stats.items():
            row[f"stats/{key}/{stat_name}"] = stat_value


def array_stats(arr: np.ndarray) -> dict[str, list[float] | list[int]]:
    values = np.asarray(arr)
    if values.ndim == 1:
        values = values.reshape(-1, 1)
    values = values.astype(np.float64, copy=False)
    out: dict[str, list[float] | list[int]] = {
        "min": values.min(axis=0).astype(float).tolist(),
        "max": values.max(axis=0).astype(float).tolist(),
        "mean": values.mean(axis=0).astype(float).tolist(),
        "std": values.std(axis=0).astype(float).tolist(),
        "count": [int(values.shape[0])],
    }
    for name, q in QUANTILES.items():
        out[name] = np.quantile(values, q, axis=0).astype(float).tolist()
    return out


def collect_numeric_stats(data_files: list[Path]) -> dict[str, dict[str, Any]]:
    arrays: dict[str, list[np.ndarray]] = {key: [] for key in NUMERIC_STATS_KEYS}
    for data_path in data_files:
        schema = set(pq.read_schema(str(data_path)).names)
        columns = [key for key in NUMERIC_STATS_KEYS if key in schema]
        table = pq.read_table(str(data_path), columns=columns)
        for key in columns:
            values = table.column(key).to_pylist()
            if key in (ACTION_KEY, STATE_KEY):
                dtype = np.float32
            elif key == "timestamp":
                dtype = np.float64
            else:
                dtype = np.int64
            arrays[key].append(np.asarray(values, dtype=dtype))
    stats = {}
    for key, parts in arrays.items():
        if parts:
            stats[key] = array_stats(np.concatenate(parts, axis=0))
    return stats


def aggregate_image_stats_from_episodes(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Approximate merged image stats from source per-episode stats.

    The converter does not decode videos. These image stats are metadata only;
    RLinf/OpenPI normalization uses action/state stats.
    """

    merged: dict[str, dict[str, Any]] = {}
    for key in VIDEO_KEYS:
        count_key = f"stats/{key}/count"
        valid = [row for row in rows if count_key in row]
        if not valid:
            continue
        counts = np.asarray([float((row[count_key] or [0])[0]) for row in valid], dtype=np.float64)
        total = float(counts.sum())
        if total <= 0:
            continue

        def arr(stat_name: str) -> np.ndarray:
            return np.asarray([row[f"stats/{key}/{stat_name}"] for row in valid], dtype=np.float64)

        mean = (arr("mean") * counts[:, None, None, None]).sum(axis=0) / total
        second = ((arr("std") ** 2 + arr("mean") ** 2) * counts[:, None, None, None]).sum(axis=0) / total
        std = np.sqrt(np.maximum(second - mean**2, 0.0))
        out = {
            "min": arr("min").min(axis=0).astype(float).tolist(),
            "max": arr("max").max(axis=0).astype(float).tolist(),
            "mean": mean.astype(float).tolist(),
            "std": std.astype(float).tolist(),
            "count": [int(total)],
        }
        for stat_name in QUANTILES:
            out[stat_name] = (
                (arr(stat_name) * counts[:, None, None, None]).sum(axis=0) / total
            ).astype(float).tolist()
        merged[key] = out
    return merged


def openpi_norm_stats_from_lerobot_stats(stats: dict[str, Any]) -> dict[str, Any]:
    return {
        "norm_stats": {
            "state": {
                "mean": stats[STATE_KEY]["mean"],
                "std": stats[STATE_KEY]["std"],
                "q01": stats[STATE_KEY]["q01"],
                "q99": stats[STATE_KEY]["q99"],
            },
            "actions": {
                "mean": stats[ACTION_KEY]["mean"],
                "std": stats[ACTION_KEY]["std"],
                "q01": stats[ACTION_KEY]["q01"],
                "q99": stats[ACTION_KEY]["q99"],
            },
        }
    }


def replace_column(table: pa.Table, name: str, values: np.ndarray) -> pa.Table:
    if name not in table.column_names:
        return table
    idx = table.column_names.index(name)
    return table.set_column(idx, name, pa.array(values, type=table.schema.field(name).type))


def rewrite_data_file(
    src_path: Path,
    dst_path: Path,
    assignments: list[dict[str, int]],
    *,
    hardlink_data: bool,
) -> None:
    if hardlink_data:
        link_or_copy(src_path, dst_path, mode="hardlink")
        return

    table = pq.read_table(str(src_path))
    n = table.num_rows
    episode_index = np.asarray(table.column("episode_index").to_pylist(), dtype=np.int64)
    index = np.asarray(table.column("index").to_pylist(), dtype=np.int64)
    task_index = np.asarray(table.column("task_index").to_pylist(), dtype=np.int64)
    frame_index = np.asarray(table.column("frame_index").to_pylist(), dtype=np.int64)
    for item in assignments:
        start, stop = int(item["data_from"]), int(item["data_to"])
        if start < 0 or stop > n or stop <= start:
            raise ValueError(f"Bad row span for {src_path}: {start}:{stop}, rows={n}")
        length = stop - start
        episode_index[start:stop] = int(item["episode_index"])
        index[start:stop] = np.arange(
            int(item["global_frame_start"]),
            int(item["global_frame_start"]) + length,
            dtype=np.int64,
        )
        task_index[start:stop] = int(item["task_index"])
        frame_index[start:stop] = np.arange(length, dtype=np.int64)

    table = replace_column(table, "episode_index", episode_index)
    table = replace_column(table, "index", index)
    table = replace_column(table, "task_index", task_index)
    table = replace_column(table, "frame_index", frame_index)
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, str(dst_path))


def write_episodes(rows: list[dict[str, Any]], meta_dir: Path, *, chunks_size: int) -> None:
    episodes_root = meta_dir / "episodes"
    episodes_root.mkdir(parents=True, exist_ok=True)
    for chunk_index in sorted({i // chunks_size for i in range(len(rows))}):
        file_index = chunk_index
        start = chunk_index * chunks_size
        chunk_rows = rows[start : start + chunks_size]
        for row in chunk_rows:
            row["meta/episodes/chunk_index"] = int(chunk_index)
            row["meta/episodes/file_index"] = int(file_index)
        out = episodes_root / f"chunk-{chunk_index:03d}" / f"file-{file_index:03d}.parquet"
        out.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.Table.from_pylist(chunk_rows), str(out))


def write_tasks(tasks: OrderedDict[str, int], meta_dir: Path) -> None:
    rows = [{"task_index": index, "task": task} for task, index in tasks.items()]
    pq.write_table(pa.Table.from_pylist(rows), str(meta_dir / "tasks.parquet"))


def merge_task(
    task_name: str,
    sessions: list[Path],
    dst_task_dir: Path,
    *,
    video_mode: str,
    hardlink_data: bool,
) -> tuple[list[Path], list[dict[str, Any]], dict[str, Any]]:
    template_info = read_json(sessions[0] / "meta" / "info.json")
    chunks_size = int(template_info.get("chunks_size") or 1000)
    data_tmpl = str(template_info["data_path"])
    video_tmpl = str(template_info["video_path"])

    data_map: dict[Path, tuple[int, int, Path]] = {}
    video_map: dict[tuple[str, Path], tuple[int, int, Path]] = {}
    data_assignments: dict[Path, list[dict[str, int]]] = defaultdict(list)
    task_indices: OrderedDict[str, int] = OrderedDict()
    episode_rows: list[dict[str, Any]] = []
    target_data_files: list[Path] = []
    episode_index = 0
    frame_index_global = 0

    def assign_task(prompt: str) -> int:
        if prompt not in task_indices:
            task_indices[prompt] = len(task_indices)
        return task_indices[prompt]

    def assign_data_file(src: Path) -> tuple[int, int, Path]:
        if src not in data_map:
            file_index = len(data_map)
            chunk_index = file_index // chunks_size
            dst = format_lerobot_path(dst_task_dir, data_tmpl, chunk=chunk_index, file=file_index)
            data_map[src] = (chunk_index, file_index, dst)
            target_data_files.append(dst)
        return data_map[src]

    def assign_video_file(video_key: str, src: Path) -> tuple[int, int, Path]:
        map_key = (video_key, src)
        if map_key not in video_map:
            file_index = sum(1 for key, _ in video_map if key == video_key)
            chunk_index = file_index // chunks_size
            dst = format_lerobot_path(
                dst_task_dir,
                video_tmpl,
                chunk=chunk_index,
                file=file_index,
                video_key=video_key,
            )
            link_or_copy(src, dst, mode=video_mode)
            video_map[map_key] = (chunk_index, file_index, dst)
        return video_map[map_key]

    for session in sessions:
        info = read_json(session / "meta" / "info.json")
        task_map = load_tasks(session)
        rows = load_episode_rows(session)
        for row in rows:
            length = int(row["length"])
            prompt = first_task(row, task_map, task_name.replace("_", " "))
            target_task_index = assign_task(prompt)

            data_chunk = int(row.get("data/chunk_index", 0))
            data_file = int(row.get("data/file_index", 0))
            src_data = format_lerobot_path(session, info["data_path"], chunk=data_chunk, file=data_file)
            target_data_chunk, target_data_file, _ = assign_data_file(src_data)

            new_row = dict(row)
            new_row["episode_index"] = int(episode_index)
            new_row["tasks"] = [prompt]
            new_row["data/chunk_index"] = int(target_data_chunk)
            new_row["data/file_index"] = int(target_data_file)

            for video_key in VIDEO_KEYS:
                prefix = f"videos/{video_key}"
                src_v_chunk = int(row.get(f"{prefix}/chunk_index", data_chunk))
                src_v_file = int(row.get(f"{prefix}/file_index", data_file))
                src_video = format_lerobot_path(
                    session,
                    info["video_path"],
                    chunk=src_v_chunk,
                    file=src_v_file,
                    video_key=video_key,
                )
                dst_v_chunk, dst_v_file, _ = assign_video_file(video_key, src_video)
                new_row[f"{prefix}/chunk_index"] = int(dst_v_chunk)
                new_row[f"{prefix}/file_index"] = int(dst_v_file)

            update_episode_index_stats(
                new_row,
                new_episode_index=episode_index,
                global_frame_start=frame_index_global,
                length=length,
                global_task_index=target_task_index,
            )
            episode_rows.append(new_row)
            data_assignments[src_data].append(
                {
                    "data_from": int(row["dataset_from_index"]),
                    "data_to": int(row["dataset_to_index"]),
                    "episode_index": int(episode_index),
                    "global_frame_start": int(frame_index_global),
                    "task_index": int(target_task_index),
                }
            )
            episode_index += 1
            frame_index_global += length

    for src, (_, _, dst) in sorted(data_map.items(), key=lambda item: item[1][1]):
        rewrite_data_file(src, dst, data_assignments[src], hardlink_data=hardlink_data)

    meta_dir = dst_task_dir / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    write_tasks(task_indices, meta_dir)
    write_episodes(episode_rows, meta_dir, chunks_size=chunks_size)

    stats = collect_numeric_stats(target_data_files)
    stats.update(aggregate_image_stats_from_episodes(episode_rows))
    write_json(meta_dir / "stats.json", stats)
    write_json(dst_task_dir / "norm_stats.json", openpi_norm_stats_from_lerobot_stats(stats))

    info = dict(template_info)
    info["total_episodes"] = int(len(episode_rows))
    info["total_frames"] = int(frame_index_global)
    info["total_tasks"] = int(len(task_indices))
    info["total_videos"] = int(len(video_map))
    info["total_chunks"] = int(max([0, *(chunk for chunk, _, _ in data_map.values())]) + 1 if data_map else 0)
    info["splits"] = {"train": f"0:{len(episode_rows)}"}
    info["data_path"] = data_tmpl
    info["video_path"] = video_tmpl
    write_json(meta_dir / "info.json", info)
    return target_data_files, episode_rows, info


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src-root", type=Path, required=True)
    parser.add_argument("--dst-root", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--video-mode", choices=("hardlink", "copy"), default="hardlink")
    parser.add_argument(
        "--include-task",
        action="append",
        default=None,
        help="Only merge this task directory name. Can be passed multiple times.",
    )
    parser.add_argument(
        "--hardlink-data",
        action="store_true",
        help="Hardlink data parquet files instead of rewriting index columns. Faster, but less self-consistent.",
    )
    args = parser.parse_args()

    src_root = args.src_root.resolve()
    dst_root = args.dst_root.resolve()
    if not src_root.is_dir():
        raise FileNotFoundError(src_root)
    if dst_root.exists():
        if not args.overwrite:
            raise FileExistsError(f"{dst_root} exists; pass --overwrite to replace it")
        shutil.rmtree(dst_root)
    dst_root.mkdir(parents=True, exist_ok=True)

    sessions_by_task: dict[str, list[Path]] = defaultdict(list)
    include_tasks = None
    if args.include_task:
        include_tasks = {str(task) for task in args.include_task}
    for session in list_session_dirs(src_root):
        task_name = session.parent.name
        if include_tasks is not None and task_name not in include_tasks:
            continue
        sessions_by_task[task_name].append(session)
    if not sessions_by_task:
        raise RuntimeError(f"No LeRobot v3 sessions found under {src_root}")

    root_data_files: list[Path] = []
    total_episodes = 0
    total_frames = 0
    summary = {}
    for task_name in sorted(sessions_by_task):
        sessions = sorted(sessions_by_task[task_name])
        print(f"[merge] task={task_name} sessions={len(sessions)}")
        data_files, episode_rows, info = merge_task(
            task_name,
            sessions,
            dst_root / task_name,
            video_mode=args.video_mode,
            hardlink_data=bool(args.hardlink_data),
        )
        root_data_files.extend(data_files)
        total_episodes += int(info["total_episodes"])
        total_frames += int(info["total_frames"])
        summary[task_name] = {
            "sessions": len(sessions),
            "episodes": int(info["total_episodes"]),
            "frames": int(info["total_frames"]),
            "data_files": len(data_files),
        }

    root_stats = collect_numeric_stats(root_data_files)
    write_json(dst_root / "stats.json", root_stats)
    write_json(dst_root / "norm_stats.json", openpi_norm_stats_from_lerobot_stats(root_stats))
    write_json(
        dst_root / "merge_summary.json",
        {
            "src_root": str(src_root),
            "dst_root": str(dst_root),
            "video_mode": args.video_mode,
            "hardlink_data": bool(args.hardlink_data),
            "tasks": summary,
            "total_tasks": len(summary),
            "total_episodes": total_episodes,
            "total_frames": total_frames,
        },
    )
    print(f"[merge] wrote {dst_root}")
    print(f"[merge] total_tasks={len(summary)} total_episodes={total_episodes} total_frames={total_frames}")
    print(f"[merge] root stats: {dst_root / 'stats.json'}")
    print(f"[merge] OpenPI norm stats: {dst_root / 'norm_stats.json'}")


if __name__ == "__main__":
    main()
