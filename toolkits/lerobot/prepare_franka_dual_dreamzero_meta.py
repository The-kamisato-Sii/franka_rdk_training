#!/usr/bin/env python3
# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0

"""Prepare DreamZero metadata sidecars for Franka dual LeRobot v3 sessions.

The Franka dual data root is organized as:

    franka_dual/<task_name>/<session_timestamp>/{meta,data,videos}

This script writes the small sidecar files consumed by RLinf's DreamZero
real_world_joint reader when they are missing:

    meta/modality.json
    meta/embodiment.json

It does not touch parquet/video files and does not overwrite existing metadata
unless the corresponding --overwrite-* flag is passed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_ROOT = Path(
    "/inspire/qb-ilm2/project/robot-body/public/hairuoliu/franka_dual"
)
FRANKA_DUAL_TAG = "real_world_franka_dual"
STATE_KEY = "observation.state"
ACTION_KEY = "action"
VIDEO_KEYS = (
    "observation.images.middle_zed",
    "observation.images.left_camera",
    "observation.images.right_camera",
)


def _read_json(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        f.write("\n")


def _feature_dim(info: dict[str, Any], key: str) -> int:
    feature = (info.get("features") or {}).get(key) or {}
    shape = feature.get("shape") or []
    return int(shape[0]) if shape else 0


def _validate_session(session_dir: Path, info: dict[str, Any]) -> None:
    features = info.get("features") or {}
    missing = [key for key in (STATE_KEY, ACTION_KEY, *VIDEO_KEYS) if key not in features]
    if missing:
        raise ValueError(f"{session_dir} missing features: {missing}")
    state_dim = _feature_dim(info, STATE_KEY)
    action_dim = _feature_dim(info, ACTION_KEY)
    if state_dim < 16 or action_dim < 16:
        raise ValueError(
            f"{session_dir} expected state/action dim >= 16, "
            f"got state={state_dim} action={action_dim}"
        )


def _franka_dual_modality() -> dict[str, Any]:
    state_parts = {
        "left_joint_angle": (0, 7),
        "left_joint_gripper": (7, 8),
        "right_joint_angle": (8, 15),
        "right_joint_gripper": (15, 16),
    }
    action_parts = dict(state_parts)
    return {
        "state": {
            name: {
                "start": start,
                "end": end,
                "dtype": "float32",
                "absolute": True,
                "original_key": STATE_KEY,
            }
            for name, (start, end) in state_parts.items()
        },
        "action": {
            name: {
                "start": start,
                "end": end,
                "dtype": "float32",
                "absolute": True,
                "original_key": ACTION_KEY,
            }
            for name, (start, end) in action_parts.items()
        },
        "video": {
            "middle_zed": {"original_key": "observation.images.middle_zed"},
            "left_camera": {"original_key": "observation.images.left_camera"},
            "right_camera": {"original_key": "observation.images.right_camera"},
        },
        "annotation": {
            "task_index": {"original_key": "task_index"},
            "language.task": {"original_key": "task_index"},
        },
    }


def _discover_sessions(root: Path) -> list[Path]:
    if (root / "meta" / "info.json").is_file():
        return [root]
    sessions = sorted(path.parent.parent for path in root.glob("*/*/meta/info.json"))
    return [session for session in sessions if session.is_dir()]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate missing DreamZero sidecar metadata for Franka dual v3 data."
    )
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--max-sessions", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite-modality", action="store_true")
    parser.add_argument("--overwrite-embodiment", action="store_true")
    args = parser.parse_args()

    root = args.root.expanduser().resolve()
    sessions = _discover_sessions(root)
    if args.max_sessions is not None:
        sessions = sessions[: int(args.max_sessions)]
    if not sessions:
        raise SystemExit(f"No Franka dual LeRobot v3 sessions found under {root}")

    wrote_modality = 0
    wrote_embodiment = 0
    skipped = 0
    for session_dir in sessions:
        info_path = session_dir / "meta" / "info.json"
        info = _read_json(info_path)
        _validate_session(session_dir, info)

        modality_path = session_dir / "meta" / "modality.json"
        embodiment_path = session_dir / "meta" / "embodiment.json"

        if args.overwrite_modality or not modality_path.exists():
            if not args.dry_run:
                _write_json(modality_path, _franka_dual_modality())
            wrote_modality += 1
        else:
            skipped += 1

        if args.overwrite_embodiment or not embodiment_path.exists():
            if not args.dry_run:
                _write_json(embodiment_path, {"embodiment_tag": FRANKA_DUAL_TAG})
            wrote_embodiment += 1

    mode = "would write" if args.dry_run else "wrote"
    print(f"scanned_sessions={len(sessions)} root={root}")
    print(f"{mode}_modality_json={wrote_modality}")
    print(f"{mode}_embodiment_json={wrote_embodiment}")
    print(f"existing_modality_json_skipped={skipped}")


if __name__ == "__main__":
    main()
