#!/usr/bin/env python
"""Small msgpack client for the Franka OpenPI server."""

from __future__ import annotations

import argparse
from typing import Any

import msgpack
import numpy as np
import requests


def _pack_array(obj: Any):
    if isinstance(obj, np.ndarray):
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="http://127.0.0.1:8000")
    parser.add_argument("--prompt", default="test dual franka action")
    args = parser.parse_args()

    rng = np.random.default_rng(0)
    raw_state = np.zeros((16,), dtype=np.float32)
    payload = {
        "state": raw_state,
        "images": {
            "middle_zed": rng.integers(0, 255, (376, 672, 3), dtype=np.uint8),
            "left_camera": rng.integers(0, 255, (480, 640, 3), dtype=np.uint8),
            "right_camera": rng.integers(0, 255, (480, 640, 3), dtype=np.uint8),
        },
        "openpi": {
            "preprocessed": True,
            "format": "openpi_pi05_client_preprocessed_v1",
            "state": np.zeros((32,), dtype=np.float32),
            "state_mask": np.r_[np.ones(16, dtype=bool), np.zeros(16, dtype=bool)],
            "images": {
                "base_0_rgb": rng.uniform(-1.0, 1.0, (3, 224, 224)).astype(np.float32),
                "left_wrist_0_rgb": rng.uniform(-1.0, 1.0, (3, 224, 224)).astype(np.float32),
                "right_wrist_0_rgb": rng.uniform(-1.0, 1.0, (3, 224, 224)).astype(np.float32),
            },
            "image_mask": {
                "base_0_rgb": np.bool_(True),
                "left_wrist_0_rgb": np.bool_(True),
                "right_wrist_0_rgb": np.bool_(True),
            },
            "action_q01": np.r_[-np.ones(16, dtype=np.float32), np.zeros(16, dtype=np.float32)],
            "action_q99": np.r_[np.ones(16, dtype=np.float32), np.zeros(16, dtype=np.float32)],
        },
        "prompt": args.prompt,
    }
    resp = requests.post(
        args.host.rstrip("/") + "/infer",
        data=packb(payload),
        headers={"Content-Type": "application/msgpack"},
        timeout=120,
    )
    resp.raise_for_status()
    data = unpackb(resp.content)
    actions = np.asarray(data["actions"], dtype=np.float32)
    print("actions.shape =", actions.shape)
    print("actions[0] =", actions[0].tolist())


if __name__ == "__main__":
    main()
