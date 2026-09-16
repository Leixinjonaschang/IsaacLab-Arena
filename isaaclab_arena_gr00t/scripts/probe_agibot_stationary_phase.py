# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Probe GR00T across nearly stationary dataset rows with phase-shifted action labels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from gr00t.policy.server_client import PolicyClient
from isaaclab_arena_gr00t.scripts.probe_agibot_visual_contract import (
    ACTION_KEYS,
    VIDEO_KEYS,
    _action_metrics,
    _anchor_translation,
    _dataset_state,
    _dataset_target,
    _dataset_video,
    _median_action,
    _processor_spatial_transform,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parquet", type=Path, required=True)
    parser.add_argument("--video-root", type=Path, required=True)
    parser.add_argument("--episode-index", type=int, required=True)
    parser.add_argument("--rows", type=int, nargs="+", required=True)
    parser.add_argument("--reference-row", type=int, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5558)
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--denoising-steps", type=int, default=8)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _flatten_state(state: dict[str, np.ndarray]) -> np.ndarray:
    return np.concatenate([state[key].reshape(-1) for key in state])


def _image_distance(
    video: dict[str, np.ndarray], reference: dict[str, np.ndarray]
) -> dict[str, dict[str, float]]:
    result = {}
    for key in VIDEO_KEYS:
        image = video[key][0, 0]
        ref = reference[key][0, 0]
        processed = _processor_spatial_transform(image)
        processed_ref = _processor_spatial_transform(ref)
        result[key] = {
            "raw_mae_uint8": float(np.abs(image.astype(np.float64) - ref.astype(np.float64)).mean()),
            "processor_spatial_mae_uint8": float(
                np.abs(processed.astype(np.float64) - processed_ref.astype(np.float64)).mean()
            ),
        }
    return result


def _motion_summary(action: dict[str, np.ndarray], side: str = "left") -> dict[str, float | int | None]:
    xyz = np.asarray(action[f"{side}_eef_9d"])[0, :, :3]
    displacement_mm = np.linalg.norm(xyz - xyz[:1], axis=-1) * 1000.0
    moving = np.flatnonzero(displacement_mm > 5.0)
    return {
        "first_over_5mm": int(moving[0]) if len(moving) else None,
        "step_7_displacement_mm": float(displacement_mm[7]),
        "step_15_displacement_mm": float(displacement_mm[15]),
        "step_39_displacement_mm": float(displacement_mm[39]),
    }


def _target_as_batched(target: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {key: value[None] for key, value in target.items()}


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def main() -> None:
    args = _parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    dataframe = pd.read_parquet(args.parquet)
    reference_video = _dataset_video(args.video_root, args.episode_index, args.reference_row)
    reference_state = _dataset_state(dataframe, args.reference_row)
    reference_state_flat = _flatten_state(reference_state)

    client = PolicyClient(host=args.host, port=args.port, timeout_ms=180_000, strict=False)
    if not client.ping():
        raise ConnectionError(f"GR00T server is not reachable at {args.host}:{args.port}")

    results: dict[str, Any] = {
        "config": {
            "parquet": str(args.parquet.resolve()),
            "video_root": str(args.video_root.resolve()),
            "episode_index": args.episode_index,
            "rows": args.rows,
            "reference_row": args.reference_row,
            "host": args.host,
            "port": args.port,
            "seed": args.seed,
            "denoising_steps": args.denoising_steps,
            "samples": args.samples,
        },
        "rows": {},
    }
    arrays: dict[str, np.ndarray] = {}
    for row in args.rows:
        video = _dataset_video(args.video_root, args.episode_index, row)
        state = _dataset_state(dataframe, row)
        target = _dataset_target(dataframe, row)
        observation = {
            "language": {"annotation.human.action.task_description": [["Stack the bowls together."]]},
            "video": video,
            "state": state,
        }
        reset_info = client.reset(options={"seed": args.seed})
        samples = []
        infos = []
        for _ in range(args.samples):
            action, info = client.get_action(
                observation, options={"num_inference_timesteps": args.denoising_steps}
            )
            samples.append({key: np.asarray(action[key], dtype=np.float32) for key in ACTION_KEYS})
            infos.append(info)
        median = _anchor_translation(_median_action(samples), state)
        target_batched = _target_as_batched(target)
        state_delta = _flatten_state(state) - reference_state_flat
        entry = {
            "reset_info": reset_info,
            "server_info": infos,
            "state_rms_from_reference": float(np.sqrt(np.mean(state_delta**2))),
            "state_max_abs_from_reference": float(np.abs(state_delta).max()),
            "image_distance_from_reference": _image_distance(video, reference_video),
            "prediction_metrics": {
                f"prefix_{horizon}": _action_metrics(median, target, horizon) for horizon in (1, 8, 16, 40)
            },
            "prediction_motion": _motion_summary(median),
            "target_motion": _motion_summary(target_batched),
        }
        results["rows"][str(row)] = entry
        for key, value in median.items():
            arrays[f"row_{row:04d}__median__{key}"] = value
        metrics = entry["prediction_metrics"]
        print(
            f"row={row}: state_rms={entry['state_rms_from_reference']:.2e} "
            f"L8={metrics['prefix_8']['left_position_error_mm']:.1f} "
            f"L16={metrics['prefix_16']['left_position_error_mm']:.1f} "
            f"L40={metrics['prefix_40']['left_position_error_mm']:.1f} "
            f"onset pred/target={entry['prediction_motion']['first_over_5mm']}/"
            f"{entry['target_motion']['first_over_5mm']}",
            flush=True,
        )

    np.savez_compressed(args.output_dir / "actions.npz", **arrays)
    (args.output_dir / "results.json").write_text(json.dumps(_json_safe(results), indent=2) + "\n")
    print(f"Wrote {args.output_dir / 'results.json'}", flush=True)


if __name__ == "__main__":
    main()
