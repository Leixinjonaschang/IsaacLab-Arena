# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Compare two AgiBot GR00T rollouts that differ only in camera warmup length."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd


ACTION_KEYS = ("left_eef_9d", "right_eef_9d", "left_hand", "right_hand")
VIDEO_KEYS = ("ego_view", "left_wrist_view", "right_wrist_view")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parquet", type=Path, required=True)
    parser.add_argument("--video-root", type=Path)
    parser.add_argument("--episode-index", type=int, required=True)
    parser.add_argument("--baseline-traces", type=Path, required=True)
    parser.add_argument("--candidate-traces", type=Path, required=True)
    parser.add_argument("--baseline-start-row", type=int, required=True)
    parser.add_argument("--candidate-start-row", type=int, required=True)
    parser.add_argument("--query-stride", type=int, default=16)
    parser.add_argument("--nearest-row-limit", type=int, default=300)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _dataset_target(dataframe: pd.DataFrame, row: int) -> dict[str, np.ndarray]:
    eef = np.stack(dataframe["action.eef_9d"].to_numpy()).astype(np.float32)[row : row + 40]
    joints = np.stack(dataframe["action"].to_numpy()).astype(np.float32)[row : row + 40]
    if len(eef) != 40 or len(joints) != 40:
        raise ValueError(f"Dataset row {row} does not have a complete 40-step target")
    return {
        "left_eef_9d": eef[:, :9],
        "right_eef_9d": eef[:, 9:18],
        "left_hand": joints[:, 7:10],
        "right_hand": joints[:, 17:20],
    }


def _prediction(trace: np.lib.npyio.NpzFile) -> dict[str, np.ndarray]:
    return {key: np.asarray(trace[f"server_action__{key}"])[0] for key in ACTION_KEYS}


def _state(trace: np.lib.npyio.NpzFile) -> tuple[np.ndarray, np.ndarray]:
    eef = np.concatenate(
        (
            trace["state__left_eef_9d"][0, 0],
            trace["state__right_eef_9d"][0, 0],
        )
    )
    joints = np.concatenate(
        (
            trace["state__left_arm"][0, 0],
            trace["state__left_hand"][0, 0],
            trace["state__right_arm"][0, 0],
            trace["state__right_hand"][0, 0],
        )
    )
    return eef, joints


def _action_metrics(
    prediction: dict[str, np.ndarray], target: dict[str, np.ndarray], horizon: int
) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for side in ("left", "right"):
        predicted_xyz = prediction[f"{side}_eef_9d"][:horizon, :3]
        target_xyz = target[f"{side}_eef_9d"][:horizon, :3]
        metrics[f"{side}_position_error_mm"] = float(
            np.linalg.norm(predicted_xyz - target_xyz, axis=-1).mean() * 1000.0
        )
        metrics[f"{side}_shape_error_mm"] = float(
            np.linalg.norm(
                (predicted_xyz - predicted_xyz[:1]) - (target_xyz - target_xyz[:1]), axis=-1
            ).mean()
            * 1000.0
        )
        metrics[f"{side}_hand_mae"] = float(
            np.abs(prediction[f"{side}_hand"][:horizon] - target[f"{side}_hand"][:horizon]).mean()
        )
    return metrics


def _motion_onset(xyz: np.ndarray, threshold_m: float = 0.005) -> int | None:
    moving = np.flatnonzero(np.linalg.norm(xyz - xyz[:1], axis=-1) > threshold_m)
    return int(moving[0]) if len(moving) else None


def _processor_spatial_transform(image: np.ndarray) -> np.ndarray:
    resized = cv2.resize(image, (256, 256), interpolation=cv2.INTER_AREA)
    crop_size = int(256 * 0.95)
    start = (256 - crop_size) // 2
    crop = resized[start : start + crop_size, start : start + crop_size]
    return cv2.resize(crop, (256, 256), interpolation=cv2.INTER_AREA)


def _read_video_frame(path: Path, row: int) -> np.ndarray:
    capture = cv2.VideoCapture(str(path))
    capture.set(cv2.CAP_PROP_POS_FRAMES, row)
    ok, bgr = capture.read()
    capture.release()
    if not ok:
        raise RuntimeError(f"Failed to read frame {row} from {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _initial_image_metrics(
    trace_path: Path, video_root: Path, episode_index: int, row: int
) -> dict[str, dict[str, float]]:
    trace = np.load(trace_path, allow_pickle=False)
    metrics = {}
    for key in VIDEO_KEYS:
        live = trace[f"video__{key}"][0, 0]
        video_path = (
            video_root / f"observation.images.{key}" / f"episode_{episode_index:06d}.mp4"
        )
        dataset = _read_video_frame(video_path, row)
        raw_delta = live.astype(np.float64) - dataset.astype(np.float64)
        processed_live = _processor_spatial_transform(live)
        processed_dataset = _processor_spatial_transform(dataset)
        processed_delta = processed_live.astype(np.float64) - processed_dataset.astype(np.float64)
        metrics[key] = {
            "raw_mae_uint8": float(np.abs(raw_delta).mean()),
            "processor_spatial_mae_uint8": float(np.abs(processed_delta).mean()),
        }
    return metrics


def _analyze_case(
    trace_dir: Path,
    start_row: int,
    query_stride: int,
    dataframe: pd.DataFrame,
    observation_eef: np.ndarray,
    observation_joints: np.ndarray,
    nearest_row_limit: int,
) -> dict[str, Any]:
    trace_paths = sorted(trace_dir.glob("inference_*.npz"))
    if not trace_paths:
        raise ValueError(f"No inference traces found in {trace_dir}")
    traces = [np.load(path, allow_pickle=False) for path in trace_paths]
    arm_indices = np.r_[0:7, 10:17]
    queries = []
    for query_index, (trace_path, trace) in enumerate(zip(trace_paths, traces, strict=True)):
        row = start_row + query_stride * query_index
        state_eef, state_joints = _state(trace)
        target = _dataset_target(dataframe, row)
        prediction = _prediction(trace)
        left_distances = np.linalg.norm(
            observation_eef[:nearest_row_limit, :3] - state_eef[:3], axis=-1
        )
        right_distances = np.linalg.norm(
            observation_eef[:nearest_row_limit, 9:12] - state_eef[9:12], axis=-1
        )
        eef_distances_mm = (left_distances + right_distances) * 500.0
        arm_distances = np.sqrt(
            np.mean(
                (
                    observation_joints[:nearest_row_limit, arm_indices]
                    - state_joints[arm_indices]
                )
                ** 2,
                axis=-1,
            )
        )
        query: dict[str, Any] = {
            "query_index": query_index,
            "trace": str(trace_path),
            "dataset_row": row,
            "state_error": {
                "left_position_mm": float(
                    np.linalg.norm(state_eef[:3] - observation_eef[row, :3]) * 1000.0
                ),
                "right_position_mm": float(
                    np.linalg.norm(state_eef[9:12] - observation_eef[row, 9:12]) * 1000.0
                ),
                "arm_joint_rms_rad": float(
                    np.sqrt(
                        np.mean(
                            (
                                state_joints[arm_indices]
                                - observation_joints[row, arm_indices]
                            )
                            ** 2
                        )
                    )
                ),
                "nearest_eef_row": int(np.argmin(eef_distances_mm)),
                "nearest_eef_mean_position_mm": float(np.min(eef_distances_mm)),
                "nearest_arm_row": int(np.argmin(arm_distances)),
                "nearest_arm_joint_rms_rad": float(np.min(arm_distances)),
            },
            "action_metrics": {
                str(horizon): _action_metrics(prediction, target, horizon)
                for horizon in (8, 16, 40)
            },
            "motion_onset_over_5mm": {
                side: {
                    "prediction": _motion_onset(prediction[f"{side}_eef_9d"][:, :3]),
                    "target": _motion_onset(target[f"{side}_eef_9d"][:, :3]),
                }
                for side in ("left", "right")
            },
            "scene_positions": {
                f"bowl{index}": trace[f"scene__bowl{index}__root_pose_w"][0, :3].tolist()
                for index in range(3)
            },
        }
        if query_index + 1 < len(traces):
            next_row = row + query_stride
            next_eef, _ = _state(traces[query_index + 1])
            transition = {}
            # The delayed state at the next query corresponds to the command at index stride - 2.
            command_index = query_stride - 2
            for side, dataset_slice in (("left", slice(0, 3)), ("right", slice(9, 12))):
                command = prediction[f"{side}_eef_9d"][command_index, :3]
                actual = next_eef[dataset_slice]
                expected = observation_eef[next_row, dataset_slice]
                transition[side] = {
                    "command_index": command_index,
                    "model_command_error_mm": float(np.linalg.norm(command - expected) * 1000.0),
                    "controller_tracking_error_mm": float(np.linalg.norm(actual - command) * 1000.0),
                    "total_next_state_error_mm": float(np.linalg.norm(actual - expected) * 1000.0),
                }
            query["transition_to_next_query"] = transition
        queries.append(query)
    return {"trace_count": len(trace_paths), "start_row": start_row, "queries": queries}


def _aligned_comparison(baseline: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    baseline_by_row = {query["dataset_row"]: query for query in baseline["queries"]}
    candidate_by_row = {query["dataset_row"]: query for query in candidate["queries"]}
    rows = sorted(set(baseline_by_row) & set(candidate_by_row))
    comparisons = []
    for row in rows:
        baseline_query = baseline_by_row[row]
        candidate_query = candidate_by_row[row]
        comparisons.append(
            {
                "dataset_row": row,
                "baseline_state_left_position_mm": baseline_query["state_error"][
                    "left_position_mm"
                ],
                "candidate_state_left_position_mm": candidate_query["state_error"][
                    "left_position_mm"
                ],
                "baseline_state_right_position_mm": baseline_query["state_error"][
                    "right_position_mm"
                ],
                "candidate_state_right_position_mm": candidate_query["state_error"][
                    "right_position_mm"
                ],
            }
        )
    post_initial = comparisons[1:]
    summary: dict[str, float | list[int]] = {
        "dataset_rows": [item["dataset_row"] for item in post_initial]
    }
    for side in ("left", "right"):
        baseline_values = np.asarray(
            [item[f"baseline_state_{side}_position_mm"] for item in post_initial]
        )
        candidate_values = np.asarray(
            [item[f"candidate_state_{side}_position_mm"] for item in post_initial]
        )
        if len(baseline_values):
            baseline_mean = float(baseline_values.mean())
            candidate_mean = float(candidate_values.mean())
            summary[f"baseline_mean_{side}_position_mm"] = baseline_mean
            summary[f"candidate_mean_{side}_position_mm"] = candidate_mean
            summary[f"candidate_reduction_{side}_percent"] = float(
                (1.0 - candidate_mean / baseline_mean) * 100.0
            )
    return {
        "common_dataset_rows": rows,
        "state_error_by_row": comparisons,
        "post_initial_summary": summary,
    }


def main() -> None:
    args = _parse_args()
    dataframe = pd.read_parquet(args.parquet)
    observation_eef = np.stack(dataframe["observation.eef_9d"].to_numpy()).astype(np.float32)
    observation_joints = np.stack(dataframe["observation.state"].to_numpy()).astype(np.float32)
    baseline = _analyze_case(
        args.baseline_traces,
        args.baseline_start_row,
        args.query_stride,
        dataframe,
        observation_eef,
        observation_joints,
        args.nearest_row_limit,
    )
    candidate = _analyze_case(
        args.candidate_traces,
        args.candidate_start_row,
        args.query_stride,
        dataframe,
        observation_eef,
        observation_joints,
        args.nearest_row_limit,
    )
    result: dict[str, Any] = {
        "parquet": str(args.parquet),
        "episode_index": args.episode_index,
        "query_stride": args.query_stride,
        "baseline": baseline,
        "candidate": candidate,
        "aligned_comparison": _aligned_comparison(baseline, candidate),
    }
    if args.video_root is not None:
        baseline_first = sorted(args.baseline_traces.glob("inference_*.npz"))[0]
        candidate_first = sorted(args.candidate_traces.glob("inference_*.npz"))[0]
        result["initial_live_vs_dataset_image_metrics"] = {
            "baseline": _initial_image_metrics(
                baseline_first,
                args.video_root,
                args.episode_index,
                args.baseline_start_row,
            ),
            "candidate": _initial_image_metrics(
                candidate_first,
                args.video_root,
                args.episode_index,
                args.candidate_start_row,
            ),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
