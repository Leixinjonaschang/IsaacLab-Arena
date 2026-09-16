# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Measure how AgiBot camera substitutions change a GR00T N1.7 action chunk.

The probe fixes the dataset state, language, target action, diffusion seed, and
denoising steps.  It then substitutes every combination of the three camera
views from two simulator traces.  A typical pair is an oracle-controlled trace
at the exact demonstration pose and a model-controlled trace after one chunk.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd

from gr00t.policy.server_client import PolicyClient


VIDEO_KEYS = ("ego_view", "left_wrist_view", "right_wrist_view")
STATE_KEYS = ("left_eef_9d", "right_eef_9d", "left_hand", "right_hand", "left_arm", "right_arm")
ACTION_KEYS = ("left_eef_9d", "right_eef_9d", "left_hand", "right_hand")
PREFIXES = (1, 8, 16, 40)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parquet", type=Path, required=True)
    parser.add_argument("--video-root", type=Path, required=True)
    parser.add_argument("--episode-index", type=int, required=True)
    parser.add_argument("--dataset-row", type=int, required=True)
    parser.add_argument("--exact-trace", type=Path, required=True)
    parser.add_argument("--drifted-trace", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5558)
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--denoising-steps", type=int, default=8)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument(
        "--suite",
        choices=("camera_matrix", "preprocessing"),
        default="camera_matrix",
        help="Run the 2^3 source-camera matrix or exact-render preprocessing ablations.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _read_rgb_frame(path: Path, index: int) -> np.ndarray:
    capture = cv2.VideoCapture(str(path))
    capture.set(cv2.CAP_PROP_POS_FRAMES, index)
    ok, bgr = capture.read()
    capture.release()
    if not ok:
        raise RuntimeError(f"Could not decode frame {index} from {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _dataset_video(video_root: Path, episode_index: int, row: int) -> dict[str, np.ndarray]:
    episode_name = f"episode_{episode_index:06d}.mp4"
    return {
        key: _read_rgb_frame(video_root / f"observation.images.{key}" / episode_name, row)[None, None]
        for key in VIDEO_KEYS
    }


def _trace_video(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as trace:
        video = {key: np.array(trace[f"video__{key}"], copy=True) for key in VIDEO_KEYS}
    for key, value in video.items():
        if value.shape != (1, 1, 512, 512, 3) or value.dtype != np.uint8:
            raise ValueError(f"Trace {path} camera {key} has unexpected shape/dtype {value.shape}/{value.dtype}")
    return video


def _dataset_state(dataframe: pd.DataFrame, row: int) -> dict[str, np.ndarray]:
    joints = np.asarray(dataframe.iloc[row]["observation.state"], dtype=np.float32)
    eef = np.asarray(dataframe.iloc[row]["observation.eef_9d"], dtype=np.float32)
    return {
        "left_eef_9d": eef[:9][None, None],
        "right_eef_9d": eef[9:18][None, None],
        "left_hand": joints[7:10][None, None],
        "right_hand": joints[17:20][None, None],
        "left_arm": joints[:7][None, None],
        "right_arm": joints[10:17][None, None],
    }


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


def _rotation_6d_to_matrix(value: np.ndarray) -> np.ndarray:
    first = np.asarray(value[..., :3], dtype=np.float64)
    first /= np.linalg.norm(first, axis=-1, keepdims=True)
    second = np.asarray(value[..., 3:6], dtype=np.float64)
    second -= np.sum(first * second, axis=-1, keepdims=True) * first
    second /= np.linalg.norm(second, axis=-1, keepdims=True)
    third = np.cross(first, second, axis=-1)
    return np.stack((first, second, third), axis=-1)


def _action_metrics(
    prediction: dict[str, np.ndarray], target: dict[str, np.ndarray], horizon: int
) -> dict[str, float]:
    predicted = {key: np.asarray(prediction[key])[0, :horizon] for key in ACTION_KEYS}
    metrics: dict[str, float] = {}
    for side in ("left", "right"):
        key = f"{side}_eef_9d"
        predicted_xyz = predicted[key][:, :3]
        target_xyz = target[key][:horizon, :3]
        metrics[f"{side}_position_error_mm"] = float(
            np.linalg.norm(predicted_xyz - target_xyz, axis=-1).mean() * 1000.0
        )
        predicted_delta = predicted_xyz - predicted_xyz[:1]
        target_delta = target_xyz - target_xyz[:1]
        metrics[f"{side}_shape_error_mm"] = float(
            np.linalg.norm(predicted_delta - target_delta, axis=-1).mean() * 1000.0
        )
        predicted_rotation = _rotation_6d_to_matrix(predicted[key][:, 3:9])
        target_rotation = _rotation_6d_to_matrix(target[key][:horizon, 3:9])
        relative_rotation = np.swapaxes(predicted_rotation, -1, -2) @ target_rotation
        cosine = np.clip((np.trace(relative_rotation, axis1=-2, axis2=-1) - 1.0) / 2.0, -1.0, 1.0)
        metrics[f"{side}_rotation_error_degrees"] = float(np.degrees(np.arccos(cosine)).mean())
        metrics[f"{side}_hand_mae"] = float(
            np.abs(predicted[f"{side}_hand"] - target[f"{side}_hand"][:horizon]).mean()
        )
    return metrics


def _median_action(samples: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    return {
        key: np.median(np.stack([sample[key] for sample in samples]), axis=0).astype(np.float32)
        for key in ACTION_KEYS
    }


def _anchor_translation(
    action: dict[str, np.ndarray], state: dict[str, np.ndarray]
) -> dict[str, np.ndarray]:
    anchored = {key: np.array(value, copy=True) for key, value in action.items()}
    for side in ("left", "right"):
        key = f"{side}_eef_9d"
        current_xyz = state[key][:, -1, :3]
        first_offset = anchored[key][:, 0, :3] - current_xyz
        anchored[key][..., :3] -= first_offset[:, None]
    return anchored


def _processor_spatial_transform(image: np.ndarray) -> np.ndarray:
    """Mirror this checkpoint's deterministic 256 -> 95% center crop -> 256 eval transform."""
    resized = cv2.resize(image, (256, 256), interpolation=cv2.INTER_AREA)
    crop_size = int(256 * 0.95)
    start = (256 - crop_size) // 2
    cropped = resized[start : start + crop_size, start : start + crop_size]
    return cv2.resize(cropped, (256, 256), interpolation=cv2.INTER_AREA)


def _ssim(image_a: np.ndarray, image_b: np.ndarray) -> float:
    a = image_a.astype(np.float64)
    b = image_b.astype(np.float64)
    mu_a = cv2.GaussianBlur(a, (11, 11), 1.5)
    mu_b = cv2.GaussianBlur(b, (11, 11), 1.5)
    sigma_a = cv2.GaussianBlur(a * a, (11, 11), 1.5) - mu_a * mu_a
    sigma_b = cv2.GaussianBlur(b * b, (11, 11), 1.5) - mu_b * mu_b
    sigma_ab = cv2.GaussianBlur(a * b, (11, 11), 1.5) - mu_a * mu_b
    c1 = (0.01 * 255.0) ** 2
    c2 = (0.03 * 255.0) ** 2
    score = ((2.0 * mu_a * mu_b + c1) * (2.0 * sigma_ab + c2)) / (
        (mu_a * mu_a + mu_b * mu_b + c1) * (sigma_a + sigma_b + c2)
    )
    return float(score.mean())


def _image_metrics(image_a: np.ndarray, image_b: np.ndarray) -> dict[str, float]:
    delta = image_a.astype(np.float64) - image_b.astype(np.float64)
    mse = float(np.mean(delta * delta))
    return {
        "mae_uint8": float(np.abs(delta).mean()),
        "rmse_uint8": float(np.sqrt(mse)),
        "psnr_db": float("inf") if mse == 0.0 else float(20.0 * np.log10(255.0 / np.sqrt(mse))),
        "ssim": _ssim(image_a, image_b),
    }


def _save_image_diagnostics(
    output_dir: Path,
    dataset_video: dict[str, np.ndarray],
    exact_video: dict[str, np.ndarray],
    drifted_video: dict[str, np.ndarray],
) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    for key in VIDEO_KEYS:
        images = {
            "dataset": dataset_video[key][0, 0],
            "exact": exact_video[key][0, 0],
            "drifted": drifted_video[key][0, 0],
        }
        metrics[key] = {}
        for left, right in (("dataset", "exact"), ("dataset", "drifted"), ("exact", "drifted")):
            pair = f"{left}__{right}"
            metrics[key][pair] = {
                "raw": _image_metrics(images[left], images[right]),
                "processor_spatial": _image_metrics(
                    _processor_spatial_transform(images[left]), _processor_spatial_transform(images[right])
                ),
            }

        titled = []
        for name in ("dataset", "exact", "drifted"):
            bgr = cv2.cvtColor(images[name], cv2.COLOR_RGB2BGR)
            cv2.putText(bgr, name, (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
            titled.append(bgr)
        exact_diff = np.clip(np.abs(images["exact"].astype(np.int16) - images["dataset"].astype(np.int16)) * 5, 0, 255)
        drift_diff = np.clip(
            np.abs(images["drifted"].astype(np.int16) - images["dataset"].astype(np.int16)) * 5, 0, 255
        )
        for label, diff in (("5x |exact-dataset|", exact_diff), ("5x |drifted-dataset|", drift_diff)):
            bgr = cv2.cvtColor(diff.astype(np.uint8), cv2.COLOR_RGB2BGR)
            cv2.putText(bgr, label, (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            titled.append(bgr)
        cv2.imwrite(str(output_dir / f"{key}_comparison.png"), np.concatenate(titled, axis=1))
    return metrics


def _camera_variants(
    dataset_video: dict[str, np.ndarray], source_video: dict[str, np.ndarray], source_name: str
) -> dict[str, dict[str, np.ndarray]]:
    variants = {}
    for bits in itertools.product((0, 1), repeat=len(VIDEO_KEYS)):
        labels = "_".join(f"{key[:1].upper()}{bit}" for key, bit in zip(VIDEO_KEYS, bits, strict=True))
        name = f"{source_name}__{labels}"
        variants[name] = {
            key: source_video[key] if bit else dataset_video[key]
            for key, bit in zip(VIDEO_KEYS, bits, strict=True)
        }
    return variants


def _map_video(video: dict[str, np.ndarray], transform) -> dict[str, np.ndarray]:
    return {key: transform(video[key][0, 0], key)[None, None] for key in VIDEO_KEYS}


def _yuv420_round_trip(image: np.ndarray, _: str) -> np.ndarray:
    height, width = image.shape[:2]
    if height % 2 or width % 2:
        raise ValueError(f"YUV420 requires even dimensions, got {image.shape}")
    yuv = cv2.cvtColor(image, cv2.COLOR_RGB2YUV_I420)
    return cv2.cvtColor(yuv, cv2.COLOR_YUV2RGB_I420)


def _h264_round_trip_factory(output_dir: Path):
    cache: dict[str, np.ndarray] = {}

    def transform(image: np.ndarray, key: str) -> np.ndarray:
        if key in cache:
            return cache[key]
        import av

        path = output_dir / f"exact_{key}_h264_round_trip.mp4"
        container = av.open(str(path), mode="w")
        stream = container.add_stream("libx264", rate=15)
        stream.width = image.shape[1]
        stream.height = image.shape[0]
        stream.pix_fmt = "yuv420p"
        try:
            # Row 26 is preceded by a long stationary ready pose. Repeated frames reproduce
            # the relevant inter-frame coding regime without needing all original raw frames.
            for _ in range(32):
                frame = av.VideoFrame.from_ndarray(image, format="rgb24")
                for packet in stream.encode(frame):
                    container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)
        finally:
            container.close()
        cache[key] = _read_rgb_frame(path, 26)
        return cache[key]

    return transform


def _histogram_match(source: np.ndarray, reference: np.ndarray) -> np.ndarray:
    matched = np.empty_like(source)
    for channel in range(3):
        values, inverse, counts = np.unique(source[..., channel], return_inverse=True, return_counts=True)
        reference_values, reference_counts = np.unique(reference[..., channel], return_counts=True)
        source_quantiles = np.cumsum(counts).astype(np.float64)
        source_quantiles /= source_quantiles[-1]
        reference_quantiles = np.cumsum(reference_counts).astype(np.float64)
        reference_quantiles /= reference_quantiles[-1]
        mapped = np.interp(source_quantiles, reference_quantiles, reference_values)
        matched[..., channel] = mapped[inverse].reshape(source.shape[:2]).round().astype(np.uint8)
    return matched


def _dataset_transform_factory(dataset_video: dict[str, np.ndarray], transform_name: str):
    def transform(image: np.ndarray, key: str) -> np.ndarray:
        reference = dataset_video[key][0, 0]
        if transform_name == "histogram":
            return _histogram_match(image, reference)
        if transform_name in ("ecc", "ecc_histogram"):
            template_gray = cv2.cvtColor(reference, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
            image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
            warp = np.eye(2, 3, dtype=np.float32)
            _, warp = cv2.findTransformECC(
                template_gray,
                image_gray,
                warp,
                cv2.MOTION_EUCLIDEAN,
                (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 200, 1e-7),
                None,
                5,
            )
            image = cv2.warpAffine(
                image,
                warp,
                (reference.shape[1], reference.shape[0]),
                flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                borderMode=cv2.BORDER_REFLECT,
            )
            if transform_name == "ecc_histogram":
                image = _histogram_match(image, reference)
            return image
        raise ValueError(f"Unknown dataset-guided transform: {transform_name}")

    return transform


def _preprocessing_variants(
    output_dir: Path,
    dataset_video: dict[str, np.ndarray],
    exact_video: dict[str, np.ndarray],
) -> dict[str, dict[str, np.ndarray]]:
    transformed = {
        "raw": exact_video,
        "yuv420": _map_video(exact_video, _yuv420_round_trip),
        "h264": _map_video(exact_video, _h264_round_trip_factory(output_dir)),
        "histogram": _map_video(exact_video, _dataset_transform_factory(dataset_video, "histogram")),
        "ecc": _map_video(exact_video, _dataset_transform_factory(dataset_video, "ecc")),
        "ecc_histogram": _map_video(
            exact_video, _dataset_transform_factory(dataset_video, "ecc_histogram")
        ),
    }
    variants = {"dataset_all": dataset_video}
    for transform_name, video in transformed.items():
        variants[f"exact_{transform_name}_all"] = video
        variants[f"exact_{transform_name}_right_only"] = {
            key: video[key] if key == "right_wrist_view" else dataset_video[key] for key in VIDEO_KEYS
        }
    return variants


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return str(value)
    return value


def main() -> None:
    args = _parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    dataframe = pd.read_parquet(args.parquet)
    dataset_video = _dataset_video(args.video_root, args.episode_index, args.dataset_row)
    exact_video = _trace_video(args.exact_trace)
    drifted_video = _trace_video(args.drifted_trace)
    state = _dataset_state(dataframe, args.dataset_row)
    target = _dataset_target(dataframe, args.dataset_row)

    results: dict[str, Any] = {
        "config": {
            "parquet": str(args.parquet.resolve()),
            "video_root": str(args.video_root.resolve()),
            "episode_index": args.episode_index,
            "dataset_row": args.dataset_row,
            "exact_trace": str(args.exact_trace),
            "drifted_trace": str(args.drifted_trace),
            "host": args.host,
            "port": args.port,
            "seed": args.seed,
            "denoising_steps": args.denoising_steps,
            "samples": args.samples,
            "suite": args.suite,
        },
        "image_metrics": _save_image_diagnostics(args.output_dir, dataset_video, exact_video, drifted_video),
        "variants": {},
    }

    if args.suite == "camera_matrix":
        variants = _camera_variants(dataset_video, exact_video, "exact")
        drifted_variants = _camera_variants(dataset_video, drifted_video, "drifted")
        drifted_variants.pop("drifted__E0_L0_R0")
        variants.update(drifted_variants)
    else:
        variants = _preprocessing_variants(args.output_dir, dataset_video, exact_video)

    client = PolicyClient(host=args.host, port=args.port, timeout_ms=180_000, strict=False)
    if not client.ping():
        raise ConnectionError(f"GR00T server is not reachable at {args.host}:{args.port}")
    observations = {
        name: {
            "language": {"annotation.human.action.task_description": [["Stack the bowls together."]]},
            "video": video,
            "state": state,
        }
        for name, video in variants.items()
    }

    arrays: dict[str, np.ndarray] = {}
    for index, (name, observation) in enumerate(observations.items(), start=1):
        reset_info = client.reset(options={"seed": args.seed})
        samples = []
        infos = []
        for _ in range(args.samples):
            action, info = client.get_action(
                observation, options={"num_inference_timesteps": args.denoising_steps}
            )
            samples.append({key: np.asarray(action[key], dtype=np.float32) for key in ACTION_KEYS})
            infos.append(info)
        if any(info.get("num_inference_timesteps") != args.denoising_steps for info in infos):
            raise RuntimeError(f"Server did not acknowledge denoising_steps={args.denoising_steps} for {name}: {infos}")
        median = _median_action(samples)
        anchored = _anchor_translation(median, state)
        entry = {
            "reset_info": reset_info,
            "server_info": infos,
            "raw_median": {f"prefix_{horizon}": _action_metrics(median, target, horizon) for horizon in PREFIXES},
            "anchored_median": {
                f"prefix_{horizon}": _action_metrics(anchored, target, horizon) for horizon in PREFIXES
            },
        }
        results["variants"][name] = entry
        for sample_index, sample in enumerate(samples):
            for key, value in sample.items():
                arrays[f"{name}__sample_{sample_index:02d}__{key}"] = value
        for key, value in median.items():
            arrays[f"{name}__raw_median__{key}"] = value
            arrays[f"{name}__anchored_median__{key}"] = anchored[key]
        compact = entry["anchored_median"]
        print(
            f"[{index:02d}/{len(observations)}] {name}: "
            f"L8={compact['prefix_8']['left_position_error_mm']:.1f} "
            f"L40={compact['prefix_40']['left_position_error_mm']:.1f} "
            f"R8={compact['prefix_8']['right_position_error_mm']:.1f} "
            f"R40={compact['prefix_40']['right_position_error_mm']:.1f} mm",
            flush=True,
        )

    np.savez_compressed(args.output_dir / "actions.npz", **arrays)
    (args.output_dir / "results.json").write_text(json.dumps(_json_safe(results), indent=2) + "\n")
    print(f"Wrote {args.output_dir / 'results.json'}", flush=True)


if __name__ == "__main__":
    main()
