# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import torch

import cv2

from isaaclab_arena_gr00t.utils.io_utils import to_numpy


def resize_frames_with_padding(
    frames: torch.Tensor | np.ndarray, target_image_size: tuple, bgr_conversion: bool = False, pad_img: bool = True
) -> np.ndarray:
    """Process batch of frames with padding and resizing vectorized
    Args:
        frames: np.ndarray of shape [N, 256, 160, 3]
        target_image_size: target size (height, width)
        bgr_conversion: whether to convert BGR to RGB
        pad_img: whether to resize images
    """
    if not isinstance(frames, (torch.Tensor, np.ndarray)):
        raise ValueError(f"Invalid frame type: {type(frames)}")
    frames = to_numpy(frames)

    if bgr_conversion:
        frames = cv2.cvtColor(frames, cv2.COLOR_BGR2RGB)

    if pad_img:
        top_padding = (frames.shape[2] - frames.shape[1]) // 2
        bottom_padding = top_padding

        # Add padding to all frames at once
        frames = np.pad(
            frames,
            pad_width=((0, 0), (top_padding, bottom_padding), (0, 0), (0, 0)),
            mode="constant",
            constant_values=0,
        )

    # Resize all frames at once
    # frames.shape is (N, height, width, channels)
    # target_image_size is (height, width)
    # cv2.resize expects (width, height)
    if frames.shape[1:3] != target_image_size:
        target_size_cv2 = (target_image_size[1], target_image_size[0])  # Convert to (width, height)
        frames = np.stack([cv2.resize(f, target_size_cv2) for f in frames])

    return frames


def resize_frames_preserving_aspect(
    frames: torch.Tensor | np.ndarray, target_image_size: tuple, bgr_conversion: bool = False
) -> np.ndarray:
    """Resize frames to the policy's input size without distorting them.

    Reimplements GR00T's ``resize_with_pad`` (``examples/DROID/utils.py``), which is what the
    pretrained DROID checkpoints were fed: scale by ``max(width_ratio, height_ratio)`` so the whole
    frame fits, then centre it on a zero canvas. Only the short side is padded, and only when the
    aspect ratios genuinely differ -- a 720x1280 camera going to 180x320 scales by exactly 4 and
    needs no padding at all.

    This differs from :func:`resize_frames_with_padding`, which pads the frame to a square *before*
    resizing and therefore squashes a 16:9 camera by ~1.8x vertically. That function is kept as-is
    because the LeRobot conversion path has produced datasets with it.

    Args:
        frames: (N, H, W, C) uint8 frames.
        target_image_size: Target (height, width[, channels]).
        bgr_conversion: Whether the input is BGR and must be converted to RGB.

    Returns:
        (N, target_height, target_width, C) uint8 frames.
    """
    from PIL import Image

    if not isinstance(frames, (torch.Tensor, np.ndarray)):
        raise ValueError(f"Invalid frame type: {type(frames)}")
    frames = to_numpy(frames)
    if bgr_conversion:
        frames = frames[..., ::-1]

    target_height, target_width = int(target_image_size[0]), int(target_image_size[1])
    if frames.shape[1] == target_height and frames.shape[2] == target_width:
        return frames

    resized = []
    for frame in frames:
        current_height, current_width = frame.shape[:2]
        ratio = max(current_width / target_width, current_height / target_height)
        scaled_height, scaled_width = int(current_height / ratio), int(current_width / ratio)
        # PIL's BILINEAR reduction is area-weighted, so it antialiases the 4x downsample the way
        # the reference client does; cv2's INTER_LINEAR would alias instead.
        scaled = Image.fromarray(frame).resize((scaled_width, scaled_height), resample=Image.BILINEAR)
        canvas = Image.new(scaled.mode, (target_width, target_height), 0)
        canvas.paste(scaled, (max(0, (target_width - scaled_width) // 2), max(0, (target_height - scaled_height) // 2)))
        resized.append(np.asarray(canvas))
    return np.stack(resized)
