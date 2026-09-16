# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""AgiBot end-effector translation for the Arena GR00T N1.7 checkpoint."""

from __future__ import annotations

import numpy as np

AGIBOT_GR00T_ACTION_DIM = 20
"""Simulator action layout: left pose (7), left hand (3), right pose (7), right hand (3)."""

AGIBOT_GR00T_STATE_WIDTHS = {
    "left_eef_9d": 9,
    "right_eef_9d": 9,
    "left_hand": 3,
    "right_hand": 3,
    "left_arm": 7,
    "right_arm": 7,
}
"""Resolved ``new_embodiment`` state contract shipped with the fine-tuned checkpoint."""

AGIBOT_GR00T_ACTION_WIDTHS = {
    "left_eef_9d": 9,
    "right_eef_9d": 9,
    "left_hand": 3,
    "right_hand": 3,
}
"""Resolved ``new_embodiment`` action contract shipped with the fine-tuned checkpoint."""

# FrameTransformer and RMPFlow use scalar-last (x, y, z, w) quaternions in the pinned Isaac Lab.
# The final fine-tuning dataset stores raw USD gripper-center orientations. The left controller frame
# has this fixed offset from that raw frame; the right controller frame has no offset.
_CONTROL_LEFT_OFFSET_XYZW = np.array([0.0, -0.7071, 0.0, 0.7071], dtype=np.float64)

# Dataset min/max documented by checkpoint-10000/README.md. Clamping avoids sending an
# out-of-distribution diffusion sample to a stiff position controller.
_WORKSPACE_BOUNDS = {
    "left": (
        np.array([0.544, -0.132, 0.663], dtype=np.float64),
        np.array([1.157, 0.963, 1.314], dtype=np.float64),
    ),
    "right": (
        np.array([0.666, -0.890, 0.663], dtype=np.float64),
        np.array([1.148, 0.202, 1.583], dtype=np.float64),
    ),
}


def _rotation_class():
    from scipy.spatial.transform import Rotation

    return Rotation


def _normalized_quaternion_xyzw(quaternion: np.ndarray) -> np.ndarray:
    quaternion = np.asarray(quaternion, dtype=np.float64)
    assert quaternion.shape[-1] == 4, f"Expected (..., 4) quaternion, got {quaternion.shape}"
    norms = np.linalg.norm(quaternion, axis=-1, keepdims=True)
    assert np.all(norms > 1e-8), "Cannot normalize a zero-length quaternion"
    return quaternion / norms


def _quaternion_xyzw_to_matrix(quaternion: np.ndarray) -> np.ndarray:
    quaternion = _normalized_quaternion_xyzw(quaternion)
    shape = quaternion.shape[:-1]
    return _rotation_class().from_quat(quaternion.reshape(-1, 4)).as_matrix().reshape(*shape, 3, 3)


def _matrix_to_quaternion_xyzw(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64)
    assert matrix.shape[-2:] == (3, 3), f"Expected (..., 3, 3) matrix, got {matrix.shape}"
    shape = matrix.shape[:-2]
    quaternion = _rotation_class().from_matrix(matrix.reshape(-1, 3, 3)).as_quat().reshape(*shape, 4)
    # Pick one of q/-q deterministically; RMPFlow accepts either, but stable chunks are easier to inspect.
    return np.where(quaternion[..., 3:4] < 0.0, -quaternion, quaternion)


def rotation_6d_to_matrix(rotation_6d: np.ndarray) -> np.ndarray:
    """Decode concatenated first-two rotation columns with Gram-Schmidt orthogonalization."""
    rotation_6d = np.asarray(rotation_6d, dtype=np.float64)
    assert rotation_6d.shape[-1] == 6, f"Expected (..., 6) rotation, got {rotation_6d.shape}"
    first = rotation_6d[..., :3]
    first_norm = np.linalg.norm(first, axis=-1, keepdims=True)
    assert np.all(first_norm > 1e-8), "Rotation-6D first column is degenerate"
    first = first / first_norm
    second = rotation_6d[..., 3:]
    second = second - np.sum(first * second, axis=-1, keepdims=True) * first
    second_norm = np.linalg.norm(second, axis=-1, keepdims=True)
    assert np.all(second_norm > 1e-8), "Rotation-6D second column is degenerate"
    second = second / second_norm
    third = np.cross(first, second, axis=-1)
    return np.stack((first, second, third), axis=-1)


def _matrix_to_rotation_6d(matrix: np.ndarray) -> np.ndarray:
    return np.concatenate((matrix[..., :, 0], matrix[..., :, 1]), axis=-1)


def _control_offset_matrix(side: str) -> np.ndarray:
    assert side in ("left", "right"), f"Unsupported AgiBot side: {side}"
    if side == "left":
        return _quaternion_xyzw_to_matrix(_CONTROL_LEFT_OFFSET_XYZW)
    return np.eye(3, dtype=np.float64)


def control_pose_to_training_eef_9d(
    position: np.ndarray,
    quaternion_xyzw: np.ndarray,
    side: str,
) -> np.ndarray:
    """Encode a runtime control-frame pose in the final fine-tuning dataset's raw USD frame.

    The final dataset uses normal scalar-last quaternions and removed the obsolete left training
    offset. Only the left controller's fixed frame offset must be undone here.
    """
    position = np.asarray(position, dtype=np.float64)
    quaternion_xyzw = _normalized_quaternion_xyzw(quaternion_xyzw)
    assert position.shape[:-1] == quaternion_xyzw.shape[:-1] and position.shape[-1] == 3

    control_matrix = _quaternion_xyzw_to_matrix(quaternion_xyzw)
    training_matrix = control_matrix @ _control_offset_matrix(side).T
    return np.concatenate((position, _matrix_to_rotation_6d(training_matrix)), axis=-1).astype(np.float32)


def training_eef_9d_to_control_pose(eef_9d: np.ndarray, side: str) -> tuple[np.ndarray, np.ndarray]:
    """Decode checkpoint EEF-9D into an absolute RMPFlow control-frame pose."""
    eef_9d = np.asarray(eef_9d, dtype=np.float64)
    assert eef_9d.shape[-1] == 9, f"Expected (..., 9) EEF action, got {eef_9d.shape}"
    assert np.isfinite(eef_9d).all(), "AgiBot EEF action contains NaN or infinity"

    training_matrix = rotation_6d_to_matrix(eef_9d[..., 3:])
    control_matrix = training_matrix @ _control_offset_matrix(side)
    control_quaternion_xyzw = _matrix_to_quaternion_xyzw(control_matrix)
    return eef_9d[..., :3].astype(np.float32), control_quaternion_xyzw.astype(np.float32)


def build_agibot_gr00t_action_np(robot_action_policy: dict[str, np.ndarray]) -> np.ndarray:
    """Convert the checkpoint's four action groups to the AgiBot 20D simulator action."""
    missing = AGIBOT_GR00T_ACTION_WIDTHS.keys() - robot_action_policy.keys()
    assert not missing, f"GR00T response omitted AgiBot action groups: {sorted(missing)}"

    decoded: dict[str, np.ndarray] = {}
    batch_horizon: tuple[int, int] | None = None
    for key, width in AGIBOT_GR00T_ACTION_WIDTHS.items():
        value = np.asarray(robot_action_policy[key], dtype=np.float64)
        assert (
            value.ndim == 3 and value.shape[-1] == width
        ), f"AgiBot action '{key}' must have shape (N, H, {width}), got {value.shape}"
        if batch_horizon is None:
            batch_horizon = value.shape[:2]
        assert (
            value.shape[:2] == batch_horizon
        ), f"AgiBot action '{key}' has inconsistent shape {value.shape}"
        assert np.isfinite(
            value
        ).all(), f"AgiBot action '{key}' contains NaN or infinity"
        decoded[key] = value

    left_pos, left_quat = training_eef_9d_to_control_pose(
        decoded["left_eef_9d"], "left"
    )
    right_pos, right_quat = training_eef_9d_to_control_pose(
        decoded["right_eef_9d"], "right"
    )
    left_pos = np.clip(left_pos, *_WORKSPACE_BOUNDS["left"])
    right_pos = np.clip(right_pos, *_WORKSPACE_BOUNDS["right"])
    left_hand = np.clip(decoded["left_hand"], 0.0, 0.994)
    right_hand = np.clip(decoded["right_hand"], 0.0, 0.994)
    return np.concatenate(
        (left_pos, left_quat, left_hand, right_pos, right_quat, right_hand), axis=-1
    ).astype(np.float32)
