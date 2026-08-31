# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""End-effector state in the form the GR00T DROID embodiment expects.

GR00T N1.7's DROID tag takes an ``eef_9d`` state (XYZ plus a 6D rotation) alongside the joint
positions, and uses it as the reference frame when the server converts its relative end-effector
actions back to absolute ones.

The rotation half has to match the egocentric convention the checkpoint was trained with. GR00T
keeps that convention in ``examples/DROID/main_gr00t.py``, which is a standalone robot script rather
than an importable module, so the matrix below is a mirror of it. Upstream warns that a drifted
copy produces wrong rotations silently, with no crash; ``tests/test_droid_eef.py`` therefore checks
this implementation against the reference formula rather than trusting the mirror.
"""

from __future__ import annotations

import numpy as np

DROID_EEF_ROTATION_CORRECT = np.array(
    [[0, 0, -1], [-1, 0, 0], [0, 1, 0]],
    dtype=np.float64,
)
"""Egocentric frame correction post-multiplied onto the end-effector rotation (TFG convention).

Mirrors ``DROID_EEF_ROTATION_CORRECT`` in GR00T's ``examples/DROID/main_gr00t.py``.
"""

USD_BASE_FROM_DROID_BASE_ROTATION = np.array(
    [[-1, 0, 0], [0, -1, 0], [0, 0, 1]],
    dtype=np.float64,
)
"""Rotation from DROID's semantic base into the DROID USD's ``panda_link0`` frame.

Arena's DROID USD uses the opposite X/Y base-axis convention from the Polymetis model that
generated the training states. The asset is therefore spawned with a 180-degree Z rotation by
default. Treating that asset rotation as the semantic DROID base flips every workspace X/Y state;
this fixed transform separates the two conventions.
"""


def quaternion_wxyz_to_matrix(quaternion_wxyz: np.ndarray) -> np.ndarray:
    """Convert (N, 4) quaternions in Isaac Lab's (w, x, y, z) order to (N, 3, 3) matrices.

    SciPy is scalar-last, so the scalar component is moved to the back first. Skipping that step is
    silent rather than fatal: the permuted quaternion is still unit-norm, so it yields a valid but
    wrong rotation.
    """
    from scipy.spatial.transform import Rotation

    quaternion_wxyz = np.asarray(quaternion_wxyz, dtype=np.float64)
    return Rotation.from_quat(np.roll(quaternion_wxyz, -1, axis=-1)).as_matrix()


def droid_base_pose_from_usd_root(
    root_pos_w: np.ndarray,
    root_quat_w_wxyz: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert the DROID USD root pose to the Polymetis semantic base pose.

    Args:
        root_pos_w: (N, 3) USD ``panda_link0`` position in world coordinates.
        root_quat_w_wxyz: (N, 4) USD ``panda_link0`` orientation in world coordinates.

    Returns:
        Semantic DROID base position and orientation as (N, 3) and (N, 4) arrays. Quaternions use
        Isaac Lab's (w, x, y, z) order.
    """
    from scipy.spatial.transform import Rotation

    root_pos_w = np.asarray(root_pos_w, dtype=np.float64)
    usd_root_rot_w = quaternion_wxyz_to_matrix(root_quat_w_wxyz)
    droid_base_rot_w = usd_root_rot_w @ USD_BASE_FROM_DROID_BASE_ROTATION
    droid_base_quat_xyzw = Rotation.from_matrix(droid_base_rot_w).as_quat()
    return root_pos_w, np.roll(droid_base_quat_xyzw, 1, axis=-1)


def eef_pose_in_base_frame(
    eef_pos_w: np.ndarray,
    eef_quat_w_wxyz: np.ndarray,
    base_pos_w: np.ndarray | None = None,
    base_quat_w_wxyz: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Express a world-frame end-effector pose in the robot base frame.

    DROID reports its ``cartesian_position`` relative to the robot base, while Arena's DROID
    observation terms report the gripper pose in world coordinates. The two agree only while the
    robot base sits at the world origin, so the base pose is subtracted explicitly.

    Args:
        eef_pos_w: (N, 3) end-effector position in world coordinates.
        eef_quat_w_wxyz: (N, 4) end-effector orientation in world coordinates, (w, x, y, z).
        base_pos_w: (N, 3) robot base position in world coordinates. Defaults to the origin.
        base_quat_w_wxyz: (N, 4) robot base orientation, (w, x, y, z). Defaults to identity.

    Returns:
        Tuple of (N, 3) position and (N, 3, 3) rotation matrix, both in the robot base frame.
    """
    eef_pos_w = np.asarray(eef_pos_w, dtype=np.float64)
    eef_rot_w = quaternion_wxyz_to_matrix(eef_quat_w_wxyz)
    if base_pos_w is None and base_quat_w_wxyz is None:
        return eef_pos_w, eef_rot_w

    base_pos_w = np.zeros_like(eef_pos_w) if base_pos_w is None else np.asarray(base_pos_w, dtype=np.float64)
    if base_quat_w_wxyz is None:
        base_rot_w = np.broadcast_to(np.eye(3), eef_rot_w.shape).copy()
    else:
        base_rot_w = quaternion_wxyz_to_matrix(base_quat_w_wxyz)

    base_rot_w_transposed = np.swapaxes(base_rot_w, -1, -2)
    eef_pos_b = np.einsum("nij,nj->ni", base_rot_w_transposed, eef_pos_w - base_pos_w)
    eef_rot_b = base_rot_w_transposed @ eef_rot_w
    return eef_pos_b, eef_rot_b


def rotation_matrix_to_droid_rot6d(rotation_matrix: np.ndarray) -> np.ndarray:
    """Convert (N, 3, 3) base-frame rotations to the 6D rotation GR00T's DROID model expects.

    Equivalent to GR00T's ``compute_eef_9d`` rotation path, without its detour through euler angles:
    that helper takes euler angles only because the real DROID robot reports them, and
    ``from_euler("XYZ", as_euler("XYZ", R))`` reconstructs ``R`` exactly.
    """
    rotation_matrix = np.asarray(rotation_matrix, dtype=np.float64)
    corrected = rotation_matrix @ DROID_EEF_ROTATION_CORRECT
    return corrected[:, :2, :].reshape(rotation_matrix.shape[0], 6)


def compute_eef_9d_state(
    eef_pos_w: np.ndarray,
    eef_quat_w_wxyz: np.ndarray,
    base_pos_w: np.ndarray | None = None,
    base_quat_w_wxyz: np.ndarray | None = None,
) -> np.ndarray:
    """Build the ``eef_9d`` state GR00T's DROID embodiment expects.

    Args:
        eef_pos_w: (N, 3) end-effector position in world coordinates.
        eef_quat_w_wxyz: (N, 4) end-effector orientation in world coordinates, (w, x, y, z).
        base_pos_w: (N, 3) robot base position in world coordinates. Defaults to the origin.
        base_quat_w_wxyz: (N, 4) robot base orientation, (w, x, y, z). Defaults to identity.

    Returns:
        (N, 9) float32 array of XYZ followed by the 6D rotation, in GR00T's egocentric DROID
        convention. The intermediate maths stays in float64; only the result is narrowed, because
        the policy server rejects a state that is not float32.
    """
    eef_pos_b, eef_rot_b = eef_pose_in_base_frame(eef_pos_w, eef_quat_w_wxyz, base_pos_w, base_quat_w_wxyz)
    eef_9d = np.concatenate([eef_pos_b, rotation_matrix_to_droid_rot6d(eef_rot_b)], axis=-1)
    return eef_9d.astype(np.float32)
