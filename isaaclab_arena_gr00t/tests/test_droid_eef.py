# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Guards the ``eef_9d`` conversion GR00T N1.7's DROID embodiment conditions on.

``isaaclab_arena_gr00t/utils/droid_eef.py`` mirrors a rotation correction that GR00T keeps in a
standalone robot script, and upstream warns that a drifted copy of it produces wrong rotations with
no crash. These tests re-derive the expected values from GR00T's published formula instead of from
the mirror, so drift fails loudly here rather than silently degrading a rollout.

The other silent failure mode is the quaternion convention at the boundary: the helpers take Isaac
Lab's (w, x, y, z) order while SciPy is scalar-last, and a permuted quaternion is still unit-norm,
so a mix-up yields a valid but wrong rotation. Inputs here are therefore built in Isaac Lab's order,
and ``test_quaternion_order_is_isaac_labs_wxyz`` pins that order against analytic rotations rather
than against SciPy's own round trip.
"""

from __future__ import annotations

import numpy as np

import pytest

from isaaclab_arena_gr00t.utils.droid_eef import (
    DROID_EEF_ROTATION_CORRECT,
    USD_BASE_FROM_DROID_BASE_ROTATION,
    compute_eef_9d_state,
    droid_base_pose_from_usd_root,
    eef_pose_in_base_frame,
    quaternion_wxyz_to_matrix,
)

pytestmark = pytest.mark.gr00t_policy


def _as_wxyz(rotation) -> np.ndarray:
    """Return a SciPy rotation as an (N, 4) quaternion in Isaac Lab's (w, x, y, z) order."""
    return np.roll(rotation.as_quat(), 1, axis=-1)


def _reference_eef_9d(xyz: np.ndarray, euler_xyz: np.ndarray) -> np.ndarray:
    """GR00T's own formula, transcribed from ``examples/DROID/main_gr00t.py``."""
    from scipy.spatial.transform import Rotation

    rotation = Rotation.from_euler("XYZ", euler_xyz).as_matrix() @ DROID_EEF_ROTATION_CORRECT
    return np.concatenate([xyz, rotation[:2, :].reshape(6)])


def test_matches_gr00t_reference_formula_on_random_poses():
    """The matrix path must agree with GR00T's euler-based formula to numerical precision."""
    from scipy.spatial.transform import Rotation

    rng = np.random.default_rng(0)
    for seed in range(16):
        position = rng.uniform(-1.0, 1.0, (1, 3))
        rotation = Rotation.random(1, random_state=seed)
        ours = compute_eef_9d_state(position, _as_wxyz(rotation))
        expected = _reference_eef_9d(position[0], rotation.as_euler("XYZ")[0])
        assert ours.shape == (1, 9)
        np.testing.assert_allclose(ours[0], expected, atol=1e-9)


def test_pose_is_expressed_relative_to_the_robot_base():
    """DROID states are base-relative, so moving robot and gripper together changes nothing."""
    from scipy.spatial.transform import Rotation

    base_position = np.array([[0.3, -0.2, 0.1]])
    base_quaternion = _as_wxyz(Rotation.from_euler("z", 0.7))[None]
    eef_position = np.array([[0.5, 0.1, 0.4]])
    eef_quaternion = _as_wxyz(Rotation.random(1, random_state=99))

    at_origin = compute_eef_9d_state(eef_position, eef_quaternion, base_position, base_quaternion)
    shift = np.array([[1.0, 2.0, 3.0]])
    shifted = compute_eef_9d_state(eef_position + shift, eef_quaternion, base_position + shift, base_quaternion)
    np.testing.assert_allclose(at_origin, shifted, atol=1e-12)


def test_base_at_the_world_origin_is_a_no_op():
    """The DROID environments put the robot at the origin, where both frames coincide."""
    from scipy.spatial.transform import Rotation

    eef_position = np.array([[0.6, -0.1, 0.25]])
    eef_quaternion = _as_wxyz(Rotation.random(1, random_state=7))
    identity_quaternion = np.array([[1.0, 0.0, 0.0, 0.0]])

    explicit = compute_eef_9d_state(eef_position, eef_quaternion, np.zeros((1, 3)), identity_quaternion)
    implicit = compute_eef_9d_state(eef_position, eef_quaternion)
    np.testing.assert_allclose(explicit, implicit, atol=1e-12)


def test_default_usd_root_rotation_maps_to_identity_semantic_base():
    """Arena's default 180-degree USD placement must not flip DROID workspace X/Y."""
    from scipy.spatial.transform import Rotation

    root_position = np.array([[0.0, 0.0, 0.0]])
    root_quaternion = _as_wxyz(Rotation.from_euler("z", np.pi))[None]
    base_position, base_quaternion = droid_base_pose_from_usd_root(root_position, root_quaternion)

    np.testing.assert_allclose(base_position, root_position, atol=1e-12)
    np.testing.assert_allclose(quaternion_wxyz_to_matrix(base_quaternion)[0], np.eye(3), atol=1e-12)


def test_usd_to_semantic_base_transform_composes_with_root_placement():
    """The fixed axis correction must remain valid when a scene rotates the whole robot."""
    from scipy.spatial.transform import Rotation

    root_rotation = Rotation.from_euler("XYZ", [0.2, -0.3, 0.7])
    _, base_quaternion = droid_base_pose_from_usd_root(
        np.array([[1.0, 2.0, 3.0]]),
        _as_wxyz(root_rotation)[None],
    )
    expected = root_rotation.as_matrix() @ USD_BASE_FROM_DROID_BASE_ROTATION
    np.testing.assert_allclose(quaternion_wxyz_to_matrix(base_quaternion)[0], expected, atol=1e-12)


def test_droid_observations_select_polymetis_end_effector(monkeypatch):
    """DROID state must use panda_link8, not the Robotiq base link with a different pose."""
    import torch
    from types import SimpleNamespace

    from isaaclab_arena.embodiments.droid import observations as droid_observations

    body_positions = torch.tensor([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]])
    body_quaternions = torch.tensor([[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]])
    robot = SimpleNamespace(
        data=SimpleNamespace(
            body_names=["base_link", "panda_link8"],
            body_pos_w=body_positions,
            body_quat_w=body_quaternions,
        )
    )
    env = SimpleNamespace(scene={"robot": robot})
    monkeypatch.setattr(droid_observations.wp, "to_torch", lambda value: value)

    torch.testing.assert_close(droid_observations.ee_pos(env), body_positions[:, 1, :])
    torch.testing.assert_close(droid_observations.ee_quat(env), body_quaternions[:, 1, :])


def test_base_frame_rotation_is_orthonormal():
    """The base-frame rotation stays a proper rotation, which the rot6d slice relies on."""
    from scipy.spatial.transform import Rotation

    base_quaternion = _as_wxyz(Rotation.random(1, random_state=3))
    eef_quaternion = _as_wxyz(Rotation.random(1, random_state=4))
    _, rotation = eef_pose_in_base_frame(np.zeros((1, 3)), eef_quaternion, np.zeros((1, 3)), base_quaternion)
    np.testing.assert_allclose(rotation[0] @ rotation[0].T, np.eye(3), atol=1e-12)
    assert np.linalg.det(rotation[0]) == pytest.approx(1.0)


def test_quaternion_order_is_isaac_labs_wxyz():
    """Pin the (w, x, y, z) input order against analytic rotations, not a SciPy round trip.

    Isaac Lab's observation terms are scalar-first while SciPy is scalar-last, and feeding one to
    the other is silent: the permuted quaternion stays unit-norm, so the model would receive a
    valid but wrong orientation. Both cases below fail loudly under the scalar-last reading.
    """
    identity_wxyz = np.array([[1.0, 0.0, 0.0, 0.0]])
    np.testing.assert_allclose(quaternion_wxyz_to_matrix(identity_wxyz)[0], np.eye(3), atol=1e-12)

    half = np.sqrt(0.5)
    quarter_turn_about_z_wxyz = np.array([[half, 0.0, 0.0, half]])
    expected = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    np.testing.assert_allclose(quaternion_wxyz_to_matrix(quarter_turn_about_z_wxyz)[0], expected, atol=1e-12)


def test_identity_orientation_yields_the_bare_frame_correction():
    """An identity gripper rotation must reduce eef_9d's rotation half to the correction itself."""
    eef_9d = compute_eef_9d_state(np.zeros((1, 3)), np.array([[1.0, 0.0, 0.0, 0.0]]))
    np.testing.assert_allclose(eef_9d[0, 3:], DROID_EEF_ROTATION_CORRECT[:2, :].reshape(6), atol=1e-7)
