# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Checks that the request Arena sends satisfies GR00T N1.7's DROID contract.

GR00T's policy server hard-asserts the video horizon and the state keys it receives, so a mismatch
only shows up as a server-side assertion in the middle of a rollout. These tests pin the contract
Arena-side against the modality config the pinned GR00T checkout publishes.
"""

from __future__ import annotations

import numpy as np
import sys
import torch
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

import isaaclab_arena_gr00t.policy.gr00t_remote_closedloop_policy as gr00t_policy
from isaaclab_arena_gr00t.policy.video_history import VideoHistoryBuffer

pytestmark = pytest.mark.gr00t_policy

N17_DROID_TAG = "OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT"
N17_CONFIG = "isaaclab_arena_gr00t/policy/config/droid_manip_n17_gr00t_closedloop_config.yaml"
NUM_ENVS = 2
CAMERAS = ["external_camera_rgb", "wrist_camera_rgb"]
NUM_SIM_JOINTS = 13
# droid_abs_joint_pos drives 7 arm joints plus one binary gripper command.
SIM_ACTION_DIM = 8
POLICY_ACTION_GROUPS = {"joint_position": 7, "gripper_position": 1}


def _droid_remote_config(video_delta_indices=(-15, 0), action_horizon=40):
    """Return the checkpoint contract exposed by GR00T's PolicyClient."""
    return {
        "video": SimpleNamespace(
            delta_indices=list(video_delta_indices),
            modality_keys=["exterior_image_1_left", "wrist_image_left"],
        ),
        "state": SimpleNamespace(
            delta_indices=[0],
            modality_keys=["eef_9d", "gripper_position", "joint_position"],
        ),
        "action": SimpleNamespace(
            delta_indices=list(range(action_horizon)),
            modality_keys=["eef_9d", "gripper_position", "joint_position"],
        ),
        "language": SimpleNamespace(
            delta_indices=[0],
            modality_keys=["annotation.language.language_instruction"],
        ),
    }


def _skip_unless_n17_checkout():
    """Skip when the pinned GR00T checkout predates N1.7 and has no DROID relative tag."""
    from gr00t.data.embodiment_tags import EmbodimentTag

    if N17_DROID_TAG not in [tag.name for tag in EmbodimentTag]:
        pytest.skip(f"pinned submodules/Isaac-GR00T has no {N17_DROID_TAG}; checkout predates GR00T N1.7")


@pytest.fixture
def droid_observation():
    """Env-shaped DROID observation: two cameras, joint positions and an end-effector pose."""
    return {
        "camera_obs": {name: torch.randint(0, 255, (NUM_ENVS, 720, 1280, 3), dtype=torch.uint8) for name in CAMERAS},
        "policy": {
            "robot_joint_pos": torch.randn(NUM_ENVS, NUM_SIM_JOINTS, dtype=torch.float32),
            # The DROID embodiment publishes the gripper separately, already normalised to 0-1.
            "gripper_pos": torch.zeros(NUM_ENVS, 1, dtype=torch.float32),
            "eef_pos": torch.tensor([[0.4, 0.1, 0.3]] * NUM_ENVS, dtype=torch.float32),
            # Isaac Lab quaternions are (w, x, y, z), so this is the identity orientation.
            "eef_quat": torch.tensor([[1.0, 0.0, 0.0, 0.0]] * NUM_ENVS, dtype=torch.float32),
        },
    }


class _FakeClient:
    modality_config_override = None

    def __init__(self, *args, **kwargs):
        self.last_observation: dict[str, Any] | None = None
        self.modality_configs = self.modality_config_override or _droid_remote_config()

    def ping(self) -> bool:
        return True

    def get_modality_config(self):
        return self.modality_configs

    def get_action(self, observation):
        self.last_observation = observation
        horizon = len(self.modality_configs["action"].delta_indices)
        response = {
            group: np.random.randn(NUM_ENVS, horizon, size).astype(np.float32)
            for group, size in POLICY_ACTION_GROUPS.items()
        }
        # The server also returns eef_9d for this embodiment; Arena drives the sim from the joints.
        response["eef_9d"] = np.random.randn(NUM_ENVS, horizon, 9).astype(np.float32)
        return response, None

    def reset(self):
        pass


@pytest.fixture
def fake_client(monkeypatch):
    created: list[_FakeClient] = []

    def _ctor(*args, **kwargs):
        client = _FakeClient(*args, **kwargs)
        created.append(client)
        return client

    module = ModuleType("gr00t.policy.server_client")
    module.PolicyClient = _ctor
    monkeypatch.setitem(sys.modules, "gr00t.policy.server_client", module)
    return created


def _build_policy():
    cfg = gr00t_policy.Gr00tRemoteClosedloopPolicyCfg(
        policy_config_yaml_path=N17_CONFIG,
        policy_device="cpu",
        num_envs=NUM_ENVS,
        remote_host="unused",
        remote_port=0,
    )
    return gr00t_policy.Gr00tRemoteClosedloopPolicy(cfg)


def test_request_matches_n17_droid_modality_contract(droid_observation, fake_client):
    """Video horizon, state keys and state widths must match what the N1.7 DROID tag declares."""
    _skip_unless_n17_checkout()
    policy = _build_policy()
    policy.set_task_description("pick up the peg and insert it into the hole")

    video_config = policy.modality_configs["video"]
    state_keys = policy.modality_configs["state"].modality_keys

    # Run enough control steps that the buffer can serve the oldest requested frame.
    for _ in range(policy._video_history.history_length):
        policy._video_history.push(policy._resized_frames(droid_observation, CAMERAS))
    policy._get_action_chunk(None, droid_observation, CAMERAS)

    sent = fake_client[0].last_observation
    assert sent is not None

    expected_horizon = len(video_config.delta_indices)
    assert expected_horizon == 2
    for video_key in video_config.modality_keys:
        video = sent["video"][video_key]
        assert video.shape == (NUM_ENVS, expected_horizon, 180, 320, 3)
        assert video.dtype == np.uint8

    assert set(sent["state"].keys()) == set(state_keys)
    expected_widths = {"eef_9d": 9, "gripper_position": 1, "joint_position": 7}
    for state_key, width in expected_widths.items():
        state = sent["state"][state_key]
        assert state.shape == (NUM_ENVS, 1, width)
        # The server rejects any state that is not float32. eef_9d is computed rather than sliced
        # out of the observation, so it is the one that naturally arrives as float64.
        assert state.dtype == np.float32, f"state '{state_key}' must be float32, got {state.dtype}"


def test_computed_eef_state_is_narrowed_to_float32(droid_observation, fake_client):
    """A float64 state provider must still reach the server as float32."""
    _skip_unless_n17_checkout()
    from isaaclab_arena_gr00t.utils.droid_eef import compute_eef_9d_state

    eef_9d = compute_eef_9d_state(np.zeros((NUM_ENVS, 3)), np.tile([1.0, 0.0, 0.0, 0.0], (NUM_ENVS, 1)))
    assert eef_9d.dtype == np.float32, "compute_eef_9d_state must match GR00T's own float32 output"

    policy = _build_policy()
    policy.set_task_description("insert the peg")
    policy._video_history.push(policy._resized_frames(droid_observation, CAMERAS))
    policy._get_action_chunk(None, droid_observation, CAMERAS)

    for state in fake_client[0].last_observation["state"].values():
        assert state.dtype == np.float32


def test_action_response_is_decoded_into_the_sim_action_space(droid_observation, fake_client):
    """The server returns absolute joints for this tag, so the sim action tensor is built as usual."""
    _skip_unless_n17_checkout()
    policy = _build_policy()
    policy.set_task_description("pick up the peg and insert it into the hole")

    policy._video_history.push(policy._resized_frames(droid_observation, CAMERAS))
    action = policy._get_action_chunk(None, droid_observation, CAMERAS)

    assert isinstance(action, torch.Tensor)
    assert action.shape == (NUM_ENVS, 40, SIM_ACTION_DIM)


def test_remote_checkpoint_contract_overrides_static_n17_defaults(droid_observation, fake_client, monkeypatch):
    """A post-trained checkpoint can expose different video and action horizons under the same tag."""
    monkeypatch.setattr(
        _FakeClient,
        "modality_config_override",
        _droid_remote_config(video_delta_indices=[0], action_horizon=24),
    )
    policy = _build_policy()
    policy.set_task_description("insert the peg")

    assert policy._video_history.delta_indices == [0]
    assert policy.action_horizon == 24
    policy._video_history.push(policy._resized_frames(droid_observation, CAMERAS))
    action = policy._get_action_chunk(None, droid_observation, CAMERAS)

    sent = fake_client[0].last_observation
    assert sent is not None
    for video in sent["video"].values():
        assert video.shape == (NUM_ENVS, 1, 180, 320, 3)
    assert action.shape == (NUM_ENVS, 24, SIM_ACTION_DIM)


def test_images_reach_the_policy_undistorted():
    """A pretrained DROID checkpoint only recognises geometry it saw in training.

    GR00T's own client resizes with ``max(width_ratio, height_ratio)`` and pads the short side, so a
    16:9 camera going to 180x320 is scaled by exactly 4 with no padding. Padding to a square first
    would squash it by ~1.8x vertically, which is silent: the request still validates.
    """
    import cv2

    from isaaclab_arena_gr00t.policy.gr00t_core import resize_rgb_for_policy

    frame = np.zeros((1, 720, 1280, 3), np.uint8)
    cv2.circle(frame[0], (640, 360), 100, (255, 255, 255), -1)

    resized = resize_rgb_for_policy(rgb_list_np=[frame], target_image_size=(180, 320, 3))[0]
    rows, columns = np.where(resized[0, :, :, 0] > 127)
    width = columns.max() - columns.min() + 1
    height = rows.max() - rows.min() + 1
    assert resized.shape == (1, 180, 320, 3)
    assert abs(width - height) <= 2, f"a circle came out {width}x{height}; the aspect ratio was not preserved"


def test_gripper_state_is_normalised_not_radians(droid_observation, fake_client):
    """DROID checkpoints were trained on a 0-1 gripper signal, not finger_joint radians."""
    _skip_unless_n17_checkout()
    # finger_joint's closed command is pi/4, so radians would top out at 0.785 and the policy would
    # never observe a fully closed gripper.
    droid_observation["policy"]["gripper_pos"] = torch.full((NUM_ENVS, 1), 1.0)
    droid_observation["policy"]["robot_joint_pos"][:, 7] = torch.pi / 4

    policy = _build_policy()
    policy.set_task_description("close the gripper")
    policy._video_history.push(policy._resized_frames(droid_observation, CAMERAS))
    policy._get_action_chunk(None, droid_observation, CAMERAS)

    sent = fake_client[0].last_observation["state"]["gripper_position"]
    assert sent.shape == (NUM_ENVS, 1, 1)
    assert float(sent[0, 0, 0]) == pytest.approx(
        1.0
    ), "a fully closed gripper must reach the policy as 1.0, not as pi/4 radians"


def test_video_history_serves_the_oldest_frame_until_it_fills():
    """Before an episode is 15 steps old the past frame clamps to the oldest one available."""
    buffer = VideoHistoryBuffer(delta_indices=[-15, 0], num_envs=1)
    assert buffer.history_length == 16

    frames = [np.full((1, 2, 2, 3), step, dtype=np.uint8) for step in range(20)]
    buffer.push([frames[0]])
    history, current = buffer.stack()[0][0]
    assert history.max() == 0 and current.max() == 0, "a fresh episode reuses its only frame"

    for step in range(1, 5):
        buffer.push([frames[step]])
    history, current = buffer.stack()[0][0]
    assert history.max() == 0, "still clamped to the episode's first frame"
    assert current.max() == 4

    for step in range(5, 20):
        buffer.push([frames[step]])
    history, current = buffer.stack()[0][0]
    assert current.max() == 19
    assert history.max() == 4, "once full, the past frame is exactly 15 steps back"


def test_video_history_reset_is_per_environment():
    """Resetting one environment must not disturb the other's history."""
    buffer = VideoHistoryBuffer(delta_indices=[-2, 0], num_envs=2)
    for step in range(3):
        frame = np.stack([np.full((2, 2, 3), step, dtype=np.uint8) for _ in range(2)])
        buffer.push([frame])

    buffer.reset(np.array([0]))
    frame = np.stack([np.full((2, 2, 3), 9, dtype=np.uint8) for _ in range(2)])
    buffer.push([frame])

    stacked = buffer.stack()[0]
    assert stacked[0][0].max() == 9, "the reset env restarts from its first post-reset frame"
    assert stacked[1][0].max() == 1, "the untouched env keeps its own history"


def test_eef_state_reaches_the_server_in_isaac_labs_quaternion_order(droid_observation, fake_client):
    """The observation term is (w, x, y, z); a scalar-last reading of it is silent, not fatal.

    The fixture holds the gripper at the identity orientation with the robot base at the world
    origin, so the rotation half of ``eef_9d`` must collapse to the bare frame correction. Reading
    the quaternion scalar-last instead would yield a valid unit-norm rotation and a plausible
    state, so only the value pins the convention.
    """
    _skip_unless_n17_checkout()
    from isaaclab_arena_gr00t.utils.droid_eef import DROID_EEF_ROTATION_CORRECT

    policy = _build_policy()
    policy.set_task_description("insert the peg")
    policy._video_history.push(policy._resized_frames(droid_observation, CAMERAS))
    policy._get_action_chunk(None, droid_observation, CAMERAS)

    eef_9d = fake_client[0].last_observation["state"]["eef_9d"]
    expected_position = droid_observation["policy"]["eef_pos"].numpy()
    np.testing.assert_allclose(eef_9d[:, 0, :3], expected_position, atol=1e-6)
    np.testing.assert_allclose(
        eef_9d[:, 0, 3:],
        np.tile(DROID_EEF_ROTATION_CORRECT[:2, :].reshape(6), (NUM_ENVS, 1)),
        atol=1e-6,
    )
