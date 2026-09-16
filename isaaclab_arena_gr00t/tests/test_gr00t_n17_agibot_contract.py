# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Pin Arena's request/action translation to the fine-tuned AgiBot N1.7 contract."""

from __future__ import annotations

import sys
from enum import Enum
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch
import yaml

from isaaclab_arena.embodiments.agibot.agibot import (
    AGIBOT_LEFT_ARM_ARENA_RMPFLOW_CFG,
    AGIBOT_LEFT_ARM_GR00T_RMPFLOW_CFG,
    AGIBOT_RIGHT_ARM_ARENA_RMPFLOW_CFG,
    AGIBOT_RIGHT_ARM_GR00T_RMPFLOW_CFG,
    AgibotDualArmActionsCfg,
    AgibotGr00tActionsCfg,
    AgibotGr00tDiffIkActionsCfg,
)
from isaaclab_arena.embodiments.common.control_rate_diffik_actions import (
    ControlRateDifferentialIKAction,
    ControlRateDifferentialIKActionCfg,
)
from isaaclab_arena.embodiments.common.low_pass_rmpflow_actions import (
    LowPassRMPFlowAction,
    LowPassRMPFlowActionCfg,
)
from isaaclab_arena.terms.events import reset_joint_position_and_velocity_to_pose
import isaaclab_arena_gr00t.policy.gr00t_remote_closedloop_policy as gr00t_policy
from isaaclab_arena_gr00t.utils.agibot_eef import (
    AGIBOT_GR00T_ACTION_DIM,
    AGIBOT_GR00T_STATE_WIDTHS,
    build_agibot_gr00t_action_np,
    control_pose_to_training_eef_9d,
    training_eef_9d_to_control_pose,
)


def test_repeated_action_samples_use_dtype_preserving_elementwise_median():
    samples = [
        {"left_eef_9d": np.full((1, 40, 9), value, dtype=np.float32)}
        for value in (1.0, 100.0, 2.0)
    ]

    result = gr00t_policy.Gr00tRemoteClosedloopPolicy._median_action_samples(samples)

    assert result["left_eef_9d"].dtype == np.float32
    np.testing.assert_array_equal(
        result["left_eef_9d"], np.full((1, 40, 9), 2.0, dtype=np.float32)
    )


def test_agibot_action_chunk_translation_anchor_preserves_relative_motion():
    policy = object.__new__(gr00t_policy.Gr00tRemoteClosedloopPolicy)
    policy.policy_config = SimpleNamespace(action_chunk_translation_anchor_alpha=1.0)
    policy.task_mode = gr00t_policy.TaskMode.AGIBOT_BIMANUAL_MANIPULATION
    actions = {
        "left_eef_9d": np.asarray(
            [[[0.10, 0.20, 0.30] + [0.0] * 6, [0.14, 0.18, 0.33] + [0.0] * 6]],
            dtype=np.float32,
        ),
        "right_eef_9d": np.asarray(
            [[[0.70, -0.20, 0.40] + [0.0] * 6, [0.68, -0.17, 0.41] + [0.0] * 6]],
            dtype=np.float32,
        ),
        "left_hand": np.full((1, 2, 3), 0.25, dtype=np.float32),
        "right_hand": np.full((1, 2, 3), 0.75, dtype=np.float32),
    }
    observations = {
        "state": {
            "left_eef_9d": np.asarray([[[0.08, 0.22, 0.31] + [0.0] * 6]], dtype=np.float32),
            "right_eef_9d": np.asarray([[[0.72, -0.21, 0.39] + [0.0] * 6]], dtype=np.float32),
        }
    }
    original = {key: value.copy() for key, value in actions.items()}

    anchored = policy._anchor_agibot_action_chunk_translation(actions, observations)

    for side in ("left", "right"):
        key = f"{side}_eef_9d"
        np.testing.assert_allclose(
            anchored[key][:, 0, :3], observations["state"][key][:, -1, :3]
        )
        np.testing.assert_allclose(
            np.diff(anchored[key][..., :3], axis=1),
            np.diff(actions[key][..., :3], axis=1),
        )
    np.testing.assert_array_equal(anchored["left_hand"], actions["left_hand"])
    np.testing.assert_array_equal(anchored["right_hand"], actions["right_hand"])
    for key in actions:
        np.testing.assert_array_equal(actions[key], original[key])


pytestmark = pytest.mark.gr00t_policy

N17_CONFIG = (
    "isaaclab_arena_gr00t/policy/config/agibot_n17_gr00t_closedloop_config.yaml"
)
N17_RTC_DIAGNOSTIC_CONFIG = "isaaclab_arena_gr00t/policy/config/agibot_n17_gr00t_closedloop_rtc_frozen8_config.yaml"
N17_EXPERIMENT_CONFIG = Path(
    "isaaclab_arena_environments/experiment_configs/agibot_stack_bowls_gr00t_n17_experiment.yaml"
)
DENOISE_SWEEP_EXPERIMENT_CONFIG = Path(
    "isaaclab_arena_environments/experiment_configs/"
    "agibot_stack_bowls_gr00t_n17_denoise_sweep_20ep.yaml"
)
DENOISE_SWEEP_POLICY_CONFIGS = {
    4: Path(
        "isaaclab_arena_gr00t/policy/config/"
        "agibot_n17_gr00t_closedloop_denoise4_config.yaml"
    ),
    8: Path(N17_CONFIG),
    16: Path(
        "isaaclab_arena_gr00t/policy/config/"
        "agibot_n17_gr00t_closedloop_denoise16_config.yaml"
    ),
    32: Path(
        "isaaclab_arena_gr00t/policy/config/"
        "agibot_n17_gr00t_closedloop_denoise32_config.yaml"
    ),
}
NUM_ENVS = 2
ACTION_HORIZON = 40
CAMERAS = ["head_cam_rgb", "left_wrist_cam_rgb", "right_wrist_cam_rgb"]
VIDEO_KEYS = ["ego_view", "left_wrist_view", "right_wrist_view"]


def _agibot_remote_config():
    return {
        "video": SimpleNamespace(delta_indices=[0], modality_keys=VIDEO_KEYS),
        "state": SimpleNamespace(
            delta_indices=[0], modality_keys=list(AGIBOT_GR00T_STATE_WIDTHS)
        ),
        "action": SimpleNamespace(
            delta_indices=list(range(ACTION_HORIZON)),
            modality_keys=["left_eef_9d", "right_eef_9d", "left_hand", "right_hand"],
        ),
        "language": SimpleNamespace(
            delta_indices=[0],
            modality_keys=["annotation.human.action.task_description"],
        ),
    }


def _normalized(value: list[float]) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    return array / np.linalg.norm(array)


LEFT_CONTROL_QUATERNION = _normalized([0.1, -0.2, 0.3, 0.9])
RIGHT_CONTROL_QUATERNION = _normalized([-0.2, 0.1, -0.1, 0.95])
LEFT_CONTROL_POSITION = np.array([0.75, 0.25, 0.90], dtype=np.float32)
RIGHT_CONTROL_POSITION = np.array([0.80, -0.25, 0.90], dtype=np.float32)
LEFT_HAND_TARGET = np.array([0.20, 0.20, 0.20], dtype=np.float32)
RIGHT_HAND_TARGET = np.array([0.70, 0.70, 0.70], dtype=np.float32)

# Episode 0, row 1 from the finalized stack-bowls dataset. These values make the
# regression independent of an encoder/decoder pair accidentally sharing a bug.
DATASET_LEFT_CONTROL_POSITION = np.array(
    [0.853971, 0.186802, 0.908332], dtype=np.float32
)
DATASET_RIGHT_CONTROL_POSITION = np.array(
    [0.962955, -0.307514, 0.958940], dtype=np.float32
)
DATASET_LEFT_CONTROL_QUATERNION = _normalized(
    [-0.9106907, 0.4074864, 0.03074062, 0.06043366]
)
DATASET_RIGHT_CONTROL_QUATERNION = _normalized(
    [-0.6827974, -0.17681661, -0.7063635, 0.05978523]
)
DATASET_LEFT_EEF_9D = np.array(
    [
        0.853972,
        0.186803,
        0.908331,
        0.006739,
        -0.135125,
        0.990806,
        -0.745904,
        -0.660605,
        -0.085020,
    ],
    dtype=np.float32,
)
DATASET_RIGHT_EEF_9D = np.array(
    [
        0.962956,
        -0.307522,
        0.958924,
        -0.060399,
        0.157004,
        0.985749,
        0.325920,
        -0.930324,
        0.168146,
    ],
    dtype=np.float32,
)


@pytest.fixture
def agibot_observation():
    joint_position = torch.arange(34, dtype=torch.float32).repeat(NUM_ENVS, 1) / 100.0
    return {
        "camera_obs": {
            name: torch.randint(0, 255, (NUM_ENVS, 512, 512, 3), dtype=torch.uint8)
            for name in CAMERAS
        },
        "policy": {
            "robot_joint_pos": joint_position,
            "left_eef_pos": torch.tensor(np.tile(LEFT_CONTROL_POSITION, (NUM_ENVS, 1))),
            "left_eef_quat": torch.tensor(
                np.tile(LEFT_CONTROL_QUATERNION, (NUM_ENVS, 1))
            ),
            "right_eef_pos": torch.tensor(
                np.tile(RIGHT_CONTROL_POSITION, (NUM_ENVS, 1))
            ),
            "right_eef_quat": torch.tensor(
                np.tile(RIGHT_CONTROL_QUATERNION, (NUM_ENVS, 1))
            ),
            "left_hand_pos": torch.full((NUM_ENVS, 3), 0.4, dtype=torch.float32),
            "right_hand_pos": torch.full((NUM_ENVS, 3), 0.6, dtype=torch.float32),
        },
    }


class _FakeClient:
    def __init__(self, *args, **kwargs):
        self.modality_configs = _agibot_remote_config()
        self.last_observation: dict[str, Any] | None = None
        self.last_options: dict[str, Any] | None = None
        self.last_reset_options: dict[str, Any] | None = None
        self.action_request_count = 0

    def ping(self) -> bool:
        return True

    def get_modality_config(self):
        return self.modality_configs

    def get_action(self, observation, options=None):
        self.action_request_count += 1
        self.last_observation = observation
        self.last_options = options
        shape = (NUM_ENVS, ACTION_HORIZON)
        left_position = np.broadcast_to(LEFT_CONTROL_POSITION, (*shape, 3)).copy()
        left_quaternion = np.broadcast_to(LEFT_CONTROL_QUATERNION, (*shape, 4)).copy()
        right_position = np.broadcast_to(RIGHT_CONTROL_POSITION, (*shape, 3)).copy()
        right_quaternion = np.broadcast_to(RIGHT_CONTROL_QUATERNION, (*shape, 4)).copy()
        response = {
            "left_eef_9d": control_pose_to_training_eef_9d(
                left_position, left_quaternion, "left"
            ),
            "right_eef_9d": control_pose_to_training_eef_9d(
                right_position, right_quaternion, "right"
            ),
            "left_hand": np.broadcast_to(LEFT_HAND_TARGET, (*shape, 3)).copy(),
            "right_hand": np.broadcast_to(RIGHT_HAND_TARGET, (*shape, 3)).copy(),
        }
        rtc_applied = options is not None and "action" in observation
        return response, {
            "rtc_applied": rtc_applied,
            "rtc_model_input_present": rtc_applied,
            "rtc_time_normalization_aligned": rtc_applied,
            "num_inference_timesteps": (
                None if options is None else options.get("num_inference_timesteps")
            ),
        }

    def reset(self, options=None):
        self.last_reset_options = options
        return {} if options is None else dict(options)


@pytest.fixture
def fake_client(monkeypatch):
    created: list[_FakeClient] = []

    def _constructor(*args, **kwargs):
        client = _FakeClient(*args, **kwargs)
        created.append(client)
        return client

    class _EmbodimentTag(Enum):
        NEW_EMBODIMENT = "new_embodiment"

    gr00t_module = ModuleType("gr00t")
    gr00t_module.__path__ = []
    data_module = ModuleType("gr00t.data")
    data_module.__path__ = []
    tags_module = ModuleType("gr00t.data.embodiment_tags")
    tags_module.EmbodimentTag = _EmbodimentTag
    policy_module = ModuleType("gr00t.policy")
    policy_module.__path__ = []
    server_module = ModuleType("gr00t.policy.server_client")
    server_module.PolicyClient = _constructor
    monkeypatch.setitem(sys.modules, "gr00t", gr00t_module)
    monkeypatch.setitem(sys.modules, "gr00t.data", data_module)
    monkeypatch.setitem(sys.modules, "gr00t.data.embodiment_tags", tags_module)
    monkeypatch.setitem(sys.modules, "gr00t.policy", policy_module)
    monkeypatch.setitem(sys.modules, "gr00t.policy.server_client", server_module)
    return created


def _build_policy(policy_config_yaml_path=N17_CONFIG):
    config = gr00t_policy.Gr00tRemoteClosedloopPolicyCfg(
        policy_config_yaml_path=policy_config_yaml_path,
        policy_device="cpu",
        num_envs=NUM_ENVS,
        remote_host="unused",
        remote_port=0,
    )
    return gr00t_policy.Gr00tRemoteClosedloopPolicy(config)


def _request_action(policy, observation):
    policy.set_task_description("Stack the bowls together.")
    policy._video_history.push(policy._resized_frames(observation, CAMERAS))
    return policy._get_action_chunk(None, observation, CAMERAS)


def test_request_matches_local_checkpoint_contract(agibot_observation, fake_client):
    policy = _build_policy()
    _request_action(policy, agibot_observation)
    sent = fake_client[0].last_observation
    assert sent is not None

    assert list(sent["video"]) == VIDEO_KEYS
    for video in sent["video"].values():
        assert video.shape == (NUM_ENVS, 1, 512, 512, 3)
        assert video.dtype == np.uint8

    assert list(sent["state"]) == list(AGIBOT_GR00T_STATE_WIDTHS)
    for key, width in AGIBOT_GR00T_STATE_WIDTHS.items():
        assert sent["state"][key].shape == (NUM_ENVS, 1, width)
        assert sent["state"][key].dtype == np.float32

    joint_position = agibot_observation["policy"]["robot_joint_pos"].numpy()
    np.testing.assert_allclose(
        sent["state"]["left_arm"][:, 0], joint_position[:, [4, 6, 8, 10, 12, 14, 16]]
    )
    np.testing.assert_allclose(
        sent["state"]["right_arm"][:, 0], joint_position[:, [5, 7, 9, 11, 13, 15, 17]]
    )
    np.testing.assert_allclose(
        sent["state"]["left_hand"][:, 0], joint_position[:, [19, 26, 27]]
    )
    np.testing.assert_allclose(
        sent["state"]["right_hand"][:, 0], joint_position[:, [21, 28, 29]]
    )


def test_action_response_becomes_absolute_20d_control_action(
    agibot_observation, fake_client
):
    policy = _build_policy()
    action = _request_action(policy, agibot_observation)

    assert action.shape == (NUM_ENVS, ACTION_HORIZON, AGIBOT_GR00T_ACTION_DIM)
    shape = (NUM_ENVS, ACTION_HORIZON)
    np.testing.assert_allclose(
        action[:, :, 0:3],
        np.broadcast_to(LEFT_CONTROL_POSITION, (*shape, 3)),
        atol=1e-6,
    )
    np.testing.assert_allclose(
        action[:, :, 3:7],
        np.broadcast_to(LEFT_CONTROL_QUATERNION, (*shape, 4)),
        atol=1e-6,
    )
    np.testing.assert_allclose(
        action[:, :, 7:10], np.broadcast_to(LEFT_HAND_TARGET, (*shape, 3)), atol=1e-6
    )
    np.testing.assert_allclose(
        action[:, :, 10:13],
        np.broadcast_to(RIGHT_CONTROL_POSITION, (*shape, 3)),
        atol=1e-6,
    )
    np.testing.assert_allclose(
        action[:, :, 13:17],
        np.broadcast_to(RIGHT_CONTROL_QUATERNION, (*shape, 4)),
        atol=1e-6,
    )
    np.testing.assert_allclose(
        action[:, :, 17:20], np.broadcast_to(RIGHT_HAND_TARGET, (*shape, 3)), atol=1e-6
    )


def test_rtc_sends_previous_horizon_and_native_options_on_second_inference(
    agibot_observation, fake_client
):
    policy = _build_policy(N17_RTC_DIAGNOSTIC_CONFIG)

    _request_action(policy, agibot_observation)
    first_action = {
        key: value.copy() for key, value in policy._previous_policy_action.items()
    }
    assert fake_client[0].last_options == {"num_inference_timesteps": 4}

    _request_action(policy, agibot_observation)

    sent = fake_client[0].last_observation
    assert sent is not None
    assert set(sent["action"]) == set(first_action)
    for key, expected in first_action.items():
        np.testing.assert_array_equal(sent["action"][key], expected)
    assert fake_client[0].last_options == {
        "num_inference_timesteps": 4,
        "action_horizon": ACTION_HORIZON,
        "rtc_overlap_steps": ACTION_HORIZON - policy.action_chunk_length,
        "rtc_frozen_steps": 8,
        "rtc_ramp_rate": 4.0,
    }

    policy.reset()
    assert policy._previous_policy_action is None
    assert not policy._rtc_valid_envs.any()


@pytest.mark.parametrize("side", ["left", "right"])
def test_checkpoint_eef_encoding_round_trips_control_frame(side):
    shape = (2, 3)
    position = np.broadcast_to(
        LEFT_CONTROL_POSITION if side == "left" else RIGHT_CONTROL_POSITION, (*shape, 3)
    ).copy()
    quaternion = np.broadcast_to(
        LEFT_CONTROL_QUATERNION if side == "left" else RIGHT_CONTROL_QUATERNION,
        (*shape, 4),
    ).copy()
    encoded = control_pose_to_training_eef_9d(position, quaternion, side)
    decoded_position, decoded_quaternion = training_eef_9d_to_control_pose(
        encoded, side
    )

    np.testing.assert_allclose(decoded_position, position, atol=1e-6)
    # Quaternion signs are equivalent, so compare rotation matrices through absolute dot products.
    dots = np.abs(np.sum(decoded_quaternion * quaternion, axis=-1))
    np.testing.assert_allclose(dots, np.ones(shape), atol=1e-6)


@pytest.mark.parametrize(
    ("side", "control_position", "control_quaternion", "expected_eef_9d"),
    [
        (
            "left",
            DATASET_LEFT_CONTROL_POSITION,
            DATASET_LEFT_CONTROL_QUATERNION,
            DATASET_LEFT_EEF_9D,
        ),
        (
            "right",
            DATASET_RIGHT_CONTROL_POSITION,
            DATASET_RIGHT_CONTROL_QUATERNION,
            DATASET_RIGHT_EEF_9D,
        ),
    ],
)
def test_checkpoint_eef_encoding_matches_finalized_dataset_row(
    side, control_position, control_quaternion, expected_eef_9d
):
    encoded = control_pose_to_training_eef_9d(
        control_position, control_quaternion, side
    )

    np.testing.assert_allclose(encoded, expected_eef_9d, atol=3e-5)


def test_finalized_dataset_eef_action_decodes_to_20d_control_layout():
    policy_action = {
        "left_eef_9d": DATASET_LEFT_EEF_9D.reshape(1, 1, 9),
        "right_eef_9d": DATASET_RIGHT_EEF_9D.reshape(1, 1, 9),
        "left_hand": LEFT_HAND_TARGET.reshape(1, 1, 3),
        "right_hand": RIGHT_HAND_TARGET.reshape(1, 1, 3),
    }

    action = build_agibot_gr00t_action_np(policy_action)

    assert action.shape == (1, 1, AGIBOT_GR00T_ACTION_DIM)
    np.testing.assert_allclose(
        action[0, 0, 0:3], DATASET_LEFT_CONTROL_POSITION, atol=3e-5
    )
    np.testing.assert_allclose(action[0, 0, 7:10], LEFT_HAND_TARGET, atol=1e-6)
    np.testing.assert_allclose(
        action[0, 0, 10:13], DATASET_RIGHT_CONTROL_POSITION, atol=3e-5
    )
    np.testing.assert_allclose(action[0, 0, 17:20], RIGHT_HAND_TARGET, atol=1e-6)
    left_dot = abs(np.dot(action[0, 0, 3:7], DATASET_LEFT_CONTROL_QUATERNION))
    right_dot = abs(np.dot(action[0, 0, 13:17], DATASET_RIGHT_CONTROL_QUATERNION))
    np.testing.assert_allclose(left_dot, 1.0, atol=3e-5)
    np.testing.assert_allclose(right_dot, 1.0, atol=3e-5)


def test_hold_action_uses_current_absolute_pose_and_hand_targets(
    agibot_observation, fake_client
):
    policy = _build_policy()
    hold = policy._extract_hold_action(agibot_observation).numpy()

    assert hold.shape == (NUM_ENVS, AGIBOT_GR00T_ACTION_DIM)
    np.testing.assert_allclose(
        hold[:, 0:3], np.broadcast_to(LEFT_CONTROL_POSITION, (NUM_ENVS, 3))
    )
    np.testing.assert_allclose(
        hold[:, 3:7], np.broadcast_to(LEFT_CONTROL_QUATERNION, (NUM_ENVS, 4))
    )
    np.testing.assert_allclose(hold[:, 7:10], 0.4)
    np.testing.assert_allclose(
        hold[:, 10:13], np.broadcast_to(RIGHT_CONTROL_POSITION, (NUM_ENVS, 3))
    )
    np.testing.assert_allclose(
        hold[:, 13:17], np.broadcast_to(RIGHT_CONTROL_QUATERNION, (NUM_ENVS, 4))
    )
    np.testing.assert_allclose(hold[:, 17:20], 0.6)


def test_agibot_config_holds_through_reset_transient_before_inference(
    agibot_observation, fake_client, monkeypatch
):
    monkeypatch.delenv("ISAACLAB_ARENA_GR00T_WARMUP_STEPS", raising=False)
    policy = _build_policy()
    policy.set_task_description("Stack the bowls together.")

    assert policy.policy_config.initial_camera_warmup_steps == 10
    assert policy.policy_config.state_delay_steps == 1
    ready_eef_9d = np.asarray(
        policy.policy_config.initial_agibot_ready_eef_9d, dtype=np.float32
    )
    assert ready_eef_9d.shape == (18,)
    ready_left_position, ready_left_quaternion = training_eef_9d_to_control_pose(
        ready_eef_9d[:9], "left"
    )
    ready_right_position, ready_right_quaternion = training_eef_9d_to_control_pose(
        ready_eef_9d[9:], "right"
    )
    hold = policy.get_action(None, agibot_observation).numpy()
    assert fake_client[0].last_observation is None
    np.testing.assert_allclose(
        hold[:, 0:3], np.broadcast_to(ready_left_position, (NUM_ENVS, 3)), atol=1e-6
    )
    np.testing.assert_allclose(
        np.abs(np.sum(hold[:, 3:7] * ready_left_quaternion, axis=-1)),
        np.ones(NUM_ENVS),
        atol=1e-6,
    )
    np.testing.assert_allclose(hold[:, 7:10], 0.994)
    np.testing.assert_allclose(
        hold[:, 10:13], np.broadcast_to(ready_right_position, (NUM_ENVS, 3)), atol=1e-6
    )
    np.testing.assert_allclose(
        np.abs(np.sum(hold[:, 13:17] * ready_right_quaternion, axis=-1)),
        np.ones(NUM_ENVS),
        atol=1e-6,
    )
    np.testing.assert_allclose(hold[:, 17:20], 0.994)

    rendered_observation = dict(agibot_observation)
    rendered_observation["camera_obs"] = {
        name: torch.full_like(image, 17)
        for name, image in agibot_observation["camera_obs"].items()
    }
    rendered_observation["policy"] = dict(agibot_observation["policy"])
    rendered_observation["policy"]["robot_joint_pos"] = (
        agibot_observation["policy"]["robot_joint_pos"] + 1.0
    )
    rendered_observation["policy"]["left_eef_pos"] = (
        agibot_observation["policy"]["left_eef_pos"] + 0.1
    )
    rendered_observation["policy"]["right_eef_pos"] = (
        agibot_observation["policy"]["right_eef_pos"] + 0.1
    )
    for _ in range(9):
        policy.get_action(None, rendered_observation)
        assert fake_client[0].last_observation is None

    policy.get_action(None, rendered_observation)
    sent = fake_client[0].last_observation
    assert sent is not None
    for video in sent["video"].values():
        np.testing.assert_array_equal(video, np.full_like(video, 17))
    # This checkpoint's sidecar wrist streams lead proprioception by one control step: the
    # newly rendered video is paired with the policy group from the previous warmup step.
    np.testing.assert_allclose(
        sent["state"]["left_eef_9d"][..., :3],
        np.broadcast_to(LEFT_CONTROL_POSITION + 0.1, (NUM_ENVS, 1, 3)),
    )
    np.testing.assert_allclose(
        sent["state"]["right_eef_9d"][..., :3],
        np.broadcast_to(RIGHT_CONTROL_POSITION + 0.1, (NUM_ENVS, 1, 3)),
    )
    joint_position = rendered_observation["policy"]["robot_joint_pos"].numpy()
    np.testing.assert_allclose(
        sent["state"]["left_arm"][:, 0], joint_position[:, [4, 6, 8, 10, 12, 14, 16]]
    )
    np.testing.assert_allclose(
        sent["state"]["right_arm"][:, 0], joint_position[:, [5, 7, 9, 11, 13, 15, 17]]
    )


def test_eval_config_starts_from_recorded_ready_joint_branch():
    config = yaml.safe_load(N17_EXPERIMENT_CONFIG.read_text())
    environment = config["runs"]["agibot_stack_bowls_gr00t_n17"]["environment"]
    ready = np.asarray(
        environment["policy_ready_arm_joint_positions"], dtype=np.float32
    )
    recorded_reset = np.asarray(
        [
            -1.08169997,
            0.59069997,
            0.34419999,
            -1.28190005,
            0.69279999,
            0.69999999,
            0.0,
            1.08169997,
            -0.59069997,
            -0.34419999,
            1.28190005,
            -0.69279999,
            -0.69999999,
            0.0,
        ],
        dtype=np.float32,
    )
    recorded_ready_relative = np.asarray(
        [
            -3.46899033e-05,
            2.68220901e-06,
            -1.03801489e-04,
            -1.28269196e-04,
            -3.54647636e-05,
            1.27853274e-01,
            1.95965504e-06,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
        dtype=np.float32,
    )

    assert ready.shape == (14,)
    np.testing.assert_allclose(
        ready - recorded_reset, recorded_ready_relative, atol=2e-7
    )


def test_production_eval_uses_validated_diffik_and_short_fresh_chunks(
    agibot_observation, fake_client, monkeypatch
):
    monkeypatch.delenv("ISAACLAB_ARENA_GR00T_ACTION_SAMPLE_COUNT", raising=False)
    experiment = yaml.safe_load(N17_EXPERIMENT_CONFIG.read_text())
    run = experiment["runs"]["agibot_stack_bowls_gr00t_n17"]

    assert run["environment"]["action_mode"] == "gr00t_diffik"

    policy = _build_policy()
    assert policy.action_chunk_length == 16
    assert policy.policy_config.denoising_steps == 8
    assert policy._action_sample_count == 5
    assert policy.policy_config.action_chunk_translation_anchor_alpha == 1.0
    assert policy._rtc_enabled is False

    _request_action(policy, agibot_observation)
    assert fake_client[0].action_request_count == 5
    assert fake_client[0].last_options == {"num_inference_timesteps": 8}


def test_denoise_sweep_runs_twenty_paired_episodes_and_changes_only_denoise():
    policy_configs = {
        steps: yaml.safe_load(path.read_text())
        for steps, path in DENOISE_SWEEP_POLICY_CONFIGS.items()
    }
    baseline = policy_configs[8]
    for steps, config in policy_configs.items():
        assert config["denoising_steps"] == steps
        assert config.keys() == baseline.keys()
        for key in config.keys() - {"denoising_steps"}:
            assert config[key] == baseline[key]

    experiment = yaml.safe_load(DENOISE_SWEEP_EXPERIMENT_CONFIG.read_text())
    assert experiment["shared"]["rollout_limit"] == {"num_episodes": 20}
    assert experiment["shared"]["environment_builder"]["num_envs"] == 1
    assert experiment["shared"]["environment_builder"]["seed"] == 42
    assert experiment["shared"]["num_rebuilds"] == 1
    assert len(experiment["runs"]) == 4

    configured_paths = {
        Path(run["policy"]["policy_config_yaml_path"])
        for run in experiment["runs"].values()
    }
    assert configured_paths == set(DENOISE_SWEEP_POLICY_CONFIGS.values())


def test_action_sample_count_environment_override_takes_precedence(
    fake_client, monkeypatch
):
    monkeypatch.setenv("ISAACLAB_ARENA_GR00T_ACTION_SAMPLE_COUNT", "3")

    policy = _build_policy()

    assert policy._action_sample_count == 3


def test_full_policy_reset_seeds_remote_diffusion_but_partial_reset_does_not(
    fake_client,
):
    policy = _build_policy()

    policy.reset()
    assert fake_client[0].last_reset_options == {"seed": 10}

    policy.reset(env_ids=torch.tensor([0]))
    assert fake_client[0].last_reset_options is None

    policy.reset(env_ids=torch.tensor([1, 0]))
    assert fake_client[0].last_reset_options == {"seed": 10}


def test_diffik_action_config_pins_the_measured_branch_continuity_contract():
    actions = AgibotGr00tDiffIkActionsCfg()

    for arm_action in (actions.left_arm_action, actions.right_arm_action):
        assert isinstance(arm_action, ControlRateDifferentialIKActionCfg)
        assert arm_action.class_type is ControlRateDifferentialIKAction
        assert arm_action.controller.command_type == "pose"
        assert arm_action.controller.use_relative_mode is False
        assert arm_action.controller.ik_method == "dls"
        assert arm_action.controller.ik_params["lambda_val"] == pytest.approx(0.05)
        assert arm_action.max_joint_delta_per_control_step == pytest.approx(0.1)
        assert arm_action.nullspace_joint_weights == (0.0,) * 6 + (1.0,)
        assert arm_action.nullspace_pinv_rtol == pytest.approx(1.0e-4)


def test_diffik_nullspace_uses_the_episode_joint_reference_without_eef_leakage():
    action = object.__new__(ControlRateDifferentialIKAction)
    action.cfg = SimpleNamespace(nullspace_gain=0.2, nullspace_pinv_rtol=1.0e-4)
    action._num_joints = 7
    action._nullspace_joint_weights = torch.ones(1, 7)
    action._joint_reference = torch.zeros(1, 7)
    jacobian = torch.cat((torch.eye(6), torch.zeros(6, 1)), dim=1).unsqueeze(0)
    joint_pos = torch.tensor([[0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 2.0]])

    delta = action._nullspace_delta(jacobian, joint_pos)

    torch.testing.assert_close(delta[:, :6], torch.zeros(1, 6), atol=1.0e-6, rtol=0.0)
    torch.testing.assert_close(delta[:, 6], torch.tensor([-0.4]), atol=1.0e-6, rtol=0.0)
    torch.testing.assert_close(
        (jacobian @ delta.unsqueeze(-1)).squeeze(-1),
        torch.zeros(1, 6),
        atol=1.0e-6,
        rtol=0.0,
    )


def test_diffik_joint_rate_limit_does_not_throttle_other_joints():
    action = object.__new__(ControlRateDifferentialIKAction)
    action.cfg = SimpleNamespace(max_joint_delta_per_control_step=0.1)
    delta = torch.tensor([[0.2, -0.1, 0.05], [0.02, -0.03, 0.01]])

    limited = action._limit_joint_delta(delta)

    torch.testing.assert_close(
        limited,
        torch.tensor([[0.1, -0.1, 0.05], [0.02, -0.03, 0.01]]),
    )


def test_diffik_command_comparison_treats_quaternion_sign_as_the_same_pose():
    action = object.__new__(ControlRateDifferentialIKAction)
    action.cfg = SimpleNamespace(command_tolerance=1.0e-6)
    current = torch.tensor([[0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 1.0]])
    same_pose = torch.tensor([[0.1, 0.2, 0.3, 0.0, 0.0, 0.0, -1.0]])
    moved_pose = same_pose.clone()
    moved_pose[0, 0] += 1.0e-3

    assert not action._command_changed(current, same_pose).item()
    assert action._command_changed(current, moved_pose).item()


def test_explicit_joint_reset_pose_preserves_default_observation_reference():
    class FakeAsset:
        def __init__(self):
            self.data = SimpleNamespace(
                default_joint_pos=SimpleNamespace(torch=torch.full((4, 5), 7.0))
            )
            self.calls = []

        def find_joints(self, names, preserve_order):
            assert names == ["joint_b", "joint_d"]
            assert preserve_order is True
            return [1, 3], names

        def write_joint_position_to_sim_index(self, **kwargs):
            self.calls.append(("position", kwargs))

        def write_joint_velocity_to_sim_index(self, **kwargs):
            self.calls.append(("velocity", kwargs))

        def set_joint_position_target_index(self, **kwargs):
            self.calls.append(("position_target", kwargs))

        def set_joint_velocity_target_index(self, **kwargs):
            self.calls.append(("velocity_target", kwargs))

    asset = FakeAsset()
    original_default = asset.data.default_joint_pos.torch.clone()
    env = SimpleNamespace(scene={"robot": asset}, device="cpu")
    env_ids = torch.tensor([1, 3])

    reset_joint_position_and_velocity_to_pose(
        env,
        env_ids,
        joint_names=["joint_b", "joint_d"],
        joint_positions=[0.25, -0.5],
    )

    torch.testing.assert_close(asset.data.default_joint_pos.torch, original_default)
    assert [name for name, _ in asset.calls] == [
        "position",
        "velocity",
        "position_target",
        "velocity_target",
    ]
    for name, kwargs in asset.calls:
        assert kwargs["joint_ids"] == [1, 3]
        torch.testing.assert_close(kwargs["env_ids"], env_ids)
        expected = (
            torch.tensor([[0.25, -0.5], [0.25, -0.5]])
            if "position" in name
            else torch.zeros((2, 2))
        )
        value = (
            kwargs["position"]
            if name == "position"
            else kwargs.get("velocity", kwargs.get("target"))
        )
        torch.testing.assert_close(value, expected)


@pytest.mark.parametrize(
    ("controller", "arena_controller", "filename", "target_p_gain", "target_d_gain"),
    [
        (
            AGIBOT_LEFT_ARM_GR00T_RMPFLOW_CFG,
            AGIBOT_LEFT_ARM_ARENA_RMPFLOW_CFG,
            "agibot_left_arm_gr00t_rmpflow_config.yaml",
            480.0,
            40.0,
        ),
        (
            AGIBOT_RIGHT_ARM_GR00T_RMPFLOW_CFG,
            AGIBOT_RIGHT_ARM_ARENA_RMPFLOW_CFG,
            "agibot_right_arm_gr00t_rmpflow_config.yaml",
            400.0,
            60.0,
        ),
    ],
)
def test_gr00t_uses_oracle_calibrated_rmpflow_config(
    controller, arena_controller, filename, target_p_gain, target_d_gain
):
    """Keep the absolute-target tuning isolated from relative-EEF control."""
    config_path = Path(controller.config_file)
    assert config_path.name == filename
    assert config_path.is_file()
    assert controller.config_file != arena_controller.config_file
    assert controller.collision_file == arena_controller.collision_file

    config = yaml.safe_load(config_path.read_text())
    rmp_params = config["rmp_params"]
    assert rmp_params["cspace_target_rmp"]["metric_scalar"] == pytest.approx(1.0e-6)
    assert rmp_params["target_rmp"]["accel_p_gain"] == pytest.approx(target_p_gain)
    assert rmp_params["target_rmp"]["accel_d_gain"] == pytest.approx(target_d_gain)
    assert config["canonical_resolve"]["max_acceleration_norm"] == pytest.approx(200.0)


def test_oracle_rmpflow_tuning_is_scoped_to_gr00t_action_instances():
    gr00t_actions = AgibotGr00tActionsCfg()
    teleop_actions = AgibotDualArmActionsCfg()

    assert (
        gr00t_actions.left_arm_action.controller.config_file
        == AGIBOT_LEFT_ARM_GR00T_RMPFLOW_CFG.config_file
    )
    assert (
        gr00t_actions.right_arm_action.controller.config_file
        == AGIBOT_RIGHT_ARM_GR00T_RMPFLOW_CFG.config_file
    )
    assert (
        teleop_actions.left_arm_action.controller.config_file
        == AGIBOT_LEFT_ARM_ARENA_RMPFLOW_CFG.config_file
    )
    assert (
        teleop_actions.right_arm_action.controller.config_file
        == AGIBOT_RIGHT_ARM_ARENA_RMPFLOW_CFG.config_file
    )
    for arm_action in (gr00t_actions.left_arm_action, gr00t_actions.right_arm_action):
        assert isinstance(arm_action, LowPassRMPFlowActionCfg)
        assert arm_action.class_type is LowPassRMPFlowAction
        assert arm_action.joint_target_ema_alpha == pytest.approx(0.2)
    assert not isinstance(teleop_actions.left_arm_action, LowPassRMPFlowActionCfg)
    assert not isinstance(teleop_actions.right_arm_action, LowPassRMPFlowActionCfg)


def test_gr00t_gripper_drive_adds_preload_without_changing_raw_action_endpoints():
    gr00t_actions = AgibotGr00tActionsCfg()
    teleop_actions = AgibotDualArmActionsCfg()

    for hand_action in (
        gr00t_actions.left_hand_action,
        gr00t_actions.right_hand_action,
    ):
        assert hand_action.offset == pytest.approx(-0.1)
        assert hand_action.scale * 0.0 + hand_action.offset == pytest.approx(-0.1)
        assert hand_action.scale * 0.994 + hand_action.offset == pytest.approx(0.994)
    assert teleop_actions.left_gripper_action.open_command_expr[
        "left_hand_joint1"
    ] == pytest.approx(0.994)
    assert teleop_actions.left_gripper_action.close_command_expr[
        "left_hand_joint1"
    ] == pytest.approx(0.0)


def test_low_pass_rmpflow_passes_first_target_then_filters_and_zeros_velocity():
    class FakeAsset:
        def __init__(self):
            self.data = SimpleNamespace(
                joint_pos=SimpleNamespace(torch=torch.zeros((1, 2))),
                joint_vel=SimpleNamespace(torch=torch.zeros((1, 2))),
            )
            self.position_targets = []
            self.velocity_targets = []

        def set_joint_position_target_index(self, *, target, joint_ids):
            self.position_targets.append(target.clone())

        def set_joint_velocity_target_index(self, *, target, joint_ids):
            self.velocity_targets.append(target.clone())

    raw_targets = iter(
        (
            torch.tensor([[1.0, -1.0]]),
            torch.tensor([[3.0, 1.0]]),
        )
    )
    action = object.__new__(LowPassRMPFlowAction)
    action.cfg = SimpleNamespace(joint_target_ema_alpha=0.1)
    action._asset = FakeAsset()
    action._joint_ids = slice(None)
    action._rmpflow_controller = SimpleNamespace(
        compute=lambda joint_pos, joint_vel: (
            next(raw_targets),
            torch.full((1, 2), 99.0),
        )
    )
    action._compute_frame_pose = lambda: (
        torch.zeros((1, 3)),
        torch.tensor([[0.0, 0.0, 0.0, 1.0]]),
    )
    action._filtered_joint_target = torch.zeros((1, 2))
    action._filter_initialized = torch.zeros(1, dtype=torch.bool)

    action.apply_actions()
    action.apply_actions()

    torch.testing.assert_close(
        action._asset.position_targets[0], torch.tensor([[1.0, -1.0]])
    )
    torch.testing.assert_close(
        action._asset.position_targets[1], torch.tensor([[1.2, -0.8]])
    )
    for velocity_target in action._asset.velocity_targets:
        torch.testing.assert_close(velocity_target, torch.zeros((1, 2)))
