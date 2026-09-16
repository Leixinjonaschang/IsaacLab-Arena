# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""GR00T remote closed-loop policy using GR00T's native PolicyClient.

This policy connects to a GR00T policy server (launched via
``gr00t/eval/run_gr00t_server.py``) and uses its own observation/action translation pipeline.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import torch

from isaaclab_arena.assets.register import register_policy
from isaaclab_arena.policy.action_scheduling import (
    ActionChunkScheduler,
    ActionScheduler,
    SyncedBatchActionScheduler,
)
from isaaclab_arena.policy.policy_base import PolicyBase
from isaaclab_arena_gr00t.policy.config.gr00t_closedloop_policy_config import (
    Gr00tClosedloopPolicyCfg,
    TaskMode,
)
from isaaclab_arena_gr00t.policy.gr00t_core import (
    Gr00tBasePolicyCfg,
    build_gr00t_action_tensor,
    build_gr00t_policy_observations,
    compute_action_dim,
    extract_eef_pose_from_nested_obs,
    extract_obs_numpy_from_torch,
    load_gr00t_joint_configs,
    resize_rgb_for_policy,
)
from isaaclab_arena_gr00t.policy.state_history import StateHistoryBuffer
from isaaclab_arena_gr00t.policy.video_history import VideoHistoryBuffer
from isaaclab_arena_gr00t.utils.io_utils import create_config_from_yaml, to_numpy

logger = logging.getLogger(__name__)


class ActionSchedulerType(str, Enum):
    """Action scheduler used to consume a policy's inference chunks."""

    CHUNK = "chunk"
    SYNCED_BATCH = "synced_batch"

    def get_scheduler_cls(self) -> type[ActionScheduler]:
        """Return the action-scheduler class this type selects."""
        return {
            ActionSchedulerType.CHUNK: ActionChunkScheduler,
            ActionSchedulerType.SYNCED_BATCH: SyncedBatchActionScheduler,
        }[self]


# TODO(xinjieyao, 2026-04-27): Consider adding RemotePolicyCfg and deriving this config from it.
@dataclass
class Gr00tRemoteClosedloopPolicyCfg(Gr00tBasePolicyCfg):
    """Configuration for Gr00tRemoteClosedloopPolicy.

    Inherits policy_config_yaml_path and policy_device from Gr00tBasePolicyCfg,
    and adds remote server connection parameters and num_envs.
    """

    num_envs: int = 1
    """Number of parallel environments served by the policy."""

    remote_host: str = "localhost"
    """GR00T policy server hostname."""

    remote_port: int = 5555
    """GR00T policy server port."""

    remote_api_token: str | None = None
    """Optional policy-server API token."""

    scheduler: ActionSchedulerType = ActionSchedulerType.CHUNK
    """Action scheduler used to consume inference chunks."""


@register_policy
class Gr00tRemoteClosedloopPolicy(PolicyBase[Gr00tRemoteClosedloopPolicyCfg]):
    """GR00T closed-loop policy that delegates inference to a remote GR00T server.

    Uses GR00T's native ``PolicyClient`` (from ``gr00t.policy.server_client``)
    to communicate with a GR00T policy server.
    """

    name = "gr00t_remote_closedloop"

    def __init__(self, config: Gr00tRemoteClosedloopPolicyCfg):
        super().__init__(config)

        action_scheduler_cls = ActionSchedulerType(config.scheduler).get_scheduler_cls()

        # Policy config (for obs/action translation — no model loading)
        # TODO(xinjieyao, 2026-04-27): to be refactored
        self.policy_config: Gr00tClosedloopPolicyCfg = create_config_from_yaml(
            config.policy_config_yaml_path, Gr00tClosedloopPolicyCfg
        )
        self.num_envs = config.num_envs
        self.device = config.policy_device
        self.task_mode = TaskMode(self.policy_config.task_mode_name)

        # Joint configs (for sim from/to policy joint space remapping)
        (
            self.policy_joints_config,
            self.robot_action_joints_config,
            self.robot_state_joints_config,
        ) = load_gr00t_joint_configs(self.policy_config)

        # Connect before allocating any temporal buffers: the checkpoint's processor is the source
        # of truth for video/state/action horizons, and base and post-trained checkpoints can expose
        # different contracts under the same embodiment tag.
        from gr00t.policy.server_client import PolicyClient

        client = PolicyClient(
            host=config.remote_host,
            port=config.remote_port,
            api_token=config.remote_api_token,
            strict=False,
        )
        self._client: Any | None = client
        if not client.ping():
            raise ConnectionError(
                f"Cannot reach GR00T policy server at {config.remote_host}:{config.remote_port}"
            )

        self.modality_configs = client.get_modality_config()
        required_modalities = {"video", "state", "action", "language"}
        assert isinstance(
            self.modality_configs, dict
        ), f"GR00T policy server returned {type(self.modality_configs).__name__}, expected a modality-config dict"
        missing_modalities = required_modalities - self.modality_configs.keys()
        assert not missing_modalities, (
            f"GR00T policy server omitted required modalities {sorted(missing_modalities)};"
            f" got {sorted(self.modality_configs)}"
        )
        assert len(self.policy_config.pov_cam_name_sim) == len(
            self.modality_configs["video"].modality_keys
        ), (
            f"Arena config provides cameras {self.policy_config.pov_cam_name_sim}, but the remote checkpoint expects"
            f" {self.modality_configs['video'].modality_keys}"
        )

        # Video history. GR00T checkpoints declare how many past frames they want through the video
        # modality config's delta_indices; the server rejects a request that carries a different
        # number, so the buffer is sized from the config rather than assumed.
        self._video_history = VideoHistoryBuffer(
            delta_indices=self.modality_configs["video"].delta_indices,
            num_envs=self.num_envs,
        )
        self._state_history = StateHistoryBuffer(
            delay_steps=self.policy_config.state_delay_steps,
            num_envs=self.num_envs,
        )

        # GR00T N1.7's DROID embodiment additionally conditions on the end-effector pose.
        self._requires_eef_state = (
            "eef_9d" in self.modality_configs["state"].modality_keys
        )
        self._requires_agibot_eef_state = (
            self.task_mode == TaskMode.AGIBOT_BIMANUAL_MANIPULATION
            and {
                "left_eef_9d",
                "right_eef_9d",
            }.issubset(self.modality_configs["state"].modality_keys)
        )
        # Every DROID checkpoint, N1.6 and N1.7 alike, was trained on a 0-1 gripper signal.
        self._normalizes_gripper_state = (
            self.task_mode == TaskMode.DROID_MANIPULATION
            and ("gripper_position" in self.modality_configs["state"].modality_keys)
        )

        # Action / chunk shapes
        self.action_dim = compute_action_dim(
            self.task_mode, self.robot_action_joints_config
        )
        self.action_chunk_length = self.policy_config.action_chunk_length
        self.action_horizon = len(self.modality_configs["action"].delta_indices)
        assert 1 <= self.action_chunk_length <= self.action_horizon, (
            f"action_chunk_length={self.action_chunk_length} must be between 1 and the remote checkpoint's"
            f" action horizon {self.action_horizon}"
        )
        self._rtc_enabled = self.policy_config.rtc_enabled
        self._rtc_overlap_steps = self.action_horizon - self.action_chunk_length
        if self._rtc_enabled:
            assert (
                self._rtc_overlap_steps > 0
            ), "GR00T RTC requires action_chunk_length to be smaller than the remote action horizon"
            assert self.policy_config.rtc_frozen_steps <= self._rtc_overlap_steps, (
                f"rtc_frozen_steps={self.policy_config.rtc_frozen_steps} exceeds the remote-contract "
                f"overlap of {self._rtc_overlap_steps} steps"
            )
        self._previous_policy_action: dict[str, np.ndarray] | None = None
        self._rtc_valid_envs = np.zeros(self.num_envs, dtype=bool)

        self._chunking_state: ActionScheduler | None = action_scheduler_cls(
            num_envs=self.num_envs,
            action_chunk_length=self.action_chunk_length,
            action_horizon=self.action_horizon,
            action_dim=self.action_dim,
            device=self.device,
            dtype=torch.float,
        )

        logger.info(
            "GR00T remote contract: video=%s delta=%s state=%s delay=%d action=%s horizon=%d execution=%d"
            " samples=%d denoise=%d anchor=%.3f rtc=%s overlap=%d frozen=%d ramp=%.3f",
            self.modality_configs["video"].modality_keys,
            self.modality_configs["video"].delta_indices,
            self.modality_configs["state"].modality_keys,
            self.policy_config.state_delay_steps,
            self.modality_configs["action"].modality_keys,
            self.action_horizon,
            self.action_chunk_length,
            self.policy_config.action_sample_count,
            self.policy_config.denoising_steps,
            self.policy_config.action_chunk_translation_anchor_alpha,
            self._rtc_enabled,
            self._rtc_overlap_steps,
            self.policy_config.rtc_frozen_steps,
            self.policy_config.rtc_ramp_rate,
        )

        self.task_description: str | None = None
        trace_dir = os.environ.get("ISAACLAB_ARENA_GR00T_TRACE_DIR")
        self._trace_dir = Path(trace_dir) if trace_dir else None
        self._trace_limit = int(
            os.environ.get("ISAACLAB_ARENA_GR00T_TRACE_LIMIT", "10")
        )
        self._trace_index = 0
        if self._trace_dir is not None:
            self._trace_dir.mkdir(parents=True, exist_ok=True)
            logger.info(
                "GR00T diagnostic traces will be written to %s", self._trace_dir
            )

        # Mounted cameras can still contain their pre-articulation poses in the observation returned
        # by reset. Hold the robot while the configured number of post-reset frames are rendered.
        # Keep the environment override for controlled diagnostics.
        self._warmup_steps = int(
            os.environ.get(
                "ISAACLAB_ARENA_GR00T_WARMUP_STEPS",
                str(self.policy_config.initial_camera_warmup_steps),
            )
        )
        assert (
            self._warmup_steps >= 0
        ), "ISAACLAB_ARENA_GR00T_WARMUP_STEPS must be non-negative"
        self._warmup_steps_remaining = self._warmup_steps
        self._reuse_initial_state_after_warmup = (
            os.environ.get("ISAACLAB_ARENA_GR00T_WARMUP_REUSE_INITIAL_STATE", "0")
            == "1"
        )
        assert (
            not self._reuse_initial_state_after_warmup or self._warmup_steps > 0
        ), "ISAACLAB_ARENA_GR00T_WARMUP_REUSE_INITIAL_STATE requires ISAACLAB_ARENA_GR00T_WARMUP_STEPS > 0"
        self._warmup_initial_policy_observation: dict[str, Any] | None = None
        self._warmup_hold_action: torch.Tensor | None = None
        ready_eef_9d_override = os.environ.get(
            "ISAACLAB_ARENA_GR00T_WARMUP_AGIBOT_READY_EEF_9D"
        )
        if ready_eef_9d_override is not None:
            self._warmup_agibot_ready_eef_9d = np.fromstring(
                ready_eef_9d_override, dtype=np.float32, sep=","
            )
        elif self.policy_config.initial_agibot_ready_eef_9d is not None:
            self._warmup_agibot_ready_eef_9d = np.asarray(
                self.policy_config.initial_agibot_ready_eef_9d, dtype=np.float32
            )
        else:
            self._warmup_agibot_ready_eef_9d = None
        if self._warmup_agibot_ready_eef_9d is not None:
            assert (
                self.task_mode == TaskMode.AGIBOT_BIMANUAL_MANIPULATION
            ), "ISAACLAB_ARENA_GR00T_WARMUP_AGIBOT_READY_EEF_9D only supports AgiBot"
            assert self._warmup_agibot_ready_eef_9d.shape == (
                18,
            ), "ISAACLAB_ARENA_GR00T_WARMUP_AGIBOT_READY_EEF_9D must contain 18 comma-separated values"
        self._idle_arm_hold_threshold_m = float(
            os.environ.get("ISAACLAB_ARENA_GR00T_IDLE_ARM_HOLD_THRESHOLD_M", "0")
        )
        assert (
            self._idle_arm_hold_threshold_m >= 0.0
        ), "ISAACLAB_ARENA_GR00T_IDLE_ARM_HOLD_THRESHOLD_M must be non-negative"
        self._action_sample_count = int(
            os.environ.get(
                "ISAACLAB_ARENA_GR00T_ACTION_SAMPLE_COUNT",
                str(self.policy_config.action_sample_count),
            )
        )
        assert (
            self._action_sample_count >= 1
        ), "ISAACLAB_ARENA_GR00T_ACTION_SAMPLE_COUNT must be positive"
        if self._action_sample_count > 1:
            logger.info(
                "GR00T diagnostic action sampling: taking the elementwise median of %d server responses",
                self._action_sample_count,
            )

    # ---------------------- Policy interface -------------------

    def set_task_description(self, task_description: str | None) -> str:
        if task_description is None:
            task_description = self.policy_config.language_instruction
        if not task_description:
            raise ValueError(
                "No language instruction provided. Set 'language_instruction' in the job config, "
                "pass --language_instruction on the CLI, or define 'task_description' on the task class."
            )
        self.task_description = task_description
        return self.task_description

    def get_action(self, env: gym.Env, observation: dict[str, Any]) -> torch.Tensor:
        assert self._chunking_state is not None, "GR00T remote policy has been closed"

        # Record every control step, not just the steps that trigger inference: the video
        # delta_indices are expressed in control steps, so a buffer fed only on inference steps
        # would hand the model the wrong point in the past.
        self._video_history.push(
            self._resized_frames(observation, self.policy_config.pov_cam_name_sim)
        )
        self._state_history.push(
            {key: to_numpy(value) for key, value in observation["policy"].items()}
        )

        if self._warmup_steps_remaining > 0:
            if self._warmup_initial_policy_observation is None:
                self._warmup_initial_policy_observation = {
                    key: value.clone() if isinstance(value, torch.Tensor) else value
                    for key, value in observation["policy"].items()
                }
            self._warmup_steps_remaining -= 1
            if self._warmup_hold_action is None:
                self._warmup_hold_action = self._extract_hold_action(observation)
                if self._warmup_agibot_ready_eef_9d is not None:
                    eef_9d = self._warmup_agibot_ready_eef_9d
                    ready_policy_action = {
                        "left_eef_9d": np.tile(
                            eef_9d[None, None, :9], (self.num_envs, 1, 1)
                        ),
                        "right_eef_9d": np.tile(
                            eef_9d[None, None, 9:], (self.num_envs, 1, 1)
                        ),
                        "left_hand": np.full(
                            (self.num_envs, 1, 3), 0.994, dtype=np.float32
                        ),
                        "right_hand": np.full(
                            (self.num_envs, 1, 3), 0.994, dtype=np.float32
                        ),
                    }
                    self._warmup_hold_action = build_gr00t_action_tensor(
                        robot_action_policy=ready_policy_action,
                        task_mode=self.task_mode,
                        policy_joints_config=self.policy_joints_config,
                        robot_action_joints_config=self.robot_action_joints_config,
                        device=self.device,
                        embodiment_tag=self.policy_config.embodiment_tag,
                    )[:, 0]
                if self.task_mode == TaskMode.AGIBOT_BIMANUAL_MANIPULATION:
                    # AgiBot's measured hand state is zero at the open reset pose, while its action
                    # target for that same pose is 0.994 in the demonstrations.
                    self._warmup_hold_action[:, 7:10] = 0.994
                    self._warmup_hold_action[:, 17:20] = 0.994
            warmup_action = self._warmup_hold_action.clone()
            logger.info("GR00T camera warmup: returning hold action before inference")
            return warmup_action

        def fetch_chunk() -> torch.Tensor:
            inference_observation = dict(observation)
            inference_observation["policy"] = self._state_history.delayed()
            if (
                self._reuse_initial_state_after_warmup
                and self._warmup_initial_policy_observation is not None
            ):
                inference_observation["policy"] = (
                    self._warmup_initial_policy_observation
                )
                self._warmup_initial_policy_observation = None
                logger.info(
                    "GR00T diagnostic warmup: first inference reuses the saved pre-step state"
                )
            return self._get_action_chunk(
                env,
                inference_observation,
                self.policy_config.pov_cam_name_sim,
                current_observation=observation,
            )

        return self._chunking_state.get_action(
            fetch_chunk,
            hold_action=self._extract_hold_action(observation),
        )

    def _resized_frames(
        self, observation: dict[str, Any], camera_names: list[str] | str
    ) -> list[np.ndarray]:
        """Return this step's camera frames, resized once to the policy's input size."""
        if isinstance(camera_names, str):
            camera_names = [camera_names]
        rgb_list_np, _ = extract_obs_numpy_from_torch(
            nested_obs=observation, camera_names=camera_names
        )
        target_image_size = getattr(self.policy_config, "target_image_size", None)
        if target_image_size is not None:
            rgb_list_np = resize_rgb_for_policy(
                rgb_list_np=rgb_list_np, target_image_size=target_image_size
            )
        return rgb_list_np

    def _build_extra_state(
        self, env: gym.Env, observation: dict[str, Any]
    ) -> dict[str, np.ndarray]:
        """Return state entries the joint remapping cannot express or would express wrongly.

        Two DROID state keys need this. ``eef_9d`` is a Cartesian pose rather than a set of joints.
        ``gripper_position`` *is* a joint, but DROID reports it normalised to 0-1 (confirmed by the
        dataset statistics shipped with GR00T, ``demo_data/droid_sample/meta/stats.json``), whereas
        the raw ``finger_joint`` the joint remapping would pick up is in radians over 0-pi/4. Sending
        radians means the policy never sees a fully closed gripper.
        """
        extra_state: dict[str, np.ndarray] = {}
        if self._requires_agibot_eef_state:
            from isaaclab_arena_gr00t.utils.agibot_eef import (
                control_pose_to_training_eef_9d,
            )

            policy_observation = observation.get("policy", {})
            for side in ("left", "right"):
                position = policy_observation.get(f"{side}_eef_pos")
                quaternion = policy_observation.get(f"{side}_eef_quat")
                assert (
                    position is not None and quaternion is not None
                ), f"An AgiBot policy needs '{side}_eef_pos' and '{side}_eef_quat' observation terms"
                extra_state[f"{side}_eef_9d"] = control_pose_to_training_eef_9d(
                    to_numpy(position), to_numpy(quaternion), side
                )

        if self._normalizes_gripper_state:
            gripper_position = observation["policy"].get("gripper_pos")
            assert gripper_position is not None, (
                "A DROID policy needs a normalised 'gripper_pos' observation term, which this"
                " embodiment does not publish."
            )
            extra_state["gripper_position"] = to_numpy(gripper_position).reshape(
                self.num_envs, -1
            )

        if self._requires_eef_state:
            from isaaclab_arena_gr00t.utils.droid_eef import compute_eef_9d_state

            eef_pose = extract_eef_pose_from_nested_obs(observation)
            assert eef_pose is not None, (
                "The policy asks for an 'eef_9d' state but the embodiment publishes no 'eef_pos' /"
                " 'eef_quat' observation terms."
            )
            eef_pos_w, eef_quat_w_wxyz = eef_pose
            base_pos_w, base_quat_w_wxyz = self._robot_base_pose(env)
            extra_state["eef_9d"] = compute_eef_9d_state(
                eef_pos_w, eef_quat_w_wxyz, base_pos_w, base_quat_w_wxyz
            )
        return extra_state

    @staticmethod
    def _robot_base_pose(env: gym.Env) -> tuple[np.ndarray | None, np.ndarray | None]:
        """Return the semantic DROID base pose, or (None, None) when no environment is available.

        The DROID USD's ``panda_link0`` uses the opposite X/Y convention from Polymetis. The fixed
        asset-to-semantic transform is composed here so root placement still works without exposing
        the USD convention to the policy.
        """
        import warp as wp

        from isaaclab_arena_gr00t.utils.droid_eef import droid_base_pose_from_usd_root

        try:
            robot = env.unwrapped.scene["robot"]
        except (AttributeError, KeyError):
            return None, None
        return droid_base_pose_from_usd_root(
            wp.to_torch(robot.data.root_pos_w).detach().cpu().numpy(),
            wp.to_torch(robot.data.root_quat_w).detach().cpu().numpy(),
        )

    def _extract_hold_action(self, observation: dict[str, Any]) -> torch.Tensor:
        """Build the action vector that waiting envs should hold: their current sim joint positions
        copied into the action slots that share a joint name with the state config."""
        if self.task_mode == TaskMode.AGIBOT_BIMANUAL_MANIPULATION:
            policy_observation = observation.get("policy", {})

            def _term(name: str, width: int) -> torch.Tensor:
                value = policy_observation.get(name)
                assert (
                    value is not None
                ), f"AgiBot hold action needs observation term '{name}'"
                value = value.to(device=self.device, dtype=torch.float).reshape(
                    self.num_envs, -1
                )
                assert (
                    value.shape[1] == width
                ), f"AgiBot observation '{name}' must have width {width}, got {tuple(value.shape)}"
                return value

            return torch.cat(
                (
                    _term("left_eef_pos", 3),
                    _term("left_eef_quat", 4),
                    _term("left_hand_pos", 3),
                    _term("right_eef_pos", 3),
                    _term("right_eef_quat", 4),
                    _term("right_hand_pos", 3),
                ),
                dim=1,
            )

        joint_pos_sim = observation["policy"]["robot_joint_pos"].to(
            device=self.device, dtype=torch.float
        )
        hold_action = torch.zeros(
            (self.num_envs, self.action_dim), dtype=torch.float, device=self.device
        )
        for joint_name, action_idx in self.robot_action_joints_config.items():
            state_idx = self.robot_state_joints_config.get(joint_name)
            if state_idx is not None:
                hold_action[:, action_idx] = joint_pos_sim[:, state_idx]
        return hold_action

    def _get_action_chunk(
        self,
        env: gym.Env,
        observation: dict[str, Any],
        camera_names: list[str] | str = "robot_head_cam_rgb",
        current_observation: dict[str, Any] | None = None,
    ) -> torch.Tensor:
        """Get an action chunk from the remote GR00T server.

        Calls GR00T's PolicyClient to get the action chunk.
        """
        if isinstance(camera_names, str):
            camera_names = [camera_names]

        # 1. Reuse the same obs translation as local policy
        assert self.task_description is not None, "Task description is not set"
        assert self._client is not None, "GR00T remote policy has been closed"
        _, joint_pos_sim_np = extract_obs_numpy_from_torch(
            nested_obs=observation, camera_names=camera_names
        )
        policy_observations = build_gr00t_policy_observations(
            rgb_list_np=self._video_history.stack(),
            joint_pos_sim_np=joint_pos_sim_np,
            task_description=self.task_description,
            policy_config=self.policy_config,
            robot_state_joints_config=self.robot_state_joints_config,
            policy_joints_config=self.policy_joints_config,
            modality_configs=self.modality_configs,
            extra_state_np=self._build_extra_state(env, observation),
        )
        missing_state = (
            set(self.modality_configs["state"].modality_keys)
            - policy_observations["state"].keys()
        )
        assert (
            not missing_state
        ), f"Arena could not build required GR00T state groups: {sorted(missing_state)}"

        # 2. Call GR00T's own client.  On every replan after the first, N1.7 can use the
        # unexecuted tail of the previous absolute action horizon as an inpainting prior.  The
        # server processor converts this old plan back into the current state's relative/normalised
        # action frame before the model consumes it.
        inference_options: dict[str, int | float] = {
            "num_inference_timesteps": self.policy_config.denoising_steps
        }
        if (
            self._rtc_enabled
            and self._previous_policy_action is not None
            and self._rtc_valid_envs.all()
        ):
            policy_observations["action"] = {
                key: np.array(value, dtype=np.float32, copy=True)
                for key, value in self._previous_policy_action.items()
            }
            inference_options.update(
                {
                    "action_horizon": self.action_horizon,
                    "rtc_overlap_steps": self._rtc_overlap_steps,
                    "rtc_frozen_steps": self.policy_config.rtc_frozen_steps,
                    "rtc_ramp_rate": self.policy_config.rtc_ramp_rate,
                }
            )

        action_samples: list[dict[str, np.ndarray]] = []
        for _ in range(self._action_sample_count):
            action, info = self._client.get_action(
                policy_observations, options=inference_options
            )
            assert info.get("num_inference_timesteps") == self.policy_config.denoising_steps, (
                "The GR00T server did not acknowledge the requested denoising steps. Restart the "
                "server with the patched gr00t.policy.gr00t_policy implementation."
            )
            if "action" in policy_observations:
                assert (
                    info.get("rtc_applied") is True
                    and info.get("rtc_model_input_present") is True
                    and info.get("rtc_time_normalization_aligned") is True
                ), (
                    "The GR00T server did not acknowledge RTC. Restart the server with the patched "
                    "gr00t.policy.gr00t_policy implementation before running this policy."
                )
            action_samples.append(action)
        robot_action_policy = self._median_action_samples(action_samples)
        robot_action_policy = self._anchor_agibot_action_chunk_translation(
            robot_action_policy, policy_observations
        )
        robot_action_policy = self._hold_idle_agibot_arms(
            robot_action_policy, policy_observations
        )
        if self._rtc_enabled:
            expected_action_keys = set(self.modality_configs["action"].modality_keys)
            assert set(robot_action_policy) == expected_action_keys, (
                "GR00T returned action groups that do not match the checkpoint's RTC contract: "
                f"expected {sorted(expected_action_keys)}, got {sorted(robot_action_policy)}"
            )
            self._previous_policy_action = {
                key: np.array(value, dtype=np.float32, copy=True)
                for key, value in robot_action_policy.items()
            }
            for key, value in self._previous_policy_action.items():
                assert value.shape[:2] == (self.num_envs, self.action_horizon), (
                    f"RTC action '{key}' must have leading shape "
                    f"({self.num_envs}, {self.action_horizon}), got {value.shape}"
                )
            self._rtc_valid_envs[:] = True

        # 3. Action translation from policy output to sim action tensor
        action_tensor = build_gr00t_action_tensor(
            robot_action_policy=robot_action_policy,
            task_mode=self.task_mode,
            policy_joints_config=self.policy_joints_config,
            robot_action_joints_config=self.robot_action_joints_config,
            device=self.device,
            embodiment_tag=self.policy_config.embodiment_tag,
        )
        self._write_diagnostic_trace(
            env,
            policy_observations,
            robot_action_policy,
            action_samples,
            action_tensor,
            current_observation=current_observation,
        )

        assert (
            action_tensor.shape[0] == self.num_envs
            and action_tensor.shape[1] >= self.action_chunk_length
        )
        return action_tensor

    @staticmethod
    def _median_action_samples(
        action_samples: list[dict[str, np.ndarray]],
    ) -> dict[str, np.ndarray]:
        """Return an elementwise median while preserving single-sample outputs and array dtypes."""
        if len(action_samples) == 1:
            return action_samples[0]

        expected_keys = action_samples[0].keys()
        assert all(
            sample.keys() == expected_keys for sample in action_samples
        ), "Repeated GR00T action samples returned inconsistent keys"
        return {
            key: np.median(
                np.stack([sample[key] for sample in action_samples]), axis=0
            ).astype(action_samples[0][key].dtype, copy=False)
            for key in expected_keys
        }

    def _anchor_agibot_action_chunk_translation(
        self,
        robot_action_policy: dict[str, np.ndarray],
        policy_observations: dict[str, dict[str, np.ndarray]],
    ) -> dict[str, np.ndarray]:
        """Anchor a decoded AgiBot chunk to its observed initial XYZ without changing its trajectory shape."""
        alpha = self.policy_config.action_chunk_translation_anchor_alpha
        if alpha == 0.0 or self.task_mode != TaskMode.AGIBOT_BIMANUAL_MANIPULATION:
            return robot_action_policy

        anchored = {
            key: np.array(value, copy=True)
            for key, value in robot_action_policy.items()
        }
        for side in ("left", "right"):
            key = f"{side}_eef_9d"
            current_xyz = np.asarray(policy_observations["state"][key])[:, -1, :3]
            predicted_xyz = anchored[key][..., :3]
            first_frame_offset = predicted_xyz[:, 0] - current_xyz
            predicted_xyz -= alpha * first_frame_offset[:, None]
        return anchored

    def _hold_idle_agibot_arms(
        self,
        robot_action_policy: dict[str, np.ndarray],
        policy_observations: dict[str, dict[str, np.ndarray]],
    ) -> dict[str, np.ndarray]:
        """Optionally suppress small open-hand horizon drift on an otherwise idle AgiBot arm."""
        threshold = self._idle_arm_hold_threshold_m
        if threshold == 0.0 or self.task_mode != TaskMode.AGIBOT_BIMANUAL_MANIPULATION:
            return robot_action_policy

        filtered = {
            key: np.array(value, copy=True)
            for key, value in robot_action_policy.items()
        }
        for side in ("left", "right"):
            eef_key = f"{side}_eef_9d"
            hand_key = f"{side}_hand"
            current_eef = np.asarray(policy_observations["state"][eef_key])[:, 0]
            predicted_eef = filtered[eef_key]
            translation_distance = np.linalg.norm(
                predicted_eef[..., :3] - current_eef[:, None, :3], axis=-1
            )
            hand_stays_open = np.asarray(filtered[hand_key]).min(axis=(1, 2)) > 0.9
            idle = (translation_distance.max(axis=1) < threshold) & hand_stays_open
            if not idle.any():
                continue
            predicted_eef[idle] = current_eef[idle, None, :]
            filtered[hand_key][idle] = 0.994
            logger.info(
                "GR00T diagnostic idle-arm hold: side=%s envs=%s max_translation_m=%s",
                side,
                np.flatnonzero(idle).tolist(),
                np.round(translation_distance.max(axis=1)[idle], 4).tolist(),
            )
        return filtered

    def _write_diagnostic_trace(
        self,
        env: gym.Env,
        policy_observations: dict[str, Any],
        robot_action_policy: dict[str, Any],
        action_samples: list[dict[str, np.ndarray]],
        action_tensor: torch.Tensor,
        current_observation: dict[str, Any] | None = None,
    ) -> None:
        """Persist bounded inference I/O when explicitly enabled through the environment."""
        if self._trace_dir is None or self._trace_index >= self._trace_limit:
            return

        trace: dict[str, np.ndarray] = {
            "task_description": np.asarray(self.task_description or ""),
            "sim_action": action_tensor.detach().cpu().numpy(),
        }
        for key, value in policy_observations["video"].items():
            trace[f"video__{key}"] = np.asarray(value)
        for key, value in policy_observations["state"].items():
            trace[f"state__{key}"] = np.asarray(value)
        if current_observation is not None:
            current_policy = current_observation.get("policy", {})
            for key in (
                "robot_joint_pos",
                "left_eef_pos",
                "left_eef_quat",
                "right_eef_pos",
                "right_eef_quat",
                "left_hand_pos",
                "right_hand_pos",
            ):
                if key in current_policy:
                    trace[f"current_policy__{key}"] = np.asarray(
                        to_numpy(current_policy[key])
                    )
        for key, value in policy_observations.get("action", {}).items():
            trace[f"rtc_action__{key}"] = np.asarray(value)
        for key, value in robot_action_policy.items():
            trace[f"server_action__{key}"] = np.asarray(value)
        for sample_index, sample in enumerate(action_samples):
            for key, value in sample.items():
                trace[f"server_sample_{sample_index:03d}__{key}"] = np.asarray(value)

        # Object poses turn a visual rollout failure into a measurable grasp-geometry failure.
        # Keep this trace-only and generic: every rigid object in the compiled scene is small
        # compared with the camera/action arrays already stored above.
        scene = getattr(getattr(env, "unwrapped", env), "scene", None)
        for name, asset in getattr(scene, "rigid_objects", {}).items():
            root_pose = asset.data.root_pose_w.torch
            trace[f"scene__{name}__root_pose_w"] = root_pose.detach().cpu().numpy()

        path = self._trace_dir / f"inference_{self._trace_index:04d}.npz"
        np.savez_compressed(path, **trace)
        logger.info("Wrote GR00T diagnostic trace %s", path)
        self._trace_index += 1

    def reset(self, env_ids: torch.Tensor | None = None):
        full_batch_reset = env_ids is None
        if env_ids is None:
            env_ids = slice(None)
        else:
            reset_env_ids = np.sort(np.asarray(to_numpy(env_ids), dtype=np.int64))
            full_batch_reset = np.array_equal(
                reset_env_ids, np.arange(self.num_envs, dtype=np.int64)
            )
        assert self._client is not None, "GR00T remote policy has been closed"
        assert self._chunking_state is not None, "GR00T remote policy has been closed"
        # N1.7 samples the initial flow-matching noise from PyTorch's process-global RNG. The
        # upstream reset endpoint is otherwise stateless, so pass the configured seed whenever
        # the whole inference batch starts a new rollout. Without this, results depend on every
        # request previously served by this long-lived process.
        reset_options = {"seed": self.policy_config.seed} if full_batch_reset else None
        self._client.reset(options=reset_options)
        self._chunking_state.reset(env_ids)
        self._video_history.reset(
            env_ids if isinstance(env_ids, slice) else to_numpy(env_ids)
        )
        self._state_history.reset(
            env_ids if isinstance(env_ids, slice) else to_numpy(env_ids)
        )
        if isinstance(env_ids, slice):
            self._previous_policy_action = None
            self._rtc_valid_envs[:] = False
        else:
            self._rtc_valid_envs[to_numpy(env_ids)] = False
        self._warmup_steps_remaining = self._warmup_steps
        self._warmup_initial_policy_observation = None
        self._warmup_hold_action = None

    def close(self) -> None:
        """Release Arena-side resources for the remote GR00T policy client."""
        client = self._client
        try:
            if client is not None:
                socket = getattr(client, "socket", None)
                context = getattr(client, "context", None)
                try:
                    if socket is not None:
                        socket.close(linger=0)
                finally:
                    if context is not None:
                        context.term()
        finally:
            self._client = None
            self._chunking_state = None
            self.modality_configs = None
