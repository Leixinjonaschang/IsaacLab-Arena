# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Diagnostic GR00T policy that substitutes demonstration video for live camera input."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import torch

from isaaclab_arena.assets.register import register_policy
from isaaclab_arena.policy.policy_base import PolicyBase
from isaaclab_arena_gr00t.policy.gr00t_remote_closedloop_policy import (
    ActionSchedulerType,
    Gr00tRemoteClosedloopPolicy,
    Gr00tRemoteClosedloopPolicyCfg,
)


@dataclass
class AgibotDatasetVideoClosedloopProbePolicyCfg(Gr00tRemoteClosedloopPolicyCfg):
    """Configure a closed-loop probe whose camera tensors come from a dataset episode."""

    dataset_video_root: str = ""
    """Directory containing ``observation.images.<modality>/episode_XXXXXX.mp4`` files."""

    dataset_episode_index: int = 0
    """Episode whose three camera streams are used."""

    dataset_start_row: int = 0
    """Dataset frame paired with the first simulator control step."""

    dataset_modalities: list[str] = field(
        default_factory=lambda: ["ego_view", "left_wrist_view", "right_wrist_view"]
    )
    """Video modalities replaced by dataset frames; unlisted cameras remain live."""


@register_policy
class AgibotDatasetVideoClosedloopProbePolicy(
    PolicyBase[AgibotDatasetVideoClosedloopProbePolicyCfg]
):
    """Keep state and execution live while replacing only the three camera streams."""

    name = "agibot_dataset_video_closedloop_probe"

    def __init__(self, config: AgibotDatasetVideoClosedloopProbePolicyCfg):
        PolicyBase.__init__(self, config)
        assert config.num_envs == 1, "The dataset-video probe only supports one environment"
        assert config.dataset_start_row >= 0
        self._dataset_video_config = config
        self._dataset_frame_index = 0
        self._dataset_captures: dict[str, cv2.VideoCapture] = {}
        self._probe = Gr00tRemoteClosedloopPolicy(
            Gr00tRemoteClosedloopPolicyCfg(
                policy_config_yaml_path=config.policy_config_yaml_path,
                policy_device=config.policy_device,
                num_envs=config.num_envs,
                remote_host=config.remote_host,
                remote_port=config.remote_port,
                remote_api_token=config.remote_api_token,
                scheduler=ActionSchedulerType(config.scheduler),
            )
        )
        self._camera_names = self._probe.policy_config.pov_cam_name_sim
        self._modality_keys = self._probe.modality_configs["video"].modality_keys
        selected_modalities = set(config.dataset_modalities)
        unknown_modalities = selected_modalities - set(self._modality_keys)
        assert selected_modalities, "dataset_modalities must not be empty"
        assert not unknown_modalities, (
            f"Unknown dataset modalities {sorted(unknown_modalities)}; "
            f"expected a subset of {self._modality_keys}"
        )
        video_root = Path(config.dataset_video_root)
        assert video_root.is_dir(), f"Dataset video root does not exist: {video_root}"
        for modality_key in self._modality_keys:
            if modality_key not in selected_modalities:
                continue
            path = (
                video_root
                / f"observation.images.{modality_key}"
                / f"episode_{config.dataset_episode_index:06d}.mp4"
            )
            assert path.is_file(), f"Dataset camera video does not exist: {path}"
            capture = cv2.VideoCapture(str(path))
            assert capture.isOpened(), f"Failed to open dataset camera video: {path}"
            capture.set(cv2.CAP_PROP_POS_FRAMES, config.dataset_start_row)
            self._dataset_captures[modality_key] = capture
        assert len(self._camera_names) == len(self._modality_keys)

    def set_task_description(self, task_description: str | None) -> str:
        description = self._probe.set_task_description(task_description)
        self.task_description = description
        return description

    def get_action(self, env: Any, observation: dict[str, Any]) -> torch.Tensor:
        camera_observation = dict(observation["camera_obs"])
        expected_row = (
            self._dataset_video_config.dataset_start_row + self._dataset_frame_index
        )
        for camera_name, modality_key in zip(
            self._camera_names, self._modality_keys, strict=True
        ):
            capture = self._dataset_captures.get(modality_key)
            if capture is None:
                continue
            ok, bgr = capture.read()
            if not ok:
                raise RuntimeError(f"Failed to decode dataset camera frame {expected_row}")
            live = camera_observation[camera_name]
            frame = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            replacement = torch.as_tensor(frame, dtype=live.dtype, device=live.device)
            if live.ndim == 4:
                replacement = replacement.unsqueeze(0)
            assert replacement.shape == live.shape, (
                f"Dataset frame for {camera_name} has shape {replacement.shape}, "
                f"but live observation has shape {live.shape}"
            )
            camera_observation[camera_name] = replacement
        self._dataset_frame_index += 1
        substituted_observation = dict(observation)
        substituted_observation["camera_obs"] = camera_observation
        return self._probe.get_action(env, substituted_observation)

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        self._probe.reset(env_ids)
        if env_ids is None or len(env_ids) == self.config.num_envs:
            self._dataset_frame_index = 0
            for capture in self._dataset_captures.values():
                capture.set(
                    cv2.CAP_PROP_POS_FRAMES,
                    self._dataset_video_config.dataset_start_row,
                )

    def close(self) -> None:
        for capture in self._dataset_captures.values():
            capture.release()
        self._dataset_captures.clear()
        self._probe.close()
