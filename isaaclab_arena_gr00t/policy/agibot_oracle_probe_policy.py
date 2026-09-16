# Copyright (c) 2026, The Isaac Lab Arena Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Replay AgiBot demonstration actions while probing a remote GR00T policy.

This diagnostic policy deliberately ignores the remote policy's actions.  The
simulator follows the recorded demonstration, while the wrapped production
policy records the predictions it would have made from the live observations.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import pandas as pd
import torch

from isaaclab_arena.assets.register import register_policy
from isaaclab_arena.policy.policy_base import PolicyBase, PolicyCfg
from isaaclab_arena_gr00t.policy.config.gr00t_closedloop_policy_config import (
    Gr00tClosedloopPolicyCfg,
    TaskMode,
)
from isaaclab_arena_gr00t.policy.gr00t_core import (
    build_gr00t_action_tensor,
    load_gr00t_joint_configs,
)
from isaaclab_arena_gr00t.policy.gr00t_remote_closedloop_policy import (
    Gr00tRemoteClosedloopPolicy,
    Gr00tRemoteClosedloopPolicyCfg,
)
from isaaclab_arena_gr00t.utils.io_utils import create_config_from_yaml


@dataclass
class AgibotOracleProbePolicyCfg(PolicyCfg):
    """Configuration for an oracle rollout with non-controlling model probes."""

    parquet_path: str
    policy_config_yaml_path: str
    remote_host: str = "127.0.0.1"
    remote_port: int = 5557
    num_steps: int = 300
    policy_device: str = "cuda:0"
    execution_mode: str = "eef"
    query_remote: bool = True
    object_trace_path: str | None = None
    state_trace_path: str | None = None

    def __post_init__(self) -> None:
        assert self.num_steps > 0
        assert self.execution_mode in ("eef", "joint")
        assert Path(
            self.parquet_path
        ).is_file(), f"Dataset parquet does not exist: {self.parquet_path}"
        assert Path(
            self.policy_config_yaml_path
        ).is_file(), f"Policy config does not exist: {self.policy_config_yaml_path}"
        assert (
            self.state_trace_path is None
            or Path(self.state_trace_path).suffix == ".npz"
        ), "state_trace_path must use the .npz extension"


@register_policy
class AgibotOracleProbePolicy(PolicyBase[AgibotOracleProbePolicyCfg]):
    """Execute recorded actions and query GR00T on the resulting live observations."""

    name = "agibot_oracle_probe"

    def __init__(self, config: AgibotOracleProbePolicyCfg):
        super().__init__(config)
        dataframe = pd.read_parquet(config.parquet_path)
        assert config.num_steps <= len(dataframe)

        eef_actions = np.stack(dataframe["action.eef_9d"].to_numpy()).astype(
            np.float32
        )[: config.num_steps]
        joint_actions = np.stack(dataframe["action"].to_numpy()).astype(np.float32)[
            : config.num_steps
        ]
        translation_config: Gr00tClosedloopPolicyCfg = create_config_from_yaml(
            config.policy_config_yaml_path, Gr00tClosedloopPolicyCfg
        )
        policy_joints, robot_action_joints, _ = load_gr00t_joint_configs(
            translation_config
        )
        if config.execution_mode == "joint":
            self._oracle_actions = torch.as_tensor(
                joint_actions[None], dtype=torch.float32, device=config.policy_device
            )
        else:
            self._oracle_actions = build_gr00t_action_tensor(
                robot_action_policy={
                    "left_eef_9d": eef_actions[None, :, :9],
                    "right_eef_9d": eef_actions[None, :, 9:18],
                    "left_hand": joint_actions[None, :, 7:10],
                    "right_hand": joint_actions[None, :, 17:20],
                },
                task_mode=TaskMode.AGIBOT_BIMANUAL_MANIPULATION,
                policy_joints_config=policy_joints,
                robot_action_joints_config=robot_action_joints,
                device=config.policy_device,
                embodiment_tag=translation_config.embodiment_tag,
            )
        assert self._oracle_actions.shape == (1, config.num_steps, 20)

        self._probe = (
            Gr00tRemoteClosedloopPolicy(
                Gr00tRemoteClosedloopPolicyCfg(
                    policy_config_yaml_path=config.policy_config_yaml_path,
                    policy_device=config.policy_device,
                    remote_host=config.remote_host,
                    remote_port=config.remote_port,
                    num_envs=1,
                )
            )
            if config.query_remote
            else None
        )
        self._index = 0
        self._object_trace: list[dict[str, float | int]] = []
        self._state_trace: dict[str, list[np.ndarray]] = {}

    def set_task_description(self, task_description: str | None) -> str:
        description = (
            self._probe.set_task_description(task_description)
            if self._probe is not None
            else task_description or "Stack the bowls together."
        )
        self.task_description = description
        return description

    def get_action(self, env: gym.Env, observation: dict[str, Any]) -> torch.Tensor:
        if self.config.state_trace_path is not None:
            policy_observation = observation["policy"]
            for key in (
                "robot_joint_pos",
                "left_eef_pos",
                "left_eef_quat",
                "right_eef_pos",
                "right_eef_quat",
                "left_hand_pos",
                "right_hand_pos",
            ):
                value = policy_observation.get(key)
                assert (
                    value is not None
                ), f"Oracle state trace requires observation term '{key}'"
                array = (
                    value.detach().cpu().numpy()
                    if isinstance(value, torch.Tensor)
                    else np.asarray(value)
                )
                assert (
                    array.shape[0] == 1
                ), f"Oracle state trace only supports one env, got {array.shape}"
                self._state_trace.setdefault(key, []).append(
                    np.array(array[0], copy=True)
                )

        if self.config.object_trace_path is not None:
            row: dict[str, float | int] = {"step": self._index}
            for bowl_index in range(3):
                position = (
                    env.unwrapped.scene[f"bowl{bowl_index}"]
                    .data.root_pos_w.torch[0]
                    .detach()
                    .cpu()
                    .numpy()
                )
                for axis, value in zip("xyz", position, strict=True):
                    row[f"bowl{bowl_index}_{axis}"] = float(value)
            self._object_trace.append(row)

        # Update every production input buffer and issue normal server requests, but
        # never let the returned model action affect this oracle-controlled rollout.
        if self._probe is not None:
            self._probe.get_action(env, observation)
        action_index = min(self._index, self._oracle_actions.shape[1] - 1)
        action = self._oracle_actions[:, action_index]
        self._index += 1
        return action

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        self._index = 0
        if self._probe is not None:
            self._probe.reset(env_ids)

    def has_length(self) -> bool:
        return True

    def length(self) -> int:
        return self.config.num_steps

    def close(self) -> None:
        if self.config.state_trace_path is not None and self._state_trace:
            output_path = Path(self.config.state_trace_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                output_path,
                step=np.arange(
                    len(next(iter(self._state_trace.values()))), dtype=np.int32
                ),
                **{key: np.stack(values) for key, values in self._state_trace.items()},
            )
        if self.config.object_trace_path is not None and self._object_trace:
            output_path = Path(self.config.object_trace_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(self._object_trace).to_csv(output_path, index=False)
        if self._probe is not None:
            self._probe.close()
