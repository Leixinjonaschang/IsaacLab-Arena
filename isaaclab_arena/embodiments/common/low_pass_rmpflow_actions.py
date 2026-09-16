# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""RMPFlow actions with a physics-rate low-pass filter on joint targets."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.envs.mdp.actions.rmpflow_actions_cfg import RMPFlowActionCfg
from isaaclab.envs.mdp.actions.rmpflow_task_space_actions import RMPFlowAction
from isaaclab.managers.action_manager import ActionTerm
from isaaclab.utils.configclass import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class LowPassRMPFlowAction(RMPFlowAction):
    """Filter RMPFlow's joint-position output before sending it to stiff PD drives.

    Lula RMPFlow integrates an internal joint trajectory at every physics substep.  With a
    low-rate stream of aggressive absolute Cartesian targets, that trajectory can alternate
    direction between adjacent substeps even while the end effector looks accurate at the
    control rate.  Passing those targets directly to stiff drives creates saturated torque
    reversals that can shake a grasped object loose.

    This term applies an exponential moving average at the physics rate and sends a zero joint
    velocity target, matching the damping semantics of Arena's stable joint-position replay.
    The first target after reset is passed through unchanged so initialization does not lag.
    """

    cfg: LowPassRMPFlowActionCfg

    def __init__(self, cfg: LowPassRMPFlowActionCfg, env: ManagerBasedEnv) -> None:
        if not 0.0 < cfg.joint_target_ema_alpha <= 1.0:
            raise ValueError(
                "joint_target_ema_alpha must be in (0, 1], got "
                f"{cfg.joint_target_ema_alpha}"
            )
        super().__init__(cfg, env)
        self._filtered_joint_target = torch.zeros(
            (self.num_envs, self._num_joints), device=self.device
        )
        self._filter_initialized = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    def apply_actions(self) -> None:
        _, ee_quat_curr = self._compute_frame_pose()
        joint_pos = self._asset.data.joint_pos.torch[:, self._joint_ids]
        if ee_quat_curr.norm() != 0:
            raw_joint_target, _ = self._rmpflow_controller.compute(
                self._asset.data.joint_pos.torch, self._asset.data.joint_vel.torch
            )
        else:
            raw_joint_target = joint_pos.clone()

        filtered_target = torch.lerp(
            self._filtered_joint_target,
            raw_joint_target,
            self.cfg.joint_target_ema_alpha,
        )
        filtered_target = torch.where(
            self._filter_initialized.unsqueeze(-1),
            filtered_target,
            raw_joint_target,
        )
        self._filtered_joint_target.copy_(filtered_target)
        self._filter_initialized.fill_(True)

        self._asset.set_joint_position_target_index(
            target=self._filtered_joint_target, joint_ids=self._joint_ids
        )
        self._asset.set_joint_velocity_target_index(
            target=torch.zeros_like(self._filtered_joint_target), joint_ids=self._joint_ids
        )

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        super().reset(env_ids)
        if env_ids is None:
            self._filter_initialized.fill_(False)
        else:
            self._filter_initialized[env_ids] = False


@configclass
class LowPassRMPFlowActionCfg(RMPFlowActionCfg):
    """Configuration for physics-rate low-pass filtering of RMPFlow joint targets."""

    class_type: type[ActionTerm] = LowPassRMPFlowAction

    joint_target_ema_alpha: float = 0.1
    """EMA coefficient applied once per physics step; one disables filtering."""
