# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""A joint position action applied as a first-order hold across the control step."""

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.envs.mdp.actions.actions_cfg import JointPositionActionCfg
from isaaclab.envs.mdp.actions.joint_actions import JointPositionAction
from isaaclab.managers.action_manager import ActionTerm
from isaaclab.utils.configclass import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class SmoothJointPositionAction(JointPositionAction):
    """Joint position targets walked linearly across the control step's physics substeps.

    A plain ``JointPositionAction`` holds each commanded target for the whole control step (a
    zero-order hold), so a stiff PD arm is asked for the entire step's motion at the first
    physics substep. At low control rates that per-step jolt is mechanically visible: at Arena's
    15 Hz it repeatedly worked a pinched object out of a gripper mid-carry. RMPFlow terms do not
    have this problem because they recompute their joint targets every substep; this term gives
    the same smoothness to direct joint commands by interpolating from the joints' measured
    positions at command time to the commanded target, reaching it on the last substep.

    Linear interpolation, on measured evidence: cubic smoothstep was tried against the same
    A/B (a rim-held bowl carried by the left arm) on the theory that linear's velocity jump at
    each control-step boundary was rattling the load, and made it *worse* -- carry drift 124 mm
    against linear's 97 mm and direct per-physics-step targets' 13 mm. The remaining gap to
    direct targets is not closed by within-step shaping alone; see the task notes before
    changing this again.
    """

    def __init__(self, cfg: SmoothJointPositionActionCfg, env: ManagerBasedEnv) -> None:
        super().__init__(cfg, env)
        self._substeps = max(1, int(getattr(env.cfg, "decimation", 1)))
        self._substep = 0
        self._start: torch.Tensor | None = None

    def process_actions(self, actions: torch.Tensor) -> None:
        super().process_actions(actions)
        self._start = self._asset.data.joint_pos.torch[:, self._joint_ids].clone()
        self._substep = 0

    def apply_actions(self) -> None:
        if self._start is None:
            super().apply_actions()
            return
        self._substep = min(self._substep + 1, self._substeps)
        target = torch.lerp(self._start, self.processed_actions, self._substep / self._substeps)
        self._asset.set_joint_position_target_index(target=target, joint_ids=self._joint_ids)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        super().reset(env_ids)
        self._start = None


@configclass
class SmoothJointPositionActionCfg(JointPositionActionCfg):
    """Configuration for the first-order-hold joint position action term."""

    class_type: type[ActionTerm] = SmoothJointPositionAction
