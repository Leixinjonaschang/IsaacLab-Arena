# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Give an idle RMPFlow arm a target to hold, so it stops walking away from its reset pose.

``RMPFlowAction`` in relative mode rebuilds its target every step as *current pose + delta*, so a
zero delta means "hold wherever you are **now**", not "hold where you were". There is no restoring
force, and any displacement — a reset transient, a contact impulse — is written into the target and
never given back.

On the Agibot that is not a small effect. In dual-arm mode the arm nobody is driving leaves its
reset pose within ~10 control steps (0.67 s at 15 Hz) and never returns; the right hand exits a
head-mounted viewport entirely. Measured displacement of the right end-effector over 800
zero-command steps:

    single-arm (the idle arm has no action term at all)   0.2 mm
    dual-arm, stock behaviour                             88 mm within 10 steps, settling ~64 mm
    dual-arm, with this hold                              0.2 mm

The underlying transient is upstream and unexplained: with the target equal to the current pose,
``default_q`` equal to the current joints, joint velocities exactly zero, the nearest joint limit
1.07 rad away and lula's FK matching the sim to 0.0 mm / 0.00°, the first evaluation after a reset
still moves a joint 0.4660 rad (~7 rad/s, against the RMPFlow config's own
``joint_velocity_cap_rmp.max_velocity: 3.14``). Ruled out as causes:
``ignore_robot_state_updates`` (turning it *off* diverges to 890 mm), the self-collision
``repulsion_gain`` (zeroing it changes nothing digit-for-digit), and a lula/USD kinematic mismatch.
Holding the target does not fix that jump — it stops it from being latched in forever.

This does **not** guard against contact. An earlier version of this module also clamped the
commanded descent at the work surface; it was removed because the table already stops the hand
physically (3.7 mm of penetration unguarded), and bounding the action space is not something a
policy evaluation can be asked to respect. What the clamp did buy, if it is ever wanted back, was
lower contact impulse while pressing: finger speed 1.036 m/s commanded 110 mm below the table
against 0.567 m/s clamped.
"""

from __future__ import annotations

import torch
from collections.abc import Sequence
from dataclasses import fields

from isaaclab.envs import ManagerBasedEnv
from isaaclab.envs.mdp.actions.rmpflow_actions_cfg import RMPFlowActionCfg
from isaaclab.envs.mdp.actions.rmpflow_task_space_actions import RMPFlowAction
from isaaclab.utils.configclass import configclass


class TargetHoldingRMPFlowAction(RMPFlowAction):
    """An RMPFlow arm term that holds its previous target while commanded exactly zero."""

    cfg: TargetHoldingRMPFlowActionCfg

    def __init__(self, cfg: TargetHoldingRMPFlowActionCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        assert self.cfg.use_relative_mode, "TargetHoldingRMPFlowAction only makes sense in relative mode"
        self._held_pos: torch.Tensor | None = None
        self._held_quat: torch.Tensor | None = None
        print(f"[arm target hold] {self._body_name}: enabled={cfg.hold_on_zero_command}")

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        """Drop the held target so the first command after a reset re-reads the robot."""
        super().reset(env_ids)
        self._held_pos = None
        self._held_quat = None

    def process_actions(self, actions: torch.Tensor):
        """Run the usual RMPFlow processing, then re-issue the held target where idle.

        Args:
            actions: This term's slice of the action vector, a delta pose of shape (num_envs, 6).
        """
        super().process_actions(actions)
        if self.cfg.hold_on_zero_command:
            self._hold_target_where_idle(actions)

    def _hold_target_where_idle(self, actions: torch.Tensor) -> None:
        """Replace the freshly computed target with the previous one, per idle environment."""
        idle = (actions.abs() < 1e-9).all(dim=1)
        if self._held_pos is None:
            self._held_pos = self.ee_pos_des.clone()
            self._held_quat = self.ee_quat_des.clone()
            return
        if idle.any():
            self.ee_pos_des[idle] = self._held_pos[idle]
            self.ee_quat_des[idle] = self._held_quat[idle]
            self.ee_pose_des = torch.cat([self.ee_pos_des, self.ee_quat_des], dim=1)
            self._rmpflow_controller.set_command(self.ee_pose_des)
        # A commanded arm re-latches onto wherever it was just told to go.
        moving = ~idle
        if moving.any():
            self._held_pos[moving] = self.ee_pos_des[moving]
            self._held_quat[moving] = self.ee_quat_des[moving]


@configclass
class TargetHoldingRMPFlowActionCfg(RMPFlowActionCfg):
    """Configuration for an RMPFlow arm term that holds its target while idle."""

    class_type: type[RMPFlowAction] = TargetHoldingRMPFlowAction

    hold_on_zero_command: bool = True
    """Whether an arm commanded exactly zero holds its previous target instead of its current pose."""


def install_arm_target_hold(env_cfg) -> None:
    """Swap every relative-mode RMPFlow arm term in ``env_cfg`` for the target-holding subclass.

    Intended as an ``env_cfg_callback``, so the behaviour is part of the compiled config and
    applies to whatever drives the environment — teleoperation, dataset replay, policy evaluation.

    Args:
        env_cfg: The compiled environment configuration, patched in place.
    """
    held_terms = []
    for term_name, term_cfg in vars(env_cfg.actions).items():
        if not isinstance(term_cfg, RMPFlowActionCfg) or isinstance(term_cfg, TargetHoldingRMPFlowActionCfg):
            continue
        if not term_cfg.use_relative_mode:
            continue
        held = TargetHoldingRMPFlowActionCfg(**{f.name: getattr(term_cfg, f.name) for f in fields(term_cfg)})
        held.class_type = TargetHoldingRMPFlowAction
        setattr(env_cfg.actions, term_name, held)
        held_terms.append(term_name)

    assert held_terms, "install_arm_target_hold found no relative-mode RMPFlow action terms"
