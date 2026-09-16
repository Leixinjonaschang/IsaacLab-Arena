# Copyright (c) 2026, The Isaac Lab Arena Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Branch-continuous differential IK evaluated once per environment control step."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

import isaaclab.utils.math as math_utils
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from isaaclab.envs.mdp.actions.task_space_actions import (
    DifferentialInverseKinematicsAction,
)
from isaaclab.managers.action_manager import ActionTerm
from isaaclab.utils.configclass import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class ControlRateDifferentialIKAction(DifferentialInverseKinematicsAction):
    """Convert an absolute EEF target to one bounded joint target per control step.

    Isaac Lab's stock differential-IK term recomputes a complete joint correction at every physics
    substep.  On AgiBot's stiff, zero-damping arm drives, even a nominally stationary Cartesian
    target then chases the reset transient through several redundant joint solutions.  This term
    instead computes once at policy rate and linearly walks to that fixed joint target over the
    physics substeps, matching the timing of the joint-position demonstrations.

    If the Cartesian command has not changed, the previous joint target is retained exactly.  That
    prevents physical tracking error or sub-millimetre frame-conversion residuals from being
    reinterpreted as repeated IK commands along a redundant joint direction.

    The first command after reset establishes a joint-space hold without running IK.  Reset events
    write joint state before FrameTransformer and articulation FK buffers necessarily agree; using
    that transient Cartesian snapshot can cache a completely wrong redundant-arm solution.  The
    normal AgiBot camera warmup repeats its ready-pose command, so the arm stays at the explicitly
    reset joint pose until the first genuinely different model command arrives.
    """

    cfg: ControlRateDifferentialIKActionCfg

    def __init__(
        self, cfg: ControlRateDifferentialIKActionCfg, env: ManagerBasedEnv
    ) -> None:
        if cfg.max_joint_delta_per_control_step <= 0.0:
            raise ValueError("max_joint_delta_per_control_step must be positive")
        if cfg.command_tolerance < 0.0:
            raise ValueError("command_tolerance must be non-negative")
        if cfg.nullspace_gain < 0.0:
            raise ValueError("nullspace_gain must be non-negative")
        if cfg.nullspace_pinv_rtol <= 0.0:
            raise ValueError("nullspace_pinv_rtol must be positive")
        if cfg.controller.command_type != "pose" or cfg.controller.use_relative_mode:
            raise ValueError(
                "ControlRateDifferentialIKAction requires absolute pose commands"
            )
        if cfg.controller.ik_method != "dls" or cfg.controller.ik_params is None:
            raise ValueError(
                "ControlRateDifferentialIKAction requires damped-least-squares IK"
            )
        super().__init__(cfg, env)
        if len(cfg.nullspace_joint_weights) != self._num_joints:
            raise ValueError(
                "nullspace_joint_weights must have one value per controlled joint; got "
                f"{len(cfg.nullspace_joint_weights)} for {self._num_joints} joints"
            )
        self._nullspace_joint_weights = torch.tensor(
            cfg.nullspace_joint_weights, device=self.device, dtype=torch.float32
        ).unsqueeze(0)
        self._substeps = max(1, int(getattr(env.cfg, "decimation", 1)))
        shape = (self.num_envs, self._num_joints)
        self._joint_start = torch.zeros(shape, device=self.device)
        self._joint_target = torch.zeros(shape, device=self.device)
        self._joint_reference = torch.zeros(shape, device=self.device)
        self._target_initialized = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self._last_command = torch.zeros_like(self.processed_actions)
        self._substep = 0

    def process_actions(self, actions: torch.Tensor) -> None:
        """Resolve one bounded joint target and freeze it for this control step."""
        super().process_actions(actions)
        ee_pos_curr, ee_quat_curr = self._compute_frame_pose()
        joint_pos = self._asset.data.joint_pos.torch[:, self._joint_ids]
        self._joint_start.copy_(joint_pos)

        uninitialized = ~self._target_initialized
        self._joint_reference.copy_(
            torch.where(
                uninitialized.unsqueeze(-1), joint_pos, self._joint_reference
            )
        )
        if uninitialized.all():
            self._joint_target.copy_(joint_pos)
            self._last_command.copy_(self.processed_actions)
            self._target_initialized.fill_(True)
            self._substep = 0
            return

        jacobian = self._compute_frame_jacobian()
        candidate = self._ik_controller.compute(
            ee_pos_curr, ee_quat_curr, jacobian, joint_pos
        )
        candidate += self._nullspace_delta(jacobian, joint_pos)
        delta = self._limit_joint_delta(candidate - joint_pos)
        candidate = joint_pos + delta

        command_changed = self._command_changed(
            self.processed_actions, self._last_command
        )
        recompute = self._target_initialized & command_changed
        self._joint_target.copy_(
            torch.where(
                uninitialized.unsqueeze(-1),
                joint_pos,
                torch.where(recompute.unsqueeze(-1), candidate, self._joint_target),
            )
        )
        self._last_command.copy_(self.processed_actions)
        self._target_initialized.fill_(True)
        self._substep = 0

    def apply_actions(self) -> None:
        """Interpolate to the fixed control-rate target over the physics substeps."""
        if not self._target_initialized.any():
            joint_pos = self._asset.data.joint_pos.torch[:, self._joint_ids]
            self._asset.set_joint_position_target_index(
                target=joint_pos, joint_ids=self._joint_ids
            )
            return
        self._substep = min(self._substep + 1, self._substeps)
        target = torch.lerp(
            self._joint_start, self._joint_target, self._substep / self._substeps
        )
        if not self._target_initialized.all():
            joint_pos = self._asset.data.joint_pos.torch[:, self._joint_ids]
            target = torch.where(
                self._target_initialized.unsqueeze(-1), target, joint_pos
            )
        self._asset.set_joint_position_target_index(
            target=target, joint_ids=self._joint_ids
        )
        self._asset.set_joint_velocity_target_index(
            target=torch.zeros_like(target), joint_ids=self._joint_ids
        )

    def _compute_frame_jacobian(self) -> torch.Tensor:
        """Return the target-frame Jacobian without rotating its angular velocity.

        The upstream action rotates the angular rows by ``body_offset.rot``.  Angular velocity is
        identical at rigidly attached frames when expressed in the same base coordinates, so that
        operation makes a rotational tool offset drive IK in the wrong direction.  Only a
        translational offset changes the geometric Jacobian's linear rows.
        """
        jacobian = self.jacobian_b.clone()
        if self.cfg.body_offset is None or self._offset_pos is None:
            return jacobian

        _, body_quat_b = math_utils.subtract_frame_transforms(
            self._asset.data.root_pos_w.torch,
            self._asset.data.root_quat_w.torch,
            self._asset.data.body_pos_w.torch[:, self._body_idx],
            self._asset.data.body_quat_w.torch[:, self._body_idx],
        )
        offset_pos_b = math_utils.quat_apply(body_quat_b, self._offset_pos)
        jacobian[:, :3, :] += torch.bmm(
            -math_utils.skew_symmetric_matrix(offset_pos_b), jacobian[:, 3:, :]
        )
        return jacobian

    def _limit_joint_delta(self, delta: torch.Tensor) -> torch.Tensor:
        """Bound each joint correction without throttling the other joints."""
        limit = self.cfg.max_joint_delta_per_control_step
        return torch.clamp(delta, min=-limit, max=limit)

    def _nullspace_delta(
        self, jacobian: torch.Tensor, joint_pos: torch.Tensor
    ) -> torch.Tensor:
        """Bias the redundant DoF toward the reset branch without changing the EEF motion."""
        if self.cfg.nullspace_gain == 0.0:
            return torch.zeros_like(joint_pos)
        jacobian_pinv = torch.linalg.pinv(
            jacobian, rtol=self.cfg.nullspace_pinv_rtol
        )
        identity = torch.eye(
            self._num_joints, device=jacobian.device, dtype=jacobian.dtype
        ).unsqueeze(0)
        nullspace_projector = identity - jacobian_pinv @ jacobian
        return self.cfg.nullspace_gain * (
            nullspace_projector
            @ (
                (self._joint_reference - joint_pos)
                * self._nullspace_joint_weights
            ).unsqueeze(-1)
        ).squeeze(-1)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        super().reset(env_ids)
        if env_ids is None:
            self._target_initialized.fill_(False)
        else:
            self._target_initialized[env_ids] = False
        self._substep = 0

    def _command_changed(
        self, current: torch.Tensor, previous: torch.Tensor
    ) -> torch.Tensor:
        """Compare poses while treating quaternion ``q`` and ``-q`` as equivalent."""
        position_changed = (
            torch.amax(torch.abs(current[:, :3] - previous[:, :3]), dim=1)
            > self.cfg.command_tolerance
        )
        current_quat = torch.nn.functional.normalize(current[:, 3:7], dim=1)
        previous_quat = torch.nn.functional.normalize(previous[:, 3:7], dim=1)
        quaternion_changed = (
            1.0 - torch.abs(torch.sum(current_quat * previous_quat, dim=1))
            > self.cfg.command_tolerance
        )
        return position_changed | quaternion_changed


@configclass
class ControlRateDifferentialIKActionCfg(DifferentialInverseKinematicsActionCfg):
    """Configuration for control-rate, rate-limited differential IK."""

    class_type: type[ActionTerm] = ControlRateDifferentialIKAction

    max_joint_delta_per_control_step: float = 0.1
    """Largest absolute change, in radians, allowed for one joint in one control step."""

    command_tolerance: float = 1.0e-6
    """Tolerance used to decide whether an absolute Cartesian command changed."""

    nullspace_gain: float = 0.2
    """Per-control-step gain that keeps the redundant arm near its episode reset branch."""

    nullspace_joint_weights: tuple[float, ...] = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)
    """Joint-wise posture cost around the joint pose captured on the first command after reset."""

    nullspace_pinv_rtol: float = 1.0e-4
    """Relative singular-value tolerance for the true null-space projector."""
