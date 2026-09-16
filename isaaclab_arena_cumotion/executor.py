# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Run cuMotion trajectories on an Arena environment, and record what happened.

The environment's own action terms are bypassed: joint position targets are written straight to
the articulation and the simulation is stepped directly, so RMPFlow never runs and never fights
the planned trajectory.

Three details here are not cosmetic, each having been arrived at from a specific failure:

* **Trajectories are played back slower than time-optimal.** The arms are servoed with very high
  stiffness and zero damping; at full speed the arm lags far enough behind its setpoint that the
  gripper closes before it has arrived.
* **Gripper commands are ramped.** Stepping the target straight to closed drives the closed-loop
  finger linkage hard enough that the reaction throws the whole arm off its held pose.
* **Retreats retrace the approach in reverse** rather than being re-planned. Once an object is in
  the fingers the collision world is stale, and planning from a just-disturbed state is what fails.
"""

from __future__ import annotations

import numpy as np
import torch
from collections.abc import Callable
from typing import TYPE_CHECKING

import warp as wp

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

    from isaaclab_arena_cumotion.planner import CumotionArmPlanner

DEFAULT_TRAJECTORY_SPEED = 0.35
"""Fraction of cuMotion's time-optimal speed. Above ~0.5 the stiff arm cannot keep up."""

DEFAULT_GRIPPER_RAMP_SECONDS = 200 / 120
"""Time a gripper command is ramped over, rather than stepped. Stated in seconds so the ramp is
the same wall-clock impulse at any control or physics rate; the value is the original 200 steps
at the 1/120 s physics dt the grasps were tuned at."""

DEFAULT_SETTLE_SECONDS = 40 / 120
"""Time a followed path holds its final configuration for; 40 steps at the original 1/120 s dt."""


class ArmExecutor:
    """Drives one arm along planned paths and holds the simulation loop.

    Args:
        env: The unwrapped Isaac Lab environment.
        planner: Planner whose joint indices and trajectory generator are reused.
        on_step: Called after every simulation step, e.g. to grab a camera frame.
    """

    def __init__(
        self,
        env: ManagerBasedEnv,
        planner: CumotionArmPlanner,
        on_step: Callable[[], None] | None = None,
    ) -> None:
        self.env = env
        self.planner = planner
        self.robot = planner.robot
        self.dt = env.sim.get_physics_dt()
        self.on_step = on_step
        self._gripper_target = planner.cfg.gripper_open_pos
        self.gripper_ramp_steps = max(1, round(DEFAULT_GRIPPER_RAMP_SECONDS / self.dt))
        """Default steps a gripper ramp is spread over, one per ``step`` call."""
        self.settle_steps = max(1, round(DEFAULT_SETTLE_SECONDS / self.dt))
        """Default steps a followed path holds its final configuration for."""

    def step(self, arm_target: np.ndarray | None = None, gripper_target: float | None = None, steps: int = 1) -> None:
        """Write joint targets and advance the simulation.

        Args:
            arm_target: Arm joint targets; the previous ones are held when omitted.
            gripper_target: Gripper joint target; the previous one is held when omitted.
            steps: How many physics steps to take.
        """
        if gripper_target is not None:
            self._gripper_target = gripper_target
        for _ in range(steps):
            if arm_target is not None:
                self.robot.set_joint_position_target(
                    torch.as_tensor(arm_target, dtype=torch.float32, device=self.env.device).unsqueeze(0),
                    joint_ids=self.planner.arm_joint_ids,
                )
            self.robot.set_joint_position_target(
                torch.full(
                    (1, len(self.planner.gripper_joint_ids)), float(self._gripper_target), device=self.env.device
                ),
                joint_ids=self.planner.gripper_joint_ids,
            )
            self.env.scene.write_data_to_sim()
            self.env.sim.step(render=True)
            self.env.scene.update(self.dt)
            if self.on_step is not None:
                self.on_step()

    def follow(
        self,
        path,
        reverse: bool = False,
        speed: float = DEFAULT_TRAJECTORY_SPEED,
        settle_steps: int | None = None,
    ) -> np.ndarray:
        """Time-parameterise a planned path and servo the arm along it.

        Args:
            path: A cuMotion ``Path``, as returned by the planner.
            reverse: Retrace the path from its end to its start.
            speed: Fraction of the time-optimal speed to play it back at.
            settle_steps: Extra steps holding the final configuration; the executor's default
                when omitted.

        Returns:
            The final commanded arm configuration.
        """
        if settle_steps is None:
            settle_steps = self.settle_steps
        waypoints = path.get_waypoints().numpy().astype(np.float64)
        if reverse:
            waypoints = waypoints[::-1].copy()
        trajectory = self.planner.trajectory_generator.generate_trajectory_from_cspace_waypoints(waypoints)
        assert trajectory is not None, "cuMotion could not time-parameterise the planned path"

        num_steps = max(2, int(np.ceil(trajectory.duration / (self.dt * speed))))
        q = waypoints[0]
        for i in range(num_steps):
            state = trajectory.get_target_state(trajectory.duration * i / (num_steps - 1))
            if state is not None:
                q = state.joints.positions.numpy()
            self.step(arm_target=q)
        self.step(arm_target=q, steps=settle_steps)
        return np.asarray(q, dtype=np.float64)

    def set_gripper(self, target: float, hold_arm_at: np.ndarray | None = None, ramp_steps: int | None = None) -> None:
        """Ramp the gripper to a target, holding the arm where it is.

        Args:
            target: Gripper joint target.
            hold_arm_at: Arm configuration to hold; the current one is used when omitted.
            ramp_steps: Steps over which to ramp; the executor's default when omitted.
        """
        if ramp_steps is None:
            ramp_steps = self.gripper_ramp_steps
        hold = self.planner.joint_positions() if hold_arm_at is None else hold_arm_at
        start = self._gripper_target
        for i in range(ramp_steps):
            alpha = (i + 1) / ramp_steps
            self.step(arm_target=hold, gripper_target=start + alpha * (target - start))

    def open_gripper(self, hold_arm_at: np.ndarray | None = None, ramp_steps: int | None = None) -> None:
        """Ramp the gripper open."""
        self.set_gripper(self.planner.cfg.gripper_open_pos, hold_arm_at, ramp_steps)

    def close_gripper(self, hold_arm_at: np.ndarray | None = None, ramp_steps: int | None = None) -> None:
        """Ramp the gripper closed."""
        self.set_gripper(self.planner.cfg.gripper_closed_pos, hold_arm_at, ramp_steps)


class JointActionInterface:
    """One full joint-space action vector, shared by every executor driving the same env.

    ``env.step`` consumes actions for the whole robot at once, so two arms cannot each call it
    with only their own targets: whichever called last would zero the other. This holds the
    complete vector -- initialised from the measured joint positions, so the first step holds the
    robot where it stands -- and each executor writes only its own slices before stepping.

    Requires the environment's action terms to all be joint-position terms (e.g.
    ``AgibotDualArmJointActionsCfg``); RMPFlow terms would re-solve motions cuMotion already
    planned.
    """

    def __init__(self, env) -> None:
        assert env.num_envs == 1, "JointActionInterface drives a single environment"
        self.env = env
        manager = env.action_manager
        self.action = torch.zeros((1, manager.total_action_dim), device=env.device)
        self._slices: dict[str, slice] = {}
        self._joint_names: dict[str, list[str]] = {}
        self._joint_ids: dict[str, list[int]] = {}
        offset = 0
        for name, dim in zip(manager.active_terms, manager.action_term_dim):
            term = manager.get_term(name)
            joint_names = getattr(term, "_joint_names", None)
            joint_ids = getattr(term, "_joint_ids", None)
            assert joint_names is not None, f"action term '{name}' is not a joint term"
            self._slices[name] = slice(offset, offset + dim)
            self._joint_names[name] = list(joint_names)
            robot = env.scene.articulations[term.cfg.asset_name]
            if isinstance(joint_ids, slice):
                joint_ids = list(range(robot.num_joints))[joint_ids]
            self._joint_ids[name] = list(joint_ids)
            offset += dim
        self.sync_from_robot()

    def joint_names(self, term_name: str) -> list[str]:
        """The joint names one term drives, in the order its action slice expects."""
        return self._joint_names[term_name]

    def sync_from_robot(self) -> None:
        """Reset every term's targets to the measured joint positions, e.g. after an env reset."""
        robot = self.env.scene.articulations["robot"]
        joint_pos = wp.to_torch(robot.data.joint_pos)[0]
        for name, term_slice in self._slices.items():
            self.action[0, term_slice] = joint_pos[self._joint_ids[name]]

    def set(self, term_name: str, values: np.ndarray) -> None:
        """Write one term's targets, given in the term's own joint order."""
        term_slice = self._slices[term_name]
        self.action[0, term_slice] = torch.as_tensor(values, dtype=torch.float32, device=self.env.device)

    def set_scalar(self, term_name: str, value: float) -> None:
        """Write the same target to every joint of one term, e.g. a gripper."""
        self.action[0, self._slices[term_name]] = float(value)

    def step(self) -> None:
        """Advance the environment one control step with the current action vector.

        An episode must not end mid-demonstration: a termination here means the env has already
        auto-reset and whatever was being recorded is gone, so it is an error, not a result.
        """
        _, _, terminated, truncated, _ = self.env.step(self.action)
        if bool(terminated[0]) or bool(truncated[0]):
            # The env has already auto-reset, so whatever was being demonstrated is gone. Raised
            # rather than asserted so a driver can fail one demonstration and carry on.
            raise RuntimeError(
                "the episode terminated mid-demonstration -- check episode_length_s and the termination terms"
            )


class EnvActionExecutor(ArmExecutor):
    """An ArmExecutor that drives its arm through ``env.step`` instead of writing to the sim.

    Same interface as ``ArmExecutor``, so ``PickAndPlace`` and scripts run unchanged; the
    difference is that every step goes through the action manager, which is what lets Isaac
    Lab's recorder hooks (HDF5 demo recording) see the actions and observations. One ``step``
    here is one *control* step -- ``decimation`` physics steps -- so the pacing attributes are
    rescaled to keep trajectory playback and gripper ramps at the same wall-clock rate.
    """

    def __init__(
        self,
        env,
        planner: CumotionArmPlanner,
        interface: JointActionInterface,
        arm_term: str,
        gripper_term: str,
        on_step: Callable[[], None] | None = None,
    ) -> None:
        """
        Args:
            env: The unwrapped Isaac Lab environment.
            planner: Planner whose joint order and trajectory generator are reused.
            interface: Shared action vector; each arm's executor writes its own slices.
            arm_term: Action-term name of this arm, e.g. ``"right_arm_action"``.
            gripper_term: Action-term name of this arm's gripper.
            on_step: Called after every control step, e.g. to grab a camera frame.
        """
        super().__init__(env, planner, on_step)
        self.interface = interface
        # One step here is one control step; re-derive the second-based pacing at that rate.
        self.dt = float(env.step_dt)
        self.gripper_ramp_steps = max(1, round(DEFAULT_GRIPPER_RAMP_SECONDS / self.dt))
        self.settle_steps = max(1, round(DEFAULT_SETTLE_SECONDS / self.dt))
        self._arm_term = arm_term
        self._gripper_term = gripper_term
        # The action slice is in the term's joint order, the planner reports cspace order.
        term_order = interface.joint_names(arm_term)
        planner_order = list(planner.cfg.arm_joint_names)
        assert sorted(term_order) == sorted(
            planner_order
        ), f"'{arm_term}' drives {term_order}, the planner {planner_order}"
        self._permutation = [planner_order.index(name) for name in term_order]

    def step(self, arm_target: np.ndarray | None = None, gripper_target: float | None = None, steps: int = 1) -> None:
        """Write this arm's slices of the shared action and advance the env.

        Args:
            arm_target: Arm joint targets in planner order; the previous ones held when omitted.
            gripper_target: Gripper joint target; the previous one is held when omitted.
            steps: How many control steps to take.
        """
        if gripper_target is not None:
            self._gripper_target = gripper_target
        if arm_target is not None:
            self.interface.set(self._arm_term, np.asarray(arm_target, dtype=np.float64)[self._permutation])
        self.interface.set_scalar(self._gripper_term, self._gripper_target)
        for _ in range(steps):
            self.interface.step()
            if self.on_step is not None:
                self.on_step()
