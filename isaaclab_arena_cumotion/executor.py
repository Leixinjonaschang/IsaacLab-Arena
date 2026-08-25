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
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

    from isaaclab_arena_cumotion.planner import CumotionArmPlanner

DEFAULT_TRAJECTORY_SPEED = 0.35
"""Fraction of cuMotion's time-optimal speed. Above ~0.5 the stiff arm cannot keep up."""

DEFAULT_GRIPPER_RAMP_STEPS = 200
"""Steps over which a gripper command is ramped, rather than stepped."""


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
                torch.full((1, len(self.planner.gripper_joint_ids)), float(self._gripper_target), device=self.env.device),
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
        settle_steps: int = 40,
    ) -> np.ndarray:
        """Time-parameterise a planned path and servo the arm along it.

        Args:
            path: A cuMotion ``Path``, as returned by the planner.
            reverse: Retrace the path from its end to its start.
            speed: Fraction of the time-optimal speed to play it back at.
            settle_steps: Extra steps holding the final configuration.

        Returns:
            The final commanded arm configuration.
        """
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

    def set_gripper(
        self, target: float, hold_arm_at: np.ndarray | None = None, ramp_steps: int = DEFAULT_GRIPPER_RAMP_STEPS
    ) -> None:
        """Ramp the gripper to a target, holding the arm where it is.

        Args:
            target: Gripper joint target.
            hold_arm_at: Arm configuration to hold; the current one is used when omitted.
            ramp_steps: Steps over which to ramp.
        """
        hold = self.planner.joint_positions() if hold_arm_at is None else hold_arm_at
        start = self._gripper_target
        for i in range(ramp_steps):
            alpha = (i + 1) / ramp_steps
            self.step(arm_target=hold, gripper_target=start + alpha * (target - start))

    def open_gripper(self, hold_arm_at: np.ndarray | None = None, ramp_steps: int = DEFAULT_GRIPPER_RAMP_STEPS) -> None:
        """Ramp the gripper open."""
        self.set_gripper(self.planner.cfg.gripper_open_pos, hold_arm_at, ramp_steps)

    def close_gripper(
        self, hold_arm_at: np.ndarray | None = None, ramp_steps: int = DEFAULT_GRIPPER_RAMP_STEPS
    ) -> None:
        """Ramp the gripper closed."""
        self.set_gripper(self.planner.cfg.gripper_closed_pos, hold_arm_at, ramp_steps)
