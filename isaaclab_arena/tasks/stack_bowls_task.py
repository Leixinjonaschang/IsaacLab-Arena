# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import math
import numpy as np
import torch
from dataclasses import MISSING

import isaaclab.envs.mdp as mdp_isaac_lab
from isaaclab.envs.common import ViewerCfg
from isaaclab.envs.manager_based_env import ManagerBasedEnv
from isaaclab.managers import TerminationTermCfg
from isaaclab.utils.configclass import configclass

from isaaclab_arena.assets.object_base import ObjectBase
from isaaclab_arena.assets.register import register_task
from isaaclab_arena.embodiments.common.arm_mode import ArmMode
from isaaclab_arena.metrics.metric_base import MetricBase
from isaaclab_arena.metrics.success_rate import SuccessRateMetric
from isaaclab_arena.tasks.predicates.spatial import (
    lowest_object_upright,
    objects_at_rest,
    objects_stacked,
    objects_upright,
)
from isaaclab_arena.tasks.stack_bowls_trace import StackBowlsTrace
from isaaclab_arena.tasks.task_base import TaskBase
from isaaclab_arena.utils.cameras import get_viewer_cfg_look_at_object

DEFAULT_UPRIGHT_THRESHOLD_RAD = math.radians(45.0)
"""How far any bowl may tilt, matching RoboDojo's stack_bowls."""

DEFAULT_BASE_UPRIGHT_THRESHOLD_RAD = math.radians(7.0)
"""How far the bottom bowl may tilt. Stricter, since it carries the pile."""

DEFAULT_STACK_XY_THRESHOLD_M = 0.04
"""How far apart neighbouring bowls in the pile may be horizontally."""

DEFAULT_MIN_Z_GAP_M = 0.005
"""Minimum height difference between neighbouring bowls, so side-by-side does not count."""

DEFAULT_MAX_Z_GAP_M = 0.06
"""Maximum height difference between neighbouring bowls: one bowl height.

Not in RoboDojo, which relies on its ``all_robot_back_to_origin`` condition to rule this out.
Without an upper bound a bowl still in the gripper, hovering above the pile, reads as the top
of it -- observed on a recorded demo where the run was flagged solved with the last bowl 197 mm
up and moving at 0.23 m/s. Bowls nest 23 mm apart, so 60 mm passes real stacks comfortably."""

DEFAULT_REST_VELOCITY_M_S = 0.1
"""Speed below which a bowl counts as settled.

Well above the physical noise floor on purpose. PhysX's TGS solver reports a standing velocity
for bodies in a resting contact stack -- it warns about this at startup -- and that reading does
not decay, so a threshold taken from a lone object (0.001 m/s) can never be satisfied by a
stacked one. Measured on a pile that is geometrically frozen, bottom to top:

    sim dt 1/200   0.001, 0.014, 0.026 m/s
    sim dt 1/120   0.001, 0.034, 0.046 m/s   (Arena's 15 Hz default)

The coarser physics step nearly doubles the floor, so re-measure this if the control rate moves
again. A bowl actually being carried reads 0.2 m/s and up, so 0.1 still sits between the two
cases -- 2.2x above the resting artifact and 2x below a carried bowl."""


@register_task
class StackBowlsTask(TaskBase):
    """Stack several bowls into a single pile.

    Ported from RoboDojo's ``stack_bowls``. Success needs every bowl upright to within
    ``upright_threshold_rad``, the lowest one upright to the tighter
    ``base_upright_threshold_rad``, all of them forming one pile (sorted by height, neighbouring
    bowls within ``stack_xy_threshold_m`` horizontally and ``min_z_gap_m`` to ``max_z_gap_m``
    apart vertically), and every bowl settled below ``rest_velocity_m_s``.

    The upper gap bound and the at-rest check are not in RoboDojo. RoboDojo instead requires
    ``all_robot_back_to_origin``, which implies the robot has let go and everything has come to
    rest; Arena has no equivalent predicate. Reproducing only the stacking half let a bowl still
    held in the gripper above the pile count as its top -- a recorded demo was flagged solved
    with the last bowl 197 mm up and travelling at 0.23 m/s. These two conditions close that
    hole at the point where it actually matters.
    """

    def __init__(
        self,
        bowls: list[ObjectBase],
        upright_threshold_rad: float = DEFAULT_UPRIGHT_THRESHOLD_RAD,
        base_upright_threshold_rad: float = DEFAULT_BASE_UPRIGHT_THRESHOLD_RAD,
        stack_xy_threshold_m: float = DEFAULT_STACK_XY_THRESHOLD_M,
        min_z_gap_m: float = DEFAULT_MIN_Z_GAP_M,
        max_z_gap_m: float | None = DEFAULT_MAX_Z_GAP_M,
        rest_velocity_m_s: float = DEFAULT_REST_VELOCITY_M_S,
        episode_length_s: float | None = None,
        task_description: str | None = None,
        viewer_cfg: ViewerCfg | None = None,
    ):
        super().__init__(
            episode_length_s=episode_length_s,
            task_description="Stack the bowls together." if task_description is None else task_description,
        )
        assert len(bowls) >= 2, f"Stacking needs at least two bowls, got {len(bowls)}"
        self.bowls = bowls
        self.bowl_names = [bowl.name for bowl in bowls]
        self.upright_threshold_rad = upright_threshold_rad
        self.base_upright_threshold_rad = base_upright_threshold_rad
        self.stack_xy_threshold_m = stack_xy_threshold_m
        self.min_z_gap_m = min_z_gap_m
        self.max_z_gap_m = max_z_gap_m
        self.rest_velocity_m_s = rest_velocity_m_s
        self.viewer_cfg = viewer_cfg
        """Overrides the default over-the-shoulder view, e.g. with a robot head-mounted one."""
        self.trace = StackBowlsTrace(self.bowl_names)
        """Diagnostic per-step CSV, off unless ``ARENA_STACK_BOWLS_TRACE`` names a path."""

    def is_success(self, env: ManagerBasedEnv) -> torch.Tensor:
        """Returns whether the bowls form an acceptable pile."""
        self.trace.record(env)
        return (
            objects_upright(env, self.bowl_names, self.upright_threshold_rad)
            & lowest_object_upright(env, self.bowl_names, self.base_upright_threshold_rad)
            & self._stacked(env)
            & objects_at_rest(env, self.bowl_names, self.rest_velocity_m_s)
        )

    def _stacked(self, env: ManagerBasedEnv) -> torch.Tensor:
        """Whether the bowls form one pile, bounded above as well as below."""
        return objects_stacked(
            env,
            self.bowl_names,
            self.stack_xy_threshold_m,
            min_z_gap=self.min_z_gap_m,
            max_z_gap=self.max_z_gap_m,
        )

    def get_scene_cfg(self):
        return None

    def get_termination_cfg(self):
        return TerminationsCfg(success=TerminationTermCfg(func=self.is_success, params={}))

    def get_events_cfg(self):
        return None

    def get_mimic_env_cfg(self, arm_mode: ArmMode):
        raise NotImplementedError("Mimic is not set up for StackBowlsTask yet.")

    def get_metrics(self) -> list[MetricBase]:
        return [SuccessRateMetric()]

    def get_viewer_cfg(self) -> ViewerCfg:
        if self.viewer_cfg is not None:
            return self.viewer_cfg
        return get_viewer_cfg_look_at_object(lookat_object=self.bowls[0], offset=np.array([-0.9, -0.9, 0.85]))

    def apply_reachability_constraints(self) -> None:
        """The robot has to reach every bowl, both to pick it up and to place it on the pile."""
        self._apply_reachability_constraints(self.bowls)


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out: TerminationTermCfg = TerminationTermCfg(func=mdp_isaac_lab.time_out, time_out=True)

    # Depends on the bowls, so the task passes it in at construction time.
    success: TerminationTermCfg = MISSING
