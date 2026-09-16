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
from isaaclab_arena.tasks.predicates.joints import joint_past_travel_fraction
from isaaclab_arena.tasks.predicates.spatial import (
    any_object_in_frame_box,
    count_objects_in_frame_box,
    objects_upright_about_any_axis,
)
from isaaclab_arena.tasks.task_base import TaskBase
from isaaclab_arena.utils.cameras import get_viewer_cfg_look_at_object

DEFAULT_UPRIGHT_THRESHOLD_RAD = math.radians(30.0)
"""How far a slice may tilt off vertical, matching RoboDojo's make_toast ``AXIS_UP_THRESHOLD``."""

DEFAULT_SLOT_Z_LOWER_M = 0.03
DEFAULT_SLOT_Z_UPPER_M = 0.065
"""How far above a slot's floor the slice's origin has to sit, from RoboDojo's ``BREAD_Z_*``.

Below the lower bound the slice has not gone in; above the upper bound it is resting on the
toaster's lid rather than dropped into the slot, or still hanging from the gripper."""

DEFAULT_SLICES_LEFT_IN_SHELF = 2
"""How many of the four slices stay in the rack. Two go in the toaster, so two remain."""


@register_task
class MakeToastTask(TaskBase):
    """Put two slices of bread into a toaster and push its lever down.

    Ported from RoboDojo's ``make_toast``. Success needs a slice in each of the toaster's two
    slots -- inside the slot's footprint and between ``slot_z_lower_m`` and ``slot_z_upper_m``
    above its floor -- every slice still standing on edge to within ``upright_threshold_rad``,
    ``slices_left_in_shelf`` of them still in the rack, and the toaster's lever pressed past
    its ``pressedness_threshold``.

    RoboDojo additionally requires ``all_robot_back_to_origin``. That is dropped here for the
    same reason as in ``StackBowlsTask``: Arena has no equivalent predicate, and reproducing it
    would mean teleoperators had to park both arms before a finished run counted.
    """

    def __init__(
        self,
        breads: list[ObjectBase],
        toaster: ObjectBase,
        bread_shelf: ObjectBase,
        upright_threshold_rad: float = DEFAULT_UPRIGHT_THRESHOLD_RAD,
        slot_z_lower_m: float = DEFAULT_SLOT_Z_LOWER_M,
        slot_z_upper_m: float = DEFAULT_SLOT_Z_UPPER_M,
        slices_left_in_shelf: int = DEFAULT_SLICES_LEFT_IN_SHELF,
        episode_length_s: float | None = None,
        task_description: str | None = None,
        viewer_cfg: ViewerCfg | None = None,
    ):
        super().__init__(
            episode_length_s=episode_length_s,
            task_description=(
                "Pick up two slices of bread, place them into the toaster, and press the lever down."
                if task_description is None
                else task_description
            ),
        )
        assert (
            len(breads) > slices_left_in_shelf
        ), f"Two slices have to leave the rack, so more than {slices_left_in_shelf} are needed, got {len(breads)}"
        self.breads = breads
        self.bread_names = [bread.name for bread in breads]
        self.toaster = toaster
        self.bread_shelf = bread_shelf
        self.upright_threshold_rad = upright_threshold_rad
        self.slot_z_lower_m = slot_z_lower_m
        self.slot_z_upper_m = slot_z_upper_m
        self.slices_left_in_shelf = slices_left_in_shelf
        self.viewer_cfg = viewer_cfg
        """Overrides the default over-the-shoulder view, e.g. with a robot head-mounted one."""

    def is_success(self, env: ManagerBasedEnv) -> torch.Tensor:
        """Returns whether both slots are loaded, the rest of the bread is undisturbed and the
        lever is down."""
        return (
            self._slot_loaded(env, "toast_slot1")
            & self._slot_loaded(env, "toast_slot2")
            & objects_upright_about_any_axis(env, self.bread_names, self.upright_threshold_rad)
            & self._slices_remaining_in_shelf(env)
            & self._lever_down(env)
        )

    def _lever_down(self, env: ManagerBasedEnv) -> torch.Tensor:
        """Whether the toaster's lever has been pushed home."""
        toaster_type = type(self.toaster)
        return joint_past_travel_fraction(
            env,
            self.toaster.name,
            toaster_type.LEVER_JOINT_NAME,
            toaster_type.LEVER_PRESSED_FRACTION,
        )

    def _slot_loaded(self, env: ManagerBasedEnv, slot_tag: str) -> torch.Tensor:
        """Whether some slice sits in the named toaster slot."""
        x_range, y_range = type(self.toaster).SLOT_RECT_LOCAL_M[slot_tag]
        slot_z = type(self.toaster).SLOT_Z_LOCAL_M
        return any_object_in_frame_box(
            env,
            self.bread_names,
            self.toaster.name,
            x_range=x_range,
            y_range=y_range,
            z_range=(slot_z + self.slot_z_lower_m, slot_z + self.slot_z_upper_m),
        )

    def _slices_remaining_in_shelf(self, env: ManagerBasedEnv) -> torch.Tensor:
        """Whether the expected number of slices are still in the rack.

        RoboDojo's ``is_A_in_B`` puts the slice's origin inside the rack's bounding box in XY and
        anywhere above its underside, which is what the box here spells out.
        """
        half_x, half_y, half_z = type(self.bread_shelf).HALF_EXTENTS_M
        return count_objects_in_frame_box(
            env,
            self.bread_names,
            self.bread_shelf.name,
            x_range=(-half_x, half_x),
            y_range=(-half_y, half_y),
            z_range=(-half_z, None),
            count=self.slices_left_in_shelf,
        )

    def get_scene_cfg(self):
        return None

    def get_termination_cfg(self):
        return TerminationsCfg(success=TerminationTermCfg(func=self.is_success, params={}))

    def get_events_cfg(self):
        return None

    def get_mimic_env_cfg(self, arm_mode: ArmMode):
        raise NotImplementedError("Mimic is not set up for MakeToastTask yet.")

    def get_metrics(self) -> list[MetricBase]:
        return [SuccessRateMetric()]

    def get_viewer_cfg(self) -> ViewerCfg:
        if self.viewer_cfg is not None:
            return self.viewer_cfg
        return get_viewer_cfg_look_at_object(lookat_object=self.toaster, offset=np.array([-0.9, -0.9, 0.85]))

    def apply_reachability_constraints(self) -> None:
        """The robot has to reach every slice in the rack, and the toaster it loads them into."""
        self._apply_reachability_constraints([*self.breads, self.toaster])


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out: TerminationTermCfg = TerminationTermCfg(func=mdp_isaac_lab.time_out, time_out=True)

    # Depends on the scene objects, so the task passes it in at construction time.
    success: TerminationTermCfg = MISSING
