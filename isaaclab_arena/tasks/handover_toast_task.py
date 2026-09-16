# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

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
from isaaclab_arena.tasks.predicates.spatial import any_object_near_body, count_objects_in_frame_box
from isaaclab_arena.tasks.task_base import TaskBase
from isaaclab_arena.utils.cameras import get_viewer_cfg_look_at_object

DEFAULT_RECEIVE_DISTANCE_M = 0.10
"""How close a slice's origin has to sit to the receiving gripper's tool body to count as held.

A held slice measures ~18 mm from the tool frame, and the slice itself is 58 mm from origin to
edge, so 100 mm accepts any real grip while a slice merely brushed by the hand stays outside."""

DEFAULT_LIFT_CLEARANCE_M = 0.10
"""How far above the table a held slice has to be.

The rack stands the slices ~75 mm over the table and the handover happens at chest height, so
this floor is comfortably between "still racked" and "held in the air" -- it exists so a slice
knocked onto the table next to a lowered gripper cannot count as handed over."""


@register_task
class HandoverToastTask(TaskBase):
    """Take one slice of bread out of the rack and hand it to the receiving arm.

    The first stage of ``MakeToastTask`` on its own, for data collection: the rack is reachable
    only by one arm and the toaster only by the other, so every demonstration of make_toast runs
    through this handover. Success needs some slice held at the receiving gripper -- within
    ``receive_distance_m`` of ``receive_body_name`` and at least ``lift_clearance_m`` above the
    table -- with ``slices_left_in_shelf`` slices still in the rack.
    """

    def __init__(
        self,
        breads: list[ObjectBase],
        bread_shelf: ObjectBase,
        receive_body_name: str,
        table_top_z_m: float,
        receive_distance_m: float = DEFAULT_RECEIVE_DISTANCE_M,
        lift_clearance_m: float = DEFAULT_LIFT_CLEARANCE_M,
        slices_left_in_shelf: int | None = None,
        episode_length_s: float | None = None,
        task_description: str | None = None,
        viewer_cfg: ViewerCfg | None = None,
    ):
        super().__init__(
            episode_length_s=episode_length_s,
            task_description=(
                "Pick up a slice of bread with one hand and pass it to the other hand."
                if task_description is None
                else task_description
            ),
        )
        assert len(breads) >= 1, "The rack needs at least one slice to hand over"
        self.breads = breads
        self.bread_names = [bread.name for bread in breads]
        self.bread_shelf = bread_shelf
        self.receive_body_name = receive_body_name
        """Robot body the slice ends up held at -- the receiving gripper's tool frame."""
        self.table_top_z_m = table_top_z_m
        self.receive_distance_m = receive_distance_m
        self.lift_clearance_m = lift_clearance_m
        self.slices_left_in_shelf = len(breads) - 1 if slices_left_in_shelf is None else slices_left_in_shelf
        """How many slices stay racked. One leaves by default, which also pins down that the held
        slice really came out of the rack rather than the count being made up by a stray."""
        self.viewer_cfg = viewer_cfg
        """Overrides the default over-the-shoulder view, e.g. with a robot head-mounted one."""

    def is_success(self, env: ManagerBasedEnv) -> torch.Tensor:
        """Returns whether a slice is held at the receiving gripper with the rack undisturbed."""
        return self._slice_received(env) & self._slices_remaining_in_shelf(env)

    def _slice_received(self, env: ManagerBasedEnv) -> torch.Tensor:
        """Whether some slice hangs at the receiving gripper, clear of the table."""
        return any_object_near_body(
            env,
            self.bread_names,
            self.receive_body_name,
            self.receive_distance_m,
            min_height_m=self.table_top_z_m + self.lift_clearance_m,
        )

    def _slices_remaining_in_shelf(self, env: ManagerBasedEnv) -> torch.Tensor:
        """Whether the expected number of slices are still in the rack.

        The same containment box as ``MakeToastTask``: RoboDojo's ``is_A_in_B`` puts the slice's
        origin inside the rack's bounding box in XY and anywhere above its underside.
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
        raise NotImplementedError("Mimic is not set up for HandoverToastTask yet.")

    def get_metrics(self) -> list[MetricBase]:
        return [SuccessRateMetric()]

    def get_viewer_cfg(self) -> ViewerCfg:
        if self.viewer_cfg is not None:
            return self.viewer_cfg
        return get_viewer_cfg_look_at_object(lookat_object=self.bread_shelf, offset=np.array([-0.9, -0.9, 0.85]))

    def apply_reachability_constraints(self) -> None:
        """The robot has to reach every slice in the rack."""
        self._apply_reachability_constraints(self.breads)


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out: TerminationTermCfg = TerminationTermCfg(func=mdp_isaac_lab.time_out, time_out=True)

    # Depends on the scene objects, so the task passes it in at construction time.
    success: TerminationTermCfg = MISSING
