# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np
from dataclasses import MISSING
from typing import TYPE_CHECKING

import isaaclab.envs.mdp as mdp_isaac_lab
from isaaclab.envs.common import ViewerCfg
from isaaclab.managers import SceneEntityCfg, TerminationTermCfg
from isaaclab.utils.configclass import configclass

from isaaclab_arena.assets.asset import Asset
from isaaclab_arena.assets.register import register_task
from isaaclab_arena.embodiments.common.arm_mode import ArmMode
from isaaclab_arena.metrics.metric_base import MetricBase
from isaaclab_arena.metrics.object_moved import ObjectMovedRateMetric
from isaaclab_arena.metrics.success_rate import SuccessRateMetric
from isaaclab_arena.tasks.predicates.spatial import objects_in_proximity
from isaaclab_arena.tasks.task_base import TaskBase
from isaaclab_arena.utils.cameras import get_viewer_cfg_look_at_object

if TYPE_CHECKING:
    from isaaclab.envs.mimic_env_cfg import MimicEnvCfg


@register_task
class InsertObjectTask(TaskBase):
    """Pick an object up and seat it inside a receptacle.

    Success compares the two root frames axis by axis, so the tolerances describe the assembled
    pose directly: the lateral pair says the object is on the receptacle's axis, and the vertical
    one says it has actually descended into the receptacle rather than come to rest on top of it.
    Unlike a contact-based pick-and-place check, resting the object on the receptacle's rim does
    not count.

    The task owns no reset events; the objects keep their own pose randomisation, so an environment
    can place the receptacle and the object independently.
    """

    def __init__(
        self,
        held_object: Asset,
        receptacle: Asset,
        background_scene: Asset,
        max_lateral_separation_m: float = 0.01,
        max_vertical_separation_m: float = 0.02,
        episode_length_s: float | None = None,
        task_description: str | None = None,
    ):
        """
        Args:
            held_object: Object the robot has to pick up and insert.
            receptacle: Object that receives it. Its root frame defines the assembled pose.
            background_scene: Scene whose ``object_min_z`` decides when the object counts as dropped.
            max_lateral_separation_m: Success tolerance on the X and Y root separation.
            max_vertical_separation_m: Success tolerance on the Z root separation.
            episode_length_s: Budget for the whole insertion.
            task_description: Natural-language instruction handed to language-conditioned policies.
        """
        super().__init__(episode_length_s=episode_length_s)
        assert max_lateral_separation_m > 0.0, "max_lateral_separation_m must be greater than zero"
        assert max_vertical_separation_m > 0.0, "max_vertical_separation_m must be greater than zero"
        self.held_object = held_object
        """Object the robot inserts."""
        self.receptacle = receptacle
        """Object the held object has to end up inside."""
        self.background_scene = background_scene
        """Scene that defines the height below which the held object counts as dropped."""
        self.max_lateral_separation_m = max_lateral_separation_m
        self.max_vertical_separation_m = max_vertical_separation_m
        self.termination_cfg = self._make_termination_cfg()
        self.task_description = (
            f"Pick up the {held_object.name} and insert it into the {receptacle.name}"
            if task_description is None
            else task_description
        )

    def _make_termination_cfg(self) -> TerminationsCfg:
        """Build the success and drop terminations."""
        success = TerminationTermCfg(
            func=objects_in_proximity,
            params={
                "object_cfg": SceneEntityCfg(self.held_object.name),
                "target_object_cfg": SceneEntityCfg(self.receptacle.name),
                "max_x_separation": self.max_lateral_separation_m,
                "max_y_separation": self.max_lateral_separation_m,
                "max_z_separation": self.max_vertical_separation_m,
            },
        )
        object_dropped = TerminationTermCfg(
            func=mdp_isaac_lab.root_height_below_minimum,
            params={
                "minimum_height": self.background_scene.object_min_z,
                "asset_cfg": SceneEntityCfg(self.held_object.name),
            },
        )
        return TerminationsCfg(success=success, object_dropped=object_dropped)

    def get_scene_cfg(self):
        """Return no extra scene entities; the task reads root poses that already exist."""

    def get_termination_cfg(self) -> TerminationsCfg:
        return self.termination_cfg

    def get_events_cfg(self):
        """Return no events; the objects carry their own pose randomisation."""

    def apply_reachability_constraints(self) -> None:
        """The robot must reach both the object it picks up and the receptacle it inserts into."""
        self._apply_reachability_constraints([self.held_object, self.receptacle])

    def get_metrics(self) -> list[MetricBase]:
        return [SuccessRateMetric(), ObjectMovedRateMetric(self.held_object)]

    def get_viewer_cfg(self) -> ViewerCfg:
        """Frame the receptacle, where the decisive part of the task happens."""
        return get_viewer_cfg_look_at_object(lookat_object=self.receptacle, offset=np.array([-0.9, -0.9, 0.8]))

    def get_prompt(self) -> str | None:
        """Return the task description used to prompt language-conditioned policies."""
        return self.task_description

    def get_mimic_env_cfg(self, arm_mode: ArmMode) -> MimicEnvCfg:
        """Reuse the pick-and-place Mimic layout: approach the object, then the receptacle."""
        from isaaclab_arena.tasks.pick_and_place_task import PickPlaceMimicEnvCfg

        return PickPlaceMimicEnvCfg(
            arm_mode=arm_mode,
            pick_up_object_name=self.held_object.name,
            destination_location_name=self.receptacle.name,
        )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out: TerminationTermCfg = TerminationTermCfg(func=mdp_isaac_lab.time_out, time_out=True)

    success: TerminationTermCfg = MISSING

    object_dropped: TerminationTermCfg = MISSING
