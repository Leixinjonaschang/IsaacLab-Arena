# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np
from typing import TYPE_CHECKING

from isaaclab.envs.common import ViewerCfg

from isaaclab_arena.affordances.openable import Openable
from isaaclab_arena.embodiments.common.arm_mode import ArmMode
from isaaclab_arena.tasks.open_door_task import OpenDoorTask
from isaaclab_arena.tasks.pick_and_place_task import PickAndPlaceTask
from isaaclab_arena.tasks.sequential_task_base import SequentialTaskBase
from isaaclab_arena.utils.cameras import get_viewer_cfg_look_at_object

if TYPE_CHECKING:
    from isaaclab.envs.mimic_env_cfg import MimicEnvCfg


class OpenDoorAndPlaceObjectTask(SequentialTaskBase):
    """Open a container, then take an object out of it and place it on a destination.

    The container must still be open and the object must still rest on the destination on the
    final step, so a policy cannot succeed by letting the door swing shut behind it.
    """

    def __init__(
        self,
        open_door_task: OpenDoorTask,
        pick_and_place_task: PickAndPlaceTask,
        episode_length_s: float | None = None,
        task_description: str | None = None,
        mimic_datagen_name: str = "open_door_and_place_object_task_D0",
    ):
        """
        Args:
            open_door_task: Subtask that opens the container the object starts in.
            pick_and_place_task: Subtask that moves the object onto its destination.
            episode_length_s: Budget for both subtasks. Defaults to the sum of the subtasks'.
            task_description: Natural-language instruction handed to language-conditioned policies.
            mimic_datagen_name: Dataset name written into the Mimic datagen config.
        """
        assert isinstance(open_door_task.openable_object, Openable), "open_door_task must act on an Openable container"
        super().__init__(
            subtasks=[open_door_task, pick_and_place_task],
            episode_length_s=episode_length_s,
            task_description=task_description,
            desired_subtask_success_state=[True, True],
        )
        self.openable_object = open_door_task.openable_object
        """The container that has to be opened before the object can be picked up."""
        self.mimic_datagen_name = mimic_datagen_name
        """Dataset name stamped onto the combined Mimic datagen config."""

    def get_viewer_cfg(self) -> ViewerCfg:
        """Frame the container, which both subtasks play out around."""
        return get_viewer_cfg_look_at_object(lookat_object=self.openable_object, offset=np.array([-1.3, -1.3, 1.3]))

    def get_prompt(self) -> str | None:
        """Return the task description used to prompt language-conditioned policies."""
        return self.task_description

    def get_mimic_env_cfg(self, arm_mode: ArmMode) -> MimicEnvCfg:
        """Return the combined Mimic config, named after this task."""
        mimic_env_cfg = super().get_mimic_env_cfg(arm_mode)
        mimic_env_cfg.datagen_config.name = self.mimic_datagen_name
        return mimic_env_cfg
