# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

from isaaclab_arena.assets.register import register_environment
from isaaclab_arena.environments.arena_environment_factory import ArenaEnvironmentFactory
from isaaclab_arena_environments.agibot_make_toast_environment import (
    _TABLE_TOP_Z,
    AgibotMakeToastEnvironment,
    AgibotMakeToastEnvironmentCfg,
)

if TYPE_CHECKING:
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment


@dataclass
class AgibotHandoverToastEnvironmentCfg(AgibotMakeToastEnvironmentCfg):
    """Configure the Agibot toast-handover environment. Same knobs as make_toast; the success
    criterion differs, and a little per-reset variety is on by default for data collection."""

    shelf_jitter_x_min_m: float = -0.05
    shelf_jitter_x_max_m: float = 0.02
    """Rack-and-slices group x range, asymmetric on purpose: the rack's nominal x of 0.40 sits
    near the outer edge of the right arm's measured reach band (0.35-0.45), and the first
    symmetric +/-50 mm sweep produced resets with zero reachable grasps outward of it. This
    keeps x in 0.35-0.42, spanning the band without stepping past it."""

    shelf_jitter_y_m: float = 0.05
    """Rack-and-slices group y offset, +/-50 mm."""

    shelf_jitter_yaw_rad: float = math.radians(15.0)
    """Rack-and-slices group yaw, +/-15 degrees about the rack's origin."""

    toaster_jitter_xy_m: float = 0.05
    """Toaster offset, +/-50 mm; the carry crosses 0.23 m above its lid, so this varies the
    scenery and the collision model rather than the handover itself."""


@register_environment
class AgibotHandoverToastEnvironment(
    AgibotMakeToastEnvironment, ArenaEnvironmentFactory[AgibotHandoverToastEnvironmentCfg]
):
    """Take a slice of bread from the rack and pass it to the other hand, with the Agibot.

    The first stage of ``agibot_make_toast``, split out for data collection: the rack is
    right-arm-only and the toaster left-arm-only (see the toaster placement notes in the parent),
    so the right-to-left handover is the piece every make_toast demonstration shares. The scene is
    identical -- toaster included, so recordings transfer -- and only the task is swapped: success
    is a slice held at the left gripper, clear of the table, with the other slices still racked.
    """

    name: str = "agibot_handover_toast"
    _legacy_argparse_cfg_type = AgibotHandoverToastEnvironmentCfg

    def build(self, cfg: AgibotHandoverToastEnvironmentCfg) -> IsaacLabArenaEnvironment:
        """Build the environment from its typed configuration."""
        from isaaclab_arena.tasks.handover_toast_task import HandoverToastTask

        environment = super().build(cfg)
        breads = [environment.scene.assets[f"bread{index}"] for index in range(cfg.num_breads)]
        environment.task = HandoverToastTask(
            breads=breads,
            bread_shelf=environment.scene.assets["bread_shelf"],
            # The Agibot's left tool body; the right one is right_gripper_center. The rack sits on
            # the robot's right, so the left hand is the receiving one.
            receive_body_name="gripper_center",
            table_top_z_m=_TABLE_TOP_Z,
            episode_length_s=120.0,
            viewer_cfg=environment.embodiment.get_head_viewer_cfg() if cfg.head_view else None,
        )
        return environment
