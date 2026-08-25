# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""cuMotion motion planning for Arena embodiments.

Isaac Sim 6.0 ships cuMotion as the ``isaacsim.robot_motion.cumotion`` extension, backed by a
native ``cumotion`` wheel. It is a different library from the open-source cuRobo that
:mod:`isaaclab_arena_curobo` wraps, with an incompatible Python API, so the two live side by side.

Nothing here imports Isaac Sim at module scope: the planner and executor are only usable inside a
running :class:`~isaaclab.app.AppLauncher` session, and importing them earlier would fail.
"""

from __future__ import annotations
