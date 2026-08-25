# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Registry of cuMotion descriptions keyed by embodiment name.

The description is a property of the robot hardware, so entries are registered under the
robot-family base name (e.g. ``"agibot"``) and inherited by concrete variants.
"""

from __future__ import annotations

import os
import pathlib
from typing import TYPE_CHECKING

import isaaclab_arena
from isaaclab_arena_cumotion.cumotion_embodiment_cfg import CumotionEmbodimentCfg

if TYPE_CHECKING:
    from isaaclab_arena.embodiments.embodiment_base import EmbodimentBase

_CUMOTION_EMBODIMENT_CFGS: dict[str, CumotionEmbodimentCfg] = {}


def register_cumotion_cfg(embodiment_name: str, cfg: CumotionEmbodimentCfg, arm: str = "left") -> None:
    """Register a cuMotion config for one arm of an embodiment, erroring on a duplicate key.

    A bimanual robot needs one entry per arm: cuMotion plans a single kinematic chain, and the
    arm that is not being planned is pinned at its default configuration.
    """
    key = f"{embodiment_name}:{arm}"
    assert key not in _CUMOTION_EMBODIMENT_CFGS, f"A cuMotion config is already registered for '{key}'."
    _CUMOTION_EMBODIMENT_CFGS[key] = cfg


def get_cumotion_cfg_by_name(embodiment_name: str, arm: str = "left") -> CumotionEmbodimentCfg:
    """Return the cuMotion config registered for an exact embodiment name and arm."""
    cfg = _CUMOTION_EMBODIMENT_CFGS.get(f"{embodiment_name}:{arm}")
    assert cfg is not None, (
        f"No cuMotion config registered for '{embodiment_name}:{arm}'. Register one via"
        f" register_cumotion_cfg(...). Known: {sorted(_CUMOTION_EMBODIMENT_CFGS)}."
    )
    return cfg


def get_embodiment_cumotion_cfg(embodiment: EmbodimentBase, arm: str = "left") -> CumotionEmbodimentCfg:
    """Return the cuMotion config registered for an embodiment's robot family.

    Walks the class hierarchy so a config registered under a family name (e.g. ``"agibot"``) also
    covers subclassed variants that override ``name``. The most-derived match wins, so a variant
    may register its own override.

    Args:
        embodiment: Embodiment whose robot family should be looked up.
    """
    for cls in type(embodiment).__mro__:
        name = cls.__dict__.get("name")
        cfg = _CUMOTION_EMBODIMENT_CFGS.get(f"{name}:{arm}") if name else None
        if cfg is not None:
            return cfg
    raise AssertionError(
        f"No cuMotion config registered for embodiment '{embodiment.name}' arm '{arm}', or its"
        f" robot family. Known: {sorted(_CUMOTION_EMBODIMENT_CFGS)}."
    )


# Isaac Lab downloads its RMPFlow assets from Nucleus, but cuMotion's loader reads real files off
# disk, so the local Isaac asset cache is used directly (for the URDF). The Lula robot
# descriptions come from Arena's own copies instead: Arena's default Agibot rest pose mirrors
# the left wrist (see agibot.py), and cuMotion has to read descriptions that state that pose --
# planning one arm against the stock file would place the *other* arm 0.77 rad away from where
# it actually is, and that arm is a fixed obstacle in the collision model.
ISAAC_ASSET_ROOT = os.environ.get("ISAAC_ASSET_ROOT", "/tmp/Assets/Isaac/6.0/Isaac")
_AGIBOT_RMPFLOW_DIR = f"{ISAAC_ASSET_ROOT}/IsaacLab/Controllers/RmpFlowAssets/agibot"
_ARENA_AGIBOT_RMPFLOW_DIR = pathlib.Path(isaaclab_arena.__file__).parent / "embodiments" / "agibot" / "rmpflow"

# The Agibot's arm links are named Link<n>_<side>; ``left_base_link`` is the *hand* base (child of
# Link7_l through the fixed Joint_hand_l), not the arm mount, which is ``base_link_l``.
_AGIBOT_LEFT_SELF_COLLISION_IGNORE = {
    "Link1_l": ["Link2_l", "Link3_l"],
    "Link2_l": ["Link3_l", "Link4_l"],
    "Link3_l": ["Link4_l", "Link5_l"],
    "Link4_l": ["Link5_l", "Link6_l"],
    "Link5_l": ["Link6_l", "Link7_l", "left_base_link"],
    "Link6_l": ["Link7_l", "left_base_link"],
    "Link7_l": ["left_base_link"],
}

_AGIBOT_RIGHT_SELF_COLLISION_IGNORE = {
    "Link1_r": ["Link2_r", "Link3_r"],
    "Link2_r": ["Link3_r", "Link4_r"],
    "Link3_r": ["Link4_r", "Link5_r"],
    "Link4_r": ["Link5_r", "Link6_r"],
    "Link5_r": ["Link6_r", "Link7_r", "right_base_link"],
    "Link6_r": ["Link7_r", "right_base_link"],
    "Link7_r": ["right_base_link"],
}

AGIBOT_LEFT_ARM_CUMOTION_CFG = CumotionEmbodimentCfg(
    lula_robot_description=str(_ARENA_AGIBOT_RMPFLOW_DIR / "agibot_left_arm_gripper.yaml"),
    robot_urdf=f"{_AGIBOT_RMPFLOW_DIR}/agibot.urdf",
    tool_frame="gripper_center",
    arm_joint_names=[f"left_arm_joint{i}" for i in range(1, 8)],
    gripper_joint_names=["left_hand_joint1", "left_.*_Support_Joint"],
    gripper_open_pos=0.994,
    gripper_closed_pos=0.0,
    self_collision_ignore=_AGIBOT_LEFT_SELF_COLLISION_IGNORE,
    # Both measured with scripts/probe_gripper_axes.py, not read off a rest pose: the wrist-to-tool
    # vector is +z in tool coordinates and the jaws separate along y, on both hands.
    tool_approach_axis="+z",
    jaw_axis="+y",
)

AGIBOT_RIGHT_ARM_CUMOTION_CFG = CumotionEmbodimentCfg(
    lula_robot_description=str(_ARENA_AGIBOT_RMPFLOW_DIR / "agibot_right_arm_gripper.yaml"),
    robot_urdf=f"{_AGIBOT_RMPFLOW_DIR}/agibot.urdf",
    tool_frame="right_gripper_center",
    arm_joint_names=[f"right_arm_joint{i}" for i in range(1, 8)],
    gripper_joint_names=["right_hand_joint1", "right_.*_Support_Joint"],
    gripper_open_pos=0.994,
    gripper_closed_pos=0.0,
    self_collision_ignore=_AGIBOT_RIGHT_SELF_COLLISION_IGNORE,
    tool_approach_axis="+z",
    jaw_axis="+y",
)

register_cumotion_cfg("agibot", AGIBOT_LEFT_ARM_CUMOTION_CFG, arm="left")
register_cumotion_cfg("agibot", AGIBOT_RIGHT_ARM_CUMOTION_CFG, arm="right")
