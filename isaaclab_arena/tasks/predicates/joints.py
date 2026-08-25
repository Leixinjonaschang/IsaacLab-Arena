# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg

from isaaclab_arena.tasks.predicates.predicate_utils import get_env, select
from isaaclab_arena.utils.joint_utils import (
    get_articulation_from_asset_cfg,
    get_joint_index_from_asset_cfg,
    get_joint_position_limits_from_articulation,
    get_unnormalized_joint_position,
)


def joint_travel_fraction(
    env: ManagerBasedRLEnv,
    articulation_name: str,
    joint_name: str,
) -> torch.Tensor:
    """Returns how far along its own range a joint currently sits, as a fraction in [0, 1].

    Zero is the joint's lower limit and one its upper limit, with no reference to where the joint
    rests when untouched -- so which end means "actuated" depends on the asset.

    Args:
        env: The environment.
        articulation_name: Scene name of the articulation holding the joint.
        joint_name: Name of the joint, as the articulation reports it.

    Returns:
        The fraction, one entry per environment.
    """
    asset_cfg = SceneEntityCfg(articulation_name, joint_names=[joint_name])
    unwrapped = get_env(env)
    articulation = get_articulation_from_asset_cfg(unwrapped, asset_cfg)
    joint_index = get_joint_index_from_asset_cfg(unwrapped, asset_cfg)
    lower, upper = get_joint_position_limits_from_articulation(articulation, joint_index)
    return (get_unnormalized_joint_position(unwrapped, asset_cfg) - lower) / (upper - lower)


def joint_past_travel_fraction(
    env: ManagerBasedRLEnv,
    articulation_name: str,
    joint_name: str,
    threshold: float,
    env_id: int | None = None,
) -> torch.Tensor:
    """Checks that a joint has travelled past a fraction of its range, measured from its lower limit.

    This is deliberately not built on ``Pressable.is_pressed``. That reads
    ``get_normalized_joint_position``, which flips the fraction to ``1 - fraction`` whenever the
    lower limit is negative -- a convention that suits doors hinged through zero but reverses the
    meaning of the thresholds RoboDojo's ``is_joint_position_above_ratio`` publishes.

    Args:
        env: The environment.
        articulation_name: Scene name of the articulation holding the joint.
        joint_name: Name of the joint, as the articulation reports it.
        threshold: Fraction of the joint's range that has to be exceeded.
        env_id: Restrict the result to a single environment, or None for all of them.

    Returns:
        True when the joint sits past threshold of the way from its lower to its upper limit.
    """
    return select(joint_travel_fraction(env, articulation_name, joint_name) > threshold, env_id)
