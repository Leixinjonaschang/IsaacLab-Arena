# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import torch

import warp as wp
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg

DROID_EE_BODY_NAME = "panda_link8"
"""Franka end-effector link used by DROID's Polymetis forward kinematics."""


def _ee_body_index(robot) -> int:
    """Return the Polymetis-compatible end-effector body index."""
    assert (
        DROID_EE_BODY_NAME in robot.data.body_names
    ), f"DROID robot is missing end-effector body '{DROID_EE_BODY_NAME}'. Available bodies: {robot.data.body_names}"
    return robot.data.body_names.index(DROID_EE_BODY_NAME)


def arm_joint_pos(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    robot = env.scene[asset_cfg.name]
    joint_names = [
        "panda_joint1",
        "panda_joint2",
        "panda_joint3",
        "panda_joint4",
        "panda_joint5",
        "panda_joint6",
        "panda_joint7",
    ]
    joint_indices = [i for i, name in enumerate(robot.data.joint_names) if name in joint_names]
    return wp.to_torch(robot.data.joint_pos)[:, joint_indices]


def gripper_pos(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Returns gripper position as 0 for open and 1 for closed."""
    robot = env.scene[asset_cfg.name]
    joint_names = ["finger_joint"]
    joint_indices = [i for i, name in enumerate(robot.data.joint_names) if name in joint_names]
    joint_pos = wp.to_torch(robot.data.joint_pos)[:, joint_indices]
    # rescale to 0–1
    return joint_pos / (torch.pi / 4)


def ee_pos(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Returns the end effector position (x, y, z) in the world frame."""
    robot = env.scene[asset_cfg.name]
    body_idx = _ee_body_index(robot)
    return wp.to_torch(robot.data.body_pos_w)[:, body_idx, :]


def ee_quat(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Returns the end effector orientation as quaternion (w, x, y, z) in the world frame."""
    robot = env.scene[asset_cfg.name]
    body_idx = _ee_body_index(robot)
    return wp.to_torch(robot.data.body_quat_w)[:, body_idx, :]
