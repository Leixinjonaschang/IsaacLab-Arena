# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Compare each arm's tool-frame *orientation* between cuMotion's kinematics and the simulator.

The start-up cross-check compares only the tool's position, and position agreeing says nothing
about the frame it is expressed in. A grasp is authored as a world-frame orientation and handed
to cuMotion, which resolves it against its own tool frame -- so if that frame is rotated relative
to the simulated one, the arm reaches the commanded pose to a fraction of a millimetre while the
jaws point somewhere else entirely, and only one hand need be affected.

Usage::

    OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y .venv/bin/python \\
        isaaclab_arena_cumotion/scripts/probe_tool_orientation.py --headless
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--env", type=str, default="agibot_stack_bowls")
AppLauncher.add_app_launcher_args(parser)
args, _ = parser.parse_known_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import numpy as np  # noqa: E402
import torch  # noqa: E402

import warp as wp  # noqa: E402

import isaaclab.utils.math as math_utils  # noqa: E402
import isaaclab_arena_environments  # noqa: E402,F401
from isaaclab_arena.assets.registries import EnvironmentRegistry  # noqa: E402
from isaaclab_arena.cli.isaaclab_arena_cli import (  # noqa: E402
    arena_env_builder_cfg_from_argparse,
    get_isaaclab_arena_cli_parser,
)
from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder  # noqa: E402

from isaaclab_arena_cumotion.planner import CumotionArmPlanner  # noqa: E402

arena_args = get_isaaclab_arena_cli_parser().parse_args(["--num_envs", "1"])
factory = EnvironmentRegistry().get_component_by_name(args.env)()
arena_env = factory.build(factory._legacy_argparse_cfg_type(teleop_device=None))
builder = ArenaEnvBuilder(arena_env, arena_env_builder_cfg_from_argparse(arena_args))
env = builder.make_registered().unwrapped
env.reset()

robot = env.scene.articulations["robot"]
body_names = list(robot.data.body_names)
quaternions = wp.to_torch(robot.data.body_quat_w)[0].detach().cpu().numpy()


def axis_label(vector: np.ndarray) -> str:
    axis = int(np.argmax(np.abs(vector)))
    return f"{'+' if vector[axis] > 0 else '-'}{'xyz'[axis]}"


for arm in ("left", "right"):
    planner = CumotionArmPlanner(env, arena_env.embodiment, arm=arm)
    cfg = planner.cfg
    q = planner.joint_positions()

    # Simulated tool frame.
    tool_index = body_names.index(cfg.tool_frame)
    sim_rotation = math_utils.matrix_from_quat(torch.tensor(quaternions[tool_index]).float().unsqueeze(0))[0].numpy()

    # cuMotion's tool frame for the same configuration. Its pose is in the arm's base frame, and
    # the robot is not yawed in this scene, so the rotation is directly comparable.
    lula_rotation = np.asarray(planner.kinematics.pose(np.asarray(q, dtype=np.float64), cfg.tool_frame).rotation.matrix())

    relative = sim_rotation.T @ lula_rotation
    angle_deg = float(np.degrees(np.arccos(np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0))))

    print(f"\n=== {arm} arm ({cfg.tool_frame}) ===")
    print(f"  lula description: {cfg.lula_robot_description}")
    print(f"  configured approach axis {cfg.tool_approach_axis}, jaw axis {cfg.jaw_axis}")
    print(f"  tool_correction is identity: {np.allclose(planner.tool_correction, np.eye(3))}")
    print(f"  position cross-check: {planner.kinematics_error_m() * 1000:.2f} mm")
    print("  sim  tool axes in world: " + "  ".join(f"{a} {np.round(sim_rotation[:, i], 3)}" for i, a in enumerate("xyz")))
    print("  lula tool axes in world: " + "  ".join(f"{a} {np.round(lula_rotation[:, i], 3)}" for i, a in enumerate("xyz")))
    print(f"  ORIENTATION MISMATCH: {angle_deg:.2f} deg")
    print(f"  sim  approach (tool z) points {axis_label(sim_rotation[:, 2])} in world")
    print(f"  lula approach (tool z) points {axis_label(lula_rotation[:, 2])} in world")

simulation_app.close()
