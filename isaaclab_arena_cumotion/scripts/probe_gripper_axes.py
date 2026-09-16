# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Measure a gripper's approach and finger-closing axes, in its own tool frame.

Both are needed to author a grasp pose and neither is documented: the approach axis differs
between the Agibot's two hands, and which way the jaws open is a property of the linkage that no
config states. Both are directly observable from the simulated articulation, so they are measured
here rather than guessed at -- guessing is silent, since the arm will reach a wrongly-oriented
pose to a fraction of a millimetre and simply close on nothing.

The finger axis is read as the separation between the two jaws; the approach axis is read as
whichever tool axis points from the wrist towards the fingers.

Usage::

    OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y .venv/bin/python \\
        isaaclab_arena_cumotion/scripts/probe_gripper_axes.py --headless
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

import warp as wp  # noqa: E402

import isaaclab_arena_environments  # noqa: E402,F401
from isaaclab_arena.assets.registries import EnvironmentRegistry  # noqa: E402
from isaaclab_arena.cli.isaaclab_arena_cli import (  # noqa: E402
    arena_env_builder_cfg_from_argparse,
    get_isaaclab_arena_cli_parser,
)
from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder  # noqa: E402

arena_args = get_isaaclab_arena_cli_parser().parse_args(["--num_envs", "1"])
factory = EnvironmentRegistry().get_component_by_name(args.env)()
arena_env = factory.build(factory._legacy_argparse_cfg_type(teleop_device=None))
builder = ArenaEnvBuilder(arena_env, arena_env_builder_cfg_from_argparse(arena_args))
env = builder.make_registered().unwrapped
env.reset()

robot = env.scene.articulations["robot"]
body_names = list(robot.data.body_names)
positions = wp.to_torch(robot.data.body_pos_w)[0].detach().cpu().numpy()
quaternions = wp.to_torch(robot.data.body_quat_w)[0].detach().cpu().numpy()

print("\nbodies:")
for name in body_names:
    if any(token in name for token in ("gripper", "Left", "Right", "hand", "base_link")):
        print(f"  {name:<32} {np.round(positions[body_names.index(name)], 4)}")

import isaaclab.utils.math as math_utils  # noqa: E402
import torch  # noqa: E402

for side, tool, wrist in (("left", "gripper_center", "left_base_link"), ("right", "right_gripper_center", "right_base_link")):
    if tool not in body_names or wrist not in body_names:
        print(f"\n{side}: no {tool}/{wrist} body")
        continue
    tool_index = body_names.index(tool)
    tool_quat_xyzw = quaternions[tool_index]
    rotation = math_utils.matrix_from_quat(torch.tensor(tool_quat_xyzw).float().unsqueeze(0))[0].numpy()

    jaws = [n for n in body_names if n.startswith(f"{side}_") and ("Left" in n or "Right" in n)]
    left_jaw = [n for n in jaws if "Left" in n]
    right_jaw = [n for n in jaws if "Right" in n]
    assert left_jaw and right_jaw, f"{side}: could not find both jaws among {jaws}"

    left_centroid = np.mean([positions[body_names.index(n)] for n in left_jaw], axis=0)
    right_centroid = np.mean([positions[body_names.index(n)] for n in right_jaw], axis=0)
    separation_w = left_centroid - right_centroid
    separation_tool = rotation.T @ separation_w

    wrist_to_tool_w = positions[tool_index] - positions[body_names.index(wrist)]
    approach_tool = rotation.T @ wrist_to_tool_w

    def dominant(vector: np.ndarray) -> str:
        axis = int(np.argmax(np.abs(vector)))
        return f"{'+' if vector[axis] > 0 else '-'}{'xyz'[axis]}"

    print(f"\n{side} ({tool}), {len(left_jaw)} + {len(right_jaw)} jaw bodies")
    print(f"  jaw separation  world {np.round(separation_w, 4)}  |  tool {np.round(separation_tool, 4)}"
          f"  -> closing axis {dominant(separation_tool)}  ({np.linalg.norm(separation_w) * 1000:.1f} mm apart)")
    print(f"  wrist -> tool   world {np.round(wrist_to_tool_w, 4)}  |  tool {np.round(approach_tool, 4)}"
          f"  -> approach axis {dominant(approach_tool)}")
    print(f"  tool axes in world: x {np.round(rotation[:, 0], 2)} y {np.round(rotation[:, 1], 2)}"
          f" z {np.round(rotation[:, 2], 2)}")

simulation_app.close()
