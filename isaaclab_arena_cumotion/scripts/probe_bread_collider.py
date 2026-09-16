# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Measure how thick a slice of bread's *collider* is, as against its visual mesh.

The usdz authors a 11.8 mm slab and approximates it with a convex hull, but what PhysX ends up
simulating is not readable from the asset: thin meshes get inflated to meet the GPU convex-hull
minimum, and the collision offsets add more on top. The difference matters -- a grasp is aimed at
the visual geometry, and the jaws load against the collider.

Dropping the slice flat on a known surface measures the answer directly: a slab of thickness ``t``
resting on a surface puts its centre ``t/2`` above it.

Usage::

    OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y .venv/bin/python \\
        isaaclab_arena_cumotion/scripts/probe_bread_collider.py --headless
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--env", type=str, default="agibot_make_toast")
parser.add_argument("--slice", type=str, default="bread0")
parser.add_argument("--drop-height", type=float, default=0.05, help="Height above the table to drop from, in m.")
parser.add_argument("--settle-steps", type=int, default=200)
AppLauncher.add_app_launcher_args(parser)
args, _ = parser.parse_known_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import numpy as np  # noqa: E402
import torch  # noqa: E402

import warp as wp  # noqa: E402

import isaaclab_arena_environments  # noqa: E402,F401
from isaaclab_arena.assets.registries import EnvironmentRegistry  # noqa: E402
from isaaclab_arena.cli.isaaclab_arena_cli import (  # noqa: E402
    arena_env_builder_cfg_from_argparse,
    get_isaaclab_arena_cli_parser,
)
from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder  # noqa: E402

TABLE_TOP_Z = 0.6232
VISUAL_THICKNESS_M = 0.0118

arena_args = get_isaaclab_arena_cli_parser().parse_args(["--num_envs", "1"])
factory = EnvironmentRegistry().get_component_by_name(args.env)()
arena_env = factory.build(factory._legacy_argparse_cfg_type(teleop_device=None))
builder = ArenaEnvBuilder(arena_env, arena_env_builder_cfg_from_argparse(arena_args))
env = builder.make_registered().unwrapped
env.reset()

zero_action = torch.zeros((env.num_envs, env.action_space.shape[1]), device=env.device)


def _position(key: str) -> np.ndarray:
    return wp.to_torch(env.scene[key].data.root_pos_w)[0].detach().cpu().numpy().astype(np.float64)


# Lay the slice flat on open table, well clear of the rack, and let it fall. Identity orientation
# puts its 11.8 mm axis vertical, which is what makes the resting height a thickness measurement.
asset = env.scene[args.slice]
root_state = wp.to_torch(asset.data.root_state_w).detach().cpu().numpy().copy()
root_state[0, :3] = [0.30, 0.30, TABLE_TOP_Z + args.drop_height]
root_state[0, 3:7] = [0.0, 0.0, 0.0, 1.0]
root_state[0, 7:] = 0.0
asset.write_root_state_to_sim(torch.tensor(root_state, device=env.device, dtype=torch.float32))
for _ in range(args.settle_steps):
    env.step(zero_action)

resting = _position(args.slice)
half = resting[2] - TABLE_TOP_Z
print(f"\nslice {args.slice} resting at z={resting[2]:.4f}, table top {TABLE_TOP_Z:.4f}")
print(f"  centre sits {half * 1000:.1f} mm above the surface")
print(f"  => collider is {2 * half * 1000:.1f} mm thick, against a {VISUAL_THICKNESS_M * 1000:.1f} mm visual slab")
print(f"  inflation: {(2 * half - VISUAL_THICKNESS_M) * 1000:+.1f} mm")

simulation_app.close()
