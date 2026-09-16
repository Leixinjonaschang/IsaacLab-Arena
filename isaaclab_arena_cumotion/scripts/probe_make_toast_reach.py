# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Ask which arm can reach which part of ``agibot_make_toast``, before designing the motion.

The rack sits on the robot's right and the toaster on its left, so whether one arm can carry a
slice the whole way -- or whether the two have to hand over between them -- is a reachability
question with a measurable answer. Asking it first decides the shape of the whole task; guessing
it means discovering the answer halfway through a sequence that then has to be rewritten.

Usage::

    OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y .venv/bin/python \\
        isaaclab_arena_cumotion/scripts/probe_make_toast_reach.py --headless
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--env", type=str, default="agibot_make_toast")
parser.add_argument("--settle-steps", type=int, default=60)
AppLauncher.add_app_launcher_args(parser)
args, _ = parser.parse_known_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import numpy as np  # noqa: E402
import torch  # noqa: E402

import isaaclab.utils.math as math_utils  # noqa: E402
import warp as wp  # noqa: E402

import isaaclab_arena_environments  # noqa: E402,F401
from isaaclab_arena.assets.registries import EnvironmentRegistry  # noqa: E402
from isaaclab_arena.cli.isaaclab_arena_cli import (  # noqa: E402
    arena_env_builder_cfg_from_argparse,
    get_isaaclab_arena_cli_parser,
)
from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder  # noqa: E402
from isaaclab_arena_cumotion.grasps import _rot_about, quat_wxyz_from_matrix, slab_grasps  # noqa: E402
from isaaclab_arena_cumotion.planner import CumotionArmPlanner  # noqa: E402

BREAD_BBOX_MIN_M = (-0.0584, -0.0583, -0.0003)
BREAD_BBOX_MAX_M = (0.0586, 0.0578, 0.0115)
BREAD_FACE_NORMAL_LOCAL = (0.0, 0.0, 1.0)

# Toaster slot rectangles in the toaster's own frame, from its metadata's passive.functional.
SLOTS_LOCAL = {
    "toast_slot1": ((-0.078, 0.072), (-0.041, -0.011)),
    "toast_slot2": ((-0.078, 0.072), (0.014, 0.039)),
}
SLOT_Z_LOCAL_M = -0.051

arena_args = get_isaaclab_arena_cli_parser().parse_args(["--num_envs", "1"])
factory = EnvironmentRegistry().get_component_by_name(args.env)()
arena_env = factory.build(factory._legacy_argparse_cfg_type(teleop_device=None))
builder = ArenaEnvBuilder(arena_env, arena_env_builder_cfg_from_argparse(arena_args))
env = builder.make_registered().unwrapped
env.reset()

zero_action = torch.zeros((env.num_envs, env.action_space.shape[1]), device=env.device)
for _ in range(args.settle_steps):
    env.step(zero_action)

planners = {arm: CumotionArmPlanner(env, arena_env.embodiment, arm=arm) for arm in ("left", "right")}
for arm, planner in planners.items():
    print(f"{arm} arm kinematics cross-check: {planner.kinematics_error_m() * 1000:.2f} mm")

bread_keys = sorted(k for k in env.scene.rigid_objects if k.startswith("bread") and k != "bread_shelf")


def _pose(key):
    asset = env.scene[key]
    position = wp.to_torch(asset.data.root_pos_w)[0].detach().cpu().numpy().astype(np.float64)
    quat_xyzw = wp.to_torch(asset.data.root_quat_w)[0].detach().cpu()
    rotation = math_utils.matrix_from_quat(quat_xyzw.float().unsqueeze(0))[0].numpy().astype(np.float64)
    return position, rotation


print("\n=== picking each slice out of the rack ===")
for key in bread_keys:
    proposals = slab_grasps(
        env,
        key,
        face_normal_local=BREAD_FACE_NORMAL_LOCAL,
        bbox_min_m=BREAD_BBOX_MIN_M,
        bbox_max_m=BREAD_BBOX_MAX_M,
        grasp_depth_m=(0.030,),
        lateral_offset_m=(0.0,),
        approach_offset_m=0.017,
        flip=(True,),
    )
    counts = {arm: sum(int(p.ik_reachable(q.position, q.quat_wxyz)) for q in proposals) for arm, p in planners.items()}
    print(f"  {key}: " + ", ".join(f"{arm} {n}/{len(proposals)}" for arm, n in counts.items()))

print("\n=== dropping a slice into each toaster slot ===")
toaster_pos, toaster_rot = _pose("toaster")
# The tool pose that would release a slice into a slot: same family as the grasp -- approach down,
# jaws across the slice's thickness -- positioned over the slot at a range of heights above it.
for tag, (x_range, y_range) in SLOTS_LOCAL.items():
    centre_local = np.array([0.5 * sum(x_range), 0.5 * sum(y_range), SLOT_Z_LOCAL_M])
    for height in (0.10, 0.14, 0.18, 0.22):
        point = toaster_pos + toaster_rot @ (centre_local + np.array([0.0, 0.0, height]))
        counts = {}
        for arm, planner in planners.items():
            orientations = []
            for tilt in (0, -10, -20, -30, 10, 20, 30):
                # Jaws along the toaster's own y, which is the slot's narrow axis.
                for jaw in (toaster_rot[:, 1], -toaster_rot[:, 1]):
                    approach = _rot_about(jaw, np.radians(tilt)) @ np.array([0.0, 0.0, -1.0])
                    orientation = np.column_stack([np.cross(jaw, approach), jaw, approach])
                    orientations.append(quat_wxyz_from_matrix(orientation))
            hits = sum(int(planner.ik_reachable(point, q)) for q in orientations)
            counts[arm] = f"{hits}/{len(orientations)}"
        print(f"  {tag} at {height * 1000:.0f} mm above the slot: " + ", ".join(f"{a} {n}" for a, n in counts.items()))

print("\n=== a chest-height handover region, reachable by both ===")
robot_pos = wp.to_torch(env.scene.articulations["robot"].data.root_pos_w)[0].detach().cpu().numpy().astype(np.float64)
for x in (0.20, 0.30, 0.40):
    for z in (0.95, 1.05, 1.15):
        point = robot_pos + np.array([x, 0.0, z])
        counts = {}
        for arm, planner in planners.items():
            orientations = []
            for yaw in (0, 45, 90, 135, 180, 225, 270, 315):
                c, s = np.cos(np.radians(yaw)), np.sin(np.radians(yaw))
                spin = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
                orientation = spin @ np.diag([1.0, -1.0, -1.0])
                orientations.append(quat_wxyz_from_matrix(orientation))
            hits = sum(int(planner.ik_reachable(point, q)) for q in orientations)
            counts[arm] = f"{hits}/{len(orientations)}"
        print(f"  x {x:.2f} z {z:.2f}: " + ", ".join(f"{a} {n}" for a, n in counts.items()))

# Neither arm reaching the slots where the toaster stands is a placement result, not a grasp one.
# Sweep where the toaster *could* stand and report the reach, so the layout can be fixed with a
# measured number rather than by nudging it until something works.
print("\n=== if the toaster stood at a different x (its slots' reach) ===")
for shift in (-0.16, -0.12, -0.08, -0.04, 0.0, 0.04):
    line = []
    for tag, (x_range, y_range) in SLOTS_LOCAL.items():
        centre_local = np.array([0.5 * sum(x_range), 0.5 * sum(y_range), SLOT_Z_LOCAL_M])
        point = toaster_pos + np.array([shift, 0.0, 0.0]) + toaster_rot @ (centre_local + np.array([0.0, 0.0, 0.14]))
        for arm, planner in planners.items():
            orientations = []
            for tilt in (0, -10, -20, -30, 10, 20, 30):
                for jaw in (toaster_rot[:, 1], -toaster_rot[:, 1]):
                    approach = _rot_about(jaw, np.radians(tilt)) @ np.array([0.0, 0.0, -1.0])
                    orientations.append(
                        quat_wxyz_from_matrix(np.column_stack([np.cross(jaw, approach), jaw, approach]))
                    )
            hits = sum(int(planner.ik_reachable(point, q)) for q in orientations)
            line.append(f"{tag[-1]}/{arm} {hits}")
    print(f"  toaster x {toaster_pos[0] + shift:.2f} (shift {shift * 1000:+.0f} mm): " + ", ".join(line))

# Where can the right arm actually hold a slice flat with the gripper pointing left? The staging
# pose the carry was asked for -- bearing 90, jaws down -- has no IK solution at the rack/toaster
# midpoint, and the carry has been quietly falling back to 45 deg. Sweep the neighbourhood so the
# point can be chosen from a map instead of nudged.
print("\n=== staging poses: gripper pointing left (bearing 90), jaws down, slice flat ===")
jaw_down = np.array([0.0, 0.0, -1.0])
# Bearing 90 is unreachable everywhere in this box, so report the furthest left that *is*
# reachable at each point instead of a grid of noes -- "how far left can it point" is the
# question the layout actually turns on.
BEARINGS = list(range(0, 121, 15))
for z in (0.85, 0.90, 0.95, 1.00):
    for x in (0.24, 0.30, 0.36, 0.42):
        cells = []
        for y in (-0.05, 0.0, 0.05):
            point = np.array([x, y, z])
            best = None
            for bearing in BEARINGS:
                approach = np.array([np.cos(np.radians(bearing)), np.sin(np.radians(bearing)), 0.0])
                quat = quat_wxyz_from_matrix(np.column_stack([np.cross(jaw_down, approach), jaw_down, approach]))
                if planners["right"].ik_reachable(point, quat):
                    best = bearing if best is None else max(best, bearing)
            cells.append(f"  {best:>4}" if best is not None else "   -- ")
        print(f"  z{z:.2f} x{x:.2f}  furthest-left bearing at y-0.05/0.00/+0.05:" + "".join(cells))

simulation_app.close()
