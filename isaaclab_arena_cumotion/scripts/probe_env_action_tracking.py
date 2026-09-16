# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Measure how well the joint-action path tracks, against the direct-write executor.

The recorded handover demos descend to within 50 mm of the grasp where the direct-write executor
lands within 0.1 mm, and both send the same planned joint paths. This drives one arm to the same
pose through both paths in one process and prints the per-joint error of each, which says whether
the gap is in the action plumbing (one joint mismapped or not driven) or in the dynamics (all
joints lagging a little).

Usage::

    OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y .venv/bin/python \\
        isaaclab_arena_cumotion/scripts/probe_env_action_tracking.py --headless
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--env", type=str, default="agibot_handover_toast")
parser.add_argument("--arm", type=str, default="right")
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
from isaaclab_arena.embodiments.agibot.agibot import AgibotDualArmJointActionsCfg  # noqa: E402
from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder  # noqa: E402
from isaaclab_arena.environments.isaaclab_arena_manager_based_env_cfg import set_control_rate_50hz  # noqa: E402
from isaaclab_arena_cumotion.executor import ArmExecutor, EnvActionExecutor, JointActionInterface  # noqa: E402
from isaaclab_arena_cumotion.planner import CumotionArmPlanner  # noqa: E402

arena_args = get_isaaclab_arena_cli_parser().parse_args(["--num_envs", "1"])
factory = EnvironmentRegistry().get_component_by_name(args.env)()
arena_env = factory.build(factory._legacy_argparse_cfg_type(teleop_device=None))
arena_env.embodiment.action_config = AgibotDualArmJointActionsCfg()
_prev_cb = arena_env.env_cfg_callback


def _cfg(patched):
    patched = _prev_cb(patched) if _prev_cb is not None else patched
    patched = set_control_rate_50hz(patched)
    patched.terminations.success = None
    return patched


arena_env.env_cfg_callback = _cfg
builder = ArenaEnvBuilder(arena_env, arena_env_builder_cfg_from_argparse(arena_args))
env = builder.make_registered().unwrapped
env.sim.reset()
env.reset()

robot = env.scene.articulations["robot"]
manager = env.action_manager
print("\naction terms:")
for name, dim in zip(manager.active_terms, manager.action_term_dim):
    term = manager.get_term(name)
    print(f"  {name}: dim {dim}, joints {getattr(term, '_joint_names', None)}")

planner = CumotionArmPlanner(env, arena_env.embodiment, arm=args.arm)
print(f"\nplanner arm joints: {planner.cfg.arm_joint_names}")
print(f"planner gripper joint ids: {planner.gripper_joint_ids}")

interface = JointActionInterface(env)
env_executor = EnvActionExecutor(
    env, planner, interface, f"{args.arm}_arm_action", f"{args.arm}_gripper_action", on_step=None
)
direct_executor = ArmExecutor(env, planner, on_step=None)

# The probe target is a real pregrasp pose -- the same generator and stand-off the pick uses --
# because a hand-authored orientation is too easy to get wrong for this tool (see the grasp-quat
# conventions in grasps.py), and the point here is tracking, not pose authoring.
from isaaclab_arena_cumotion.grasps import slab_grasps  # noqa: E402

env_executor.step(steps=15)  # let the rack settle a moment before measuring
bread_keys = sorted(key for key in env.scene.rigid_objects if key.startswith("bread") and key != "bread_shelf")
positions = {key: wp.to_torch(env.scene[key].data.root_pos_w)[0].cpu().numpy() for key in bread_keys}
target_key = min(bread_keys, key=lambda k: positions[k][1])
proposals = slab_grasps(
    env,
    target_key,
    face_normal_local=(0.0, 0.0, 1.0),
    bbox_min_m=(-0.0584, -0.0583, -0.0003),
    bbox_max_m=(0.0586, 0.0578, 0.0115),
    grasp_depth_m=(0.030,),
    lateral_offset_m=(0.0, -0.025, 0.025),
    approach_offset_m=0.017,
    flip=(True,),
)
plan = None
for proposal in proposals:
    goal = proposal.position + np.array([0.0, 0.0, 0.12])
    if not planner.ik_reachable(goal, proposal.quat_wxyz):
        continue
    plan = planner.plan_pose(planner.joint_positions(), goal, proposal.quat_wxyz)
    if plan is not None and plan.is_executable():
        print(f"probing with the pregrasp of '{proposal.label}'")
        break
    plan = None
assert plan is not None, "no pregrasp pose was plannable"
q_goal = plan.path.get_waypoints().numpy().astype(np.float64)[-1]


def report(label: str) -> None:
    """Per-joint tracking error against the plan's final configuration, and the tool error."""
    q_now = planner.joint_positions()
    errors = np.degrees(np.abs(q_now - q_goal))
    tool_mm = float(np.linalg.norm(planner.tool_position() - goal)) * 1000.0
    print(f"\n{label}:")
    for name, err in zip(planner.cfg.arm_joint_names, errors):
        flag = "   <-- OFF" if err > 2.0 else ""
        print(f"  {name:<18} {err:7.2f} deg{flag}")
    print(f"  tool {tool_mm:.1f} mm from the commanded pose")


print(f"\nprobe goal: {np.round(goal, 4)} above {target_key}")

# Path one: through env.step and the action manager.
env_executor.follow(plan.path, speed=0.25)
report("via env.step (joint actions)")


def gripper_state(label: str) -> None:
    """The gripper's commanded-vs-actual joint positions and the pad gap."""
    joint_pos = wp.to_torch(robot.data.joint_pos)[0].detach().cpu().numpy()
    actual = joint_pos[planner.gripper_joint_ids]
    body_names = list(robot.data.body_names)
    pads = [n for n in body_names if n.startswith(f"{args.arm}_") and "Pad" in n]
    body_pos = wp.to_torch(robot.data.body_pos_w)[0].detach().cpu().numpy()
    gap_mm = float(np.linalg.norm(body_pos[body_names.index(pads[0])] - body_pos[body_names.index(pads[1])])) * 1000
    print(f"  {label}: joints {np.round(actual, 4)}, pad gap {gap_mm:.1f} mm")


# The gripper through both paths, in free air at the pregrasp: close fully, report, reopen.
print("\ngripper close via env.step:")
q_here = planner.joint_positions()
env_executor.close_gripper(hold_arm_at=q_here)
gripper_state("closed")
env_executor.open_gripper(hold_arm_at=q_here)
gripper_state("reopened")

print("gripper close via direct writes:")
direct_executor.close_gripper(hold_arm_at=q_here)
gripper_state("closed")
direct_executor.open_gripper(hold_arm_at=q_here)
gripper_state("reopened")

# Finally a real pick through env.step: descend onto the slice, close, lift, and see if it comes.
print("\npick via env.step:")
descend = planner.plan_pose(planner.joint_positions(), proposal.position, proposal.quat_wxyz)
assert descend is not None and descend.is_executable(), "could not plan the probe descent"
start_z = float(positions[target_key][2])
env_executor.set_gripper(0.25 * planner.cfg.gripper_open_pos, hold_arm_at=q_here)
env_executor.follow(descend.path, speed=0.25)
q_grasp = planner.joint_positions()
gripper_state("at the grasp, before closing")
env_executor.close_gripper(hold_arm_at=q_grasp)
gripper_state("closed on the slice")
env_executor.follow(descend.path, reverse=True, speed=0.15)
gripper_state("after the lift")
rise_mm = (float(wp.to_torch(env.scene[target_key].data.root_pos_w)[0, 2].item()) - start_z) * 1000
print(f"  slice rose {rise_mm:+.1f} mm")

simulation_app.close()
