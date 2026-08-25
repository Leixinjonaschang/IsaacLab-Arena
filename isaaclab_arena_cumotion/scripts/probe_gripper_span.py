# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Measure how wide a gripper's jaws actually open and close, and where its tool frame sits.

``probe_gripper_axes`` answers which way the jaws move; this answers how far. Both numbers decide
whether a given object can be pinched at all, and neither is in any config: the commanded gripper
target is a joint angle, and what that means in millimetres of jaw gap is a property of the
linkage. The tool frame's offset from the finger pads matters for the same reason -- a grasp pose
puts the *tool frame* on the object, so if the pads are elsewhere the fingers close beside it.

Usage::

    OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y .venv/bin/python \\
        isaaclab_arena_cumotion/scripts/probe_gripper_span.py --headless
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--env", type=str, default="agibot_make_toast")
parser.add_argument("--arm", type=str, default="right", choices=("left", "right"))
AppLauncher.add_app_launcher_args(parser)
args, _ = parser.parse_known_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import numpy as np  # noqa: E402

import isaaclab.utils.math as math_utils  # noqa: E402
import warp as wp  # noqa: E402

import isaaclab_arena_environments  # noqa: E402,F401
from isaaclab_arena.assets.registries import EnvironmentRegistry  # noqa: E402
from isaaclab_arena.cli.isaaclab_arena_cli import (  # noqa: E402
    arena_env_builder_cfg_from_argparse,
    get_isaaclab_arena_cli_parser,
)
from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder  # noqa: E402
from isaaclab_arena_cumotion.executor import ArmExecutor  # noqa: E402
from isaaclab_arena_cumotion.planner import CumotionArmPlanner  # noqa: E402

arena_args = get_isaaclab_arena_cli_parser().parse_args(["--num_envs", "1"])
factory = EnvironmentRegistry().get_component_by_name(args.env)()
arena_env = factory.build(factory._legacy_argparse_cfg_type(teleop_device=None))
builder = ArenaEnvBuilder(arena_env, arena_env_builder_cfg_from_argparse(arena_args))
env = builder.make_registered().unwrapped
env.reset()

planner = CumotionArmPlanner(env, arena_env.embodiment, arm=args.arm)
executor = ArmExecutor(env, planner)
robot = env.scene.articulations["robot"]
body_names = list(robot.data.body_names)

tool = planner.cfg.tool_frame
tool_index = body_names.index(tool)
jaws = [n for n in body_names if n.startswith(f"{args.arm}_") and ("Left" in n or "Right" in n)]
left_jaw = [n for n in jaws if "Left" in n]
right_jaw = [n for n in jaws if "Right" in n]
assert left_jaw and right_jaw, f"{args.arm}: could not find both jaws among {jaws}"
print(f"\ntool frame {tool}; jaw bodies {left_jaw} vs {right_jaw}")


def _positions() -> np.ndarray:
    return wp.to_torch(robot.data.body_pos_w)[0].detach().cpu().numpy()


def _tool_rotation() -> np.ndarray:
    quat_xyzw = wp.to_torch(robot.data.body_quat_w)[0, tool_index].detach().cpu()
    return math_utils.matrix_from_quat(quat_xyzw.float().unsqueeze(0))[0].numpy()


# The pad links are the only ones that actually touch the object; the rest of the finger chain is
# linkage. Measuring the whole chain's centroids answers a different question.
left_pad = next(n for n in left_jaw if "Pad" in n)
right_pad = next(n for n in right_jaw if "Pad" in n)


def _report(label: str) -> None:
    positions = _positions()
    rotation = _tool_rotation()
    pad_gap = np.linalg.norm(positions[body_names.index(left_pad)] - positions[body_names.index(right_pad)])
    chain_gap = np.linalg.norm(
        np.mean([positions[body_names.index(n)] for n in left_jaw], axis=0)
        - np.mean([positions[body_names.index(n)] for n in right_jaw], axis=0)
    )
    tool_position = positions[tool_index]
    # Each pad in the tool's own frame: index 1 is the jaw axis, index 2 the approach. Reporting
    # them separately rather than as a gap is the point -- a gap says the jaws met, it does not say
    # they met where the tool frame is, and a pinch is aimed at the tool frame.
    local = {n: rotation.T @ (positions[body_names.index(n)] - tool_position) for n in (left_pad, right_pad)}
    jaw = {n: float(v[1]) for n, v in local.items()}
    centre = 0.5 * (jaw[left_pad] + jaw[right_pad])
    print(
        f"  {label:<22} pad gap {pad_gap * 1000:6.1f} mm  (chain {chain_gap * 1000:6.1f} mm) |"
        f" jaw axis L {jaw[left_pad] * 1000:+6.1f} R {jaw[right_pad] * 1000:+6.1f}"
        f"  midpoint {centre * 1000:+5.1f} mm |"
        f" approach {np.mean([float(v[2]) for v in local.values()]) * 1000:+6.1f} mm"
    )


print(f"\nopen target {planner.cfg.gripper_open_pos}, closed target {planner.cfg.gripper_closed_pos}")
hold = planner.joint_positions()
for fraction in (1.0, 0.75, 0.5, 0.25, 0.0):
    target = planner.cfg.gripper_closed_pos + fraction * (planner.cfg.gripper_open_pos - planner.cfg.gripper_closed_pos)
    executor.set_gripper(target, hold_arm_at=hold)
    executor.step(arm_target=hold, steps=20)
    _report(f"{fraction:.2f} open ({target:.3f})")

print(
    "\nThe midpoint column is what a pinch is actually aimed at. If it drifts as the jaws close,"
    " the two fingers do not meet at the tool frame, and an object centred on the tool frame gets"
    " pushed by the leading jaw instead of pinched by both."
)

simulation_app.close()
