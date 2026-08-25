# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Measure how ``agibot_make_toast`` actually settles, and which way its lever travels.

The layout is authored from asset metadata -- slot corners, rack support frames, bounding boxes --
and none of that says where things end up once physics has run, nor whether pushing the lever
raises or lowers the joint that the success check reads. Both are directly observable, so they
are measured here.

Usage::

    OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y .venv/bin/python \\
        isaaclab_arena_cumotion/scripts/probe_make_toast.py --headless
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--env", type=str, default="agibot_make_toast")
parser.add_argument("--settle_steps", type=int, default=60)
parser.add_argument("--shot", type=str, default=None, help="Write a PNG of the settled scene here.")
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

arena_args = get_isaaclab_arena_cli_parser().parse_args(["--num_envs", "1"])
factory = EnvironmentRegistry().get_component_by_name(args.env)()
arena_env = factory.build(factory._legacy_argparse_cfg_type(teleop_device=None))
builder = ArenaEnvBuilder(arena_env, arena_env_builder_cfg_from_argparse(arena_args))
env = builder.make_registered().unwrapped
env.reset()


def _to_numpy(value):
    return (wp.to_torch(value) if not isinstance(value, torch.Tensor) else value).detach().cpu().numpy()


def _pose(name):
    asset = env.scene[name]
    return _to_numpy(asset.data.root_pos_w)[0], _to_numpy(asset.data.root_quat_w)[0]


camera = None
if args.shot is not None:
    import isaaclab.sim as sim_utils  # noqa: E402
    from isaaclab.sensors import Camera, CameraCfg  # noqa: E402

    camera = Camera(
        CameraCfg(
            prim_path="/World/probe_cam",
            height=720,
            width=1280,
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(focal_length=20.0, clipping_range=(0.05, 30.0)),
        )
    )
    env.sim.reset()
    env.reset()
    camera.set_world_poses_from_view(
        eyes=torch.tensor([[-0.35, -0.95, 1.35]], device=env.device),
        targets=torch.tensor([[0.42, 0.00, 0.70]], device=env.device),
    )

zero_action = torch.zeros((env.num_envs, env.action_space.shape[1]), device=env.device)
for _ in range(args.settle_steps):
    env.step(zero_action)

if camera is not None:
    import pathlib  # noqa: E402

    import imageio.v3 as iio  # noqa: E402

    camera.update(dt=env.physics_dt, force_recompute=True)
    shot = pathlib.Path(args.shot)
    shot.parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(shot, _to_numpy(camera.data.output["rgb"])[0][..., :3].astype(np.uint8))
    print(f"\nwrote {shot}")

print("\n=== settled poses (world) ===")
names = [name for name in env.scene.rigid_objects] + [name for name in env.scene.articulations if name != "robot"]
for name in sorted(names):
    position, quaternion = _pose(name)
    print(f"  {name:<14} pos={np.round(position, 4)}  quat={np.round(quaternion, 4)}")

print("\n=== bread, in the toaster's frame ===")
toaster_position, toaster_quaternion = _pose("toaster")
# Isaac Lab 3.0 reports root_quat_w as (x, y, z, w), which is what matrix_from_quat consumes.
import isaaclab.utils.math as math_utils  # noqa: E402

rotation = math_utils.matrix_from_quat(torch.tensor(toaster_quaternion).unsqueeze(0))[0].cpu().numpy()
for name in sorted(n for n in env.scene.rigid_objects if n.startswith("bread")):
    position, _ = _pose(name)
    print(f"  {name:<14} local={np.round(rotation.T @ (position - toaster_position), 4)}")

print("\n=== toaster joints ===")
toaster = env.scene.articulations["toaster"]
joint_names = list(toaster.data.joint_names)
joint_positions = _to_numpy(toaster.data.joint_pos)[0]
lower = _to_numpy(toaster.data.joint_pos_limits)[0, :, 0]
upper = _to_numpy(toaster.data.joint_pos_limits)[0, :, 1]
for index, name in enumerate(joint_names):
    ratio = (joint_positions[index] - lower[index]) / (upper[index] - lower[index])
    print(
        f"  {name:<12} pos={joint_positions[index]: .4f}  limits=[{lower[index]: .4f},{upper[index]: .4f}] "
        f" ratio={ratio:.3f}"
    )

# Which way does the lever body travel as its joint runs to each limit? Drive the joint directly
# and watch link_1 in the toaster's own frame -- that settles the polarity the success check needs.
lever_index = joint_names.index("joint_1")
body_names = list(toaster.data.body_names)
lever_body = body_names.index("link_1")
for label, target in (("lower limit", lower[lever_index]), ("upper limit", upper[lever_index])):
    state = _to_numpy(toaster.data.joint_pos).copy()
    state[:, lever_index] = target
    toaster.write_joint_state_to_sim(
        torch.tensor(state, device=env.device), torch.zeros_like(torch.tensor(state, device=env.device))
    )
    for _ in range(5):
        env.step(zero_action)
    lever_world = _to_numpy(toaster.data.body_pos_w)[0][lever_body]
    print(f"  lever at {label:<12} local={np.round(rotation.T @ (lever_world - toaster_position), 4)}")


# Does the success check actually fire on a solved state? Nothing else in this pipeline exercises
# it: teleoperation reaching the goal is the only other way to find out, and a check that can
# never be satisfied looks exactly like a task that is merely hard.
from isaaclab_arena.tasks.predicates.spatial import objects_upright_about_any_axis  # noqa: E402

task = arena_env.task
bread_names = task.bread_names
print("\n=== success terms, as the scene stands ===")


def _report():
    slot1 = task._slot_loaded(env, "toast_slot1")[0].item()
    slot2 = task._slot_loaded(env, "toast_slot2")[0].item()
    upright = objects_upright_about_any_axis(env, bread_names, task.upright_threshold_rad)[0].item()
    in_shelf = task._slices_remaining_in_shelf(env)[0].item()
    pressed = task._lever_down(env)[0].item()
    print(
        f"  slot1={slot1}  slot2={slot2}  all_upright={upright}"
        f"  {task.slices_left_in_shelf}_in_shelf={in_shelf}  lever_pressed={pressed}"
        f"  -> success={task.is_success(env)[0].item()}"
    )
    toaster_now, toaster_quat_now = _pose("toaster")
    rotation_now = math_utils.matrix_from_quat(torch.tensor(toaster_quat_now).unsqueeze(0))[0].cpu().numpy()
    for name in bread_names:
        position, _ = _pose(name)
        print(f"    {name} toaster-local={np.round(rotation_now.T @ (position - toaster_now), 4)}")
    print(f"    joint_1={_to_numpy(toaster.data.joint_pos)[0][lever_index]: .4f}")


_report()

# Now stage the solved state directly: two slices dropped into the two slots, lever pushed home.
print("\n=== success terms, with the task staged as solved ===")
slot_centres = {
    tag: (0.5 * (x_range[0] + x_range[1]), 0.5 * (y_range[0] + y_range[1]))
    for tag, (x_range, y_range) in type(task.toaster).SLOT_RECT_LOCAL_M.items()
}
slot_z = type(task.toaster).SLOT_Z_LOCAL_M + 0.5 * (task.slot_z_lower_m + task.slot_z_upper_m)
for name, tag in zip(bread_names[:2], ("toast_slot1", "toast_slot2")):
    local = np.array([*slot_centres[tag], slot_z])
    world = toaster_position + rotation @ local
    bread = env.scene[name]
    root_state = _to_numpy(bread.data.root_state_w).copy()
    root_state[0, :3] = world
    bread.write_root_state_to_sim(torch.tensor(root_state, device=env.device, dtype=torch.float32))

lever_state = _to_numpy(toaster.data.joint_pos).copy()
lever_state[:, lever_index] = lower[lever_index] + 0.95 * (upper[lever_index] - lower[lever_index])
lever_tensor = torch.tensor(lever_state, device=env.device, dtype=torch.float32)
toaster.write_joint_state_to_sim(lever_tensor, torch.zeros_like(lever_tensor))
# Refresh the read buffers without advancing physics: a single step would drop the slices under
# gravity before anything got read, which is a property of the probe, not of the predicate.
env.scene.write_data_to_sim()
env.scene.update(env.physics_dt)
_report()

simulation_app.close()
