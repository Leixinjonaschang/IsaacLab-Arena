# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Re-render the camera observations of recorded demos with the current Agibot camera rig.

Kinematic playback: each recorded step's full sim state is written back into the scene
(``scene.reset_to`` + a real physics step -- ``render()`` alone leaves rigid bodies frozen) and
the three cameras are captured: the true-ego ``head_cam`` and the two D405 wrist cameras. The
images are exactly the recorded episode under the new viewpoints; actions and states are
untouched.

The streams are written as one 15 fps mp4 per (demo, camera) into a ``<hdf5>.cameras/`` sidecar
directory, NOT back into the HDF5: gzip image streams run ~400 MB per demo per camera and HDF5
never reclaims replaced datasets, so rewriting a 200-demo file in place needs ~330 GB the box
does not have -- while the mp4s (the form the LeRobot conversion ships anyway) total a few GB.
The HDF5 is opened read-only, so several workers may share one file, split by ``--demo-range``.
Finished demos are skipped on relaunch (interrupted renders leave only ``.part`` files).

The wrist cameras are world-posed every frame from the live tool poses: a camera prim parented
under a moving link does not track it (the sensor FrameView falls back to the USD-authored
transform for runtime prims), so a config-class mount cannot work. Mount numbers live in
``isaaclab_arena.embodiments.agibot.agibot``.

Usage::

    OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y .venv/bin/python \\
        isaaclab_arena_cumotion/scripts/rerender_demo_cameras.py --headless \\
        --env agibot_handover_toast \\
        --hdf5 /home/ubuntu/playground/datasets/handover_toast_v0_raw/handover_toast.hdf5 \\
        --demo-range 0 100
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--hdf5", type=str, nargs="+", required=True, help="HDF5 demo files to re-render, in place.")
parser.add_argument("--env", type=str, required=True, help="Arena environment the demos were recorded in.")
parser.add_argument(
    "--demo-range",
    type=int,
    nargs=2,
    default=None,
    metavar=("START", "END"),
    help="Half-open slice of the (index-sorted) demo list to process, for parallel workers.",
)
parser.add_argument("--force", action="store_true", help="Re-render demos that already have all three streams.")
AppLauncher.add_app_launcher_args(parser)
args, _ = parser.parse_known_args()
args.enable_cameras = True
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import h5py  # noqa: E402
import imageio.v2 as iio  # noqa: E402
import numpy as np  # noqa: E402
import pathlib  # noqa: E402
import torch  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
import isaaclab.utils.math as math_utils  # noqa: E402
from isaaclab.sensors import Camera, CameraCfg  # noqa: E402

import isaaclab_arena_environments  # noqa: E402,F401
from isaaclab_arena.assets.registries import EnvironmentRegistry  # noqa: E402
from isaaclab_arena.cli.isaaclab_arena_cli import (  # noqa: E402
    arena_env_builder_cfg_from_argparse,
    get_isaaclab_arena_cli_parser,
)
from isaaclab_arena.embodiments.agibot.agibot import (  # noqa: E402
    WRIST_CAM_MOUNTS,
    WRIST_CAM_VIEW_NUDGE_M,
    AgibotCameraCfg,
)
from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder  # noqa: E402

STREAMS = ("head_cam_rgb", "left_wrist_cam_rgb", "right_wrist_cam_rgb")
FPS = 15  # one frame per control step at Arena's 15 Hz control rate; the videos play in real time

# ------------------------------------------------------------------------------------- env ---
arena_args = get_isaaclab_arena_cli_parser().parse_args(["--num_envs", "1", "--enable_cameras"])
factory = EnvironmentRegistry().get_component_by_name(args.env)()
arena_env = factory.build(factory._legacy_argparse_cfg_type(teleop_device=None))
builder = ArenaEnvBuilder(arena_env, arena_env_builder_cfg_from_argparse(arena_args))
env = builder.make_registered().unwrapped

head_cfg = AgibotCameraCfg().head_cam


def make_cam(name: str) -> Camera:
    return Camera(
        CameraCfg(
            prim_path=f"/World/rerender_{name}",
            height=head_cfg.height,
            width=head_cfg.width,
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=head_cfg.spawn.focal_length,
                horizontal_aperture=head_cfg.spawn.horizontal_aperture,
                clipping_range=(0.02, 30.0),
            ),
        )
    )


cams = {name: make_cam(name) for name in STREAMS}

env.sim.reset()
env.reset()
robot = env.scene["robot"]
device = env.device
env_ids = torch.tensor([0], device=device)

# The head camera is rigid to the (static) base: compose its base-frame offset once per reset.
base_pos = robot.data.root_pos_w[0]
base_rot = math_utils.matrix_from_quat(robot.data.root_quat_w[0].unsqueeze(0))[0]
head_pos = base_pos + base_rot @ torch.tensor(head_cfg.offset.pos, device=device)
head_quat = math_utils.quat_mul(
    robot.data.root_quat_w[0].unsqueeze(0), torch.tensor([head_cfg.offset.rot], device=device)
)
cams["head_cam_rgb"].set_world_poses(head_pos.unsqueeze(0), head_quat, convention="opengl")


def aim_wrist_cams() -> None:
    """World-pose the wrist cameras from the live tool poses (view = tool +z, world-up roll)."""
    for side, (tool, offset) in WRIST_CAM_MOUNTS.items():
        index = robot.data.body_names.index(tool)
        tool_pos = robot.data.body_pos_w[0, index]
        tool_rot = math_utils.matrix_from_quat(robot.data.body_quat_w[0, index].unsqueeze(0))[0]
        view = tool_rot[:, 2]
        eye = tool_pos + tool_rot @ torch.tensor(offset, device=device) + WRIST_CAM_VIEW_NUDGE_M * view
        z_cam = -view
        up = torch.tensor([0.0, 0.0, 1.0], device=device)
        up = up - torch.dot(up, z_cam) * z_cam
        if torch.linalg.norm(up) < 0.1:  # vertical view: fall back to the tool -x as up
            up = -tool_rot[:, 0]
            up = up - torch.dot(up, z_cam) * z_cam
        up = up / torch.linalg.norm(up)
        x_cam = torch.linalg.cross(up, z_cam)
        x_cam = x_cam / torch.linalg.norm(x_cam)
        y_cam = torch.linalg.cross(z_cam, x_cam)
        quat = math_utils.quat_from_matrix(torch.stack([x_cam, y_cam, z_cam], dim=1).unsqueeze(0))
        cams[f"{side}_wrist_cam_rgb"].set_world_poses(eye.unsqueeze(0), quat, convention="opengl")


# -------------------------------------------------------------------------------- playback ---
def rerender_demo(demo, out_dir: pathlib.Path, demo_name: str) -> int:
    """Replay one demo's states and write its three camera streams as sidecar mp4s."""
    group = demo["states"]
    states = {
        kind: {
            asset: {field: np.array(group[kind][asset][field]) for field in group[kind][asset]} for asset in group[kind]
        }
        for kind in group
    }
    num_steps = next(iter(states["articulation"]["robot"].values())).shape[0]

    writers = {
        # Default libx264 quality (5), matching what convert_hdf5_to_lerobot.py's own encoder
        # produced: quality=8 made each video ~10x larger for no training benefit.
        name: iio.get_writer(out_dir / f"{demo_name}_{name}.part.mp4", fps=FPS, codec="libx264", macro_block_size=8)
        for name in STREAMS
    }

    def frame_state(step: int) -> dict:
        return {
            kind: {
                asset: {
                    field: torch.tensor(array[step], dtype=torch.float32, device=device).unsqueeze(0)
                    for field, array in fields.items()
                }
                for asset, fields in assets.items()
            }
            for kind, assets in states.items()
        }

    for step in range(num_steps):
        env.scene.reset_to(frame_state(step), env_ids, is_relative=True)
        aim_wrist_cams()
        env.sim.step(render=True)
        for name, cam in cams.items():
            cam.update(env.sim.get_physics_dt())
            writers[name].append_data(cam.data.output["rgb"][0, ..., :3].detach().cpu().numpy().astype(np.uint8))

    # Only completed renders carry the final name; a killed worker leaves .part files behind,
    # which the skip check ignores, so relaunching resumes cleanly.
    for name, writer in writers.items():
        writer.close()
        (out_dir / f"{demo_name}_{name}.part.mp4").rename(out_dir / f"{demo_name}_{name}.mp4")
    return num_steps


for path in args.hdf5:
    out_dir = pathlib.Path(f"{path}.cameras")
    out_dir.mkdir(parents=True, exist_ok=True)
    # Read-only and unlocked: the images go to the sidecar directory and nothing writes the
    # HDF5, so several workers may share one file (HDF5's default lock would bar the second).
    with h5py.File(path, "r", locking=False) as handle:
        names = sorted(handle["data"], key=lambda name: int(name.split("_")[-1]))
        if args.demo_range is not None:
            names = names[args.demo_range[0] : args.demo_range[1]]
        print(f"{path}: {len(names)} demos -> {out_dir}")
        for i, name in enumerate(names):
            if not args.force and all((out_dir / f"{name}_{stream}.mp4").exists() for stream in STREAMS):
                print(f"  [{i + 1}/{len(names)}] {name}: already rendered, skipping")
                continue
            steps = rerender_demo(handle[f"data/{name}"], out_dir, name)
            print(f"  [{i + 1}/{len(names)}] {name}: {steps} steps rendered", flush=True)

print("all files done")
simulation_app.close()
