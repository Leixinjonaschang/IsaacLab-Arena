# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Demonstrate ``agibot_handover_toast`` with cuMotion-planned motions, and record HDF5 demos.

One unconditional sequence -- pick a slice from the rack with the right arm, carry it flat over
the toaster, pass it to the left arm, and withdraw the right -- ending exactly where the task's
own success predicate looks: the slice held at the left gripper, the other slices still racked.
This is ``make_toast_cumotion.py`` cut down to the handover task's scope, with the staged flags
and their tuned defaults baked in; see that script for how each value was measured.

With ``--record-dir`` the run is captured as an ``record_demos.py``-compatible HDF5 dataset. The
arms are then driven through ``env.step`` with joint-space actions (``EnvActionExecutor``) so
Isaac Lab's recorder hooks see every step; without it, joint targets are written straight to the
articulation as before.

Usage::

    OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y .venv/bin/python \\
        isaaclab_arena_cumotion/scripts/handover_toast_cumotion.py --headless \\
        --record-dir /home/ubuntu/playground/datasets/handover_toast --num-demos 3 \\
        --video /home/ubuntu/playground/make_toast/handover.mp4
"""

import argparse
import re

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--env", type=str, default="agibot_handover_toast")
parser.add_argument("--slice", type=str, default=None, help="Scene key to pick; the rightmost by default.")
parser.add_argument("--video", type=str, default=None)
parser.add_argument("--fps", type=int, default=30)
parser.add_argument("--record-every", type=int, default=4, help="Keep one video frame in N physics steps.")
parser.add_argument("--max-frames", type=int, default=2400)
parser.add_argument("--settle-seconds", type=float, default=4.0, help="Time to let the rack settle per demo.")
parser.add_argument(
    "--record-dir",
    type=str,
    default=None,
    help=(
        "Directory to write the HDF5 demonstration dataset into. Enables the embodiment's cameras"
        " and drives the arms through env.step with joint-space actions, so the recorded episodes"
        " carry actions, robot state and camera observations."
    ),
)
parser.add_argument("--dataset-name", type=str, default="handover_toast", help="HDF5 file name, without extension.")
parser.add_argument("--num-demos", type=int, default=1, help="How many demonstrations to run (and record).")
parser.add_argument(
    "--seed",
    type=int,
    default=None,
    help=(
        "Seed for the per-reset scene jitter; drawn from the system's entropy when omitted."
        " It must NOT be left at Isaac Lab's fixed default: every process then samples the same"
        " jitter sequence, and parallel collection workers produce byte-identical demonstrations."
    ),
)
AppLauncher.add_app_launcher_args(parser)
args, _ = parser.parse_known_args()
args.enable_cameras = True
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import numpy as np  # noqa: E402
import os  # noqa: E402
import pathlib  # noqa: E402
import torch  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
import warp as wp  # noqa: E402
from isaaclab.managers.recorder_manager import DatasetExportMode  # noqa: E402
from isaaclab.sensors import Camera, CameraCfg  # noqa: E402

import isaaclab_arena_environments  # noqa: E402,F401
from isaaclab_arena.assets.registries import EnvironmentRegistry  # noqa: E402
from isaaclab_arena.cli.isaaclab_arena_cli import (  # noqa: E402
    arena_env_builder_cfg_from_argparse,
    get_isaaclab_arena_cli_parser,
)
from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder  # noqa: E402
from isaaclab_arena.utils.isaaclab_utils.recorders import ArenaEnvRecorderManagerCfg  # noqa: E402
from isaaclab_arena_cumotion.executor import ArmExecutor, EnvActionExecutor, JointActionInterface  # noqa: E402
from isaaclab_arena_cumotion.grasps import matrix_from_quat_wxyz, quat_wxyz_from_matrix, slab_grasps  # noqa: E402
from isaaclab_arena_cumotion.pick_place import PickAndPlace  # noqa: E402
from isaaclab_arena_cumotion.planner import CumotionArmPlanner  # noqa: E402

# Slice, rack and toaster geometry -- shared with make_toast_cumotion, measured there.
BREAD_EXTENTS_M = (0.1169, 0.1161, 0.0118)
BREAD_FACE_NORMAL_LOCAL = (0.0, 0.0, 1.0)
BREAD_BBOX_MIN_M = (-0.0584, -0.0583, -0.0003)
BREAD_BBOX_MAX_M = (0.0586, 0.0578, 0.0115)
SHELF_EXTENTS_M = (0.1583, 0.0835, 0.0762)
TOASTER_EXTENTS_M = (0.2267, 0.1554, 0.1619)

TABLE_TOP_Z = 0.6232
TABLE_OBSTACLE = "/obstacles/table"

# The settled values from make_toast_cumotion's staged runs, baked in: this script exists to
# reproduce one known-good demonstration shape, not to search for one.
GIVING_ARM = "right"  # the rack is right-arm-only; the toaster side, and so the receiver, is left
PREGRASP_HEIGHT_M = 0.12
PREGRASP_OPEN_FRACTION = 0.25  # 30.3 mm pad gap: fits the 25-33 mm inter-slice gaps on the way down
# Denser than make_toast's (0, +/-25): which offsets are IK-reachable shifts with how the rack
# happens to settle, and at 50 Hz the +/-25 mm ones were repeatedly all that was left. A 25 mm
# offset pinch is torque-loaded enough to shed the slice during the lift or the carry, so the
# sweep offers near-centre steps for the reachability to land on instead. The 35 mm depth is a
# firmer fallback pinch; 40 would put the pads level with the rack's rim.
GRASP_DEPTHS_M = (0.030, 0.035)
GRASP_OFFSETS_M = (0.0, -0.010, 0.010, -0.020, 0.020, -0.025, 0.025)
PAD_OFFSET_M = 0.017  # the pads sit this far behind the tool frame along the approach
LIFT_THRESHOLD_M = 0.02
MAX_CANDIDATES = 40
CARRY_XY = (0.3525, 0.0383)  # over the toaster, read off a recording's pose trace
CARRY_Z = 0.886
CARRY_TRAVEL_Z = 1.02  # cross this high or the hanging slice catches the 0.785 m toaster
CARRY_AZIMUTH_DEG = 53.0  # a full 90 has no IK solution with the slice held flat
CARRY_JAWS = np.array([0.0, 0.0, -1.0])  # jaws down puts the giving wrist camera outward
HANDOVER_JAWS = np.array([0.0, 0.0, 1.0])  # the left tool frame is relabelled, so "up" looks out
HANDOVER_STANDOFF_M = 0.06  # chest is only reachable to ~1.05, so no rack-sized clearance here
RELEASE_RETREAT_LADDER_M = (0.30, 0.25, 0.20, 0.15, 0.10)

# Playback speeds, as fractions of cuMotion's time-optimal trajectory. These are the values the
# first validated run (handover_toast_v1) used, kept at the user's direction: a 3x speed-up of
# the loaded legs was tried and coincided with the slice repeatedly working out of the pinch, so
# the demonstration crawls rather than gambles. The slice is a 12 mm slab held in a pinch; the
# close-in onto it stays slow because millimetres matter there, not seconds.
CLIMB_SPEED = 0.10
CROSS_SPEED = 0.06
LOWER_SPEED = 0.06
REACH_SPEED = 0.10
CLOSE_IN_SPEED = 0.06
RETREAT_SPEED = 0.12

# ------------------------------------------------------------------------------------- env ---
recording = args.record_dir is not None
arena_args = get_isaaclab_arena_cli_parser().parse_args(["--num_envs", "1", "--enable_cameras"])
factory = EnvironmentRegistry().get_component_by_name(args.env)()
# Cameras stay off even when recording: the raw HDF5 carries actions and per-step sim states
# only, and every image stream is re-rendered offline from those states (rerender_demo_cameras.py).
arena_env = factory.build(factory._legacy_argparse_cfg_type(teleop_device=None, enable_cameras=False))
embodiment_type = type(arena_env.embodiment)

if recording:
    from isaaclab_arena.embodiments.agibot.agibot import AgibotDualArmJointActionsCfg  # noqa: E402

    # Joint-space actions, so cuMotion's planned joint paths pass through the action manager
    # verbatim instead of being re-solved (and fought) by RMPFlow.
    arena_env.embodiment.action_config = AgibotDualArmJointActionsCfg()

    _prev_cb = arena_env.env_cfg_callback

    def _recording_env_cfg(patched):
        """Attach the demo recorder, and take the success termination out of the env's hands.

        The success predicate fires the moment the left hand holds the slice; as a termination it
        would auto-reset the env mid-recording and the episode would be exported (or dropped)
        before the giving hand has withdrawn. The script checks the predicate itself and exports
        explicitly, exactly as Isaac Lab's record_demos.py does.

        Control stays at Arena's 15 Hz default on purpose: that rate was chosen upstream to match
        the training side (DROID's 15 Hz control data; pi0.5/cosmos measured better on it), so
        demos recorded here are directly comparable. An early recording run seemed to implicate
        15 Hz -- it dropped the slice mid-carry -- but that traced to a settle-time bug, not the
        rate.
        """
        patched = _prev_cb(patched) if _prev_cb is not None else patched
        patched.recorders = ArenaEnvRecorderManagerCfg()
        # Cameras are off in states-only recording; the camera-obs recorder term would KeyError.
        patched.recorders.record_pre_step_flat_camera_observations = None
        patched.recorders.dataset_export_dir_path = args.record_dir
        patched.recorders.dataset_filename = args.dataset_name
        patched.recorders.dataset_export_mode = DatasetExportMode.EXPORT_SUCCEEDED_ONLY
        patched.terminations.success = None
        # A retry-heavy demonstration can outlast the task's teleop-sized episode length, and a
        # time_out mid-demo auto-resets the env and destroys the recording. Success is judged and
        # exported by the script, so the timeout's only job here is to never fire.
        patched.episode_length_s = 600.0
        return patched

    arena_env.env_cfg_callback = _recording_env_cfg

builder = ArenaEnvBuilder(arena_env, arena_env_builder_cfg_from_argparse(arena_args))
env = builder.make_registered().unwrapped
DECIMATION = max(1, round(env.step_dt / env.sim.get_physics_dt()))
if recording:
    # One executor step is now DECIMATION physics steps, so keep the video rate unchanged.
    args.record_every = max(1, round(args.record_every / DECIMATION))
# Settling is wall-clock physics, so it is stated in seconds and converted at the env's control
# rate. The first cut rescaled a step count by the decimation instead, which cut the recording
# mode's settle from 4 s to 0.3 s -- the grasps were then authored against slices still sliding
# into place, and every descent stalled ~50 mm short on a slice that had moved.
SETTLE_STEPS = max(1, round(args.settle_seconds / env.step_dt))

# The robot's own head view, so recordings frame every run the same way the teleop viewport does.
camera = Camera(
    CameraCfg(
        prim_path="/World/head_cam",
        height=720,
        width=1280,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(focal_length=18.147, horizontal_aperture=20.955, clipping_range=(0.05, 30.0)),
    )
)
# Reseed AFTER the env is built: Isaac Lab seeds the global RNGs with the env cfg's fixed 42
# during construction, which would give every process -- and so every parallel collection
# worker -- the same per-reset jitter sequence.
_seed = args.seed if args.seed is not None else int.from_bytes(os.urandom(4), "little")
torch.manual_seed(_seed)
np.random.seed(_seed % 2**32)
print(f"scene jitter seed: {_seed}")

env.sim.reset()
env.reset()
_robot = env.scene.articulations["robot"]
_head = (
    wp.to_torch(_robot.data.body_pos_w)[0, list(_robot.data.body_names).index(embodiment_type.HEAD_BODY_NAME)]
    .detach()
    .cpu()
    .numpy()
    .astype(np.float64)
)
camera.set_world_poses_from_view(
    eyes=torch.tensor([(_head + np.array(embodiment_type.HEAD_VIEW_EYE)).tolist()], device=env.device),
    targets=torch.tensor([(_head + np.array(embodiment_type.HEAD_VIEW_LOOKAT)).tolist()], device=env.device),
)

# Frames stream straight to the container; accumulating them in RAM gets the run OOM-killed.
writer = None
if args.video is not None:
    import imageio.v2 as iio  # noqa: E402

    video_path = pathlib.Path(args.video)
    video_path.parent.mkdir(parents=True, exist_ok=True)
    writer = iio.get_writer(video_path, fps=args.fps, codec="libx264", macro_block_size=8)

step_counter = [0]
frame_counter = [0]
truncated = [False]


def grab_frame() -> None:
    """Keep one camera frame in ``--record-every``; the servo runs far faster than 30 fps."""
    step_counter[0] += 1
    if writer is None or step_counter[0] % args.record_every != 0:
        return
    if frame_counter[0] >= args.max_frames:
        truncated[0] = True
        return
    camera.update(env.sim.get_physics_dt())
    writer.append_data(camera.data.output["rgb"][0, ..., :3].detach().cpu().numpy().astype(np.uint8))
    frame_counter[0] += 1


def object_position(key: str) -> np.ndarray:
    """An object's world position."""
    return wp.to_torch(env.scene[key].data.root_pos_w)[0].detach().cpu().numpy().astype(np.float64)


def object_pose(key: str):
    """An object's world position and rotation matrix."""
    import isaaclab.utils.math as math_utils

    asset = env.scene[key]
    position = wp.to_torch(asset.data.root_pos_w)[0].detach().cpu().numpy().astype(np.float64)
    quat_xyzw = wp.to_torch(asset.data.root_quat_w)[0].detach().cpu()
    rotation = math_utils.matrix_from_quat(quat_xyzw.float().unsqueeze(0))[0].numpy().astype(np.float64)
    return position, rotation


# -------------------------------------------------------------------------------- planners ---
embodiment = arena_env.embodiment
receiving_arm = "left" if GIVING_ARM == "right" else "right"
planners: dict[str, CumotionArmPlanner] = {}
for arm in (GIVING_ARM, receiving_arm):
    planner = CumotionArmPlanner(env, embodiment, arm=arm)
    error_m = planner.kinematics_error_m()
    print(f"{arm} arm kinematics cross-check: {error_m * 1000:.2f} mm")
    assert error_m < 1e-3, f"cuMotion's {arm}-arm kinematics disagree with the simulated robot"
    # The work surface, modelled 30 mm below the real top -- see stack_bowls_cumotion for why a
    # flush slab leaves near-table grasps unplannable while still blocking sweeps through the table.
    planner.add_box_obstacle(
        TABLE_OBSTACLE, np.array([0.185, 0.0, TABLE_TOP_Z - 0.09]), (2.2, 1.4, 0.12), safety_tolerance_m=0.0
    )
    planners[arm] = planner
giving, receiving = planners[GIVING_ARM], planners[receiving_arm]

if recording:
    interface = JointActionInterface(env)
    executors = {
        arm: EnvActionExecutor(
            env, planners[arm], interface, f"{arm}_arm_action", f"{arm}_gripper_action", on_step=grab_frame
        )
        for arm in planners
    }
else:
    interface = None
    executors = {arm: ArmExecutor(env, planners[arm], on_step=grab_frame) for arm in planners}
giving_executor, receiving_executor = executors[GIVING_ARM], executors[receiving_arm]
pick_place = PickAndPlace(giving, giving_executor, contact_obstacles=(TABLE_OBSTACLE,))

scene_obstacles: list[str] = []  # filled on the first demo, once the rack has settled


def settle(steps: int) -> None:
    """Let physics settle without commanding a motion.

    When recording this must go through the executor: a raw zero action in joint-space mode would
    command every joint to zero and the arms would sweep the table.
    """
    if recording:
        giving_executor.step(steps=steps)
    else:
        zero_action = torch.zeros((env.num_envs, env.action_space.shape[1]), device=env.device)
        for _ in range(steps):
            env.step(zero_action)
            grab_frame()


def run_demo(bread_keys: list[str], target: str) -> tuple[bool, str]:
    """Pick ``target``, carry it to the staging pose, hand it over and withdraw.

    Returns whether the demonstration reached its end, and a one-line reason when it did not.
    The task predicate is judged by the caller; this only reports how far the motion got.
    """
    pick = _pick_slice(bread_keys, target)
    if not pick.success:
        return False, f"pick failed: {pick.failure}"
    carry_quat = _carry_to_staging(pick, target)
    if carry_quat is None:
        return False, "the slice never arrived flat at the staging pose"
    q_taking = _hand_over(carry_quat, target)
    if q_taking is None:
        return False, "the receiving arm could not take the slice"
    _release_and_withdraw(carry_quat, q_taking)
    return True, ""


def _pick_slice(bread_keys: list[str], target: str):
    """Pick ``target`` out of the rack with the giving arm; returns the ``PickResult``."""
    neighbours = tuple(key for key in bread_keys if key != target)
    start_z = object_position(target)[2]
    proposals = slab_grasps(
        env,
        target,
        face_normal_local=BREAD_FACE_NORMAL_LOCAL,
        bbox_min_m=BREAD_BBOX_MIN_M,
        bbox_max_m=BREAD_BBOX_MAX_M,
        grasp_depth_m=GRASP_DEPTHS_M,
        lateral_offset_m=GRASP_OFFSETS_M,
        approach_offset_m=PAD_OFFSET_M,
        flip=(True,),  # the flipped wrist roll keeps the wrist camera looking out
    )
    reachable = [p for p in proposals if giving.ik_reachable(p.position, p.quat_wxyz)]
    print(f"  {len(reachable)}/{len(proposals)} slab grasp candidates are IK-reachable", flush=True)
    if len(reachable) > MAX_CANDIDATES:
        print(f"  planning only the first {MAX_CANDIDATES} of them")
        reachable = reachable[:MAX_CANDIDATES]

    # Near-centre candidates first, wide offsets only as a fallback: every centred grasp so far
    # has held through the carry, while every +/-25 mm one has either slipped out during the lift
    # (measured with probe_env_action_tracking: pads closed on the slice at 14.3 mm, then closed
    # through it to 2.2 mm as the lift ran) or come out marginal. The offsets exist for scenes
    # where the centre is unreachable, not as equals.
    def _offset_mm(proposal) -> float:
        match = re.search(r"offset ([+-]?\d+) mm", proposal.label)
        return abs(float(match.group(1))) if match else 0.0

    reachable.sort(key=_offset_mm)
    centred = [p for p in reachable if _offset_mm(p) <= 10.0]
    offset_fallback = [p for p in reachable if _offset_mm(p) > 10.0]
    pick = None
    for pool, note in ((centred, "centred"), (offset_fallback, "offset fallback")):
        if not pool:
            continue
        if note != "centred":
            print(f"  no centred candidate held; trying {len(pool)} offset ones")
        pick = pick_place.pick(
            pool,
            pregrasp_height_m=PREGRASP_HEIGHT_M,
            mute_during_descent=(*neighbours, "bread_shelf"),
            verify_grasp=lambda: object_position(target)[2] - start_z > LIFT_THRESHOLD_M,
            pregrasp_gripper_pos=PREGRASP_OPEN_FRACTION * giving.cfg.gripper_open_pos,
            # Slower than the default 0.15: the pinch holds a thin slab, and the offset grasps
            # in particular shed it when the lift jerks.
            lift_speed=0.10,
            # A descent that stalls tens of mm short leaves the pads pinching the slice's top
            # sliver; such a grasp can pass the lift check and still shed the slice mid-carry.
            # 20 mm rejects those while allowing the normal few-mm stall on contact -- and a
            # stall-abandoned candidate backs out with the jaws open, leaving the rack
            # undisturbed, so trying the next candidates is free where a *closed*-and-missed
            # attempt would not be.
            max_reach_error_m=0.02,
            max_attempts=2,
        )
        for line in pick.trace:
            print(f"  {line}")
        if pick.success:
            break
    if pick is None:
        pick = pick_place.pick([], max_attempts=1)  # empty; yields a clean failure result
    return pick


def _carry_to_staging(pick, target: str):
    """Carry the picked slice flat to the staging pose; returns the held wrist quat, or None.

    Climb first: the carried slice is not in the collision model, so crossing low drags it
    through the rack. The climb height is laddered -- a failed request to go higher must not come
    out lower.
    """
    print("  === carry ===")
    here = giving.tool_position()
    lift = None
    for target_z in np.arange(max(CARRY_TRAVEL_Z, here[2]), here[2] - 0.001, -0.02):
        candidate = giving.plan_pose(
            giving.joint_positions(), np.array([here[0], here[1], target_z]), pick.grasp_quat_wxyz
        )
        if candidate is not None and candidate.is_executable():
            lift = candidate
            if target_z < CARRY_TRAVEL_Z - 0.001:
                print(f"  can only climb to z {target_z:.3f}, not the {CARRY_TRAVEL_Z:.3f} asked for")
            break
    if lift is None:
        print(f"  cannot climb at all; crossing at {here[2]:.3f}")
    else:
        giving_executor.follow(lift.path, speed=CLIMB_SPEED)

    # Across and flat, jaws pinned down: which way the wrist camera faces is a requirement, the
    # crossing height only a means, so heights give way before the jaw direction does.
    carry_point = np.array([CARRY_XY[0], CARRY_XY[1], max(CARRY_TRAVEL_Z, CARRY_Z)])
    bearings = [CARRY_AZIMUTH_DEG]
    for step in range(15, 181, 15):
        bearings += [CARRY_AZIMUTH_DEG - step, CARRY_AZIMUTH_DEG + step]
    cross_heights = list(np.arange(carry_point[2], CARRY_Z - 0.001, -0.02))
    carry = carry_quat = None
    for cross_z, jaw in [(z, CARRY_JAWS) for z in cross_heights] + [(z, -CARRY_JAWS) for z in cross_heights]:
        for azimuth_deg in bearings:
            azimuth = np.radians(azimuth_deg)
            approach = np.array([np.cos(azimuth), np.sin(azimuth), 0.0])
            point = np.array([carry_point[0], carry_point[1], cross_z])
            quat = quat_wxyz_from_matrix(np.column_stack([np.cross(jaw, approach), jaw, approach]))
            if not giving.ik_reachable(point, quat):
                continue
            candidate = giving.plan_pose(giving.joint_positions(), point, quat)
            if candidate is None or not candidate.is_executable():
                continue
            carry, carry_quat, carry_point = candidate, quat, point
            warn = "" if jaw[2] == CARRY_JAWS[2] else "  (JAWS FLIPPED from the requested direction)"
            print(f"  crossing at z {cross_z:.3f}, bearing {azimuth_deg:g} deg{warn}")
            break
        if carry is not None:
            break
    if carry is None:
        print("  no reachable way to carry the slice flat to the staging pose")
        return None

    # Hold with the executor, never a bare zero action: outside the executor the arm goes limp
    # mid-air. Then straight down onto the staging height, so nothing sweeps sideways.
    q_hold = giving_executor.follow(carry.path, speed=CROSS_SPEED)
    if abs(carry_point[2] - CARRY_Z) > 0.001:
        descend = giving.plan_pose(giving.joint_positions(), np.array([*carry_point[:2], CARRY_Z]), carry_quat)
        if descend is not None and descend.is_executable():
            q_hold = giving_executor.follow(descend.path, speed=LOWER_SPEED)
        else:
            print(f"  could not lower to z {CARRY_Z:.3f}; holding at z {carry_point[2]:.3f}")
    giving_executor.step(arm_target=q_hold, steps=max(1, 60 // (DECIMATION if recording else 1)))
    held_pos, held_rot = object_pose(target)
    normal = held_rot @ np.array(BREAD_FACE_NORMAL_LOCAL)
    tilt_deg = float(np.degrees(np.arccos(abs(np.clip(normal[2], -1.0, 1.0)))))
    from_tool_mm = float(np.linalg.norm(giving.tool_position() - held_pos)) * 1000.0
    print(f"  slice staged at {np.round(held_pos, 4)}, {tilt_deg:.1f} deg off flat, {from_tool_mm:.1f} mm from tool")
    # A held slice measures 17-34 mm from the tool; one shed onto the toaster's lid measured
    # 116 mm and still slipped under the old 120 mm bound, so everything after it groped at air.
    if tilt_deg > 15.0 or from_tool_mm > 60.0:
        return None
    return carry_quat


def _hand_over(carry_quat: np.ndarray, target: str):
    """Take the staged slice with the receiving arm; returns its held configuration, or None.

    The receiving grasp mirrors the giving one through the slice -- the opposite edge of the same
    slab -- rather than asking slab_grasps, whose "top edge" choice degenerates on a slice lying
    flat.
    """
    print("  === handover ===")
    slice_pos, slice_rot = object_pose(target)
    centre = slice_pos + slice_rot @ (0.5 * (np.array(BREAD_BBOX_MIN_M) + np.array(BREAD_BBOX_MAX_M)))
    normal = slice_rot @ np.array(BREAD_FACE_NORMAL_LOCAL)
    offset = giving.tool_position() - centre
    in_plane = offset - np.dot(offset, normal) * normal
    take_point = centre - in_plane + np.dot(offset, normal) * normal
    mirrored = -matrix_from_quat_wxyz(carry_quat)[:, 2]
    base_bearing = float(np.degrees(np.arctan2(mirrored[1], mirrored[0])))

    # The mirror bearing is only the tidiest first try: any horizontal approach across the
    # thickness works on a slice in mid-air, and a narrow sweep once read as "cannot take" when it
    # was really "cannot take from behind".
    bearings = [base_bearing]
    for step in range(15, 181, 15):
        bearings += [base_bearing - step, base_bearing + step]

    taken = None
    for jaw in (HANDOVER_JAWS, -HANDOVER_JAWS):
        for bearing in bearings:
            radians = np.radians(bearing)
            approach = np.array([np.cos(radians), np.sin(radians), 0.0])
            quat = quat_wxyz_from_matrix(np.column_stack([np.cross(jaw, approach), jaw, approach]))
            if not receiving.ik_reachable(take_point, quat):
                continue
            # Stand off back along the approach, not upwards: a slice in the air has nothing
            # under it.
            stand_off = take_point - approach * HANDOVER_STANDOFF_M
            reach = receiving.plan_pose(receiving.joint_positions(), stand_off, quat)
            if reach is None or not reach.is_executable():
                continue
            close_in = receiving.plan_pose(reach.q_end, take_point, quat)
            if close_in is None or not close_in.is_executable():
                continue
            taken = (bearing, jaw, reach, close_in)
            break
        if taken is not None:
            break
    if taken is None:
        print("  the receiving arm could not plan onto the slice")
        return None

    bearing, jaw, reach, close_in = taken
    warn = "" if jaw[2] == HANDOVER_JAWS[2] else "  (JAWS FLIPPED from the requested direction)"
    print(f"  {receiving_arm} arm taking at bearing {bearing:.0f} deg{warn}")
    receiving_executor.set_gripper(PREGRASP_OPEN_FRACTION * receiving.cfg.gripper_open_pos)
    receiving_executor.follow(reach.path, speed=REACH_SPEED)
    q_taking = receiving_executor.follow(close_in.path, speed=CLOSE_IN_SPEED)
    receiving_executor.close_gripper(hold_arm_at=q_taking)
    receiving_executor.step(arm_target=q_taking, steps=max(1, 40 // (DECIMATION if recording else 1)))
    both_mm = float(np.linalg.norm(receiving.tool_position() - object_position(target))) * 1000.0
    print(f"  both hands on the slice; receiving tool {both_mm:.1f} mm from it")
    if both_mm > 60.0:
        # The jaws closed, but not on the slice -- it was lost somewhere earlier and this closed
        # on air. Without this check the giving hand goes on to "release" and the run reports a
        # handover that never happened.
        return None
    return q_taking


def _release_and_withdraw(carry_quat: np.ndarray, q_taking: np.ndarray) -> None:
    """Open the giving hand and move it out of the receiver's way.

    It withdraws sideways -- backing off along its own approach only puts 60 mm between the hands
    and leaves it in the receiver's way. The distance is laddered, since a fixed one with no plan
    would strand the hand parked against the slice with its jaws open.
    """
    giving_executor.open_gripper(hold_arm_at=giving.joint_positions())
    release_dir = matrix_from_quat_wxyz(carry_quat)[:, 2]
    sideways_sign = -1.0 if GIVING_ARM == "right" else 1.0
    back_off = None
    for distance in RELEASE_RETREAT_LADDER_M:
        for direction in (np.array([0.0, sideways_sign, 0.0]), -release_dir):
            candidate = giving.plan_pose(
                giving.joint_positions(), giving.tool_position() + direction * distance, carry_quat
            )
            if candidate is not None and candidate.is_executable():
                back_off = candidate
                print(f"  giving hand withdrawing {distance * 1000:.0f} mm")
                break
        if back_off is not None:
            break
    if back_off is not None:
        q_back = giving_executor.follow(back_off.path, speed=RETREAT_SPEED)
        giving_executor.step(arm_target=q_back, steps=max(1, 40 // (DECIMATION if recording else 1)))
    else:
        print("  the giving hand could not plan a retreat; it stays put with the jaws open")
    receiving_executor.step(arm_target=q_taking, steps=max(1, 60 // (DECIMATION if recording else 1)))


# ------------------------------------------------------------------------------- main loop ---
task = arena_env.task
report: list[str] = []
recorded = 0
for demo in range(args.num_demos):
    print(f"\n--- demo {demo + 1}/{args.num_demos} ---")
    if demo:
        # The order is record_demos.py's: discard whatever the recorder holds, then reset.
        if recording:
            env.recorder_manager.reset()
        env.reset()
        if interface is not None:
            interface.sync_from_robot()
        for executor in executors.values():
            executor._gripper_target = executor.planner.cfg.gripper_open_pos

    # Let the rack settle before anything is measured; the slices slide a few mm coming to rest.
    settle(SETTLE_STEPS)
    bread_keys = sorted(key for key in env.scene.rigid_objects if key.startswith("bread") and key != "bread_shelf")
    assert bread_keys, f"No bread in the scene, found {sorted(env.scene.rigid_objects)}"
    positions = {key: object_position(key) for key in bread_keys}
    # The rightmost slice: the only one with a free side, so its grasp threads one 21 mm gap.
    target = args.slice if args.slice is not None else min(bread_keys, key=lambda k: positions[k][1])
    assert target in bread_keys, f"--slice {target} is not one of {bread_keys}"
    print(f"picking {target} with the {GIVING_ARM} arm")

    # Obstacles reflect where things stand *now*; the slice being handed over is not one.
    if not scene_obstacles:
        for planner in planners.values():
            planner.add_scene_object_obstacle("bread_shelf", SHELF_EXTENTS_M)
            planner.add_scene_object_obstacle("toaster", TOASTER_EXTENTS_M)
            for key in bread_keys:
                planner.add_scene_object_obstacle(key, BREAD_EXTENTS_M)
        scene_obstacles = ["bread_shelf", "toaster", *bread_keys]
    for planner in planners.values():
        for key in scene_obstacles:
            planner.set_obstacle_enabled(key, True)
            planner.update_obstacle_pose(key)
        planner.set_obstacle_enabled(target, False)

    try:
        completed, reason = run_demo(bread_keys, target)
    except RuntimeError as error:
        completed, reason = False, str(error)
    success = bool(task.is_success(env)[0].item()) if completed else False
    final_pos = object_position(target)
    to_receiver_mm = float(np.linalg.norm(receiving.tool_position() - final_pos)) * 1000.0
    if completed:
        print(f"  slice at {np.round(final_pos, 4)}, {to_receiver_mm:.1f} mm from the {receiving_arm} tool")
    print(f"  task success predicate: {success}" + (f"  ({reason})" if reason else ""))

    if success and recording:
        # Mark and export by hand -- the success termination was removed from the env, so nothing
        # else will. Exactly record_demos.py's sequence.
        env.recorder_manager.record_pre_reset([0], force_export_or_skip=False)
        env.recorder_manager.set_success_to_episodes([0], torch.tensor([[True]], dtype=torch.bool, device=env.device))
        env.recorder_manager.export_episodes([0])
        recorded += 1
        print(f"  exported demo {recorded} to {args.record_dir}/{args.dataset_name}.hdf5")
    outcome = "success" if success else f"FAILED ({reason or 'predicate false'})"
    report.append(f"demo {demo + 1}: {outcome}")

# --------------------------------------------------------------------------------- verdict ---
print("\n=== outcome ===")
for line in report:
    print(f"  {line}")
if recording:
    print(f"  {recorded}/{args.num_demos} demonstrations exported to {args.record_dir}/{args.dataset_name}.hdf5")
if writer is not None:
    writer.close()
    print(f"  wrote {frame_counter[0]} frames to {video_path}")
    if truncated[0]:
        print(f"  recording stopped at --max-frames {args.max_frames}; the run continued past it")
simulation_app.close()
