# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Pick a slice of bread out of the rack in ``agibot_make_toast`` with a cuMotion-planned motion.

The first step of make_toast, on its own: one arm, one slice, grasp and lift. Getting a slab out
of a rack is the part of this task with no precedent in Arena -- a slice is not a body of
revolution, so ``rim_grasps`` does not describe it, and the asset carries no annotated grasp point
-- so it is worth establishing before anything is carried anywhere.

Usage::

    OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y .venv/bin/python \\
        isaaclab_arena_cumotion/scripts/make_toast_cumotion.py --headless \\
        --video /path/to/pick.mp4
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--env", type=str, default="agibot_make_toast")
parser.add_argument("--arm", type=str, default="right", choices=("left", "right"))
parser.add_argument("--slice", type=str, default=None, help="Scene key to pick; the rightmost by default.")
parser.add_argument("--video", type=str, default=None)
parser.add_argument("--fps", type=int, default=30)
parser.add_argument("--record-every", type=int, default=4)
parser.add_argument(
    "--max-frames",
    type=int,
    default=2400,
    help=(
        "Stop recording after this many frames, reported when it happens. The candidate sweep is"
        " long and every failure in it looks the same, so the first minute or two is what there is"
        " to see."
    ),
)
parser.add_argument("--pregrasp-height", type=float, default=0.12)
parser.add_argument("--lift-threshold", type=float, default=0.02, help="Rise counting as a successful lift, in m.")
parser.add_argument(
    "--max-candidates",
    type=int,
    default=40,
    help=(
        "Cap on the reachable candidates handed to the pick. Each one costs a full approach and"
        " descend plan, so the whole 210-pose sweep takes tens of minutes; the sweep is still run"
        " and reported in full, only the planning is capped."
    ),
)
parser.add_argument(
    "--pregrasp-open",
    type=float,
    default=0.25,
    help=(
        "How far open to hold the jaws on the way down, as a fraction of the gripper's travel."
        " Measured with probe_gripper_span.py: 1.0 is a 104.5 mm pad gap, 0.25 is 30.3 mm. The"
        " slices sit 25-33 mm apart, so descending wide open puts the far jaw through the"
        " neighbours; 30 mm clears an 11.8 mm slice by 9 mm a side and still fits the gap."
    ),
)
parser.add_argument(
    "--max-attempts",
    type=int,
    default=1,
    help=(
        "How many grasp candidates to execute. One by default: every candidate is planned against"
        " the rack as it stood at the start, so the moment an attempt nudges a slice the remaining"
        " plans are aimed at a scene that no longer exists, and watching them play out says nothing"
        " about the grasp."
    ),
)
parser.add_argument(
    "--grasp-depths",
    type=float,
    nargs="+",
    default=(0.030,),
    help=(
        "How far below the slice's top edge to pinch, in m. 30 mm is the settled value: at 20 mm"
        " the pinch caught only the top sliver and the slice stayed in the rack, because the pad"
        " contact surfaces sit about 6 mm inside their body origins -- the jaws start loading at a"
        " 24 mm origin gap on an 11.8 mm slab, not at 11.8 mm. The slice stands 59 mm proud of the"
        " rack, so 40 mm would put the pads level with the rack's own rim."
    ),
)
parser.add_argument(
    "--grasp-offsets",
    type=float,
    nargs="+",
    default=(0.0, -0.025, 0.025),
    help=(
        "Sideways offsets along the top edge, in m. The face is 116.9 mm across, so 55 mm leaves"
        " 3.5 mm of bread inside the jaws -- with only one attempt executed, a candidate like that"
        " is not worth offering at all."
    ),
)
parser.add_argument(
    "--diagnose",
    action="store_true",
    help=(
        "Drive to the grasp pose the pick would choose and report where the finger pads end up"
        " relative to the slice, instead of grasping. A pinch that closes on nothing looks"
        " identical in the trace to one that closes on the object and drops it."
    ),
)
parser.add_argument(
    "--pad-offset",
    type=float,
    default=0.017,
    help=(
        "How far the finger pads sit behind the tool frame along the approach, in m; grasp points"
        " are pushed this much deeper so the pads land where they were aimed. Measured with"
        " --diagnose: commanded 20 mm below the slice's top edge, the pads came down 3.2 mm below"
        " it, and the tool frame itself landed exactly where it was told."
    ),
)
parser.add_argument(
    "--grasp-flip",
    type=str,
    default="flipped",
    choices=("both", "upright", "flipped"),
    help=(
        "Which wrist roll to offer. 'flipped' is 'upright' rolled 180 degrees about the approach"
        " axis; measurement puts the finger pads in the same places either way, with the two"
        " fingers swapped, so this chooses how the wrist gets there rather than how well it holds."
        " Pinned to 'flipped' by default at the user's request, to keep the wrist camera looking"
        " out. Leaving it free let the planner's travel sort pick a roll per run, which made runs"
        " incomparable."
    ),
)
parser.add_argument(
    "--isolate-target",
    action="store_true",
    help=(
        "With --isolate, move the target slice out too, so the same descent closes on empty air."
        " That separates a jaw the object is blocking from one the gripper blocks itself: the"
        " free-air span measured at the home pose does not settle it, because a jam can depend on"
        " the arm configuration."
    ),
)
parser.add_argument(
    "--isolate",
    action="store_true",
    help=(
        "Teleport the other slices out of the scene once everything has settled, leaving the"
        " target alone in the rack. Bisects a jaw that jams part-way: still jammed with nothing"
        " beside it and the blocker is the gripper or the rack, not the neighbours. Teleporting"
        " rather than not spawning keeps the task's own four-slice invariant intact."
    ),
)
parser.add_argument(
    "--carry",
    action="store_true",
    help=(
        "After the pick, carry the slice to --carry-xy at --carry-z and hold it there, lying flat."
        " Nothing else: no second arm, no release. This is the staging pose a handover would"
        " happen at, on its own, so its position can be judged from a video before anything is"
        " built on top of it."
    ),
)
parser.add_argument(
    "--carry-xy",
    type=float,
    nargs=2,
    default=(0.3525, 0.0383),
    help=(
        "Where to carry the slice, in world xy. Read off --pose-trace at the moment the user"
        " picked out of a recording, not estimated from the picture: it is over the toaster rather"
        " than at the rack/toaster midpoint the first version aimed for."
    ),
)
parser.add_argument("--carry-z", type=float, default=0.886, help="Height to carry the slice to, in world z.")
parser.add_argument(
    "--carry-travel-z",
    type=float,
    default=1.02,
    help=(
        "Height to climb to and cross at, before descending onto the staging pose. Separate from"
        " --carry-z because they answer different questions: the staging height is where the slice"
        " has to end up, the travel height is what it has to clear on the way. Crossing at the"
        " staging height put the tool at 0.886 with up to 86 mm of slice hanging below it -- a"
        " bottom edge at 0.80 against a toaster 0.785 tall -- so the slice caught the toaster and"
        " the arm stalled 227 mm short. Climbing as high as the arm allows and coming down"
        " vertically at the end keeps the staging pose, and its jaws-down wrist, intact."
    ),
)
parser.add_argument(
    "--carry-jaws",
    type=str,
    default="down",
    choices=("down", "up"),
    help=(
        "Which way the jaw axis points at the staging pose, which is also which way up the slice"
        " is held and therefore where the wrist camera ends up looking. 'down' puts the camera"
        " outward, which is what the task wants. Like the bearing and the wrist roll before it,"
        " this is a property of the task and has to be stated: left to a sweep the planner took"
        " whichever solved first, and it came out with the hand upside down."
    ),
)
parser.add_argument(
    "--insert",
    action="store_true",
    help="After the handover, release with the giving arm and put the slice into --slot.",
)
parser.add_argument(
    "--slot",
    type=str,
    default="toast_slot1",
    choices=("toast_slot1", "toast_slot2"),
    help=(
        "Which toaster slot to load. toast_slot1 is the robot's left one: the toaster is yawed by"
        " pi, so its local -y lands at world y 0.186 against toast_slot2's 0.134, and the robot"
        " faces +x with its left hand towards +y."
    ),
)
parser.add_argument(
    "--insert-height",
    type=float,
    default=0.050,
    help=(
        "Where the slice's origin ends up above the slot's floor point, in m. The task counts a"
        " slot as loaded between 30 and 65 mm, so this aims at the middle of that band."
    ),
)
parser.add_argument(
    "--insert-travel-z",
    type=float,
    default=1.00,
    help=(
        "World height to turn the slice upright at, before descending into the slot. A 117 mm"
        " slice swung upright close to the toaster sweeps its lower edge down to about 60 mm below"
        " the tool, which at the old 100 mm stand-off is under the toaster's 0.785 m lid -- so the"
        " turn has to happen well clear and the descent has to be vertical. Laddered down if the"
        " arm cannot reach that high."
    ),
)
parser.add_argument(
    "--handover-jaws",
    type=str,
    default="up",
    choices=("down", "up"),
    help=(
        "Which way the *receiving* arm's jaw axis points, which is what decides where its wrist"
        " camera looks. Stated separately from --carry-jaws, and defaulting to the opposite,"
        " because the two hands are not mirror images: the generated robot description relabels"
        " the left gripper's axes, so the same jaw sign turns the camera opposite ways on the two"
        " arms. Reusing --carry-jaws for both pointed the receiving camera inward."
    ),
)
parser.add_argument(
    "--carry-azimuth",
    type=float,
    default=53.0,
    help=(
        "Which way the gripper points at the staging pose, as a world compass bearing in the xy"
        " plane: 0 is +x (straight ahead of the robot), 90 is +y (its left, the toaster side)."
        " 53 is what the arm actually holds at the chosen point -- a full 90 has no IK solution"
        " anywhere in this workspace with the slice held flat, which is why this is stated rather"
        " than swept: left to a sweep the planner picked straight ahead and reached it by rotating"
        " the wrong way round."
    ),
)

parser.add_argument(
    "--handover",
    action="store_true",
    help=(
        "After the pick, carry the slice to chest height and pass it to the other arm. The two"
        " arms' workspaces do not overlap over the table -- probe_make_toast_reach measures the"
        " rack as right-arm-only and the toaster slots as left-arm-only -- so a slice can only get"
        " from one to the other through a handover in front of the chest, where both reach."
    ),
)
parser.add_argument(
    "--handover-standoff",
    type=float,
    default=0.06,
    help=(
        "How far back along its approach the receiving arm stands off before closing in, in m."
        " Not the same as --pregrasp-height: that clearance exists to clear the rack, and a slice"
        " held in the air has nothing to clear. Carried over at 0.12 it put the stand-off at"
        " z 1.14, and the chest is only reachable to about z 1.05 -- all 19 candidates failed"
        " there, none of them even reaching the close-in."
    ),
)
parser.add_argument("--chest-x", type=float, default=0.30, help="Handover point ahead of the robot base, in m.")
parser.add_argument("--chest-z", type=float, default=1.00, help="Handover height above the robot base, in m.")
parser.add_argument(
    "--pose-trace",
    type=str,
    default=None,
    help=(
        "Write the driven arm's tool pose for every recorded video frame to this CSV. Lets a pose"
        " chosen by watching a recording be read back exactly, rather than estimated off the"
        " picture and then nudged."
    ),
)
parser.add_argument("--settle-steps", type=int, default=60)
AppLauncher.add_app_launcher_args(parser)
args, _ = parser.parse_known_args()
# Each stage aims at where the previous one left the slice, so a later flag without the earlier
# ones has nothing to act on; failing here beats a silent skip after minutes of sim startup.
assert not args.handover or args.carry, "--handover builds on --carry; pass both"
assert not args.insert or args.handover, "--insert builds on --handover; pass both"
# The head camera is created unconditionally, so the app must always bring the camera pipeline up.
args.enable_cameras = True
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import numpy as np  # noqa: E402
import pathlib  # noqa: E402
import torch  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
import warp as wp  # noqa: E402
from isaaclab.sensors import Camera, CameraCfg  # noqa: E402

import isaaclab_arena_environments  # noqa: E402,F401
from isaaclab_arena.assets.registries import EnvironmentRegistry  # noqa: E402
from isaaclab_arena.cli.isaaclab_arena_cli import (  # noqa: E402
    arena_env_builder_cfg_from_argparse,
    get_isaaclab_arena_cli_parser,
)
from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder  # noqa: E402
from isaaclab_arena_cumotion.embodiment_cumotion_registry import get_embodiment_cumotion_cfg  # noqa: E402
from isaaclab_arena_cumotion.executor import ArmExecutor  # noqa: E402
from isaaclab_arena_cumotion.grasps import (  # noqa: E402
    _rot_about,
    matrix_from_quat_wxyz,
    quat_wxyz_from_matrix,
    slab_grasps,
)
from isaaclab_arena_cumotion.pick_place import PickAndPlace  # noqa: E402
from isaaclab_arena_cumotion.planner import CumotionArmPlanner  # noqa: E402

# RoboDojo's Rigid/bread/00000 aligned bbox: a 116.9 x 116.1 mm face, 11.8 mm thick. The thin axis
# is the slice's local z, which is what makes it a slab rather than a body of revolution.
BREAD_EXTENTS_M = (0.1169, 0.1161, 0.0118)
BREAD_FACE_NORMAL_LOCAL = (0.0, 0.0, 1.0)

# Read off the usdz, not assumed: the slice's origin sits on its **bottom face**, so the slab runs
# from -0.3 mm to +11.5 mm through the origin rather than +/-5.9 mm about it. Dropping a slice flat
# on the table confirms it -- the origin comes to rest level with the surface.
BREAD_BBOX_MIN_M = (-0.0584, -0.0583, -0.0003)
BREAD_BBOX_MAX_M = (0.0586, 0.0578, 0.0115)

SHELF_EXTENTS_M = (0.1583, 0.0835, 0.0762)
TOASTER_EXTENTS_M = (0.2267, 0.1554, 0.1619)

# The slot rectangles come from the Toaster asset class, so the numbers the insertion aims at are
# the same objects the task's success check reads -- a second copy here once risked drifting
# silently whenever the task side moved.
from isaaclab_arena.assets.local_objects import Toaster  # noqa: E402

SLOTS_LOCAL = Toaster.SLOT_RECT_LOCAL_M
SLOT_Z_LOCAL_M = Toaster.SLOT_Z_LOCAL_M

TABLE_TOP_Z = 0.6232
TABLE_OBSTACLE = "/obstacles/table"

# ------------------------------------------------------------------------------------- env ---
arena_args = get_isaaclab_arena_cli_parser().parse_args(["--num_envs", "1", "--enable_cameras"])
factory = EnvironmentRegistry().get_component_by_name(args.env)()
arena_env = factory.build(factory._legacy_argparse_cfg_type(teleop_device=None))
embodiment_type = type(arena_env.embodiment)
TOOL_FRAMES = {arm: get_embodiment_cumotion_cfg(arena_env.embodiment, arm).tool_frame for arm in ("left", "right")}
builder = ArenaEnvBuilder(arena_env, arena_env_builder_cfg_from_argparse(arena_args))
env = builder.make_registered().unwrapped

# The robot's own head view, not a free-floating camera. Every run is then framed the same way
# and the same way the teleoperated viewport is, so a recording can be compared against a demo
# without first working out where the camera was. Ad-hoc placements cost a run each time: one
# aimed at the rack missed the entire handover, and the next sat behind the room's wall.
camera = Camera(
    CameraCfg(
        prim_path="/World/head_cam",
        height=720,
        width=1280,
        data_types=["rgb"],
        # Reproduces the Kit viewport's field of view, so this matches what --viz kit shows.
        spawn=sim_utils.PinholeCameraCfg(focal_length=18.147, horizontal_aperture=20.955, clipping_range=(0.05, 30.0)),
    )
)
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
print(f"head camera at {np.round(_head + np.array(embodiment_type.HEAD_VIEW_EYE), 3)}")

# Frames are streamed straight to the container rather than collected and written at the end.
# Accumulating them does not survive this script: a sweep over two dozen candidates runs tens of
# thousands of servo steps, and at 1280x720x3 that is tens of GB of RAM -- the first attempt at
# this was SIGKILLed by the OOM killer after finishing the pick but before writing anything.
writer = None
if args.video is not None:
    import imageio.v2 as iio  # noqa: E402

    video_path = pathlib.Path(args.video)
    video_path.parent.mkdir(parents=True, exist_ok=True)
    writer = iio.get_writer(video_path, fps=args.fps, codec="libx264", macro_block_size=8)

step_counter = [0]
frame_counter = [0]
truncated = [False]
pose_trace: list[tuple] = []
traced_slice: list[str | None] = [None]


def _tool_pose_row():
    """The driven arm's tool position and the bearing its approach axis points along.

    Recorded per video frame so a pose picked out of a recording can be read back as numbers
    instead of estimated from the picture.
    """
    import isaaclab.utils.math as math_utils

    robot = env.scene.articulations["robot"]
    names = list(robot.data.body_names)
    index = names.index(TOOL_FRAMES[args.arm])
    position = wp.to_torch(robot.data.body_pos_w)[0, index].detach().cpu().numpy().astype(np.float64)
    quat_xyzw = wp.to_torch(robot.data.body_quat_w)[0, index].detach().cpu()
    rotation = math_utils.matrix_from_quat(quat_xyzw.float().unsqueeze(0))[0].numpy().astype(np.float64)
    approach = rotation[:, 2]
    bearing = float(np.degrees(np.arctan2(approach[1], approach[0])))
    jaw_z = float(rotation[2, 1])

    # The slice too, and how far it is from whichever hand is nearest. Separation is the only way
    # to tell *when* a grasp let go, and its tilt at that moment says whether it was mid-turn.
    # Frames are recorded from the very first settle, before the target slice has been chosen, so
    # the slice columns stay blank until there is one.
    if traced_slice[0] is None:
        return (*np.round(position, 4), round(bearing, 1), round(jaw_z, 3), "", "", "", "", "")
    slice_p, slice_r = _object_pose_for_report(traced_slice[0])
    normal = slice_r @ np.array(BREAD_FACE_NORMAL_LOCAL)
    tilt = float(np.degrees(np.arccos(abs(np.clip(normal[2], -1.0, 1.0)))))
    hands = [names.index(n) for n in TOOL_FRAMES.values() if n in names]
    body_pos = wp.to_torch(robot.data.body_pos_w)[0].detach().cpu().numpy().astype(np.float64)
    nearest = min(float(np.linalg.norm(body_pos[i] - slice_p)) for i in hands)
    return (
        *np.round(position, 4),
        round(bearing, 1),
        round(jaw_z, 3),
        *np.round(slice_p, 4),
        round(tilt, 1),
        round(nearest * 1000, 1),
    )


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
    if args.pose_trace is not None:
        pose_trace.append((frame_counter[0], frame_counter[0] / args.fps) + _tool_pose_row())
    frame_counter[0] += 1


def object_position(key: str) -> np.ndarray:
    """An object's world position."""
    return wp.to_torch(env.scene[key].data.root_pos_w)[0].detach().cpu().numpy().astype(np.float64)


def _object_pose_for_report(key: str):
    """An object's world position and rotation matrix."""
    import isaaclab.utils.math as math_utils

    asset = env.scene[key]
    position = wp.to_torch(asset.data.root_pos_w)[0].detach().cpu().numpy().astype(np.float64)
    quat_xyzw = wp.to_torch(asset.data.root_quat_w)[0].detach().cpu()
    rotation = math_utils.matrix_from_quat(quat_xyzw.float().unsqueeze(0))[0].numpy().astype(np.float64)
    return position, rotation


# Let the rack settle before anything is measured: the slices slide a few millimetres as they
# come to rest, and a grasp authored against the spawn pose would be aimed at where they were.
zero_action = torch.zeros((env.num_envs, env.action_space.shape[1]), device=env.device)
for _ in range(args.settle_steps):
    env.step(zero_action)
    grab_frame()

bread_keys = sorted(key for key in env.scene.rigid_objects if key.startswith("bread") and key != "bread_shelf")
assert bread_keys, f"No bread in the scene, found {sorted(env.scene.rigid_objects)}"
positions = {key: object_position(key) for key in bread_keys}
print("\nsettled slices:")
for key in bread_keys:
    print(f"  {key} at {np.round(positions[key], 4)}")

# The robot faces +x, so its right hand works at -y and the "rightmost" slice is the one with the
# smallest y. Picking the outermost one first is not arbitrary: it is the only slice with a free
# side, so its grasp needs the jaws to fit into one 21 mm gap rather than two.
target = args.slice if args.slice is not None else min(bread_keys, key=lambda k: positions[k][1])
assert target in bread_keys, f"--slice {target} is not one of {bread_keys}"
traced_slice[0] = target
print(f"\npicking {target} with the {args.arm} arm")

if args.isolate:
    for key in bread_keys:
        if key == target:
            continue
        asset = env.scene[key]
        root_state = wp.to_torch(asset.data.root_state_w).detach().cpu().numpy().copy()
        root_state[0, 1] += 3.0
        asset.write_root_state_to_sim(torch.tensor(root_state, device=env.device, dtype=torch.float32))
    for _ in range(20):
        env.step(zero_action)
        grab_frame()
    positions = {key: object_position(key) for key in bread_keys}
    print(f"  isolated {target}; the other slices are now {sorted(set(bread_keys) - {target})} out of the way")

# -------------------------------------------------------------------------------- planning ---
embodiment = arena_env.embodiment
planner = CumotionArmPlanner(env, embodiment, arm=args.arm)
error_m = planner.kinematics_error_m()
print(f"{args.arm} arm kinematics cross-check: {error_m * 1000:.2f} mm")
assert error_m < 1e-3, f"cuMotion's {args.arm}-arm kinematics disagree with the simulated robot"

# The work surface, modelled 30 mm below the real top -- see stack_bowls_cumotion for why a flush
# slab leaves near-table grasps unplannable while still blocking gross sweeps through the table.
planner.add_box_obstacle(
    TABLE_OBSTACLE, np.array([0.185, 0.0, TABLE_TOP_Z - 0.09]), (2.2, 1.4, 0.12), safety_tolerance_m=0.0
)
planner.add_scene_object_obstacle("bread_shelf", SHELF_EXTENTS_M)
planner.add_scene_object_obstacle("toaster", TOASTER_EXTENTS_M)
for key in bread_keys:
    planner.add_scene_object_obstacle(key, BREAD_EXTENTS_M)

executor = ArmExecutor(env, planner, on_step=grab_frame)
pick_place = PickAndPlace(planner, executor, contact_obstacles=(TABLE_OBSTACLE,))

proposals = slab_grasps(
    env,
    target,
    face_normal_local=BREAD_FACE_NORMAL_LOCAL,
    bbox_min_m=BREAD_BBOX_MIN_M,
    bbox_max_m=BREAD_BBOX_MAX_M,
    grasp_depth_m=tuple(args.grasp_depths),
    lateral_offset_m=tuple(args.grasp_offsets),
    approach_offset_m=args.pad_offset,
    flip={"both": (False, True), "upright": (False,), "flipped": (True,)}[args.grasp_flip],
)
reachable = [p for p in proposals if planner.ik_reachable(p.position, p.quat_wxyz)]
print(f"\n{len(reachable)}/{len(proposals)} slab grasp candidates are IK-reachable", flush=True)
for proposal in reachable[:5]:
    print(f"  {proposal.label}: at {np.round(proposal.position, 4)}")
if not reachable:
    print("  none -- the arm cannot get its tool to any pinch on this slice's top edge")
if len(reachable) > args.max_candidates:
    # Reported, never silent: the sweep is ordered gentlest-lean-first, so this keeps the
    # approaches most likely to hold and drops the steep ones, but it *is* a cap.
    print(f"  planning only the first {args.max_candidates} of them (--max-candidates)")
    reachable = reachable[: args.max_candidates]

# ------------------------------------------------------------------------------- execution ---
# The slice being picked must not be an obstacle to its own grasp. Its neighbours and the rack
# stay muted only for the descent, which threads the jaws down into a 21 mm gap between slices.
planner.set_obstacle_enabled(target, False)
neighbours = tuple(key for key in bread_keys if key != target)
start_z = positions[target][2]

if args.diagnose:
    # Reproduce the pick's own choice: plan both legs for every candidate, then order exactly as
    # ``PickAndPlace.pick`` does, so what is measured here is the pose that would really be taken.
    q_start = planner.joint_positions()
    planned = []
    for index, proposal in enumerate(reachable):
        approach = planner.plan_pose(
            q_start, proposal.position + np.array([0.0, 0.0, args.pregrasp_height]), proposal.quat_wxyz
        )
        if approach is None or not approach.is_executable():
            continue
        for name in (TABLE_OBSTACLE, *neighbours, "bread_shelf"):
            planner.set_obstacle_enabled(name, False)
        descend = planner.plan_pose(approach.q_end, proposal.position, proposal.quat_wxyz)
        for name in (TABLE_OBSTACLE, *neighbours, "bread_shelf"):
            planner.set_obstacle_enabled(name, True)
        if descend is None or not descend.is_executable():
            continue
        planned.append((proposal, approach, descend, index))
    assert planned, "no candidate planned end to end"
    planned.sort(key=lambda c: (round(c[2].max_travel_rad, 1), c[3]))
    proposal, approach, descend, _ = planned[0]
    print(f"\ndiagnosing grasp '{proposal.label}'")

    robot = env.scene.articulations["robot"]
    body_names = list(robot.data.body_names)
    pads = [n for n in body_names if n.startswith(f"{args.arm}_") and "Pad" in n]
    assert len(pads) == 2, f"expected two pad links, found {pads}"

    def _slice_frame_report(stage: str) -> None:
        """Where the tool and the pads sit in the slice's own frame."""
        import isaaclab.utils.math as math_utils

        asset = env.scene[target]
        slice_pos = wp.to_torch(asset.data.root_pos_w)[0].detach().cpu().numpy().astype(np.float64)
        slice_quat = wp.to_torch(asset.data.root_quat_w)[0].detach().cpu()
        rotation = math_utils.matrix_from_quat(slice_quat.float().unsqueeze(0))[0].numpy().astype(np.float64)
        body_pos = wp.to_torch(robot.data.body_pos_w)[0].detach().cpu().numpy().astype(np.float64)
        print(f"  {stage}")
        print(f"    tool  {np.round(rotation.T @ (planner.tool_position() - slice_pos) * 1000, 1)} mm")
        for name in pads:
            local = rotation.T @ (body_pos[body_names.index(name)] - slice_pos)
            print(f"    {name:<26} {np.round(local * 1000, 1)} mm")

    print(
        "  slice frame: x is the 116.9 mm face width, y the 116.1 mm height, z the 11.8 mm"
        " thickness -- so |z| under 5.9 mm means a pad is inside the slab's own slice of space,"
        " and |x|,|y| under ~58 mm means it is over the face rather than past its edge."
    )
    if args.isolate_target:
        # Only now: the grasps were authored from this slice's pose and planned against it, so it
        # has to stay put until there is a trajectory to run.
        asset = env.scene[target]
        root_state = wp.to_torch(asset.data.root_state_w).detach().cpu().numpy().copy()
        root_state[0, 1] += 3.0
        asset.write_root_state_to_sim(torch.tensor(root_state, device=env.device, dtype=torch.float32))
        for _ in range(20):
            env.step(zero_action)
        print("  target slice moved out too: this descent closes on empty air")

    executor.set_gripper(args.pregrasp_open * planner.cfg.gripper_open_pos)
    executor.follow(approach.path)
    _slice_frame_report("at the pregrasp stand-off")
    executor.follow(descend.path)
    _slice_frame_report("at the grasp pose, before closing")
    # Settle before reading. ``close_gripper`` returns when it has finished *commanding* the ramp,
    # not when the jaws have got there, and reading immediately showed a 16.7 mm gap on a grasp
    # that in fact went on to lift the slice cleanly -- a measurement artifact, not a failed pinch.
    # Ramp the close by hand so every step can be sampled. A settled reading says the jaws stopped
    # 16.7 mm apart on an 11.8 mm slab; it cannot say whether they were blocked or never asked to
    # go further, and those need opposite fixes.
    gripper_indices = [
        i
        for i, name in enumerate(robot.data.joint_names)
        if name.startswith(f"{args.arm}_") and ("hand_joint" in name or "Support_Joint" in name)
    ]
    gripper_names = [robot.data.joint_names[i] for i in gripper_indices]
    print(f"  gripper joints: {gripper_names}")
    print("  step  commanded   actual (per joint)      pad gap   slice-z of each pad")
    q_hold = planner.joint_positions()
    open_target = args.pregrasp_open * planner.cfg.gripper_open_pos
    closed_target = planner.cfg.gripper_closed_pos
    ramp = 200
    for i in range(ramp + 40):
        alpha = min(1.0, (i + 1) / ramp)
        commanded = open_target + alpha * (closed_target - open_target)
        executor.step(arm_target=q_hold, gripper_target=commanded)
        if i % 20 and i != ramp + 39:
            continue
        import isaaclab.utils.math as math_utils

        actual = wp.to_torch(robot.data.joint_pos)[0].detach().cpu().numpy()[gripper_indices]
        body_pos = wp.to_torch(robot.data.body_pos_w)[0].detach().cpu().numpy().astype(np.float64)
        pad_world = [body_pos[list(robot.data.body_names).index(n)] for n in pads]
        gap = float(np.linalg.norm(pad_world[0] - pad_world[1]))
        asset = env.scene[target]
        slice_pos = wp.to_torch(asset.data.root_pos_w)[0].detach().cpu().numpy().astype(np.float64)
        slice_rot = (
            math_utils.matrix_from_quat(wp.to_torch(asset.data.root_quat_w)[0].detach().cpu().float().unsqueeze(0))[0]
            .numpy()
            .astype(np.float64)
        )
        pad_z = [float((slice_rot.T @ (p - slice_pos))[2]) * 1000 for p in pad_world]
        print(
            f"  {i:4d}  {commanded:9.4f}   {np.round(actual, 4)}   {gap * 1000:6.1f} mm"
            f"   {pad_z[0]:+6.1f} {pad_z[1]:+6.1f}"
        )
    # Which part of the hand is actually against the slice? The pads stop 5.7 mm clear of its face,
    # so whatever blocks the jaw is another link in the chain. Report the whole hand in the slice's
    # own frame and flag anything inside the slab's volume.
    # About the slab's own centre, which the origin is not: see BREAD_BBOX_*.
    bbox_centre = np.array([0.5 * (a + b) for a, b in zip(BREAD_BBOX_MIN_M, BREAD_BBOX_MAX_M)])
    half_x, half_y, half_z = (0.5 * e * 1000 for e in BREAD_EXTENTS_M)
    hand_links = [n for n in robot.data.body_names if n.startswith(f"{args.arm}_") and ("Left" in n or "Right" in n)]
    asset = env.scene[target]
    slice_pos = wp.to_torch(asset.data.root_pos_w)[0].detach().cpu().numpy().astype(np.float64)
    slice_rot = (
        math_utils.matrix_from_quat(wp.to_torch(asset.data.root_quat_w)[0].detach().cpu().float().unsqueeze(0))[0]
        .numpy()
        .astype(np.float64)
    )
    body_pos = wp.to_torch(robot.data.body_pos_w)[0].detach().cpu().numpy().astype(np.float64)
    print(f"\n  hand links in the slice's frame (slab is +/-{half_x:.0f} x +/-{half_y:.0f} x +/-{half_z:.1f} mm)")
    for name in sorted(hand_links):
        local = (slice_rot.T @ (body_pos[list(robot.data.body_names).index(name)] - slice_pos) - bbox_centre) * 1000
        inside = abs(local[0]) <= half_x and abs(local[1]) <= half_y and abs(local[2]) <= half_z
        print(f"    {name:<28} {np.round(local, 1)} mm{'   <-- INSIDE THE SLAB' if inside else ''}")
    _slice_frame_report("after closing, settled")
    executor.follow(descend.path, reverse=True, speed=0.15)
    _slice_frame_report("after the lift")
    print(f"    slice rose {(object_position(target)[2] - positions[target][2]) * 1000:+.1f} mm")
    if writer is not None:
        writer.close()
        print(f"\n  wrote {frame_counter[0]} frames to {video_path}")
    simulation_app.close()
    raise SystemExit(0)

# One call, not a retry loop: ``pick`` already walks its whole candidate list, taking the next
# one whenever ``verify_grasp`` rejects the last. Wrapping it in an outer retry -- which
# stack_bowls does, because there a *place* can fail after a good pick -- just repeats the
# identical sweep, and here that tripled the run time for no extra coverage.
pick = pick_place.pick(
    reachable,
    pregrasp_height_m=args.pregrasp_height,
    mute_during_descent=(*neighbours, "bread_shelf"),
    verify_grasp=lambda: object_position(target)[2] - start_z > args.lift_threshold,
    pregrasp_gripper_pos=args.pregrasp_open * planner.cfg.gripper_open_pos,
    max_attempts=args.max_attempts,
)
# Sampled now, while the slice is actually in the fingers: the outcome section prints these, and
# by the time it runs the slice may have been inserted and the hand withdrawn -- measured there,
# a successful run once reported itself "held 489.3 mm from the slice's centre".
pick_rise_mm = (object_position(target)[2] - start_z) * 1000.0
pick_held_mm = float(np.linalg.norm(planner.tool_position() - object_position(target))) * 1000.0
for line in pick.trace:
    print(f"  {line}")

# ---------------------------------------------------------------------------------- carry ---
# Hold the slice flat, above the midpoint between the rack and the toaster. "Flat" is a statement
# about the jaws, not the wrist: the slice is pinched across its thickness, so its face normal *is*
# the jaw axis, and standing that axis upright lays the slice horizontal. The approach axis is then
# horizontal and its azimuth is free, so it is swept rather than chosen -- the same trick the
# carry-to-chest needed, for the same reason.
carry_ok = None
if args.carry and pick.success:
    print("\n=== carry ===")
    carry_point = np.array([args.carry_xy[0], args.carry_xy[1], max(args.carry_travel_z, args.carry_z)])
    carry_quat = None
    print(
        f"  staging at {np.round([args.carry_xy[0], args.carry_xy[1], args.carry_z], 4)},"
        f" crossing at z {carry_point[2]:.3f}, gripper bearing {args.carry_azimuth:g} deg"
    )

    # Leg one: straight up to the carry height over wherever the hand already is, orientation
    # unchanged. The carried slice is *not* in the planner's collision model -- only the robot is --
    # so a single move to the staging pose routes the arm around the rack while dragging the slice
    # straight through the slices still in it, which is what the video showed. Getting above
    # everything first is what makes leg two safe, so it climbs to the same height it will cross at
    # rather than a fixed offset that may not be reachable.
    # Climb as high as the arm can manage, stepping down from the target until one plans, rather
    # than all-or-nothing. All-or-nothing is worse than it sounds: when the top of the ladder was
    # unreachable this skipped the climb entirely and crossed at the post-pick height, which is
    # *lower* than the previous run managed and clipped two slices still in the rack. A failed
    # request to go higher must not quietly come out lower.
    here = planner.tool_position()
    lift = None
    for target_z in np.arange(max(args.carry_travel_z, here[2]), here[2] - 0.001, -0.02):
        candidate = planner.plan_pose(
            planner.joint_positions(), np.array([here[0], here[1], target_z]), pick.grasp_quat_wxyz
        )
        if candidate is not None and candidate.is_executable():
            lift = candidate
            if target_z < args.carry_travel_z - 0.001:
                print(f"  can only climb to z {target_z:.3f}, not the {args.carry_travel_z:.3f} asked for")
            break
    if lift is None:
        print(f"  cannot climb at all; crossing at {here[2]:.3f}")
    else:
        executor.follow(lift.path, speed=0.10)
        print(f"  climbed clear to {np.round(planner.tool_position(), 4)}")

    # Leg two: across and flat.
    # Try the requested bearing first, then fall back outwards in either direction, so a fallback
    # is always the nearest thing to what was asked for rather than whatever solved first.
    bearings = [args.carry_azimuth]
    for step in range(15, 181, 15):
        bearings += [args.carry_azimuth - step, args.carry_azimuth + step]

    # The crossing height gives before the wrist does. Height is only a means -- it exists to clear
    # the toaster -- while which way the wrist camera faces is a requirement of the task, so the
    # search runs over heights with the jaw direction pinned, and only considers the other jaw
    # direction once no height at all has worked. Ordering these the other way round is what
    # silently produced a jaws-up staging pose: the code preserved the crossing height it had been
    # given and spent the wrist to do it.
    wanted = np.array([0.0, 0.0, -1.0]) if args.carry_jaws == "down" else np.array([0.0, 0.0, 1.0])
    cross_heights = list(np.arange(carry_point[2], args.carry_z - 0.001, -0.02))
    attempts = [(z, wanted) for z in cross_heights] + [(z, -wanted) for z in cross_heights]

    carry = None
    for cross_z, jaw in attempts:
        for azimuth_deg in bearings:
            azimuth = np.radians(azimuth_deg)
            approach = np.array([np.cos(azimuth), np.sin(azimuth), 0.0])
            point = np.array([carry_point[0], carry_point[1], cross_z])
            quat = quat_wxyz_from_matrix(np.column_stack([np.cross(jaw, approach), jaw, approach]))
            if not planner.ik_reachable(point, quat):
                continue
            candidate = planner.plan_pose(planner.joint_positions(), point, quat)
            if candidate is None or not candidate.is_executable():
                continue
            carry, carry_quat, carry_point = candidate, quat, point
            jaws = "up" if jaw[2] > 0 else "down"
            notes = []
            if azimuth_deg != args.carry_azimuth:
                notes.append(f"bearing asked for {args.carry_azimuth:g}")
            if jaws != args.carry_jaws:
                notes.append(f"JAWS ASKED FOR {args.carry_jaws}")
            suffix = f"  ({'; '.join(notes)})" if notes else ""
            print(f"  crossing at z {cross_z:.3f}, bearing {azimuth_deg:g} deg, jaws {jaws}{suffix}")
            break
        if carry is not None:
            break

    if carry is None:
        print("  no reachable way to hold the slice flat there")
        carry_ok = False
    else:
        # Hold with the executor, never with env.step(). The executor drives the arm by writing
        # joint targets straight to the articulation; env.step() hands control back to the
        # environment's action manager, and a zero action there wipes those targets -- the arm goes
        # limp mid-air, drops what it is holding and falls onto the table. That is what ended every
        # carry so far, and it looked exactly like the grasp failing.
        q_hold = executor.follow(carry.path, speed=0.06)
        # Final leg: straight down onto the staging height, orientation unchanged. Short and
        # vertical, so nothing sweeps sideways past the toaster on the way in.
        if abs(carry_point[2] - args.carry_z) > 0.001:
            settle_point = np.array([carry_point[0], carry_point[1], args.carry_z])
            descend = planner.plan_pose(planner.joint_positions(), settle_point, carry_quat)
            if descend is not None and descend.is_executable():
                q_hold = executor.follow(descend.path, speed=0.06)
                print(f"  lowered onto the staging pose at z {args.carry_z:.3f}")
            else:
                print(f"  could not lower to z {args.carry_z:.3f}; holding at z {carry_point[2]:.3f}")
        executor.step(arm_target=q_hold, steps=60)
        held_pos, held_rot = _object_pose_for_report(target)
        # The slice's face normal is its own local z; the angle it makes with world up says how
        # flat it ended up, which is the whole point of this pose.
        normal = held_rot @ np.array(BREAD_FACE_NORMAL_LOCAL)
        tilt_deg = float(np.degrees(np.arccos(abs(np.clip(normal[2], -1.0, 1.0)))))
        from_tool_mm = float(np.linalg.norm(planner.tool_position() - held_pos)) * 1000.0
        # Against the staging pose, which is where the slice is meant to end up -- not against the
        # crossing point it passes through on the way, which reported a good run as a 129 mm miss.
        staging = np.array([args.carry_xy[0], args.carry_xy[1], args.carry_z])
        from_target_mm = float(np.linalg.norm(held_pos - staging)) * 1000.0
        # All three, not just the tilt. Checking flatness alone once reported success for a slice
        # that had been flung across the room and come to rest on the floor -- which is, after all,
        # perfectly horizontal.
        carry_ok = tilt_deg < 15.0 and from_tool_mm < 120.0 and from_target_mm < 120.0
        print(f"  slice at {np.round(held_pos, 4)}, face {tilt_deg:.1f} deg off horizontal")
        print(f"  {from_tool_mm:.1f} mm from the tool, {from_target_mm:.1f} mm from where it was sent")

# ------------------------------------------------------------------------------- handover ---
# The receiving grasp is built by mirroring the giving one through the slice, not by asking
# ``slab_grasps`` for another proposal. That generator assumes a slab standing in something and
# approached from above: it picks the "top edge" as whichever in-plane axis stands closest to
# vertical. On a slice already lying flat both in-plane axes are horizontal, so that choice
# degenerates and the geometry it returns is meaningless. Mirroring is also what the task actually
# wants -- the second hand takes the opposite edge of the same slice.
handover_ok = None
if args.handover and carry_ok:
    print("\n=== handover ===")
    other = "left" if args.arm == "right" else "right"
    other_planner = CumotionArmPlanner(env, embodiment, arm=other)
    print(f"{other} arm kinematics cross-check: {other_planner.kinematics_error_m() * 1000:.2f} mm")
    other_planner.add_box_obstacle(
        TABLE_OBSTACLE, np.array([0.185, 0.0, TABLE_TOP_Z - 0.09]), (2.2, 1.4, 0.12), safety_tolerance_m=0.0
    )
    other_planner.add_scene_object_obstacle("bread_shelf", SHELF_EXTENTS_M)
    other_planner.add_scene_object_obstacle("toaster", TOASTER_EXTENTS_M)
    # The slices still in the rack are obstacles to this arm too; the held one is not, since the
    # whole point is to close on it.
    for key in bread_keys:
        other_planner.add_scene_object_obstacle(key, BREAD_EXTENTS_M)
    other_planner.set_obstacle_enabled(target, False)
    other_executor = ArmExecutor(env, other_planner, on_step=grab_frame)

    slice_pos, slice_rot = _object_pose_for_report(target)
    centre = slice_pos + slice_rot @ (0.5 * (np.array(BREAD_BBOX_MIN_M) + np.array(BREAD_BBOX_MAX_M)))
    normal = slice_rot @ np.array(BREAD_FACE_NORMAL_LOCAL)

    # Mirror the giving tool's in-plane offset through the slice's centre; leave its component
    # along the face normal alone, so the receiving pads straddle the same mid-plane.
    giving = planner.tool_position()
    offset = giving - centre
    in_plane = offset - np.dot(offset, normal) * normal
    take_point = centre - in_plane + np.dot(offset, normal) * normal

    giving_approach = matrix_from_quat_wxyz(carry_quat)[:, 2]
    mirrored = -giving_approach
    base_bearing = float(np.degrees(np.arctan2(mirrored[1], mirrored[0])))
    print(
        f"  giving hand at {np.round(giving, 4)}; taking the opposite edge at {np.round(take_point, 4)},"
        f" bearing {base_bearing:.0f} deg"
    )

    wanted_jaw = np.array([0.0, 0.0, -1.0]) if args.handover_jaws == "down" else np.array([0.0, 0.0, 1.0])
    # Mirroring fixes *where* the second hand grips -- the opposite edge of the same slice -- but
    # not which way it comes in from. The slice is in mid-air, so any horizontal approach that puts
    # the jaws across its thickness is as good as another, and the mirror bearing is only the
    # tidiest one to try first. Restricting the sweep to +/-60 deg of it left all 26 candidates
    # IK-unreachable, which read as "the arm cannot take the slice" when it was really "the arm
    # cannot take it from behind".
    bearings = [base_bearing]
    for step in range(15, 181, 15):
        bearings += [base_bearing - step, base_bearing + step]

    taken = None
    rejected = {"unreachable": 0, "reach_unplanned": 0, "reach_unusable": 0, "close_unplanned": 0, "close_unusable": 0}
    for jaw in (wanted_jaw, -wanted_jaw):
        for bearing in bearings:
            radians = np.radians(bearing)
            approach = np.array([np.cos(radians), np.sin(radians), 0.0])
            quat = quat_wxyz_from_matrix(np.column_stack([np.cross(jaw, approach), jaw, approach]))
            if not other_planner.ik_reachable(take_point, quat):
                rejected["unreachable"] += 1
                continue
            # Stand off *back along the approach*, not upwards: the slice is in the air, so there
            # is nothing underneath it to clear.
            stand_off = take_point - approach * args.handover_standoff
            reach = other_planner.plan_pose(other_planner.joint_positions(), stand_off, quat)
            if reach is None:
                rejected["reach_unplanned"] += 1
                continue
            if not reach.is_executable():
                rejected["reach_unusable"] += 1
                continue
            close_in = other_planner.plan_pose(reach.q_end, take_point, quat)
            if close_in is None:
                rejected["close_unplanned"] += 1
                continue
            if not close_in.is_executable():
                rejected["close_unusable"] += 1
                continue
            taken = (bearing, jaw, reach, close_in)
            break
        if taken is not None:
            break

    if taken is None:
        print(
            "  the receiving arm could not plan onto the slice; rejected "
            + ", ".join(f"{k} {v}" for k, v in rejected.items() if v)
        )
        handover_ok = False
    else:
        bearing, jaw, reach, close_in = taken
        jaws = "down" if jaw[2] < 0 else "up"
        note = "" if abs(bearing - base_bearing) < 0.5 else f"  (mirror bearing was {base_bearing:.0f})"
        warn = "" if jaws == args.handover_jaws else f"  (JAWS ASKED FOR {args.handover_jaws})"
        print(f"  {other} arm taking at bearing {bearing:.0f} deg, jaws {jaws}{note}{warn}")
        other_executor.set_gripper(args.pregrasp_open * other_planner.cfg.gripper_open_pos)
        other_executor.follow(reach.path, speed=0.10)
        q_taking = other_executor.follow(close_in.path, speed=0.06)
        other_executor.close_gripper(hold_arm_at=q_taking)
        other_executor.step(arm_target=q_taking, steps=40)
        both = float(np.linalg.norm(other_planner.tool_position() - object_position(target))) * 1000.0
        print(f"  both hands on the slice; receiving tool {both:.1f} mm from it")
        handover_ok = True


# ---------------------------------------------------------------------------------- insert ---
# Release, retreat, and put the slice into a slot. The receiving arm's target poses are worked out
# from the transform between tool and slice *captured at the moment it closed*, not by authoring a
# fresh grasp: once the slice is in the fingers, where the tool has to go is fully determined by
# where the slice has to go, and re-deriving it would only reintroduce the question of which edge
# is held.
insert_ok = None
if args.insert and handover_ok:
    print("\n=== insert ===")
    slice_pos, slice_rot = _object_pose_for_report(target)
    tool_pos = other_planner.tool_position()
    # The tool's own world rotation, read from the articulation rather than from any commanded pose.
    robot_bodies = list(env.scene.articulations["robot"].data.body_names)
    tool_index = robot_bodies.index(other_planner.cfg.tool_frame)
    tool_quat_xyzw = wp.to_torch(env.scene.articulations["robot"].data.body_quat_w)[0, tool_index].detach().cpu()
    import isaaclab.utils.math as math_utils

    tool_rot = math_utils.matrix_from_quat(tool_quat_xyzw.float().unsqueeze(0))[0].numpy().astype(np.float64)
    held_offset_local = slice_rot.T @ (tool_pos - slice_pos)
    held_rot_local = slice_rot.T @ tool_rot

    # The giving hand lets go first, and backs off along its own approach so it withdraws from the
    # slice rather than sweeping across it.
    executor.open_gripper(hold_arm_at=planner.joint_positions())
    release_dir = matrix_from_quat_wxyz(carry_quat)[:, 2]
    # Ladder the retreat distance: a single fixed 180 mm had no plan, and the arm then stayed
    # parked against the slice with its jaws open, which is worse than a shorter clean withdrawal.
    # Retreat sideways, to the robot's own right, rather than straight back along the approach.
    # Backing off along the approach only put 60 mm between the hands, and the giving arm stayed
    # in the receiving arm's way for everything that followed.
    back_off = None
    for distance in (0.30, 0.25, 0.20, 0.15, 0.10):
        for direction in (np.array([0.0, -1.0, 0.0]), -release_dir):
            candidate = planner.plan_pose(
                planner.joint_positions(), planner.tool_position() + direction * distance, carry_quat
            )
            if candidate is not None and candidate.is_executable():
                back_off = candidate
                sideways = "to the right" if direction[1] < 0 else "back along its approach"
                print(f"  giving hand withdrawing {distance * 1000:.0f} mm {sideways}")
                break
        if back_off is not None:
            break
    if back_off is not None:
        q_back = executor.follow(back_off.path, speed=0.12)
        executor.step(arm_target=q_back, steps=40)
        print(f"  giving hand released and backed off to {np.round(planner.tool_position(), 4)}")
    else:
        print("  the giving hand could not plan a retreat; it stays put with the jaws open")
    alone = object_position(target)
    print(f"  slice held by the {other} arm alone at {np.round(alone, 4)}")

    # Where the slice has to end up: standing in the slot, its face normal across the slot's narrow
    # axis, its origin the task's own 30-65 mm above the slot floor.
    toaster_pos, toaster_rot = _object_pose_for_report("toaster")
    x_range, y_range = SLOTS_LOCAL[args.slot]
    slot_local = np.array([0.5 * sum(x_range), 0.5 * sum(y_range), SLOT_Z_LOCAL_M])
    slot_world = toaster_pos + toaster_rot @ slot_local
    narrow = toaster_rot[:, 1]  # across the slot -- where the slice's faces must point
    along = toaster_rot[:, 0]  # down the slot's length
    print(f"  {args.slot} at {np.round(slot_world, 4)}")

    goal_pos = slot_world + np.array([0.0, 0.0, args.insert_height])
    insert_done = False
    # Once the face normal is pinned across the slot the slice still has a whole turn of freedom
    # about that normal, and every angle in it leaves the slice standing -- only which edge points
    # down changes. The tool is fixed to the slice, so each angle is an entirely different arm
    # pose, and sweeping them is what makes the insertion reachable at all. The first version
    # looped over an `along` axis it never actually used, so it tried two poses while appearing
    # to try four.
    for face in (narrow, -narrow):
        for spin_deg in range(0, 360, 30):
            col_z = face
            col_x = _rot_about(face, np.radians(spin_deg)) @ np.array([0.0, 0.0, -1.0])
            col_y = np.cross(col_z, col_x)
            slice_goal_rot = np.column_stack([col_x, col_y, col_z])
            target_tool_pos = goal_pos + slice_goal_rot @ held_offset_local
            target_tool_rot = slice_goal_rot @ held_rot_local
            quat = quat_wxyz_from_matrix(target_tool_rot)
            # Turn the slice upright high above the toaster, then come straight down into the slot.
            # Turning it at the old 100 mm stand-off swung a 117 mm slice whose lower edge reached
            # z 0.765 against a toaster 0.785 tall: the trace has it held at a steady 18 mm from
            # the tool through 0-55 deg of the turn and then thrown at 85 deg. Height is the only
            # thing that keeps the swept slice clear, so the turn happens at --insert-travel-z and
            # the descent is pure vertical.
            over = None
            for turn_z in np.arange(args.insert_travel_z, target_tool_pos[2] + 0.001, -0.03):
                above = np.array([target_tool_pos[0], target_tool_pos[1], turn_z])
                if not other_planner.ik_reachable(above, quat):
                    continue
                candidate = other_planner.plan_pose(other_planner.joint_positions(), above, quat)
                if candidate is not None and candidate.is_executable():
                    over, over_z = candidate, turn_z
                    break
            if over is None:
                continue
            for name in (TABLE_OBSTACLE, "toaster"):
                other_planner.set_obstacle_enabled(name, False)
            lower = other_planner.plan_pose(over.q_end, target_tool_pos, quat)
            for name in (TABLE_OBSTACLE, "toaster"):
                other_planner.set_obstacle_enabled(name, True)
            if lower is None or not lower.is_executable():
                continue
            print(
                f"  turning the slice upright at z {over_z:.3f}, then lowering"
                f" {(over_z - target_tool_pos[2]) * 1000:.0f} mm into the slot"
            )
            other_executor.follow(over.path, speed=0.08)
            q_low = other_executor.follow(lower.path, speed=0.05)
            other_executor.open_gripper(hold_arm_at=q_low)
            other_executor.step(arm_target=q_low, steps=60)
            insert_done = True
            break
        if insert_done:
            break

    if not insert_done:
        print("  no reachable way to stand the slice over the slot")
        insert_ok = False
    else:
        final_pos, final_rot = _object_pose_for_report(target)
        local = toaster_rot.T @ (final_pos - toaster_pos)
        in_x = x_range[0] <= local[0] <= x_range[1]
        in_y = y_range[0] <= local[1] <= y_range[1]
        height = local[2] - SLOT_Z_LOCAL_M
        # The same band the task's slot predicate tests, read from the task instead of restated.
        task = arena_env.task
        in_z = task.slot_z_lower_m <= height <= task.slot_z_upper_m
        insert_ok = bool(in_x and in_y and in_z)
        print(f"  slice in the toaster's frame: {np.round(local, 4)}, {height * 1000:.1f} mm above the slot floor")
        print(
            f"  inside the slot footprint: x {in_x}, y {in_y}; height in the"
            f" {task.slot_z_lower_m * 1000:.0f}-{task.slot_z_upper_m * 1000:.0f} mm band: {in_z}"
        )


print("\n=== outcome ===")
if pick.success:
    print(f"  picked {target} with '{pick.label}'; it rose {pick_rise_mm:.1f} mm")
    print(f"  held {pick_held_mm:.1f} mm from the slice's centre at the pick")
else:
    print(f"  failed to pick {target}: {pick.failure}")
    print(f"  the slice moved {pick_rise_mm:+.1f} mm in z")
for key in bread_keys:
    print(f"  {key} now at {np.round(object_position(key), 4)} (was {np.round(positions[key], 4)})")
if carry_ok is not None:
    print(f"  carry: {'held flat at the staging pose' if carry_ok else 'FAILED'}")
if handover_ok is not None:
    print(f"  handover: {'the other arm has it' if handover_ok else 'FAILED'}")
if insert_ok is not None:
    print(f"  insert: {'the slice stands in the slot' if insert_ok else 'FAILED'}")

if args.pose_trace is not None and pose_trace:
    trace_path = pathlib.Path(args.pose_trace)
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    with trace_path.open("w") as handle:
        handle.write("frame,seconds,x,y,z,bearing_deg,jaw_z,slice_x,slice_y,slice_z,slice_tilt_deg,hand_mm\n")
        for row in pose_trace:
            handle.write(",".join(str(value) for value in row) + "\n")
    print(f"  wrote {len(pose_trace)} tool poses to {trace_path}")

if writer is not None:
    writer.close()
    print(f"  wrote {frame_counter[0]} frames to {video_path}")
    if truncated[0]:
        print(f"  recording stopped at --max-frames {args.max_frames}; the run continued past it")

simulation_app.close()
