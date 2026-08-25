# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Stack the bowls in ``agibot_stack_bowls`` with cuMotion-planned motions.

The bowls carry no grasp annotation -- RoboDojo's metadata gives them only ``active.place.up`` --
but they are bodies of revolution, and that same annotation states the rim radius. So the grasp is
not a point to be guessed at but the whole rim circle, and the angle around it is handed to the
planner as a free parameter alongside the approach lean.

Usage::

    OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y .venv/bin/python \\
        isaaclab_arena_cumotion/scripts/stack_bowls_cumotion.py --headless \\
        --video /home/ubuntu/playground/stack_bowls_cumotion.mp4
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--env", type=str, default="agibot_stack_bowls")
parser.add_argument("--video", type=str, default="/home/ubuntu/playground/stack_bowls_cumotion.mp4")
parser.add_argument("--record-every", type=int, default=3, help="Keep one frame in N.")
parser.add_argument("--fps", type=int, default=30)
parser.add_argument(
    "--stack-pitch",
    type=float,
    default=0.030,
    help="Height added per bowl in the pile. Nested bowls sit lower than their 60 mm height.",
)
parser.add_argument(
    "--rim-grasp-depth",
    type=float,
    default=0.020,
    help=(
        "How far below the rim edge to close the fingers. The rim is 30 mm above the bowl's"
        " centre, so this cannot exceed that; too small and the jaws pinch the very lip, which is"
        " thin and lets the bowl pivot."
    ),
)
parser.add_argument(
    "--jaw-spin",
    type=float,
    nargs="+",
    default=(-90.0, 90.0),
    help="Spins about the tool approach axis that aim the jaws. Both signs are the same grasp.",
)
parser.add_argument(
    "--release-open",
    type=float,
    default=0.5,
    help=(
        "Fraction of the gripper's full opening to let go with. A rim pinch holds a ~3 mm wall, but"
        " opening fully sweeps each finger about a rim radius, so the inner one crosses the bowl and"
        " hooks the far wall. Pass 1.0 to open fully."
    ),
)
parser.add_argument(
    "--movers",
    type=str,
    nargs="+",
    default=None,
    help="Only carry these bowls onto the base, in this order. Defaults to all of the others.",
)
parser.add_argument(
    "--max-grasp-attempts",
    type=int,
    default=4,
    help=(
        "How many grasps to try per bowl. Whether a grasp admits a plannable carry to the pile is"
        " only discoverable once the bowl is held, so a place failure puts it back and tries the"
        " next candidate rather than giving up on the bowl."
    ),
)
parser.add_argument(
    "--base-bowl",
    type=str,
    default=None,
    help="Bowl to build the pile on. Defaults to whichever has the most reachable release poses.",
)
parser.add_argument("--jitter", type=float, default=0.0, help="Per-reset bowl xy jitter, in metres.")
parser.add_argument(
    "--stock-arm-effort",
    action="store_true",
    help="Keep the shipped 1000-2000 N m arm torque ceiling instead of the environment's 300.",
)
parser.add_argument(
    "--control-rate-50hz",
    action="store_true",
    help="Run at sim dt 1/200 with decimation 4, the rate this task was originally tuned at.",
)
AppLauncher.add_app_launcher_args(parser)
args, _ = parser.parse_known_args()
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
from isaaclab_arena.assets.registries import AssetRegistry, EnvironmentRegistry  # noqa: E402
from isaaclab_arena.cli.isaaclab_arena_cli import (  # noqa: E402
    arena_env_builder_cfg_from_argparse,
    get_isaaclab_arena_cli_parser,
)
from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder  # noqa: E402
from isaaclab_arena_cumotion.executor import ArmExecutor  # noqa: E402
from isaaclab_arena_cumotion.grasps import matrix_from_quat_wxyz, quat_wxyz_from_matrix, rim_grasps  # noqa: E402
from isaaclab_arena_cumotion.pick_place import PickAndPlace  # noqa: E402
from isaaclab_arena_cumotion.planner import CumotionArmPlanner  # noqa: E402

# RoboDojo's Rigid/bowl/00001 metadata. The projection circle is the rim (0.0551 matches the
# 0.11 m aligned-bbox width exactly); the contact circle is the much smaller foot ring the bowl
# actually stands on, which is the right scale for judging whether a pile is aligned.
BOWL_RIM_RADIUS_M = 0.0551
BOWL_FOOT_RADIUS_M = 0.0235
BOWL_HALF_HEIGHT_M = 0.0301

RIM_TILTS_DEG = (10, 20, 30, 40, 50)
"""Leans of the tool away from vertical. Straight top-down has no IK solution at this reach."""

TABLE_TOP_Z = 0.6232
TABLE_OBSTACLE = "/obstacles/table"

# ------------------------------------------------------------------------------------- env ---
arena_args = get_isaaclab_arena_cli_parser().parse_args(["--num_envs", "1", "--enable_cameras"])
factory = EnvironmentRegistry().get_component_by_name(args.env)()
env_cfg = factory._legacy_argparse_cfg_type(
    teleop_device=None,
    bowl_jitter_xy_m=args.jitter,
    **({"arm_effort_limit": None} if args.stock_arm_effort else {}),
)
arena_env = factory.build(env_cfg)
if args.control_rate_50hz:
    from isaaclab_arena.environments.isaaclab_arena_manager_based_env_cfg import set_control_rate_50hz  # noqa: E402

    _prev_cb = arena_env.env_cfg_callback

    def _fifty_hz(patched):
        patched = _prev_cb(patched) if _prev_cb is not None else patched
        return set_control_rate_50hz(patched)

    arena_env.env_cfg_callback = _fifty_hz
builder = ArenaEnvBuilder(arena_env, arena_env_builder_cfg_from_argparse(arena_args))
env = builder.make_registered().unwrapped

camera = Camera(
    CameraCfg(
        prim_path="/World/demo_cam",
        height=720,
        width=1280,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(focal_length=20.0, clipping_range=(0.05, 30.0)),
    )
)
env.sim.reset()
env.reset()
camera.set_world_poses_from_view(
    eyes=torch.tensor([[1.20, 0.85, 1.15]], device=env.device),
    targets=torch.tensor([[0.40, 0.00, 0.68]], device=env.device),
)

frames: list[np.ndarray] = []
step_counter = [0]


def grab_frame() -> None:
    """Keep one camera frame in ``--record-every``; the servo runs far faster than 30 fps.

    The sensor has to be told to update: it is not part of the scene the environment steps, so
    without this its buffer never changes and every recorded frame is identical.
    """
    step_counter[0] += 1
    if step_counter[0] % args.record_every != 0:
        return
    camera.update(env.sim.get_physics_dt())
    frames.append(camera.data.output["rgb"][0, ..., :3].detach().cpu().numpy().astype(np.uint8))


# -------------------------------------------------------------------------------- planning ---
# Both arms, because the layout needs both: RoboDojo spreads the bowls either side of the robot's
# centreline and teleoperates this task two-handed. Each bowl is handled by the arm on its own
# side, which is also what keeps the reach off the far edge of a workspace -- the -y bowl gives
# the left arm 4 rim solutions against the right arm's 38, and the +y bowl the reverse.
embodiment = arena_env.embodiment
arms = {}
for arm in ("left", "right"):
    planner = CumotionArmPlanner(env, embodiment, arm=arm)
    error_m = planner.kinematics_error_m()
    print(f"{arm} arm kinematics cross-check: {error_m * 1000:.2f} mm")
    assert error_m < 1e-3, f"cuMotion's {arm}-arm kinematics disagree with the simulated robot"
    # The work surface, modelled 30 mm below the real top. A bowl rim sits only 60 mm above the
    # table, so a flush slab plus any safety margin leaves the grasp itself unplannable, while
    # gross sweeps through the table are still blocked -- which is what the slab is for.
    planner.add_box_obstacle(
        TABLE_OBSTACLE, np.array([0.185, 0.0, TABLE_TOP_Z - 0.09]), (2.2, 1.4, 0.12), safety_tolerance_m=0.0
    )
    arms[arm] = planner

bowl_keys = sorted(key for key in env.scene.rigid_objects if key.startswith("bowl"))
assert len(bowl_keys) >= 2, f"Expected bowls in the scene, found {bowl_keys}"
for planner in arms.values():
    for key in bowl_keys:
        planner.add_scene_object_obstacle(key, (2 * BOWL_RIM_RADIUS_M, 2 * BOWL_RIM_RADIUS_M, 2 * BOWL_HALF_HEIGHT_M))

executors = {arm: ArmExecutor(env, planner, on_step=grab_frame) for arm, planner in arms.items()}
pick_places = {arm: PickAndPlace(arms[arm], executors[arm], contact_obstacles=(TABLE_OBSTACLE,)) for arm in arms}

# Captured before anything moves. An arm that stays where its last release left it is parked over
# the pile, where it is both an obstacle to the other arm and a pose no reset would ever produce;
# sending it home keeps each bowl's demonstration starting from the same configuration.
home_q = {arm: planner.joint_positions() for arm, planner in arms.items()}


def send_home(arm: str) -> None:
    """Return an arm to the configuration it started in, and reopen its gripper."""
    planner = arms[arm]
    executors[arm].open_gripper()
    plan = planner.plan_config(planner.joint_positions(), home_q[arm])
    if plan is None or not plan.is_executable():
        print(f"  the {arm} arm could not plan its way home")
        return
    executors[arm].follow(plan.path, speed=0.2)
    error_rad = float(np.max(np.abs(planner.joint_positions() - home_q[arm])))
    print(f"  the {arm} arm is home, within {np.degrees(error_rad):.1f} deg")


def bowl_position(key: str) -> np.ndarray:
    """A bowl's world position."""
    return wp.to_torch(env.scene[key].data.root_pos_w)[0].detach().cpu().numpy().astype(np.float64)


def spin_release_poses(centre: np.ndarray, offset: np.ndarray, rotation: np.ndarray, step_deg: int = 15):
    """Release poses around the vertical, for an object held ``offset`` off its own axis.

    A bowl is a body of revolution: spinning the wrist about the vertical spins the bowl and
    nothing else, so this whole circle is equivalent as far as the task is concerned and can be
    handed to the planner as one more free parameter.
    """
    poses = []
    for spin_deg in range(0, 360, step_deg):
        spin = np.radians(spin_deg)
        spin_matrix = np.array([[np.cos(spin), -np.sin(spin), 0.0], [np.sin(spin), np.cos(spin), 0.0], [0.0, 0.0, 1.0]])
        poses.append(
            (f"spin {spin_deg} deg", centre + spin_matrix @ offset, quat_wxyz_from_matrix(spin_matrix @ rotation))
        )
    return poses


# Which bowl the pile is built on is a reachability question, not a distance one: the arms sit
# either side of the robot and the bowl nearest the base can easily be the one neither can release
# over. Score each candidate base by how many release poses are reachable above it, using a
# nominal rim-held offset, and keep the best.
positions = {key: bowl_position(key) for key in bowl_keys}
nominal_offset = np.array([BOWL_RIM_RADIUS_M, 0.0, 0.0])


# The lean has to be swept here as well as the spin: a rim grasp is never taken straight from
# above (top-down IK does not solve at this reach), so a release pose scored with an unleaned
# wrist is not one the arm will ever actually be asked for.
def _leaned(tilt_deg: float) -> np.ndarray:
    tilt = np.radians(tilt_deg)
    lean = np.array([[np.cos(tilt), 0.0, np.sin(tilt)], [0.0, 1.0, 0.0], [-np.sin(tilt), 0.0, np.cos(tilt)]])
    return lean @ np.diag([1.0, -1.0, -1.0])


def _near_arm(position: np.ndarray) -> str:
    """The arm on the same side of the robot as a position."""
    return "left" if position[1] > arms["left"].base_pos[1] else "right"


base_scores = {}
for key in bowl_keys:
    centre = positions[key] + np.array([0.0, 0.0, args.stack_pitch])
    per_arm = {}
    for arm, planner in arms.items():
        per_arm[arm] = sum(
            int(planner.ik_reachable(position, quat))
            for tilt_deg in RIM_TILTS_DEG
            for _, position, quat in spin_release_poses(centre, nominal_offset, _leaned(tilt_deg))
        )
    # Score by the *weakest* arm that will actually be asked to release over this base, not by the
    # better of the two. Each bowl is carried by the arm on its own side, so a base scoring highly
    # only for the left arm is useless if a bowl on the right has to be placed onto it -- and
    # taking the max picks exactly that base, then strands the carry with no reachable release.
    carriers = {_near_arm(positions[other]) for other in bowl_keys if other != key}
    base_scores[key] = min(per_arm[arm] for arm in carriers)
    print(
        f"  {key} as base: "
        + ", ".join(f"{a} {n}" for a, n in per_arm.items())
        + f"  (carried onto by {sorted(carriers)} -> score {base_scores[key]})"
    )
if args.base_bowl is not None:
    assert args.base_bowl in bowl_keys, f"--base-bowl {args.base_bowl} is not one of {bowl_keys}"
    base_key = args.base_bowl
else:
    base_key = max(bowl_keys, key=lambda k: base_scores[k])
assert base_scores[base_key] > 0, "no bowl can be released over by either arm"
if args.movers is not None:
    assert base_key not in args.movers, f"--movers may not include the base bowl {base_key}"
    for key in args.movers:
        assert key in bowl_keys, f"--movers {key} is not one of {bowl_keys}"
    movers = list(args.movers)
else:
    # Left-side bowls first, then nearest-first within a side. The arms take turns rather than
    # working at once, so the order is free to be the legible one: each arm's whole contribution
    # is one contiguous stretch of the demonstration, from its home configuration back to it.
    movers = sorted(
        (k for k in bowl_keys if k != base_key),
        key=lambda k: (
            _near_arm(positions[k]) != "left",
            np.linalg.norm(positions[k][:2] - positions[base_key][:2]),
        ),
    )
print(f"base bowl: {base_key}; carrying {movers} onto it")
# Each bowl is aimed at the one currently on top of the pile, not at the base. The success
# predicate measures *neighbouring* bowls, so aiming every bowl at the base makes each one's error
# add to the one below it -- two placements 15 and 30 mm off the base axis can be 44 mm apart.
top_key = base_key

# ------------------------------------------------------------------------------- execution ---
report: list[str] = []
placed = 1
for mover in movers:
    print(f"\n--- {mover} -> {base_key} (pile of {placed + 1}) ---")
    others = [positions[k] for k in bowl_keys if k != mover]
    proposals = rim_grasps(
        env,
        mover,
        rim_radius_m=BOWL_RIM_RADIUS_M,
        rim_height_m=BOWL_HALF_HEIGHT_M,
        num_angles=36,
        tilt_deg=RIM_TILTS_DEG,
        z_offset_m=-args.rim_grasp_depth,
        avoid_positions=others,
        jaw_spin_deg=args.jaw_spin,
    )
    print(f"  {len(proposals)} rim grasp candidates after clearing the other bowls")

    # Hand assignment follows the side of the robot the bowl is on, not whichever arm happens to
    # have the most solutions: reaching across the centreline is what puts an arm at the edge of
    # its workspace, and it is also what would have the two arms fouling each other in a version
    # of this that ran them concurrently. The far arm is only used if the near one cannot do it,
    # and that fallback is reported rather than silently taken.
    near_arm = _near_arm(positions[mover])
    far_arm = "right" if near_arm == "left" else "left"
    pile_centre = bowl_position(base_key) + np.array([0.0, 0.0, placed * args.stack_pitch])

    # A grasp is only worth taking if the arm can also *release* from it. How the bowl ends up in
    # the fingers fixes the tool's offset and orientation relative to the bowl for the rest of the
    # move, so a candidate chosen on approach travel alone routinely lifts the bowl and then
    # strands it: the release poses that grasp implies are outside the arm's workspace even though
    # the base-bowl scan -- which sweeps its own leans against a nominal offset -- said the base was
    # reachable. Score each candidate on the release it actually leads to, using the same offset
    # and rotation the place stage will derive from it.
    def _placeable(arm: str, proposal) -> bool:
        offset = proposal.position - positions[mover]
        rotation = matrix_from_quat_wxyz(proposal.quat_wxyz)
        return any(
            arms[arm].ik_reachable(position, quat)
            for _, position, quat in spin_release_poses(pile_centre, offset, rotation)
        )

    reach = {arm: [p for p in proposals if arms[arm].ik_reachable(p.position, p.quat_wxyz)] for arm in arms}
    placeable = {arm: [p for p in candidates if _placeable(arm, p)] for arm, candidates in reach.items()}
    print(
        "  reachable rim candidates: "
        + ", ".join(f"{a} {len(n)} ({len(placeable[a])} with a reachable release)" for a, n in reach.items())
    )
    arm = near_arm if placeable[near_arm] or not placeable[far_arm] else far_arm
    if placeable[arm]:
        proposals = placeable[arm]
    elif reach[arm]:
        # Better to try and fail at the release than to refuse to pick at all.
        print("  no candidate has a reachable release; going ahead on grasp reachability alone")
        proposals = reach[arm]
    else:
        arm = far_arm if reach[far_arm] else arm
        proposals = reach[arm]
    if not proposals:
        report.append(f"{mover}: unreachable by either arm")
        break
    side = "y > 0" if near_arm == "left" else "y < 0"
    if arm == near_arm:
        print(f"  using the {arm} arm (bowl is at {side}, its own side), {len(proposals)} candidates")
    else:
        print(f"  falling back to the {arm} arm: the near {near_arm} arm has no usable candidate")
    planner, pick_place = arms[arm], pick_places[arm]

    # The bowl being picked must not be an obstacle to its own grasp.
    for other in arms.values():
        other.set_obstacle_enabled(mover, False)
    start_z = positions[mover][2]

    # A grasp that lifts cleanly can still be one the arm cannot carry from: whether the release
    # poses it implies are *plannable* -- as opposed to merely IK-reachable, which is what the
    # candidates were filtered on -- only comes out once the bowl is in the fingers. So a place
    # failure retires that grasp rather than the bowl: set it back down where it came from, and
    # pick again from the candidates not yet tried.
    pick = place = None
    tried: set[str] = set()
    for attempt in range(args.max_grasp_attempts):
        untried = [p for p in proposals if p.label not in tried]
        if not untried:
            break
        if attempt:
            print(f"  retrying with a different grasp ({len(untried)} candidates left)")
        pick = pick_place.pick(
            untried,
            mute_during_descent=(),
            verify_grasp=lambda: bowl_position(mover)[2] - start_z > 0.02,
        )
        for line in pick.trace:
            print(f"  {line}")
        if not pick.success:
            break
        tried.add(pick.label)

        # The bowl is held by its rim, so the tool sits ~one rim radius off the bowl's own axis.
        # Releasing with the tool over the pile's centre would leave the bowl that far off it,
        # which is more than the 40 mm the success predicate allows. Measure the offset from the
        # grasp that actually happened and carry it to the release.
        print(f"  bowl rose {(bowl_position(mover)[2] - positions[mover][2]) * 1000:.1f} mm during the lift")
        grasp_offset = planner.tool_position() - bowl_position(mover)
        grasp_rotation = matrix_from_quat_wxyz(pick.grasp_quat_wxyz)
        print(f"  held {np.linalg.norm(grasp_offset[:2]) * 1000:.1f} mm off the bowl's axis")
        for other in arms.values():
            other.update_obstacle_pose(base_key)
        # A bowl is a body of revolution, so spinning the wrist about the vertical on the way to
        # the pile only spins the bowl -- it stays upright and still nests. That spare degree of
        # freedom is what makes a release pose reachable at all when the grasp itself was at the
        # edge of the arm's workspace, so the whole circle is offered to the planner.
        pile_centre = bowl_position(base_key) + np.array([0.0, 0.0, placed * args.stack_pitch])
        release_targets = spin_release_poses(pile_centre, grasp_offset, grasp_rotation)
        def _bowl_report() -> str:
            here, base_here = bowl_position(mover), bowl_position(base_key)
            now_offset = planner.tool_position() - here
            # The offset is the whole basis of the release aiming: if it has changed since the
            # grasp, the bowl moved in the fingers and no amount of aiming would have helped.
            return (
                f"{mover} z {here[2]:.4f} ({(here[2] - base_here[2]) * 1000:+.0f} mm over the base),"
                f" {np.linalg.norm(here[:2] - base_here[:2]) * 1000:.1f} mm off its axis;"
                f" tool-to-bowl {np.linalg.norm(now_offset[:2]) * 1000:.1f} mm"
                f" (was {np.linalg.norm(grasp_offset[:2]) * 1000:.1f} at the grasp,"
                f" moved {np.linalg.norm(now_offset - grasp_offset) * 1000:.1f} mm since)"
            )

        place = pick_place.place(
            release_targets,
            approach_height_m=0.12,
            mute_during_descent=(base_key,),
            release_gripper_pos=args.release_open * planner.cfg.gripper_open_pos,
            retarget=lambda: pile_centre + (planner.tool_position() - bowl_position(mover)),
            on_release=_bowl_report,
        )
        for line in place.trace:
            print(f"  {line}")
        if place.success:
            break

        # Retrace the descent to put the bowl back, open, and retreat. The paths are replayed, not
        # replanned, so this needs no obstacle bookkeeping -- the way in is known to be clear.
        executors[arm].follow(pick.descend_path, speed=0.2)
        executors[arm].open_gripper(hold_arm_at=planner.joint_positions())
        executors[arm].follow(pick.descend_path, reverse=True)

    for other in arms.values():
        other.set_obstacle_enabled(mover, True)
        other.update_obstacle_pose(mover)
    send_home(arm)
    if pick is None or not pick.success:
        report.append(f"{mover}: pick failed ({pick.failure if pick else 'no candidate left'})")
        break
    if not place.success:
        report.append(f"{mover}: place failed ({place.failure}), {len(tried)} grasps tried")
        break

    executors[arm].step(steps=120)
    settled = bowl_position(mover)
    offset_mm = np.linalg.norm(settled[:2] - bowl_position(base_key)[:2]) * 1000
    report.append(f"{mover}: placed, {offset_mm:.1f} mm off the base bowl's axis, z {settled[2]:.4f}")
    print(f"  {report[-1]}")
    placed += 1

# ---------------------------------------------------------------------------------- verdict ---
print("\n=== result ===")
for line in report:
    print(f"  {line}")
print("  heights: " + ", ".join(f"{k} {bowl_position(k)[2]:.4f}" for k in bowl_keys))

# The predicate is a conjunction over every bowl, so a bare False says nothing about which motion
# to go and fix. Report the pile the way the predicate reads it -- neighbours in height order --
# against the thresholds it uses.
task = arena_env.task
by_height = sorted(bowl_keys, key=lambda k: bowl_position(k)[2])
for lower, upper in zip(by_height, by_height[1:]):
    low, high = bowl_position(lower), bowl_position(upper)
    xy = float(np.linalg.norm(high[:2] - low[:2]))
    gap = float(high[2] - low[2])
    ok_xy = xy <= task.stack_xy_threshold_m
    ok_gap = task.min_z_gap_m <= gap <= (task.max_z_gap_m if task.max_z_gap_m is not None else np.inf)
    print(
        f"  {lower} -> {upper}: {xy * 1000:.1f} mm apart (limit {task.stack_xy_threshold_m * 1000:.0f})"
        f" {'ok' if ok_xy else 'TOO FAR'}, z gap {gap * 1000:.1f} mm"
        f" (limit {task.min_z_gap_m * 1000:.0f}-{task.max_z_gap_m * 1000:.0f}) {'ok' if ok_gap else 'OUT OF RANGE'}"
    )
print(f"  task success predicate: {bool(task.is_success(env)[0].item())}")

import imageio.v3 as iio  # noqa: E402

out = pathlib.Path(args.video)
out.parent.mkdir(parents=True, exist_ok=True)
if frames:
    iio.imwrite(out, np.stack(frames), fps=args.fps, codec="libx264")
    print(f"  wrote {len(frames)} frames to {out}")
else:
    print("  no frames recorded")

simulation_app.close()
