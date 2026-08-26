# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

from isaaclab_arena.assets.register import register_environment
from isaaclab_arena.environments.arena_environment_factory import ArenaEnvironmentCfg, ArenaEnvironmentFactory

if TYPE_CHECKING:
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment

# Agibot base. Same standoff as tabletop_place_upright, the reference Agibot environment.
_ROBOT_POSITION_XYZ = (-0.60, 0.0, 0.0)

# Unlike the push-T table this one is not raised -- a 60 mm bowl is tall enough to grasp from a
# surface at the arm's natural working height, which measurement puts at 0.6232.
_TABLE_TOP_Z = 0.6232

# x places the near edge at -0.365, where the previously used SeattleLabTable's edge sat and the
# robot is known to clear it; the slab is 1.1 m deep, so its centre goes half that further out.
_TABLE_POSITION_X = -0.365 + 0.5 * 1.1

# A bowl's origin is at its geometry centre, 30.1 mm above its base.
_BOWL_Z = _TABLE_TOP_Z + 0.0301

# A triangle straddling the table centre, matching how RoboDojo's demo spreads the bowls across
# the space in front of the robot rather than off to one side.
#
# Every x stays in 0.35 to 0.46, the band the arm was measured to reach at table height. The band
# has an inner limit as well as an outer one -- an earlier layout put the near bowl at 0.30, which
# the arm cannot fold in far enough to reach -- so with the +/-0.02 jitter nothing here is placed
# outside 0.35 to 0.45.
#
# Spacing has to beat the measured 0.11 m bowl diameter *after* jitter, not before it. The first
# two were 0.14 m apart, which the jitter can close to 0.10 -- so some resets spawned two bowls
# already interpenetrating, which pre-loads a contact and makes them spring apart at the first
# touch. The closest pair here is 0.171 m, so the worst case is 0.131 m.
_BOWL_POSITIONS_XY = ((0.43, -0.16), (0.43, 0.16), (0.37, 0.00))

_BOWL_X_BAND_M = (0.35, 0.45)
"""The x band the arm was measured to reach at table height; jittered bowls are clamped into it."""

_BOWL_MIN_SEPARATION_M = 0.14
"""Smallest allowed distance between jittered bowls: the 0.11 m diameter plus a contact margin."""


def _jitter_bowls(
    env, env_ids, asset_names: list[str], nominal_xy: list[tuple[float, float]], z_m: float, xy_half_m: list[float]
) -> None:
    """Reset event: re-place the bowls with a biased xy jitter, upright, a minimum distance apart.

    Per-bowl uniform offsets, with x clamped into the arm's measured reach band -- the nominal
    positions sit near its edges, so an unbiased sample wastes resets on unreachable bowls (the
    same lesson the toast rack's asymmetric range encodes). Draws are rejected until every pair
    is at least ``_BOWL_MIN_SEPARATION_M`` apart, since bowls spawned interpenetrating pre-load a
    contact and spring apart at the first touch. Orientation stays upright; only the yaw spins,
    which on a body of revolution changes nothing physical.
    """
    import torch

    import isaaclab.utils.math as math_utils

    count = len(asset_names)
    half = torch.tensor(xy_half_m).unsqueeze(1)  # per-bowl half-extent, applied to both axes
    for cur_env in env_ids.tolist():
        xy = torch.tensor(nominal_xy)
        for _ in range(200):
            candidate = torch.tensor(nominal_xy) + (torch.rand(count, 2) * 2.0 - 1.0) * half
            candidate[:, 0] = candidate[:, 0].clamp(*_BOWL_X_BAND_M)
            separations = torch.cdist(candidate, candidate) + torch.eye(count)
            if float(separations.min()) >= _BOWL_MIN_SEPARATION_M:
                xy = candidate
                break
        for name, position_xy in zip(asset_names, xy):
            yaw = torch.rand(1) * 2.0 * math.pi - math.pi
            quat = math_utils.quat_from_euler_xyz(torch.zeros(1), torch.zeros(1), yaw).to(env.device)
            position = torch.tensor([[float(position_xy[0]), float(position_xy[1]), z_m]], device=env.device)
            root_pose = torch.cat([position + env.scene.env_origins[cur_env : cur_env + 1], quat], dim=-1).float()
            asset = env.scene[name]
            asset.write_root_pose_to_sim_index(root_pose=root_pose, env_ids=torch.tensor([cur_env], device=env.device))
            asset.write_root_velocity_to_sim_index(
                root_velocity=torch.zeros(1, 6, device=env.device),
                env_ids=torch.tensor([cur_env], device=env.device),
            )


# RoboDojo's env_cfg/scene/default.yml stages every task in the "Simple_Room_nolight" room at
# scale 0.5, lit only by an HDRI dome (``brown_photostudio_02_4k.hdr``, intensity 1000). Arena
# registers the same environment map as ``brown_photostudio_robolab``. Arena's stock flat grey
# dome at intensity 500 over a bare ground plane left the tabletop almost unreadable (mean pixel
# 25/255 against 113/255 here).
_DOME_LIGHT_HDR = "brown_photostudio_robolab"
_DOME_LIGHT_INTENSITY = 1000.0


@dataclass
class AgibotStackBowlsEnvironmentCfg(ArenaEnvironmentCfg):
    """Configure the Agibot bowl-stacking environment."""

    background: str = "robodojo_table"

    embodiment: str = "agibot"

    teleop_device: str | None = "dual_arm_keyboard"
    """Must emit as many values as the arm mode consumes: ``dual_arm_keyboard`` for two arms
    (14), plain ``keyboard`` for one (7)."""

    arm_mode: str = "dual"
    """Which arm(s) to drive: ``"left"``, ``"right"`` or ``"dual"``.

    RoboDojo teleoperates stack_bowls with two arms, and some starting layouts are hard to solve
    with one, so both are driven by default."""

    teleop_pos_sensitivity: float = 0.03
    """Metres of commanded end-effector motion per held key, per control step.

    Below the device default of 0.05. The arms are position-servoed with very high stiffness and
    zero damping, and RMPFlow runs with ``ignore_robot_state_updates``, so the arm does not yield
    when it touches something -- it drives through at whatever rate it was commanded. Descending
    onto a bowl at the default rate knocks it 18 mm sideways at 0.20 m/s and the grasp misses.

    This is per *step*, so the resulting speed follows the control rate: 0.45 m/s at Arena's
    15 Hz default, against the 1.5 m/s it meant when this value was chosen at 50 Hz. Teleop is
    correspondingly slower in wall-clock, and each approach is gentler."""

    teleop_rot_sensitivity: float = 0.1
    """Radians of commanded end-effector rotation per held key, per step.

    Five times the translation sensitivity. Rotation does not carry the risk translation does --
    it is aiming the gripper at the rim, not driving into it -- and at 0.02 reorienting the hand
    took long enough to be the slow part of a grasp."""

    num_bowls: int = 3
    """How many bowls to spawn. RoboDojo's stack_bowls uses three."""

    bowl_jitter_xy_m: float = 0.02
    """Half-extent of the per-reset random xy offset applied to each bowl.

    The sample is biased, not raw uniform: x is clamped into the arm's measured reach band
    (0.35-0.45) so large jitters do not waste resets on unreachable bowls, draws are rejected
    until every pair of bowls is 0.14 m apart, and the bowls stay upright (only their yaw
    spins). Data collection uses 0.05."""

    centre_bowl_jitter_xy_m: float = 0.02
    """Half-extent for the centre bowl alone, capped tighter than the outer two.

    The centre bowl is the one the base-selection scan almost always builds the pile on -- it is
    the only position both arms can release over -- so its jitter bounds where the *pile* goes.
    Keeping it tighter keeps the releases plannable while the outer bowls roam."""

    head_view: bool = True
    """Put the viewport on the robot's head, so teleop is driven from the robot's own view."""

    room: bool = True
    """Stage the task in RoboDojo's room instead of on a bare ground plane."""

    arm_effort_limit: float | None = 300.0
    """Torque ceiling for both arms, in N m. None keeps the shipped 1000-2000.

    The Agibot arms have damping 0 and stiffness 2e4-1e7, so they do not yield when the gripper
    reaches the table -- they hold position and saturate torque instead. Measured pressing into the
    tabletop, by effort limit, with stiffness and damping left stock:

        stock (1000-2000)   2000 N m saturated   settles 0.2 mm   rebound 0.107 m/s
        500                  500 N m             0.2 mm           0.057 m/s
        300  (this default)  300 N m             0.2 mm           0.093 m/s
        200                  200 N m             0.2 mm           0.153 m/s
        100                  100 N m             ARM COLLAPSES -- droops 332 mm and stays there

    200-500 all keep tracking precision, finger penetration and release behaviour identical to
    stock while cutting the saturated torque several-fold; 300 is the middle of that band.

    Do **not** reach for the ARX X5 triple (stiffness 4400 / damping 40 / effort 100) that RoboDojo
    runs. It hits the same 100 N m ceiling but, measured: the idle arm settles to 2.4 mm instead of
    0.2 mm, the fingers rebound at 0.647 m/s instead of 0.107, and in teleoperation the softer arm
    couples with the gripper into an oscillation that drops a bowl mid-lift. Lowering the effort
    limit alone gets the torque benefit without any of that."""

    arm_stiffness: float | None = None
    arm_damping: float | None = None
    """Left at the shipped values; see ``arm_effort_limit`` for why the X5 gains are not used."""


def _apply_arm_gains(env_cfg, cfg: AgibotStackBowlsEnvironmentCfg) -> None:
    """Overwrite the arm actuators' gains on the compiled config, before the articulation exists.

    This has to happen on ``env_cfg.scene.robot.actuators`` rather than through
    ``write_joint_*_to_sim`` at run time: those setters reach PhysX but leave Isaac Lab's
    ``ImplicitActuator`` holding its original gains, so the two disagree and ``applied_torque``
    keeps reporting the old numbers. Patching the config keeps both sides consistent.

    Args:
        env_cfg: The compiled environment configuration, patched in place.
        cfg: The environment configuration carrying the overrides.
    """
    overrides = {
        "stiffness": cfg.arm_stiffness,
        "damping": cfg.arm_damping,
        "effort_limit_sim": cfg.arm_effort_limit,
    }
    overrides = {key: value for key, value in overrides.items() if value is not None}
    if not overrides:
        return

    for name, actuator in env_cfg.scene.robot.actuators.items():
        if not name.endswith("_arm"):
            continue
        for key, value in overrides.items():
            setattr(actuator, key, value)
            # effort_limit shadows effort_limit_sim on implicit actuators; keep them equal or
            # Isaac Lab warns and picks one arbitrarily.
            if key == "effort_limit_sim":
                actuator.effort_limit = value
        print(f"[arm gains] {name}: " + ", ".join(f"{k}={v:g}" for k, v in overrides.items()))


@register_environment
class AgibotStackBowlsEnvironment(ArenaEnvironmentFactory[AgibotStackBowlsEnvironmentCfg]):
    """Stack three bowls into one pile with the Agibot."""

    name: str = "agibot_stack_bowls"
    _legacy_argparse_cfg_type = AgibotStackBowlsEnvironmentCfg

    def build(self, cfg: AgibotStackBowlsEnvironmentCfg) -> IsaacLabArenaEnvironment:
        """Build the environment from its typed configuration."""
        import isaaclab.sim as sim_utils

        from isaaclab_arena.embodiments.common.arm_mode import ArmMode
        from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
        from isaaclab_arena.scene.scene import Scene
        from isaaclab_arena.tasks.stack_bowls_task import StackBowlsTask
        from isaaclab_arena.utils.arm_target_hold import install_arm_target_hold
        from isaaclab_arena.utils.pose import Pose

        table_asset = self.asset_registry.get_asset_by_name(cfg.background)
        background = table_asset()
        background.set_initial_pose(
            Pose(
                position_xyz=(_TABLE_POSITION_X, 0.0, _TABLE_TOP_Z - table_asset.HALF_THICKNESS_M),
                rotation_xyzw=(0.0, 0.0, 0.0, 1.0),
            )
        )

        assert cfg.num_bowls <= len(
            _BOWL_POSITIONS_XY
        ), f"Only {len(_BOWL_POSITIONS_XY)} bowl positions are laid out, asked for {cfg.num_bowls}"
        # Nominal upright poses; the biased group jitter below re-places them on every reset.
        bowls = []
        for index in range(cfg.num_bowls):
            x, y = _BOWL_POSITIONS_XY[index]
            bowl = self.asset_registry.get_asset_by_name("bowl")(instance_name=f"bowl{index}")
            bowl.set_initial_pose(Pose(position_xyz=(x, y, _BOWL_Z), rotation_xyzw=(0.0, 0.0, 0.0, 1.0)))
            bowls.append(bowl)

        arm_mode = {"left": ArmMode.LEFT, "right": ArmMode.RIGHT, "dual": ArmMode.DUAL_ARM}[cfg.arm_mode]
        embodiment = self.asset_registry.get_asset_by_name(cfg.embodiment)(
            enable_cameras=cfg.enable_cameras, arm_mode=arm_mode
        )
        embodiment.set_initial_pose(Pose(position_xyz=_ROBOT_POSITION_XYZ, rotation_xyzw=(0.0, 0.0, 0.0, 1.0)))

        if cfg.teleop_device is not None:
            teleop_device = self.device_registry.get_device_by_name(cfg.teleop_device)(
                pos_sensitivity=cfg.teleop_pos_sensitivity,
                rot_sensitivity=cfg.teleop_rot_sensitivity,
            )
        else:
            teleop_device = None

        # A fresh DomeLightCfg per instance: the asset's default is a class attribute, so reusing
        # it would leak the HDR texture into every other environment built in the same process.
        light = self.asset_registry.get_asset_by_name("light")(
            spawner_cfg=sim_utils.DomeLightCfg(intensity=_DOME_LIGHT_INTENSITY),
            hdr=self.hdr_registry.get_hdr_by_name(_DOME_LIGHT_HDR)(),
        )

        if cfg.room:
            # The room brings its own floor, so it replaces the default grid ground plane.
            room_asset = self.asset_registry.get_asset_by_name("robodojo_simple_room")
            surroundings = room_asset()
            surroundings.object_cfg.spawn.scale = room_asset.SCALE
        else:
            surroundings = self.asset_registry.get_asset_by_name("ground_plane")()

        scene = Scene(assets=[background, *bowls, surroundings, light])

        def env_cfg_callback(env_cfg):
            """Swap the arm terms for ones that hold their target while idle, retune the arms,
            and attach the bowls' biased group jitter after their own reset events."""
            install_arm_target_hold(env_cfg)
            _apply_arm_gains(env_cfg, cfg)
            if cfg.bowl_jitter_xy_m:
                from isaaclab.managers import EventTermCfg

                # The centre bowl -- index 2 in the layout, the (0.37, 0.00) one -- gets its own
                # tighter half-extent; see centre_bowl_jitter_xy_m.
                per_bowl = [
                    min(cfg.bowl_jitter_xy_m, cfg.centre_bowl_jitter_xy_m) if index == 2 else cfg.bowl_jitter_xy_m
                    for index in range(cfg.num_bowls)
                ]
                env_cfg.events.jitter_bowls = EventTermCfg(
                    func=_jitter_bowls,
                    mode="reset",
                    params={
                        "asset_names": [bowl.name for bowl in bowls],
                        "nominal_xy": [list(_BOWL_POSITIONS_XY[index]) for index in range(cfg.num_bowls)],
                        "z_m": _BOWL_Z,
                        "xy_half_m": per_bowl,
                    },
                )
            return env_cfg

        return IsaacLabArenaEnvironment(
            name=self.name,
            embodiment=embodiment,
            scene=scene,
            task=StackBowlsTask(
                bowls=bowls,
                episode_length_s=120.0,
                viewer_cfg=embodiment.get_head_viewer_cfg() if cfg.head_view else None,
            ),
            teleop_device=teleop_device,
            env_cfg_callback=env_cfg_callback,
        )
