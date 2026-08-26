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

# Agibot base and table, shared with agibot_stack_bowls -- see there for how both were measured.
_ROBOT_POSITION_XYZ = (-0.60, 0.0, 0.0)
_TABLE_TOP_Z = 0.6232
_TABLE_POSITION_X = -0.365 + 0.5 * 1.1

# Where the toaster's origin goes. Measured, not guessed: probe_make_toast_reach sweeps the slots'
# reachability against this x and it falls off a cliff outward of 0.40 --
#
#     x    0.28  0.32  0.36  0.40  0.44  0.48
#     left    4     4     2     2     0     0     (reachable slot orientations, of 14)
#     right   0     0     0     0     0     0
#
# so 0.44 -- where this started, chosen to keep the *lever* inside the arm's band -- put the slots
# outside it entirely. 0.32 is the near end of the plateau. The right arm cannot reach the slots at
# any x, which is what makes the handover structural rather than stylistic: the right arm is the
# only one that reaches the rack, the left the only one that reaches the toaster.
_TOASTER_POSITION_XY = (0.32, 0.16)

# Yaw of pi, so the lever faces the robot. The lever (link_1) sticks out of the toaster's +X end
# at local x = +0.109; unrotated it would point away, and loading a slot would mean reaching over
# the body to get at it. RoboDojo yaws the toaster -75 to -105 degrees for the same reason, but
# its robot approaches from a different side.
_TOASTER_ROTATION_XYZW = (0.0, 0.0, 1.0, 0.0)

# The rack, off to the robot's right. Yawed +pi/2 so its four slots -- spaced along its own X --
# end up in a row across the robot's view rather than one behind another.
_SHELF_POSITION_XY = (0.40, -0.18)
_SHELF_ROTATION_XYZW = (0.0, 0.0, 0.7071067811865476, 0.7071067811865476)

# A slice standing in a slot: the rack's support frames rotate it by pi/2 about Y, which lays its
# face normal along the rack's local X, and the rack's own yaw then carries that to world Y. That
# is the same orientation a slice has once it is in a toaster slot, so a loaded slice can be
# lifted straight up, carried across and lowered in without the wrist ever turning.
#
# q = R_z(pi/2) * R_y(pi/2), written (x, y, z, w) as Isaac Lab 3.0 expects.
_BREAD_ROTATION_XYZW = (-0.5, 0.5, 0.5, 0.5)

# RoboDojo stages every task in "Simple_Room_nolight" lit by an HDRI dome; see
# agibot_stack_bowls_environment for the measurements behind these two.
_DOME_LIGHT_HDR = "brown_photostudio_robolab"
_DOME_LIGHT_INTENSITY = 1000.0


@dataclass
class AgibotMakeToastEnvironmentCfg(ArenaEnvironmentCfg):
    """Configure the Agibot toast-making environment."""

    background: str = "robodojo_table"

    embodiment: str = "agibot"

    teleop_device: str | None = "dual_arm_keyboard"
    """Must emit as many values as the arm mode consumes: ``dual_arm_keyboard`` for two arms
    (14), plain ``keyboard`` for one (7)."""

    arm_mode: str = "dual"
    """Which arm(s) to drive: ``"left"``, ``"right"`` or ``"dual"``.

    The rack sits on the robot's right and the toaster on its left, so both arms have work."""

    teleop_pos_sensitivity: float = 0.03
    teleop_rot_sensitivity: float = 0.1
    """Metres and radians of commanded end-effector motion per held key, per control step. Both
    carried over from agibot_stack_bowls, where they were tuned against the same arms."""

    num_breads: int = 4
    """How many slices to put in the rack. RoboDojo's make_toast uses four."""

    shelf_jitter_x_min_m: float = 0.0
    shelf_jitter_x_max_m: float = 0.0
    """Per-reset random x offset applied to the rack *and* its slices, as an explicit (min, max)
    pair so the range may be asymmetric: the rack's nominal x sits mid-way through the arm's
    reach band, and offsetting outward past the band's edge only produces unreachable resets.
    (Two scalars, not a tuple: the environment CLI cannot express tuple fields.)

    The rack moves as one piece: the slices are carried along rigidly (translated and rotated
    about the rack's origin), never jittered individually -- a slice offset on its own steps
    across the 25 mm slot spacing and settles against a neighbour instead of into its groove."""

    shelf_jitter_y_m: float = 0.0
    """Half-extent of the per-reset random y offset applied to the rack and its slices."""

    shelf_jitter_yaw_rad: float = 0.0
    """Half-extent of the per-reset random yaw applied to the rack and its slices, about the
    rack's origin."""

    toaster_jitter_xy_m: float = 0.0
    """Half-extent of the per-reset random offset applied to the toaster on both world axes; its
    yaw stays fixed.

    Keep it inside the slots' reachability plateau (x 0.28-0.40, measured by
    probe_make_toast_reach) or the insertion loses its target."""

    head_view: bool = True
    """Put the viewport on the robot's head, so teleop is driven from the robot's own view."""

    room: bool = True
    """Stage the task in RoboDojo's room instead of on a bare ground plane."""

    arm_effort_limit: float | None = 300.0
    """Torque ceiling for both arms, in N m. None keeps the shipped 1000-2000. See
    agibot_stack_bowls_environment for the measurements behind this value."""


def _jitter_rack_group(
    env,
    env_ids,
    asset_names: list[str],
    nominal_poses: list[tuple[tuple[float, float, float], tuple[float, float, float, float]]],
    anchor_xy: tuple[float, float],
    x_range_m: tuple[float, float],
    y_half_m: float,
    yaw_half_rad: float,
) -> None:
    """Reset event: move a group of assets rigidly by one sampled planar offset and yaw.

    One (dx, dy, dyaw) is drawn per reset and applied to every named asset about ``anchor_xy``,
    so the rack and the slices standing in it keep their relative arrangement exactly. Runs after
    the assets' own reset events, whose nominal poses it starts from.

    Args:
        env: The environment.
        env_ids: Environments being reset.
        asset_names: Scene keys to move together, e.g. the rack and its slices.
        nominal_poses: Each asset's nominal (position_xyz, rotation_xyzw), in asset order.
        anchor_xy: World xy point the yaw rotates about, normally the rack's origin.
        x_range_m: (min, max) of the uniform x offset; asymmetric ranges keep the group inside
            a reach band its nominal position does not sit in the middle of.
        y_half_m: Half-extent of the uniform y offset.
        yaw_half_rad: Half-extent of the uniform yaw.
    """
    import torch

    import isaaclab.utils.math as math_utils

    for cur_env in env_ids.tolist():
        offset_x = x_range_m[0] + float(torch.rand(1)) * (x_range_m[1] - x_range_m[0])
        offset_y = (float(torch.rand(1)) * 2.0 - 1.0) * y_half_m
        offset = torch.tensor([offset_x, offset_y])
        yaw = float((torch.rand(1) * 2.0 - 1.0) * yaw_half_rad)
        spin = torch.tensor([[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]])
        yaw_quat = math_utils.quat_from_euler_xyz(torch.tensor([0.0]), torch.tensor([0.0]), torch.tensor([yaw])).to(
            env.device
        )
        anchor = torch.tensor(anchor_xy)
        for name, (position_xyz, rotation_xyzw) in zip(asset_names, nominal_poses):
            position = torch.tensor(position_xyz)
            position[:2] = anchor + spin @ (position[:2] - anchor) + offset
            rotation = math_utils.quat_mul(
                yaw_quat, torch.tensor([list(rotation_xyzw)], device=env.device, dtype=torch.float32)
            )
            root_pose = torch.cat(
                [position.to(env.device).unsqueeze(0) + env.scene.env_origins[cur_env : cur_env + 1], rotation],
                dim=-1,
            ).float()
            asset = env.scene[name]
            asset.write_root_pose_to_sim_index(root_pose=root_pose, env_ids=torch.tensor([cur_env], device=env.device))
            # The kinematic rack and the fixed-base toaster reject velocity writes.
            from isaaclab_arena.terms.events import _velocity_is_writable

            if _velocity_is_writable(asset):
                asset.write_root_velocity_to_sim_index(
                    root_velocity=torch.zeros(1, 6, device=env.device),
                    env_ids=torch.tensor([cur_env], device=env.device),
                )


def _apply_arm_effort_limit(env_cfg, effort_limit: float | None) -> None:
    """Cap both arms' actuator torque on the compiled config, before the articulation exists.

    This has to happen on ``env_cfg.scene.robot.actuators`` rather than through
    ``write_joint_*_to_sim`` at run time: those setters reach PhysX but leave Isaac Lab's
    ``ImplicitActuator`` holding its original limit, so the two disagree.

    Args:
        env_cfg: The compiled environment configuration, patched in place.
        effort_limit: Torque ceiling in N m, or None to keep the shipped values.
    """
    if effort_limit is None:
        return
    for name, actuator in env_cfg.scene.robot.actuators.items():
        if not name.endswith("_arm"):
            continue
        # effort_limit shadows effort_limit_sim on implicit actuators; keep them equal or Isaac
        # Lab warns and picks one arbitrarily.
        actuator.effort_limit_sim = effort_limit
        actuator.effort_limit = effort_limit


@register_environment
class AgibotMakeToastEnvironment(ArenaEnvironmentFactory[AgibotMakeToastEnvironmentCfg]):
    """Load two slices of bread into a toaster and press its lever, with the Agibot."""

    name: str = "agibot_make_toast"
    _legacy_argparse_cfg_type = AgibotMakeToastEnvironmentCfg

    def build(self, cfg: AgibotMakeToastEnvironmentCfg) -> IsaacLabArenaEnvironment:
        """Build the environment from its typed configuration."""
        import isaaclab.sim as sim_utils

        from isaaclab_arena.embodiments.common.arm_mode import ArmMode
        from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
        from isaaclab_arena.scene.scene import Scene
        from isaaclab_arena.tasks.make_toast_task import MakeToastTask
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

        toaster_asset = self.asset_registry.get_asset_by_name("toaster")
        toaster = toaster_asset()
        toaster_z = _TABLE_TOP_Z + toaster_asset.HALF_HEIGHT_M
        # The toaster's own jitter is applied by the same event as the rack group's (below),
        # not by a PoseRange: the generic pose randomizer also writes a root velocity, which
        # PhysX rejects on a fixed-base articulation.
        toaster.set_initial_pose(
            Pose(position_xyz=(*_TOASTER_POSITION_XY, toaster_z), rotation_xyzw=_TOASTER_ROTATION_XYZW)
        )

        shelf_asset = self.asset_registry.get_asset_by_name("bread_shelf")
        bread_shelf = shelf_asset()
        shelf_origin_z = _TABLE_TOP_Z + shelf_asset.HALF_HEIGHT_M
        bread_shelf.set_initial_pose(
            Pose(position_xyz=(*_SHELF_POSITION_XY, shelf_origin_z), rotation_xyzw=_SHELF_ROTATION_XYZW)
        )

        assert cfg.num_breads <= len(
            shelf_asset.SLOT_X_M
        ), f"The rack has {len(shelf_asset.SLOT_X_M)} slots, asked for {cfg.num_breads} slices"
        breads = []
        for index in range(cfg.num_breads):
            bread = self.asset_registry.get_asset_by_name("bread")(instance_name=f"bread{index}")
            # The rack's yaw of +pi/2 sends its local X to world Y, so a slot's offset along the
            # rack's X becomes an offset along world Y.
            bread.set_initial_pose(
                Pose(
                    position_xyz=(
                        _SHELF_POSITION_XY[0],
                        _SHELF_POSITION_XY[1] + shelf_asset.SLOT_X_M[index],
                        shelf_origin_z + shelf_asset.SLOT_Z_M,
                    ),
                    rotation_xyzw=_BREAD_ROTATION_XYZW,
                )
            )
            breads.append(bread)

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

        scene = Scene(assets=[background, toaster, bread_shelf, *breads, surroundings, light])

        # The rack and its slices are jittered as one rigid group, so the nominal poses the event
        # transforms are captured here, where they are laid out.
        rack_group_names = [bread_shelf.name, *(bread.name for bread in breads)]
        rack_group_poses = [
            ((*_SHELF_POSITION_XY, shelf_origin_z), _SHELF_ROTATION_XYZW),
            *(
                (
                    (
                        _SHELF_POSITION_XY[0],
                        _SHELF_POSITION_XY[1] + shelf_asset.SLOT_X_M[index],
                        shelf_origin_z + shelf_asset.SLOT_Z_M,
                    ),
                    _BREAD_ROTATION_XYZW,
                )
                for index in range(cfg.num_breads)
            ),
        ]

        def env_cfg_callback(env_cfg):
            """Swap the arm terms for ones that hold their target while idle, cap arm torque, and
            attach the rack-group jitter after every asset's own reset event."""
            install_arm_target_hold(env_cfg)
            _apply_arm_effort_limit(env_cfg, cfg.arm_effort_limit)
            shelf_jittered = cfg.shelf_jitter_x_min_m or cfg.shelf_jitter_x_max_m or cfg.shelf_jitter_y_m
            if shelf_jittered or cfg.shelf_jitter_yaw_rad:
                from isaaclab.managers import EventTermCfg

                env_cfg.events.jitter_rack_group = EventTermCfg(
                    func=_jitter_rack_group,
                    mode="reset",
                    params={
                        "asset_names": rack_group_names,
                        "nominal_poses": rack_group_poses,
                        "anchor_xy": _SHELF_POSITION_XY,
                        "x_range_m": (cfg.shelf_jitter_x_min_m, cfg.shelf_jitter_x_max_m),
                        "y_half_m": cfg.shelf_jitter_y_m,
                        "yaw_half_rad": cfg.shelf_jitter_yaw_rad,
                    },
                )
            if cfg.toaster_jitter_xy_m:
                from isaaclab.managers import EventTermCfg

                env_cfg.events.jitter_toaster = EventTermCfg(
                    func=_jitter_rack_group,
                    mode="reset",
                    params={
                        "asset_names": [toaster.name],
                        "nominal_poses": [((*_TOASTER_POSITION_XY, toaster_z), _TOASTER_ROTATION_XYZW)],
                        "anchor_xy": _TOASTER_POSITION_XY,
                        "x_range_m": (-cfg.toaster_jitter_xy_m, cfg.toaster_jitter_xy_m),
                        "y_half_m": cfg.toaster_jitter_xy_m,
                        "yaw_half_rad": 0.0,
                    },
                )
            return env_cfg

        return IsaacLabArenaEnvironment(
            name=self.name,
            embodiment=embodiment,
            scene=scene,
            task=MakeToastTask(
                breads=breads,
                toaster=toaster,
                bread_shelf=bread_shelf,
                episode_length_s=240.0,
                viewer_cfg=embodiment.get_head_viewer_cfg() if cfg.head_view else None,
            ),
            teleop_device=teleop_device,
            env_cfg_callback=env_cfg_callback,
        )
