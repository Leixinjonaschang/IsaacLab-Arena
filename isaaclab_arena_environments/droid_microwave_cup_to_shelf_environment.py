# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from isaaclab_arena.assets.register import register_environment
from isaaclab_arena.environments.arena_environment_factory import ArenaEnvironmentCfg, ArenaEnvironmentFactory

if TYPE_CHECKING:
    from isaaclab_arena.assets.object import Object
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
    from isaaclab_arena.utils.pose import PoseRange

# Layout constants for the maple-table scene. The DROID base sits at the world origin and the
# table top of ``maple_table_robolab`` is 3 mm above it, so every asset below is placed in the
# robot's own frame.
TABLE_TOP_Z_M = 0.003
"""Height of the maple table's top surface in the DROID base frame."""

# ``Microwave039`` geometry, read off its USD in the asset's own frame (Z up, origin at the body
# centre): the shell spans +-0.1696 m in Z and the hinged door faces -Y.
MICROWAVE_HALF_HEIGHT_M = 0.1696
"""Distance from the microwave origin down to the foot of its shell."""
MICROWAVE_TURNTABLE_DROP_M = 0.1369
"""Distance from the microwave origin down to the turntable's *collision* surface.

The turntable's colliders (a thin cylinder plus a zero-thickness convex hull) sit 4.8 mm below the
visual surface they stand in for. An object spawned on the visual surface therefore drops onto a
degenerate hull and is flung across the cavity, so the cup is seated on the collision surface.
"""
MICROWAVE_TURNTABLE_OFFSET_XY_M = (0.0213, 0.0596)
"""Turntable centre relative to the microwave origin, after the -90 deg yaw below."""
MICROWAVE_TURNTABLE_RADIUS_M = 0.1419
"""Turntable collider radius, which bounds how far the cup may be nudged towards the door."""

MICROWAVE_XY_M = (0.62, 0.22)
"""Microwave footprint centre. Placed towards +Y so the door sweeps clear of the robot base."""
MICROWAVE_YAW_ROTATION_XYZW = (0.0, 0.0, -0.7071068, 0.7071068)
"""-90 deg yaw, which turns the microwave's -Y door face towards the robot (-X)."""

CUP_TOWARDS_DOOR_OFFSET_M = 0.06
"""How far the cup is nudged from the turntable centre towards the door, in -X."""
CUP_RESET_JITTER_XY_M = 0.02
"""Half-extent of the cup's per-reset position randomisation on the turntable."""
CUP_CLEARANCE_M = 0.002
"""Gap between the cup and the turntable collision surface. Keep it small: a taller drop onto the
turntable's degenerate collider throws the cup out of the microwave."""

SHELF_XY_M = (0.45, -0.28)
"""Shelf footprint centre, on the -Y half of the table opposite the microwave door swing."""


@dataclass
class DroidMicrowaveCupToShelfEnvironmentCfg(ArenaEnvironmentCfg):
    """Configure the DROID open-microwave / cup-to-shelf environment."""

    embodiment: str = "droid_abs_joint_pos"
    """DROID arm with absolute joint-position actions, the action space GR00T N1.6-DROID emits."""
    cup: str = "gregorys_coffee_cup_objaverse_robolab"
    """Object that starts on the turntable.

    The default is 70 mm wide, inside the Robotiq 2F-85's 85 mm stroke, and is the shape verified to
    settle on the microwave's turntable. Wider mugs cannot be grasped by the body, and rounder
    colliders slide off the turntable, so swapping this needs a fresh layout check.
    """
    shelf: str = "wireshelving_a01_vomp_robolab"
    """Object the cup has to end up on."""
    shelf_scale: float = 0.4
    """Uniform scale of the shelf, shrinking the floor-standing unit to a tabletop rack."""
    openness_threshold: float = 0.5
    """Fraction of the door's travel that counts as open."""
    hdr: str | None = None
    """Optional HDR dome, matching the DROID pick-and-place environment's lighting options."""
    light_intensity: float = 500.0
    """Dome light intensity."""
    episode_length_s: float = 90.0
    """Budget for opening the door, retrieving the cup and placing it."""
    teleop_device: str | None = None
    """Optional teleoperation device used to record demonstrations."""

    def __post_init__(self) -> None:
        assert self.episode_length_s > 0.0, "episode_length_s must be greater than zero"
        assert self.shelf_scale > 0.0, "shelf_scale must be greater than zero"
        assert 0.0 < self.openness_threshold < 1.0, "openness_threshold must lie strictly between 0 and 1"


@register_environment
class DroidMicrowaveCupToShelfEnvironment(ArenaEnvironmentFactory[DroidMicrowaveCupToShelfEnvironmentCfg]):
    """Open the microwave, take the cup out of it and place the cup on the shelf.

    A two-step sequential task on the DROID maple-table setup: the arm first swings the microwave
    door open, then lifts the cup off the turntable and sets it down on a tabletop shelf.
    """

    name: str = "droid_microwave_cup_to_shelf"
    _legacy_argparse_cfg_type = DroidMicrowaveCupToShelfEnvironmentCfg

    def build(self, cfg: DroidMicrowaveCupToShelfEnvironmentCfg) -> IsaacLabArenaEnvironment:
        """Build the environment from its typed configuration."""
        from isaaclab.sim.schemas import RigidBodyBaseCfg

        from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
        from isaaclab_arena.scene.scene import Scene
        from isaaclab_arena.tasks.open_door_task import OpenDoorTask
        from isaaclab_arena.tasks.pick_and_place_task import PickAndPlaceTask
        from isaaclab_arena.tasks.sequential_composite_tasks.open_door_and_place_object_task import (
            OpenDoorAndPlaceObjectTask,
        )
        from isaaclab_arena.utils.pose import Pose

        # Step 1: Retrieve assets from the registry
        background = self.asset_registry.get_asset_by_name("maple_table_robolab")()
        microwave = self.asset_registry.get_asset_by_name("microwave")()
        cup = self.asset_registry.get_asset_by_name(cfg.cup)()
        shelf = self.asset_registry.get_asset_by_name(cfg.shelf)(
            scale=(cfg.shelf_scale, cfg.shelf_scale, cfg.shelf_scale)
        )

        # A shelf is furniture: pin it so neither the swinging door nor a clumsy placement can
        # topple it, while it still reports contacts to the cup's sensor.
        shelf.object_cfg.spawn.rigid_props = RigidBodyBaseCfg(kinematic_enabled=True)

        # Step 2: Place the assets. Poses are explicit rather than relation-solved because the
        # placement solver reasons about top surfaces, and neither the microwave cavity the cup
        # starts in nor the arc its door sweeps is expressible that way.
        microwave.set_initial_pose(
            Pose(
                position_xyz=(*MICROWAVE_XY_M, TABLE_TOP_Z_M + MICROWAVE_HALF_HEIGHT_M),
                rotation_xyzw=MICROWAVE_YAW_ROTATION_XYZW,
            )
        )
        cup.set_initial_pose(self._cup_pose_range_on_turntable(cup))
        shelf.set_initial_pose(Pose(position_xyz=(*SHELF_XY_M, self._resting_z(shelf))))

        # Step 3: Configure lighting
        light = self.asset_registry.get_asset_by_name("light")()
        light.set_intensity(cfg.light_intensity)
        if cfg.hdr is not None:
            light.add_hdr(self.hdr_registry.get_hdr_by_name(cfg.hdr)())
        directional_light = self.asset_registry.get_asset_by_name("directional_light")()

        # Step 4: Select the embodiment
        embodiment = self.asset_registry.get_asset_by_name(cfg.embodiment)(enable_cameras=cfg.enable_cameras)

        if cfg.teleop_device is not None:
            teleop_device = self.device_registry.get_device_by_name(cfg.teleop_device)()
        else:
            teleop_device = None

        # Step 5: Compose the scene
        scene = Scene(assets=[background, light, directional_light, microwave, cup, shelf])

        # Step 6: Define the task
        open_door_task = OpenDoorTask(
            openable_object=microwave,
            openness_threshold=cfg.openness_threshold,
            reset_openness=0.0,
            task_description="Open the microwave door.",
        )
        pick_and_place_task = PickAndPlaceTask(
            pick_up_object=cup,
            destination_object=shelf,
            destination_location=shelf,
            background_scene=background,
            task_description=f"Take the {cup.name} out of the microwave and place it on the {shelf.name}.",
        )
        task = OpenDoorAndPlaceObjectTask(
            open_door_task=open_door_task,
            pick_and_place_task=pick_and_place_task,
            episode_length_s=cfg.episode_length_s,
            task_description="Open the microwave, take the cup out of it and place the cup on the shelf.",
            mimic_datagen_name="droid_microwave_cup_to_shelf_D0",
        )

        # Step 7: Assemble the environment
        return IsaacLabArenaEnvironment(
            name=self.name,
            embodiment=embodiment,
            scene=scene,
            task=task,
            teleop_device=teleop_device,
        )

    @staticmethod
    def _resting_z(asset: Object) -> float:
        """Return the origin height that puts the asset's lowest point on the table top."""
        return TABLE_TOP_Z_M - float(asset.get_bounding_box().min_point[0][2])

    @staticmethod
    def _cup_pose_range_on_turntable(cup: Object) -> PoseRange:
        """Return the cup's reset range: upright on the turntable, nudged towards the door.

        The cup is offset from the turntable centre so the gripper does not have to reach the back
        of the cavity; a wider cup is nudged less far so its jitter box stays inside the rim.
        """
        from isaaclab_arena.utils.pose import PoseRange

        bounding_box = cup.get_bounding_box()
        cup_radius_m = 0.5 * float(max(bounding_box.size[0][0], bounding_box.size[0][1]))
        rim_clearance_m = MICROWAVE_TURNTABLE_RADIUS_M - cup_radius_m
        assert rim_clearance_m >= CUP_RESET_JITTER_XY_M, (
            f"'{cup.name}' does not fit on the microwave turntable: it leaves {rim_clearance_m:.3f} m of rim"
            f" clearance, less than the {CUP_RESET_JITTER_XY_M} m of reset jitter"
        )
        towards_door_offset_m = min(CUP_TOWARDS_DOOR_OFFSET_M, rim_clearance_m - CUP_RESET_JITTER_XY_M)

        centre_x_m = MICROWAVE_XY_M[0] + MICROWAVE_TURNTABLE_OFFSET_XY_M[0] - towards_door_offset_m
        centre_y_m = MICROWAVE_XY_M[1] + MICROWAVE_TURNTABLE_OFFSET_XY_M[1]
        turntable_top_z_m = TABLE_TOP_Z_M + MICROWAVE_HALF_HEIGHT_M - MICROWAVE_TURNTABLE_DROP_M
        cup_z_m = turntable_top_z_m - float(bounding_box.min_point[0][2]) + CUP_CLEARANCE_M
        return PoseRange(
            position_xyz_min=(centre_x_m - CUP_RESET_JITTER_XY_M, centre_y_m - CUP_RESET_JITTER_XY_M, cup_z_m),
            position_xyz_max=(centre_x_m + CUP_RESET_JITTER_XY_M, centre_y_m + CUP_RESET_JITTER_XY_M, cup_z_m),
            rpy_min=(0.0, 0.0, 0.0),
            rpy_max=(0.0, 0.0, 0.0),
        )
