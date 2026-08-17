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
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
    from isaaclab_arena.environments.isaaclab_arena_manager_based_env_cfg import IsaacLabArenaManagerBasedRLEnvCfg

# Layout constants for the maple-table scene, shared with ``droid_microwave_cup_to_shelf``: the DROID
# base sits at the world origin and the table top of ``maple_table_robolab`` is 3 mm above it.
TABLE_TOP_Z_M = 0.003
"""Height of the maple table's top surface in the DROID base frame."""

ROBOT_FOOTPRINT_MAX_X_M = 0.441
"""Forward reach of the DROID stand and its parked arm, measured from the composed on-stand USD.

An asset placed behind this line spawns inside the robot, and PhysX ejects it across the table on
the first step, so both assets below are kept well clear of it.
"""

# The Factory assets spawn at 3x scale. The peg is a 24 mm x 150 mm pin whose origin is at its foot;
# the hole is a 120 mm block whose origin sits 9 mm above its underside and 75 mm below its top face,
# and whose bore accepts the pin with a fraction of a millimetre to spare.
PEG_HEIGHT_M = 0.15
"""Length of the peg, i.e. how much of it stands proud of the table for the gripper to take."""
PEG_DIAMETER_M = 0.024
"""Peg diameter. Comfortably inside the Robotiq 2F-85's 85 mm stroke."""
HOLE_BOTTOM_BELOW_ROOT_M = 0.009
"""Depth of the hole block's underside below its root frame."""
HOLE_TOP_ABOVE_ROOT_M = 0.075
"""Height of the hole block's top face above its root frame, i.e. how far the peg has to descend."""

PEG_CENTRE_XY_M = (0.54, 0.17)
"""Centre of the peg's reset box, within reach and clear of the stand and of the hole."""
PEG_JITTER_XY_M = 0.05
"""Half-extent of the peg's per-reset position randomisation."""
HOLE_CENTRE_XY_M = (0.67, -0.15)
"""Centre of the hole's reset box, placed on the opposite side of the table from the peg."""
HOLE_JITTER_XY_M = 0.03
"""Half-extent of the hole's per-reset position randomisation."""


@dataclass
class DroidPegInsertionEnvironmentCfg(ArenaEnvironmentCfg):
    """Configure the DROID peg-insertion environment."""

    embodiment: str = "droid_abs_joint_pos"
    """DROID arm with absolute joint-position actions, the action space GR00T N1.6-DROID emits."""
    peg: str = "peg"
    """Held asset. The default is the 8 mm Factory peg at 3x scale, 24 mm across and 150 mm tall."""
    hole: str = "hole"
    """Receptacle. Anchored to the world by the fixed joint in its USD, so the robot cannot push it."""
    max_lateral_separation_m: float = 0.01
    """Success tolerance on the peg-to-hole X and Y root separation."""
    max_vertical_separation_m: float = 0.02
    """Success tolerance on the peg-to-hole Z root separation, which is what rejects a peg merely
    balanced on the block's top face instead of seated in the bore."""
    randomize_layout: bool = True
    """Jitter both assets on every reset. Turn off for a fixed layout when recording demonstrations."""
    hdr: str | None = None
    """Optional HDR dome, matching the other DROID maple-table environments."""
    light_intensity: float = 500.0
    """Dome light intensity."""
    episode_length_s: float = 60.0
    """Budget for grasping the peg and seating it in the bore."""
    teleop_device: str | None = None
    """Optional teleoperation device used to record demonstrations."""

    def __post_init__(self) -> None:
        assert self.episode_length_s > 0.0, "episode_length_s must be greater than zero"
        assert self.max_lateral_separation_m > 0.0, "max_lateral_separation_m must be greater than zero"
        assert self.max_vertical_separation_m > 0.0, "max_vertical_separation_m must be greater than zero"


def apply_assembly_physics(env_cfg: IsaacLabArenaManagerBasedRLEnvCfg) -> IsaacLabArenaManagerBasedRLEnvCfg:
    """Raise PhysX solver precision for the tight peg/hole fit.

    Unlike ``isaaclab_arena_environments.mdp.assembly_env_cfg_callback`` this only replaces the solver
    and material settings, leaving Arena's default step and control rate intact so the environment
    keeps driving a DROID policy at the rate the rest of the DROID pipeline uses.
    """
    from isaaclab.sim.spawners.materials import RigidBodyMaterialCfg
    from isaaclab_physx.physics.physx_manager_cfg import PhysxCfg

    env_cfg.sim.physics = PhysxCfg(
        solver_type=1,
        max_position_iteration_count=192,  # Important to avoid interpenetration of the SDF meshes.
        max_velocity_iteration_count=1,
        bounce_threshold_velocity=0.2,
        friction_offset_threshold=0.01,
        friction_correlation_distance=0.00625,
        gpu_max_rigid_contact_count=2**23,
        gpu_max_rigid_patch_count=2**23,
        gpu_max_num_partitions=1,  # Important for stable simulation.
    )
    env_cfg.sim.physics_material = RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=1.0)
    return env_cfg


@register_environment
class DroidPegInsertionEnvironment(ArenaEnvironmentFactory[DroidPegInsertionEnvironmentCfg]):
    """Pick up the peg and insert it into the hole on the maple table.

    A contact-rich task on the same DROID maple-table setup as ``droid_microwave_cup_to_shelf``: the
    peg stands on its foot within reach of the arm, the hole block is anchored on the far side of the
    table, and success requires the peg to be seated in the bore rather than resting on the block.
    """

    name: str = "droid_peg_insertion"
    _legacy_argparse_cfg_type = DroidPegInsertionEnvironmentCfg

    def build(self, cfg: DroidPegInsertionEnvironmentCfg) -> IsaacLabArenaEnvironment:
        """Build the environment from its typed configuration."""
        from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
        from isaaclab_arena.scene.scene import Scene
        from isaaclab_arena.tasks.insert_object_task import InsertObjectTask

        # Step 1: Retrieve assets from the registry
        background = self.asset_registry.get_asset_by_name("maple_table_robolab")()
        peg = self.asset_registry.get_asset_by_name(cfg.peg)()
        hole = self.asset_registry.get_asset_by_name(cfg.hole)()

        # Step 2: Place the assets. Explicit poses rather than relation solving, because the solver
        # reasons about resting an object on a top surface and has nothing to say about a bore.
        peg_jitter_m = PEG_JITTER_XY_M if cfg.randomize_layout else 0.0
        hole_jitter_m = HOLE_JITTER_XY_M if cfg.randomize_layout else 0.0
        peg.set_initial_pose(
            self._upright_pose_range(PEG_CENTRE_XY_M, TABLE_TOP_Z_M, peg_jitter_m, randomize_yaw=cfg.randomize_layout)
        )
        hole.set_initial_pose(
            self._upright_pose_range(
                HOLE_CENTRE_XY_M,
                TABLE_TOP_Z_M + HOLE_BOTTOM_BELOW_ROOT_M,
                hole_jitter_m,
                randomize_yaw=cfg.randomize_layout,
            )
        )

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
        scene = Scene(assets=[background, light, directional_light, peg, hole])

        # Step 6: Define the task
        task = InsertObjectTask(
            held_object=peg,
            receptacle=hole,
            background_scene=background,
            max_lateral_separation_m=cfg.max_lateral_separation_m,
            max_vertical_separation_m=cfg.max_vertical_separation_m,
            episode_length_s=cfg.episode_length_s,
            task_description="Pick up the peg and insert it into the hole.",
        )

        # Step 7: Assemble the environment
        return IsaacLabArenaEnvironment(
            name=self.name,
            embodiment=embodiment,
            scene=scene,
            task=task,
            teleop_device=teleop_device,
            env_cfg_callback=apply_assembly_physics,
        )

    @staticmethod
    def _upright_pose_range(centre_xy_m: tuple[float, float], z_m: float, jitter_m: float, randomize_yaw: bool):
        """Return a reset range that keeps the asset upright and only moves it in the table plane.

        Roll and pitch stay at zero so the peg always stands on its foot and the bore stays vertical.
        """
        import math

        from isaaclab_arena.utils.pose import PoseRange

        yaw_limit_rad = math.pi if randomize_yaw else 0.0
        return PoseRange(
            position_xyz_min=(centre_xy_m[0] - jitter_m, centre_xy_m[1] - jitter_m, z_m),
            position_xyz_max=(centre_xy_m[0] + jitter_m, centre_xy_m[1] + jitter_m, z_m),
            rpy_min=(0.0, 0.0, -yaw_limit_rad),
            rpy_max=(0.0, 0.0, yaw_limit_rad),
        )
