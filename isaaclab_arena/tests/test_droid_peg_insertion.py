# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Layout and success-criterion tests for the DROID peg-insertion environment."""

import torch
import traceback

from isaaclab_arena.tests.utils.persistent_simulation_app import run_function_with_persistent_simulation_app

ENVIRONMENT_NAME = "droid_peg_insertion"
HEADLESS = True
NUM_SETTLE_STEPS = 30
NUM_INSERT_STEPS = 200
NUM_RESET_DRAWS = 5
# The peg is 150 mm tall and stands on its foot, so it must not have tipped over while settling.
MAX_PEG_TILT_DEGREES = 15.0


def _build_environment():
    """Build the registered environment and return the gym env plus the assets under test."""
    from isaaclab_arena.assets.registries import EnvironmentRegistry
    from isaaclab_arena.cli.isaaclab_arena_cli import arena_env_builder_cfg_from_argparse, get_isaaclab_arena_cli_parser
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena_environments.cli import ensure_environments_registered
    from isaaclab_arena_environments.droid_peg_insertion_environment import DroidPegInsertionEnvironmentCfg

    ensure_environments_registered()
    factory = EnvironmentRegistry().get_component_by_name(ENVIRONMENT_NAME)()
    arena_env = factory.build(DroidPegInsertionEnvironmentCfg())

    args_cli = get_isaaclab_arena_cli_parser().parse_args([])
    env = ArenaEnvBuilder(arena_env, arena_env_builder_cfg_from_argparse(args_cli)).make_registered()
    env.reset()

    task = arena_env.task
    return env, task.held_object, task.receptacle


def _root_position(env, asset_name: str) -> torch.Tensor:
    """Return the world position of an asset's root body for environment 0."""
    import warp as wp

    return wp.to_torch(env.unwrapped.scene[asset_name].data.root_pos_w)[0].clone()


def _tilt_degrees(env, asset_name: str) -> float:
    """Return the angle between the asset's local +Z axis and world up, in degrees."""
    import warp as wp
    from isaaclab.utils.math import matrix_from_quat

    quaternion = wp.to_torch(env.unwrapped.scene[asset_name].data.root_quat_w)[0:1]
    local_up = matrix_from_quat(quaternion)[0, :, 2]
    return float(torch.rad2deg(torch.acos(local_up[2].clamp(-1.0, 1.0))))


def _hold_action(env) -> torch.Tensor:
    """Return the absolute joint-position action that parks the arm at its spawn pose.

    ``droid_abs_joint_pos`` interprets a zero action as "drive every joint to 0 rad", which sweeps
    the arm across the table and knocks the peg over. These tests are about the scene, not about
    that action-space quirk, so they command the arm to hold still instead.
    """
    import warp as wp

    robot = env.unwrapped.scene["robot"]
    joint_names = robot.data.joint_names
    default_joint_pos = wp.to_torch(robot.data.default_joint_pos)[0]
    arm_action = torch.stack([default_joint_pos[joint_names.index(f"panda_joint{i}")] for i in range(1, 8)])
    gripper_action = torch.zeros(1, device=arm_action.device)
    return torch.cat([arm_action, gripper_action]).unsqueeze(0)


def _success(env) -> bool:
    """Return the task's own success term, which ``terminated`` mixes with time-out and drops."""
    return bool(env.unwrapped.termination_manager.get_term("success")[0])


def _test_peg_and_hole_reset_upright_and_apart(simulation_app) -> bool:
    """Every reset must leave a standing peg and an anchored hole inside the sampling box."""
    from isaaclab_arena_environments.droid_peg_insertion_environment import (
        HOLE_CENTRE_XY_M,
        HOLE_JITTER_XY_M,
        PEG_CENTRE_XY_M,
        PEG_JITTER_XY_M,
        ROBOT_FOOTPRINT_MAX_X_M,
        TABLE_TOP_Z_M,
    )

    env, peg, hole = _build_environment()

    try:
        with torch.inference_mode():
            for draw in range(NUM_RESET_DRAWS):
                env.reset()
                hold_action = _hold_action(env)
                hole_position_before = _root_position(env, hole.name)
                for _ in range(NUM_SETTLE_STEPS):
                    env.step(hold_action)

                peg_position = _root_position(env, peg.name)
                hole_position = _root_position(env, hole.name)
                separation = torch.linalg.vector_norm(peg_position[:2] - hole_position[:2]).item()
                tilt = _tilt_degrees(env, peg.name)
                print(f"Draw {draw}: peg={peg_position.tolist()} tilt={tilt:.1f} deg, peg-hole gap={separation:.3f} m")

                assert tilt < MAX_PEG_TILT_DEGREES, "The peg toppled over instead of standing on the table"
                assert (
                    abs(peg_position[2].item() - TABLE_TOP_Z_M) < 0.01
                ), f"The peg is not standing on the table top: z={peg_position[2].item():.4f} m"
                assert separation > 0.1, "The peg is too close to the hole block at reset"
                assert (
                    peg_position[0].item() > ROBOT_FOOTPRINT_MAX_X_M
                    and hole_position[0].item() > ROBOT_FOOTPRINT_MAX_X_M
                ), "An asset spawned inside the robot's footprint and will be ejected"
                for axis in range(2):
                    assert (
                        abs(peg_position[axis].item() - PEG_CENTRE_XY_M[axis]) <= PEG_JITTER_XY_M + 0.01
                    ), "The peg left its reset box"
                    assert (
                        abs(hole_position[axis].item() - HOLE_CENTRE_XY_M[axis]) <= HOLE_JITTER_XY_M + 0.01
                    ), "The hole left its reset box"

                hole_drift = torch.linalg.vector_norm(_root_position(env, hole.name) - hole_position_before).item()
                print(f"Draw {draw}: hole drifted {hole_drift * 1e3:.2f} mm")
                assert hole_drift < 5e-3, "The anchored hole should stay where it was placed"

                assert not _success(env), "A peg standing beside the block is not an insertion"

    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
        return False

    finally:
        env.close()

    return True


def _test_seating_the_peg_in_the_bore_succeeds(simulation_app) -> bool:
    """Success must require the peg to descend into the bore, not merely reach the block."""
    from isaaclab_arena_environments.droid_peg_insertion_environment import HOLE_TOP_ABOVE_ROOT_M

    env, peg, hole = _build_environment()

    def _place_peg_on_bore_axis(height_above_root_m: float) -> None:
        """Put the peg upright on the bore axis with its foot at the requested height."""
        peg_entity = env.unwrapped.scene[peg.name]
        target = _root_position(env, hole.name).unsqueeze(0)
        target[0, 2] += height_above_root_m
        upright = torch.tensor([[0.0, 0.0, 0.0, 1.0]], device=env.unwrapped.device)
        peg_entity.write_root_pose_to_sim(root_pose=torch.cat([target, upright], dim=-1))
        peg_entity.write_root_velocity_to_sim(root_velocity=torch.zeros((1, 6), device=env.unwrapped.device))

    try:
        with torch.inference_mode():
            hold_action = _hold_action(env)
            for _ in range(NUM_SETTLE_STEPS):
                env.step(hold_action)
            assert not _success(env), "The task must not start out successful"

            # A peg resting on the block's top face is aligned in X and Y but has not gone in.
            _place_peg_on_bore_axis(HOLE_TOP_ABOVE_ROOT_M + 0.001)
            env.step(hold_action)
            separation = (_root_position(env, peg.name) - _root_position(env, hole.name)).abs()
            print(f"Peg on the block's top face: z separation = {separation[2]:.4f} m, success = {_success(env)}")
            assert not _success(env), "A peg standing on top of the block must not count as inserted"

            # Released just above the mouth of the bore, the peg should drop in and register.
            _place_peg_on_bore_axis(HOLE_TOP_ABOVE_ROOT_M + 0.02)
            # The environment auto-resets on success, so read the success term rather than the
            # post-step root poses, which already belong to the next episode.
            seated = False
            for step in range(NUM_INSERT_STEPS):
                env.step(hold_action)
                if _success(env):
                    print(f"Peg registered as seated in the bore after {step + 1} step(s)")
                    seated = True
                    break

            assert seated, "The peg never registered as seated in the bore"

    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
        return False

    finally:
        env.close()

    return True


def test_peg_and_hole_reset_upright_and_apart():
    result = run_function_with_persistent_simulation_app(
        _test_peg_and_hole_reset_upright_and_apart,
        headless=HEADLESS,
    )
    assert result, f"Test {_test_peg_and_hole_reset_upright_and_apart.__name__} failed"


def test_seating_the_peg_in_the_bore_succeeds():
    result = run_function_with_persistent_simulation_app(
        _test_seating_the_peg_in_the_bore_succeeds,
        headless=HEADLESS,
    )
    assert result, f"Test {_test_seating_the_peg_in_the_bore_succeeds.__name__} failed"


if __name__ == "__main__":
    test_peg_and_hole_reset_upright_and_apart()
    test_seating_the_peg_in_the_bore_succeeds()
