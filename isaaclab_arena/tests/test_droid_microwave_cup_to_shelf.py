# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Layout, success-chain and GR00T-observation tests for the DROID cup-to-shelf environment."""

import torch
import traceback

import pytest

from isaaclab_arena.tests.utils.persistent_simulation_app import run_function_with_persistent_simulation_app

ENVIRONMENT_NAME = "droid_microwave_cup_to_shelf"
HEADLESS = True
NUM_SETTLE_STEPS = 30
NUM_DROP_STEPS = 200
NUM_RESET_DRAWS = 3
MIN_SETTLED_OPENNESS = 0.9
# Camera observation keys that isaaclab_arena_gr00t's droid_manip closed-loop config reads.
GR00T_DROID_CAMERA_KEYS = ("external_camera_rgb", "wrist_camera_rgb")


def _build_environment(enable_cameras: bool = False):
    """Build the registered environment and return the gym env plus the assets under test."""
    from isaaclab_arena.assets.registries import EnvironmentRegistry
    from isaaclab_arena.cli.isaaclab_arena_cli import arena_env_builder_cfg_from_argparse, get_isaaclab_arena_cli_parser
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena_environments.cli import ensure_environments_registered
    from isaaclab_arena_environments.droid_microwave_cup_to_shelf_environment import (
        DroidMicrowaveCupToShelfEnvironmentCfg,
    )

    ensure_environments_registered()
    factory = EnvironmentRegistry().get_component_by_name(ENVIRONMENT_NAME)()
    arena_env = factory.build(DroidMicrowaveCupToShelfEnvironmentCfg(enable_cameras=enable_cameras))

    cli_args = ["--enable_cameras"] if enable_cameras else []
    args_cli = get_isaaclab_arena_cli_parser().parse_args(cli_args)
    env = ArenaEnvBuilder(arena_env, arena_env_builder_cfg_from_argparse(args_cli)).make_registered()
    env.reset()

    open_door_task, pick_and_place_task = arena_env.task.subtasks
    return (
        env,
        open_door_task.openable_object,
        pick_and_place_task.pick_up_object,
        pick_and_place_task.destination_location,
    )


def _root_position(env, asset_name: str) -> torch.Tensor:
    """Return the world position of an asset's root body for environment 0."""
    import warp as wp

    return wp.to_torch(env.unwrapped.scene[asset_name].data.root_pos_w)[0].clone()


def _test_cup_starts_closed_inside_the_microwave(simulation_app) -> bool:
    """The episode must start with a shut microwave, a cup on its turntable and a static shelf."""
    from isaaclab_arena.tests.utils.simulation import step_zeros_and_call
    from isaaclab_arena_environments.droid_microwave_cup_to_shelf_environment import (
        MICROWAVE_HALF_HEIGHT_M,
        MICROWAVE_TURNTABLE_DROP_M,
        MICROWAVE_TURNTABLE_OFFSET_XY_M,
        MICROWAVE_TURNTABLE_RADIUS_M,
        MICROWAVE_XY_M,
        TABLE_TOP_Z_M,
    )

    env, microwave, cup, shelf = _build_environment()
    turntable_centre_xy = torch.tensor(
        [
            MICROWAVE_XY_M[0] + MICROWAVE_TURNTABLE_OFFSET_XY_M[0],
            MICROWAVE_XY_M[1] + MICROWAVE_TURNTABLE_OFFSET_XY_M[1],
        ],
        device=env.unwrapped.device,
    )
    turntable_top_z = TABLE_TOP_Z_M + MICROWAVE_HALF_HEIGHT_M - MICROWAVE_TURNTABLE_DROP_M
    cup_height = float(cup.get_bounding_box().size[0][2])

    try:
        # Every reset draws a fresh cup position on the turntable, so check more than one draw.
        for draw in range(NUM_RESET_DRAWS):
            env.reset()
            assert not microwave.is_open(env).any(), "The microwave must start closed"

            shelf_position_before = _root_position(env, shelf.name)
            step_zeros_and_call(env, NUM_SETTLE_STEPS)

            cup_position = _root_position(env, cup.name)
            radial_offset = torch.linalg.vector_norm(cup_position[:2] - turntable_centre_xy).item()
            print(f"Draw {draw}: cup settled at {cup_position.tolist()}, {radial_offset:.3f} m from the turntable")
            assert radial_offset < MICROWAVE_TURNTABLE_RADIUS_M, "The cup slid off the microwave turntable"
            assert turntable_top_z < cup_position[2].item() < turntable_top_z + cup_height, (
                f"The cup is not resting on the turntable at z={turntable_top_z:.3f} m,"
                f" it settled at z={cup_position[2].item():.3f} m"
            )

            shelf_drift = torch.linalg.vector_norm(_root_position(env, shelf.name) - shelf_position_before).item()
            print(f"Draw {draw}: shelf drifted {shelf_drift * 1e3:.2f} mm")
            assert shelf_drift < 1e-3, "The shelf is expected to be pinned in place"

            assert not microwave.is_open(env).any(), "Zero actions must not open the microwave"

    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
        return False

    finally:
        env.close()

    return True


def _test_open_then_place_on_shelf_succeeds(simulation_app) -> bool:
    """Opening the door and then landing the cup on the shelf must terminate the episode."""
    from isaaclab_arena.tests.utils.simulation import step_zeros_and_call

    env, microwave, cup, shelf = _build_environment()

    try:
        # Subtask 0: the state machine only starts scoring the pick-and-place once the door is open.
        microwave.open(env, env_ids=None)
        step_zeros_and_call(env, NUM_SETTLE_STEPS)
        openness = microwave.get_openness(env)[0].item()
        print(f"Door openness after settling: {openness:.3f}")
        # A door that swung into the robot or the shelf would be pushed back off its travel limit.
        assert openness > MIN_SETTLED_OPENNESS, "The door did not stay open; it may be colliding with the scene"

        # Subtask 1: drop the cup onto the shelf rather than driving the arm, so the test covers the
        # success chain (contact sensor -> pick-and-place success -> composite success) only.
        shelf_top_z = _root_position(env, shelf.name)[2].item() + float(shelf.get_bounding_box().max_point[0][2])
        cup_target_position = _root_position(env, shelf.name).unsqueeze(0)
        cup_target_position[0, 2] = shelf_top_z + float(cup.get_bounding_box().size[0][2])
        identity_quaternion = torch.tensor([[0.0, 0.0, 0.0, 1.0]], device=env.unwrapped.device)

        cup_entity = env.unwrapped.scene[cup.name]
        terminated = torch.zeros(1, dtype=torch.bool, device=env.unwrapped.device)
        with torch.inference_mode():
            cup_entity.write_root_pose_to_sim(root_pose=torch.cat([cup_target_position, identity_quaternion], dim=-1))
            cup_entity.write_root_velocity_to_sim(root_velocity=torch.zeros((1, 6), device=env.unwrapped.device))
            for _ in range(NUM_DROP_STEPS):
                actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
                _, _, terminated, _, _ = env.step(actions)
                if terminated.item():
                    break

        print(f"Terminated: {terminated.item()}")
        assert terminated.item(), "The composite task should succeed once the cup rests on the shelf"

    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
        return False

    finally:
        env.close()

    return True


def _test_gr00t_droid_camera_observations(simulation_app) -> bool:
    """The DROID cameras GR00T N1.6-DROID is fed must render in this environment."""
    env, _, _, _ = _build_environment(enable_cameras=True)

    try:
        with torch.inference_mode():
            observation, *_ = env.step(torch.zeros(env.action_space.shape, device=env.unwrapped.device))
        camera_observations = observation["camera_obs"]
        for camera_key in GR00T_DROID_CAMERA_KEYS:
            assert camera_key in camera_observations, f"'{camera_key}' is missing from the camera observations"
            image = camera_observations[camera_key]
            print(f"{camera_key}: shape {tuple(image.shape)}")
            assert image.shape[-1] == 3, f"'{camera_key}' is not an RGB image"
            assert image.any(), f"'{camera_key}' rendered an empty image"

    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
        return False

    finally:
        env.close()

    return True


def test_cup_starts_closed_inside_the_microwave():
    result = run_function_with_persistent_simulation_app(
        _test_cup_starts_closed_inside_the_microwave,
        headless=HEADLESS,
    )
    assert result, f"Test {_test_cup_starts_closed_inside_the_microwave.__name__} failed"


def test_open_then_place_on_shelf_succeeds():
    result = run_function_with_persistent_simulation_app(
        _test_open_then_place_on_shelf_succeeds,
        headless=HEADLESS,
    )
    assert result, f"Test {_test_open_then_place_on_shelf_succeeds.__name__} failed"


@pytest.mark.with_cameras
def test_gr00t_droid_camera_observations():
    result = run_function_with_persistent_simulation_app(
        _test_gr00t_droid_camera_observations,
        headless=HEADLESS,
        enable_cameras=True,
    )
    assert result, f"Test {_test_gr00t_droid_camera_observations.__name__} failed"


if __name__ == "__main__":
    test_cup_starts_closed_inside_the_microwave()
    test_open_then_place_on_shelf_succeeds()
    test_gr00t_droid_camera_observations()
