# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Play an RSL-RL checkpoint with the visualizer camera aimed at env 0.

Adapted from IsaacLab's ``scripts/reinforcement_learning/rsl_rl/play.py``; after env
creation it points every active visualizer (viser, kit, ...) at environment index 0
using the task's viewer configuration.
"""

import argparse
import contextlib
import importlib.metadata as metadata
import os
import sys
import time

import gymnasium as gym
import torch
from packaging import version
from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.envs import DirectMARLEnvCfg, DirectRLEnvCfg, ManagerBasedRLEnvCfg
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.string import list_intersection, string_to_callable

from isaaclab_rl.rsl_rl import (
    RslRlBaseRunnerCfg,
    RslRlVecEnvWrapper,
    handle_deprecated_rsl_rl_cfg,
)

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import add_launcher_args, get_checkpoint_path, launch_simulation
from isaaclab_tasks.utils.hydra import hydra_task_config

# Make IsaacLab's rsl_rl script helpers (cli_args) importable from this copy.
_RSL_RL_SCRIPTS_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..",
    "..",
    "submodules",
    "IsaacLab",
    "scripts",
    "reinforcement_learning",
    "rsl_rl",
)
sys.path.insert(0, _RSL_RL_SCRIPTS_DIR)
import cli_args  # isort: skip

with contextlib.suppress(ImportError):
    import isaaclab_tasks_experimental  # noqa: F401

# -- argparse ----------------------------------------------------------------
parser = argparse.ArgumentParser(description="Play an RSL-RL agent, camera on env 0.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument("--external_callback", default=None, help="Fully qualified path to an externally defined callback.")
cli_args.add_rsl_rl_args(parser)
add_launcher_args(parser)
args_cli, remaining_args = parser.parse_known_args()

# Call an external callback if requested. This gives opportunity to external code to
# register the environments (e.g. the Arena environment registration callback).
remaining_args_env_registration = None
if args_cli.external_callback:
    external_callback_function = string_to_callable(args_cli.external_callback, separator=".")
    remaining_args_env_registration = external_callback_function()

# clear out sys.argv for Hydra
remaining_args = list_intersection(remaining_args, remaining_args_env_registration)
sys.argv = [sys.argv[0]] + remaining_args

installed_version = metadata.version("rsl-rl-lib")


def point_camera_at_env(env, env_index: int) -> None:
    """Aim all active visualizers at the given environment's origin.

    Uses the env cfg's viewer eye/lookat as an env-local offset when
    ``viewer.origin_type == "env"``, so the framing chosen by the task is kept.
    """
    unwrapped = env.unwrapped
    viewer_cfg = unwrapped.cfg.viewer
    eye = torch.tensor(viewer_cfg.eye, dtype=torch.float32)
    target = torch.tensor(viewer_cfg.lookat, dtype=torch.float32)
    if viewer_cfg.origin_type == "env":
        origin = unwrapped.scene.env_origins[env_index].detach().cpu().to(torch.float32)
        eye = eye + origin
        target = target + origin
    unwrapped.sim.set_camera_view(tuple(eye.tolist()), tuple(target.tolist()))


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Play with RSL-RL agent."""
    with launch_simulation(env_cfg, args_cli):
        # override configurations with non-hydra CLI arguments
        agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
        env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
        agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)

        env_cfg.seed = agent_cfg.seed
        env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

        # resolve the checkpoint path
        log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
        print(f"[INFO] Loading experiment from directory: {log_root_path}")
        if args_cli.checkpoint:
            resume_path = retrieve_file_path(args_cli.checkpoint)
        else:
            resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
        env_cfg.log_dir = os.path.dirname(resume_path)

        # create isaac environment
        env = gym.make(args_cli.task, cfg=env_cfg)

        # point the visualizer camera (viser, kit, ...) at env 0
        point_camera_at_env(env, env_index=0)

        # wrap around environment for rsl-rl
        env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        if agent_cfg.class_name == "OnPolicyRunner":
            runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        elif agent_cfg.class_name == "DistillationRunner":
            runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        else:
            raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
        runner.load(resume_path)

        # obtain the trained policy for inference
        policy = runner.get_inference_policy(device=env.unwrapped.device)
        if version.parse(installed_version) < version.parse("4.0.0"):
            policy_nn = runner.alg.policy if version.parse(installed_version) >= version.parse("2.3.0") else runner.alg.actor_critic
        else:
            policy_nn = None

        dt = env.unwrapped.step_dt
        obs = env.get_observations()

        # simulate environment
        try:
            while True:
                start_time = time.time()
                with torch.inference_mode():
                    actions = policy(obs)
                    obs, _, dones, _ = env.step(actions)
                    # reset recurrent states for episodes that have terminated
                    if policy_nn is None:
                        policy.reset(dones)
                    else:
                        policy_nn.reset(dones)

                sleep_time = dt - (time.time() - start_time)
                if args_cli.real_time and sleep_time > 0:
                    time.sleep(sleep_time)

            env.close()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
