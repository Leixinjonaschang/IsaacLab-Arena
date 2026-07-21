# 0721 

Write a script for learning the Isaaclab Arena: IsaacLab-Arena/isaaclab_arena_environments/lift_cube_franka_joint_environment.py

Training cmd:
```
OMNI_KIT_ACCEPT_EULA=YES uv run python submodules/IsaacLab/scripts/reinforcement_learning/rsl_rl/train.py \
  --external_callback isaaclab_arena.environments.isaaclab_interop.environment_registration_callback \
  --task lift_cube_franka_joint \
  --rl_training_mode \
  --num_envs 4096 \
  --max_iterations 2000
```

Play cmd:

```
uv run python isaaclab_arena/evaluation/policy_runner.py \
  --policy_type rsl_rl --num_episodes 1024 --num_envs 64 --env_spacing 2.5 \
  --checkpoint_path <ckpt> --viz viser \
  lift_cube_franka_joint
```

