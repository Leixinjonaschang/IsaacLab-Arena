# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field
from pathlib import Path

from isaaclab_arena_gr00t.policy.config.task_mode import TaskMode


@dataclass
class Gr00tClosedloopPolicyCfg:
    """Configure GR00T closed-loop policy translation and inference."""

    language_instruction: str = field(
        default="", metadata={"description": "Instruction given to the policy in natural language."}
    )
    action_horizon: int = field(
        default=16, metadata={"description": "Number of actions in the policy's predictionhorizon."}
    )
    embodiment_tag: str = field(
        default="NEW_EMBODIMENT",
        metadata={
            "description": (
                "Identifier for the robot embodiment used in the policy inference (e.g., 'gr1' or 'new_embodiment')."
            )
        },
    )
    denoising_steps: int = field(
        default=4, metadata={"description": "Number of denoising steps used in the policy inference."}
    )
    modality_config_path: str = field(
        default=None, metadata={"description": "Path to the modality configuration file."}
    )
    original_image_size: tuple[int, int, int] = field(
        default=(480, 640, 3), metadata={"description": "Original size of input images as (height, width, channels)."}
    )
    target_image_size: tuple[int, int, int] = field(
        default=(480, 640, 3),
        metadata={"description": "Target size for images after resizing and padding as (height, width, channels)."},
    )
    policy_joints_config_path: Path = field(
        default=Path(__file__).parent.resolve() / "config" / "g1" / "gr00t_43dof_joint_space.yaml",
        metadata={"description": "Path to the YAML file specifying the joint ordering configuration for GR00T policy."},
    )
    task_mode_name: str = field(
        default=TaskMode.G1_LOCOMANIPULATION.value,
        metadata={"description": "Task option name of the policy inference."},
    )
    # robot simulation specific parameters
    action_joints_config_path: Path = field(
        default=Path(__file__).parent.parent.resolve() / "config" / "g1" / "43dof_joint_space.yaml",
        metadata={
            "description": (
                "Path to the YAML file specifying the joint ordering configuration for GR1 action space in Lab."
            )
        },
    )
    state_joints_config_path: Path = field(
        default=Path(__file__).parent.parent.resolve() / "config" / "g1" / "43dof_joint_space.yaml",
        metadata={
            "description": (
                "Path to the YAML file specifying the joint ordering configuration for GR1 state space in Lab."
            )
        },
    )
    # Default to GPU policy and CPU physics simulation
    policy_device: str = field(
        default="cuda", metadata={"description": "Device to run the policy model on (e.g., 'cuda' or 'cpu')."}
    )
    video_backend: str = field(default="decord", metadata={"description": "Video backend to use for evaluation."})
    pov_cam_name_sim: list[str] = field(
        default_factory=lambda: ["robot_head_cam_rgb"],
        metadata={"description": "Names of the POV cameras of the robot in simulation."},
    )
    # Closed loop specific parameters
    action_chunk_length: int = field(
        default=16,
        metadata={
            "description": "Number of actions to execute per inference rollout (can be less than action_horizon)."
        },
    )
    seed: int = field(default=10, metadata={"description": "Random seed for reproducibility."})

    def __post_init__(self):
        assert (
            self.action_chunk_length <= self.action_horizon
        ), "action_chunk_length must be less than or equal to action_horizon"
        # assert all paths exist
        assert Path(
            self.policy_joints_config_path
        ).exists(), f"policy_joints_config_path does not exist: {self.policy_joints_config_path}"
        assert Path(
            self.action_joints_config_path
        ).exists(), f"action_joints_config_path does not exist: {self.action_joints_config_path}"
        assert Path(
            self.state_joints_config_path
        ).exists(), f"state_joints_config_path does not exist: {self.state_joints_config_path}"
        if self.modality_config_path:
            assert Path(
                self.modality_config_path
            ).exists(), f"modality_config_path does not exist: {self.modality_config_path}"

        if isinstance(self.pov_cam_name_sim, str):
            self.pov_cam_name_sim = [self.pov_cam_name_sim]

        # embodiment_tag. Which tags exist depends on the checkpoint family the pinned
        # submodules/Isaac-GR00T checkout targets, so the set is read from GR00T rather than
        # duplicated here: N1.7 renamed the DROID tag and dropped GR1 altogether.
        from gr00t.data.embodiment_tags import EmbodimentTag

        available_tags = [tag.name for tag in EmbodimentTag]
        assert self.embodiment_tag in available_tags, (
            f"embodiment_tag '{self.embodiment_tag}' is not offered by the pinned GR00T checkout."
            f" Available tags: {', '.join(sorted(available_tags))}"
        )
        if self.task_mode_name == TaskMode.G1_LOCOMANIPULATION.value:
            assert (
                self.embodiment_tag == "NEW_EMBODIMENT"
            ), "embodiment_tag must be new_embodiment for G1 locomanipulation"
        elif self.task_mode_name == TaskMode.GR1_TABLETOP_MANIPULATION.value:
            assert self.embodiment_tag == "GR1", "embodiment_tag must be GR1 for GR1 tabletop manipulation"
        elif self.task_mode_name == TaskMode.DROID_MANIPULATION.value:
            # GR00T N1.6 calls this embodiment OXE_DROID; N1.7 calls the same robot
            # OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT. Both drive the same sim-side action space, so
            # the task mode covers both and only the state/video contract differs.
            assert self.embodiment_tag.startswith("OXE_DROID"), (
                "embodiment_tag must be a DROID tag (OXE_DROID for GR00T N1.6,"
                " OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT for N1.7) for DROID manipulation, got"
                f" '{self.embodiment_tag}'"
            )
        else:
            raise ValueError(f"Invalid inference mode: {self.task_mode_name}")
