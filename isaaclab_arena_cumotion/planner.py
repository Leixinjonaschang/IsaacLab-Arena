# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""An Arena-facing cuMotion planner: robot description, collision world, and plan selection.

The plan-selection part is the reason this is not a thin wrapper. ``plan_to_pose_target`` returns
a path whose forward kinematics land exactly on the target, and that says nothing about whether
the simulated arm can follow it. Two failure modes recur and both are silent:

* **Wrist parked on a joint limit.** JtRRT is deterministic and will happily return, say, joint 5
  at 3.077 of its 3.14 limit. The position-servoed arm jams ~0.66 rad short and the tool ends up
  300 mm from a pose the planner considers reached.
* **IK-branch flips.** A target reachable two ways may be planned the way that sweeps the elbow
  through the robot's own torso.

:meth:`CumotionArmPlanner.plan_pose` therefore scores candidates on the room they leave to the
joint limits and on how far any single joint has to travel, and rejects the ones the hardware
cannot execute rather than handing back a plan that will fail at run time.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import TYPE_CHECKING

from isaaclab_arena_cumotion.embodiment_cumotion_registry import get_embodiment_cumotion_cfg
from isaaclab_arena_cumotion.robot_description import import_cumotion, load_robot_description

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

    from isaaclab_arena.embodiments.embodiment_base import EmbodimentBase
    from isaaclab_arena_cumotion.cumotion_embodiment_cfg import CumotionEmbodimentCfg

DEFAULT_MIN_LIMIT_MARGIN_RAD = 0.30
"""Below this the position-servoed arm jams short of the planned joint angle."""

DEFAULT_MAX_JOINT_TRAVEL_RAD = 2.6
"""A single joint moving more than this is an IK-branch flip, not a reach."""


@dataclass
class PlanCandidate:
    """A planned path together with the two numbers that decide whether it is executable."""

    path: object
    """The cuMotion ``Path``."""

    q_end: np.ndarray
    """Final arm configuration."""

    limit_margin_rad: float
    """Smallest distance from any joint to its nearest limit."""

    max_travel_rad: float
    """Largest single-joint change from the start configuration."""

    def is_executable(
        self,
        min_limit_margin_rad: float = DEFAULT_MIN_LIMIT_MARGIN_RAD,
        max_joint_travel_rad: float = DEFAULT_MAX_JOINT_TRAVEL_RAD,
    ) -> bool:
        """Whether the simulated arm can be expected to follow this path."""
        return self.limit_margin_rad >= min_limit_margin_rad and self.max_travel_rad <= max_joint_travel_rad


class CumotionArmPlanner:
    """Plans single-arm motions for an Arena embodiment with cuMotion.

    Every pose argument is in world coordinates; the robot base transform is taken from the
    simulated articulation, so a relocated robot needs no extra bookkeeping.

    Args:
        env: The unwrapped Isaac Lab environment holding the robot.
        embodiment: Embodiment whose robot family has a registered cuMotion config.
        arm: Which arm to plan for, for bimanual robots.
        robot_scene_name: Scene key of the robot articulation.
    """

    def __init__(
        self,
        env: ManagerBasedEnv,
        embodiment: EmbodimentBase,
        arm: str = "left",
        robot_scene_name: str = "robot",
    ) -> None:
        from isaacsim.core.experimental.utils.app import enable_extension

        enable_extension("isaacsim.robot_motion.cumotion")

        import warp as wp
        from isaacsim.robot_motion.cumotion import CumotionWorldInterface, GraphBasedMotionPlanner, TrajectoryGenerator
        from isaacsim.robot_motion.cumotion.impl.configuration_loader import CumotionRobot

        self.env = env
        self.arm = arm
        self.cfg: CumotionEmbodimentCfg = get_embodiment_cumotion_cfg(embodiment, arm)
        self.robot = env.scene.articulations[robot_scene_name]
        self.arm_joint_ids, _ = self.robot.find_joints(self.cfg.arm_joint_names, preserve_order=True)
        self.gripper_joint_ids, _ = self.robot.find_joints(self.cfg.gripper_joint_names)
        self.tool_body_index = list(self.robot.data.body_names).index(self.cfg.tool_frame)

        self.robot_description = load_robot_description(self.cfg)
        self.kinematics = self.robot_description.kinematics()

        self.base_pos = wp.to_torch(self.robot.data.root_pos_w)[0].detach().cpu().numpy().astype(np.float64)
        base_quat_xyzw = wp.to_torch(self.robot.data.root_quat_w)[0].detach().cpu().numpy().astype(np.float64)
        self.base_quat_wxyz = np.array([base_quat_xyzw[3], *base_quat_xyzw[:3]])

        self.world = CumotionWorldInterface(
            world_to_robot_base=(
                wp.array(self.base_pos.astype(np.float32), dtype=wp.float32),
                wp.array(self.base_quat_wxyz.astype(np.float32), dtype=wp.float32),
            )
        )
        cumotion_robot = CumotionRobot(
            directory=None,
            robot_description=self.robot_description,
            kinematics=self.kinematics,
            controlled_joint_names=[
                self.robot_description.cspace_coord_name(i)
                for i in range(self.robot_description.num_cspace_coords())
            ],
        )
        self._planner = GraphBasedMotionPlanner(cumotion_robot, self.world, tool_frame=self.cfg.tool_frame)
        self._trajectory_generator = TrajectoryGenerator(cumotion_robot, robot_joint_space=self.cfg.arm_joint_names)
        self._obstacles: set[str] = set()

        self.tool_correction = self._measure_tool_correction()

        limits = self.robot.data.joint_pos_limits.torch[0].detach().cpu().numpy()
        self.joint_limits = np.array([limits[i] for i in self.arm_joint_ids], dtype=np.float64)

    @property
    def trajectory_generator(self):
        """cuMotion's time-parameterisation for planned paths."""
        return self._trajectory_generator

    def joint_positions(self) -> np.ndarray:
        """Current arm configuration."""
        return self.robot.data.joint_pos.torch[0, self.arm_joint_ids].detach().cpu().numpy().astype(np.float64)

    def tool_position(self) -> np.ndarray:
        """Current tool-frame position in world coordinates."""
        import warp as wp

        return wp.to_torch(self.robot.data.body_pos_w)[0, self.tool_body_index].detach().cpu().numpy().astype(
            np.float64
        )

    def forward_kinematics(self, q: np.ndarray) -> np.ndarray:
        """Tool-frame position in world coordinates for an arm configuration."""
        return self.base_pos + np.asarray(self.kinematics.pose(np.asarray(q, dtype=np.float64), self.cfg.tool_frame).translation)

    def kinematics_error_m(self) -> float:
        """Distance between cuMotion's forward kinematics and the simulated tool body.

        A non-zero value means the generated robot description does not describe the robot the
        simulator loaded, and no plan built on it can be trusted. Worth asserting on once at
        start-up.
        """
        return float(np.linalg.norm(self.forward_kinematics(self.joint_positions()) - self.tool_position()))

    def add_box_obstacle(
        self,
        name: str,
        position: np.ndarray,
        extents: tuple[float, float, float],
        quat_wxyz: np.ndarray | None = None,
        safety_tolerance_m: float = 0.01,
    ) -> None:
        """Add an axis- or pose-aligned box to the collision world.

        Args:
            name: Key used to enable, disable or update the obstacle later.
            position: Box centre in world coordinates.
            extents: Full side lengths.
            quat_wxyz: Box orientation, identity when omitted.
            safety_tolerance_m: Margin grown around the box.
        """
        import warp as wp

        assert name not in self._obstacles, f"Obstacle '{name}' already exists."
        quat = np.array([1.0, 0.0, 0.0, 0.0]) if quat_wxyz is None else np.asarray(quat_wxyz)
        self.world.add_cubes(
            prim_paths=[name],
            sizes=wp.array([1.0], dtype=wp.float32),
            scales=wp.array([list(extents)], dtype=wp.float32),
            safety_tolerances=wp.array([safety_tolerance_m], dtype=wp.float32),
            poses=(
                wp.array([np.asarray(position, dtype=np.float32)], dtype=wp.float32),
                wp.array([quat.astype(np.float32)], dtype=wp.float32),
            ),
            enabled_array=wp.array([True], dtype=wp.bool),
        )
        self._obstacles.add(name)

    def add_scene_object_obstacle(
        self, scene_key: str, extents: tuple[float, float, float], safety_tolerance_m: float = 0.01
    ) -> None:
        """Add a scene object's bounding box as an obstacle, at its live pose.

        Args:
            scene_key: Key of the object in the Arena scene.
            extents: Full side lengths, e.g. RoboDojo metadata's ``aligned_bbox.extents``.
            safety_tolerance_m: Margin grown around the box.
        """
        import warp as wp

        asset = self.env.scene[scene_key]
        position = wp.to_torch(asset.data.root_pos_w)[0].detach().cpu().numpy()
        quat_xyzw = wp.to_torch(asset.data.root_quat_w)[0].detach().cpu().numpy()
        self.add_box_obstacle(
            scene_key,
            position,
            extents,
            np.array([quat_xyzw[3], *quat_xyzw[:3]]),
            safety_tolerance_m,
        )

    def set_obstacle_enabled(self, name: str, enabled: bool) -> None:
        """Mute or restore one obstacle.

        Motions that deliberately go to contact -- a descent onto a work surface, a placement into
        the object being stacked on -- have to mute the thing they are approaching, or no plan
        exists.
        """
        import warp as wp

        assert name in self._obstacles, f"Unknown obstacle '{name}'. Known: {sorted(self._obstacles)}."
        self.world.update_obstacle_enables([name], wp.array([enabled], dtype=wp.bool))

    def update_obstacle_pose(self, scene_key: str) -> None:
        """Re-read a scene object's pose into the collision world."""
        import warp as wp

        asset = self.env.scene[scene_key]
        position = wp.to_torch(asset.data.root_pos_w)[0].detach().cpu().numpy().astype(np.float32)
        quat_xyzw = wp.to_torch(asset.data.root_quat_w)[0].detach().cpu().numpy()
        quat_wxyz = np.array([quat_xyzw[3], *quat_xyzw[:3]], dtype=np.float32)
        self.world.update_obstacle_transforms(
            [scene_key], (wp.array([position], dtype=wp.float32), wp.array([quat_wxyz], dtype=wp.float32))
        )

    def _measure_tool_correction(self) -> np.ndarray:
        """Rotation from the simulated tool frame to the one cuMotion plans against.

        The two describe the same physical body, but a generated robot description is free to label
        that body's axes differently from the USD, and on the Agibot the left hand's differ by 90
        degrees while the right hand's agree. Nothing catches that: the frames share an origin, so
        the position cross-check reads 0 mm either way, and the arm then reaches every commanded
        pose exactly while the jaws point somewhere else. Measuring it here rather than taking it
        from a configured axis name means the two descriptions cannot silently disagree.
        """
        import torch

        import warp as wp

        import isaaclab.utils.math as math_utils

        quat_xyzw = wp.to_torch(self.robot.data.body_quat_w)[0, self.tool_body_index].detach().cpu()
        tool_in_world = math_utils.matrix_from_quat(quat_xyzw.float().unsqueeze(0))[0].numpy()
        root_quat_xyzw = wp.to_torch(self.robot.data.root_quat_w)[0].detach().cpu()
        root_in_world = math_utils.matrix_from_quat(root_quat_xyzw.float().unsqueeze(0))[0].numpy()
        # cuMotion's pose is relative to the robot's base, so the base rotation comes out first;
        # what is left is the frame relabelling, which is fixed and configuration-independent.
        tool_in_base = root_in_world.T @ tool_in_world
        cumotion_tool_in_base = np.asarray(
            self.kinematics.pose(self.joint_positions(), self.cfg.tool_frame).rotation.matrix()
        )
        return tool_in_base.T @ cumotion_tool_in_base

    def to_tool_frame(self, canonical_quat_wxyz: np.ndarray) -> np.ndarray:
        """Correct a canonical (+z is the approach) orientation into this tool's own frame."""
        from isaaclab_arena_cumotion.grasps import matrix_from_quat_wxyz, quat_wxyz_from_matrix

        if np.allclose(self.tool_correction, np.eye(3)):
            return np.asarray(canonical_quat_wxyz)
        return quat_wxyz_from_matrix(matrix_from_quat_wxyz(canonical_quat_wxyz) @ self.tool_correction)

    def plan_pose(
        self,
        q_start: np.ndarray,
        position: np.ndarray,
        quat_wxyz: np.ndarray,
    ) -> PlanCandidate | None:
        """Plan to a world-frame tool pose, returning the candidate and its executability scores.

        Args:
            q_start: Arm configuration the motion starts from.
            position: Target tool position in world coordinates.
            quat_wxyz: Target tool orientation in world coordinates.

        Returns:
            The candidate, or None if cuMotion found no path at all.
        """
        path = self._planner.plan_to_pose_target(
            np.asarray(q_start, dtype=np.float64), position, self.to_tool_frame(quat_wxyz)
        )
        if path is None:
            return None
        q_end = path.get_waypoints().numpy()[-1].astype(np.float64)
        margin = float(np.min(np.minimum(q_end - self.joint_limits[:, 0], self.joint_limits[:, 1] - q_end)))
        travel = float(np.max(np.abs(q_end - np.asarray(q_start, dtype=np.float64))))
        return PlanCandidate(path=path, q_end=q_end, limit_margin_rad=margin, max_travel_rad=travel)

    def plan_config(self, q_start: np.ndarray, q_target: np.ndarray) -> PlanCandidate | None:
        """Plan to a target joint configuration.

        Returning an arm to a known configuration is a joint-space question, not a pose one: the
        pose it rests at is reachable several ways and only one of them is the configuration it
        started in.

        Args:
            q_start: Arm configuration the motion starts from.
            q_target: Arm configuration to end at.

        Returns:
            The candidate, or None if cuMotion found no path at all.
        """
        path = self._planner.plan_to_cspace_target(
            np.asarray(q_start, dtype=np.float64), np.asarray(q_target, dtype=np.float64)
        )
        if path is None:
            return None
        q_end = path.get_waypoints().numpy()[-1].astype(np.float64)
        margin = float(np.min(np.minimum(q_end - self.joint_limits[:, 0], self.joint_limits[:, 1] - q_end)))
        travel = float(np.max(np.abs(q_end - np.asarray(q_start, dtype=np.float64))))
        return PlanCandidate(path=path, q_end=q_end, limit_margin_rad=margin, max_travel_rad=travel)

    def plan_best_pose(
        self,
        q_start: np.ndarray,
        targets,
        min_limit_margin_rad: float = DEFAULT_MIN_LIMIT_MARGIN_RAD,
        max_joint_travel_rad: float = DEFAULT_MAX_JOINT_TRAVEL_RAD,
    ) -> tuple[object, PlanCandidate] | None:
        """Plan to whichever of several candidate poses is most comfortably executable.

        Candidates are filtered on limit margin and joint travel, then the survivor with the least
        travel wins -- staying in one IK branch matters more than maximising clearance, because a
        large swing is precisely what sends the elbow through the robot.

        Args:
            q_start: Arm configuration the motion starts from.
            targets: Iterable of ``(label, position, quat_wxyz)``.
            min_limit_margin_rad: Reject configurations closer than this to a joint limit.
            max_joint_travel_rad: Reject configurations requiring a larger single-joint move.

        Returns:
            The winning ``(label, candidate)``, or None if nothing survived.
        """
        best: tuple[object, PlanCandidate] | None = None
        for label, position, quat_wxyz in targets:
            candidate = self.plan_pose(q_start, position, quat_wxyz)
            if candidate is None or not candidate.is_executable(min_limit_margin_rad, max_joint_travel_rad):
                continue
            if best is None or candidate.max_travel_rad < best[1].max_travel_rad:
                best = (label, candidate)
        return best

    def ik_reachable(self, position: np.ndarray, quat_wxyz: np.ndarray, seed: np.ndarray | None = None) -> bool:
        """Whether a world-frame tool pose has an exact IK solution.

        Cheaper than planning, so useful for pruning a grasp-candidate sweep before any path
        search runs.
        """
        cumotion = import_cumotion()
        rotation = cumotion.Rotation3(*np.asarray(self.to_tool_frame(quat_wxyz), dtype=np.float64)).matrix()
        transform = np.eye(4)
        transform[:3, :3] = rotation
        transform[:3, 3] = np.asarray(position, dtype=np.float64) - self.base_pos
        ik_cfg = cumotion.IkConfig()
        ik_cfg.cspace_seeds = [np.asarray(self.joint_positions() if seed is None else seed, dtype=np.float64)]
        return bool(cumotion.solve_ik(self.kinematics, cumotion.Pose3(transform), self.cfg.tool_frame, ik_cfg).success)
