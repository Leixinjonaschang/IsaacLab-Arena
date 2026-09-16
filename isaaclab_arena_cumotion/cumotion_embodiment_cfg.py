# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Per-embodiment cuMotion description, owned by the cuMotion extension (not core)."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class CumotionEmbodimentCfg:
    """Per-embodiment inputs cuMotion needs to plan for one arm of this robot."""

    lula_robot_description: str
    """Path to Isaac Lab's Lula robot description YAML, from which the XRDF is generated."""

    robot_urdf: str
    """Path to the URDF cuMotion loads for kinematics."""

    tool_frame: str
    """URDF frame cuMotion drives, and the frame every target pose refers to."""

    arm_joint_names: list[str]
    """Arm joints, in the order cuMotion's cspace reports them."""

    gripper_joint_names: list[str]
    """Joints the binary gripper command drives."""

    gripper_open_pos: float
    """Joint target for the open gripper."""

    gripper_closed_pos: float
    """Joint target for the closed gripper."""

    self_collision_ignore: dict[str, list[str]]
    """Link pairs excluded from self-collision, as XRDF's ``self_collision.ignore`` map."""

    tool_approach_axis: str = "+z"
    """Which tool-frame axis the fingers point along.

    Tool frames are not consistent even between the two hands of one robot: the Agibot's left
    ``gripper_center`` has +z along the approach, its right ``right_gripper_center`` has -x, and
    Isaac Lab papers over the difference with a body offset on one arm only. Grasp poses are
    therefore authored in a canonical frame whose +z is the approach direction, and corrected into
    the tool's own frame using this. Getting it wrong is silent: the arm reaches the commanded
    pose to a fraction of a millimetre with the fingers pointing the wrong way.
    """

    jaw_axis: str = "+y"
    """Tool-frame axis the jaws separate along.

    A parallel-jaw grasp has to put this across the feature being pinched. It is not derivable
    from the approach axis -- measure it with ``scripts/probe_gripper_axes.py``.
    """

    joint_limits: list[tuple[float, float]] = field(default_factory=list)
    """Arm joint limits, used to score how much room a planned configuration leaves.

    Read from the simulated articulation when empty.
    """
