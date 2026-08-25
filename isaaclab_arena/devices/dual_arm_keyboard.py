# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Keyboard teleoperation for a bimanual robot, one arm at a time."""

from __future__ import annotations

import torch

from isaaclab.devices.keyboard import Se3Keyboard, Se3KeyboardCfg
from isaaclab.utils.configclass import configclass

ARM_ACTION_DIM = 6
"""Delta pose an arm term consumes: three translation, three rotation."""

GRIPPER_ACTION_DIM = 1
"""Binary open/close command an arm's gripper term consumes."""

ARM_SLICE_DIM = ARM_ACTION_DIM + GRIPPER_ACTION_DIM
"""Width of one arm's slice of the action vector."""

LEFT, RIGHT = 0, 1


class DualArmSe3Keyboard(Se3Keyboard):
    """Drive a two-armed robot from one keyboard, switching arms with Tab.

    The usual SE(3) keys always drive whichever arm is selected; the other arm is told to hold
    position. Emits ``2 * ARM_SLICE_DIM`` values laid out as
    ``[left delta pose, left gripper, right delta pose, right gripper]``, which has to match the
    declaration order of the action terms in the embodiment's action config -- Isaac Lab builds
    the action vector from ``cfg.__dict__.items()``, so field order is the layout.

    Each arm's gripper command is latched separately. Sending 0 for the idle arm would not be
    neutral: ``BinaryJointAction`` treats anything not below zero as *open*, so an arm holding
    something would drop it the moment the operator switched away.
    """

    def __init__(self, cfg: DualArmSe3KeyboardCfg):
        super().__init__(cfg)
        self._active_arm = LEFT
        self._gripper_closed = [False, False]
        self.add_callback("TAB", self._toggle_active_arm)

    def __str__(self) -> str:
        return super().__str__() + "\n\tSwitch arm (left/right): TAB"

    @property
    def active_arm(self) -> str:
        """Which arm the keys currently drive."""
        return "left" if self._active_arm == LEFT else "right"

    def reset(self) -> None:
        super().reset()
        self._gripper_closed = [False, False]
        self._active_arm = LEFT

    def _toggle_active_arm(self) -> None:
        """Hand control to the other arm, preserving each arm's gripper state."""
        self._gripper_closed[self._active_arm] = self._close_gripper
        self._active_arm = RIGHT if self._active_arm == LEFT else LEFT
        # Resume this arm's gripper where it was left, so K keeps toggling from there.
        self._close_gripper = self._gripper_closed[self._active_arm]
        # Drop any motion still accumulated from keys held down during the switch, otherwise
        # the incoming arm inherits a command the operator meant for the outgoing one.
        self._delta_pos[:] = 0.0
        self._delta_rot[:] = 0.0
        print(f"[teleop] active arm -> {self.active_arm}")

    def advance(self) -> torch.Tensor:
        """Return the two-arm command: live values for the active arm, hold for the other."""
        command = super().advance()
        self._gripper_closed[self._active_arm] = bool(command[ARM_ACTION_DIM] < 0)

        both = torch.zeros(2 * ARM_SLICE_DIM, dtype=command.dtype, device=command.device)
        # Zero delta pose means "stay put" for a relative-mode arm term, so only the grippers
        # need explicit values for the idle arm.
        for arm in (LEFT, RIGHT):
            both[arm * ARM_SLICE_DIM + ARM_ACTION_DIM] = -1.0 if self._gripper_closed[arm] else 1.0
        offset = self._active_arm * ARM_SLICE_DIM
        both[offset : offset + ARM_SLICE_DIM] = command[:ARM_SLICE_DIM]
        return both


@configclass
class DualArmSe3KeyboardCfg(Se3KeyboardCfg):
    """Configuration for the Tab-switching two-arm keyboard."""

    class_type: type[DualArmSe3Keyboard] = DualArmSe3Keyboard
