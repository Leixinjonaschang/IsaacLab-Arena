# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Per-step CSV trace of a bowl-stacking session, for diagnosing contact failures.

Off unless ``ARENA_STACK_BOWLS_TRACE`` names a path. Catching a teleoperated grasp going wrong
needs per-step data from inside the running session: the instant a bowl is launched is never the
instant anyone thinks to look, and a live status readout only ever holds the latest sample.

Each row records, for both arms, where the end-effector and fingers are, how far the gripper is
open, which bowl is nearest, the commanded translation delta, plus every bowl's pose and speed.
The commanded delta is what separates "the operator drove it there" from "contact dragged it".
"""

from __future__ import annotations

import os
import torch

from isaaclab_arena.tasks.predicates.predicate_utils import get_env, get_root_lin_vel_w, get_root_pos_w

TRACE_PATH_ENV_VAR = "ARENA_STACK_BOWLS_TRACE"
"""Set this to a file path to turn the trace on."""

_ARMS = (("L", "left"), ("R", "right"))


class StackBowlsTrace:
    """Append one CSV row per step describing the gripper/bowl state of environment 0."""

    def __init__(self, bowl_names: list[str]):
        self.bowl_names = bowl_names
        self.path = os.environ.get(TRACE_PATH_ENV_VAR) or None
        self._step = 0
        self._resolved = False
        self._pad_ids: dict[str, list[int]] = {}
        self._ee_ids: dict[str, int | None] = {}
        self._arm_terms: dict[str, object] = {}

    @property
    def enabled(self) -> bool:
        """Whether a trace path was configured."""
        return self.path is not None

    def _resolve(self, env) -> None:
        """Find the finger bodies and arm action terms once, on the first step."""
        robot = env.scene["robot"]
        body_names = list(robot.data.body_names)
        for key, side in _ARMS:
            self._pad_ids[key] = [i for i, b in enumerate(body_names) if b.startswith(side) and "Pad_Link" in b]
            ee_name = "gripper_center" if side == "left" else "right_gripper_center"
            self._ee_ids[key] = body_names.index(ee_name) if ee_name in body_names else None
            term_name = f"{side}_arm_action"
            self._arm_terms[key] = (
                env.action_manager.get_term(term_name) if term_name in env.action_manager.active_terms else None
            )

        columns = ["step"]
        for name in self.bowl_names:
            columns += [f"{name}_x", f"{name}_y", f"{name}_z", f"{name}_speed"]
        for key, _ in _ARMS:
            columns += [
                f"{key}_ee_x",
                f"{key}_ee_y",
                f"{key}_pad_z",
                f"{key}_span",
                f"{key}_nearest_bowl",
                f"{key}_nearest_xy",
                f"{key}_asked_dx",
                f"{key}_asked_dy",
                f"{key}_asked_dz",
            ]
        with open(self.path, "w") as handle:
            handle.write(",".join(columns) + "\n")
        self._resolved = True

    def record(self, env) -> None:
        """Append one row for the current step. Never allowed to break the simulation."""
        if not self.enabled:
            return
        try:
            # Arena deep-copies the task into several manager configs, so more than one instance
            # of this trace can exist. Let whichever one records first own the file, or the rows
            # of two writers interleave and each step counter runs independently.
            unwrapped = get_env(env)
            owner = getattr(unwrapped, "_stack_bowls_trace_owner", None)
            if owner is None:
                unwrapped._stack_bowls_trace_owner = id(self)
            elif owner != id(self):
                return

            if not self._resolved:
                self._resolve(env)
            self._step += 1

            robot = env.scene["robot"]
            body_pos = robot.data.body_pos_w.torch[0]
            cells = [str(self._step)]

            bowl_xy = {}
            for name in self.bowl_names:
                pos = get_root_pos_w(env, name)[0]
                speed = torch.linalg.vector_norm(get_root_lin_vel_w(env, name)[0]).item()
                bowl_xy[name] = pos[:2]
                cells += [f"{pos[0].item():.4f}", f"{pos[1].item():.4f}", f"{pos[2].item():.4f}", f"{speed:.4f}"]

            for key, _ in _ARMS:
                pad_ids = self._pad_ids[key]
                pads = body_pos[pad_ids] if pad_ids else None
                pad_z = pads[:, 2].min().item() if pads is not None else float("nan")
                span = (
                    torch.linalg.vector_norm(pads[0] - pads[1]).item()
                    if pads is not None and len(pad_ids) >= 2
                    else float("nan")
                )
                ee_id = self._ee_ids[key]
                ee_xy = body_pos[ee_id][:2] if ee_id is not None else None
                nearest_name, nearest_xy = "", float("nan")
                if ee_xy is not None and bowl_xy:
                    nearest_name, nearest_xy = min(
                        ((n, torch.linalg.vector_norm(ee_xy - xy).item()) for n, xy in bowl_xy.items()),
                        key=lambda pair: pair[1],
                    )
                cells += [
                    f"{ee_xy[0].item():.4f}" if ee_xy is not None else "nan",
                    f"{ee_xy[1].item():.4f}" if ee_xy is not None else "nan",
                ]
                term = self._arm_terms[key]
                if term is None:
                    asked_xyz = (float("nan"), float("nan"), float("nan"))
                else:
                    offset = self._term_offset(env, key)
                    action = env.action_manager.action[0]
                    asked_xyz = tuple(action[offset + i].item() for i in range(3))
                cells += [
                    f"{pad_z:.4f}",
                    f"{span:.4f}",
                    nearest_name,
                    f"{nearest_xy:.4f}",
                    f"{asked_xyz[0]:.4f}",
                    f"{asked_xyz[1]:.4f}",
                    f"{asked_xyz[2]:.4f}",
                ]

            with open(self.path, "a") as handle:
                handle.write(",".join(cells) + "\n")
        except Exception as error:  # noqa: BLE001 - a diagnostic must never take the sim down
            print(f"[stack bowls trace] disabled after error: {error}")
            self.path = None

    @staticmethod
    def _term_offset(env, key: str) -> int:
        """Index where this arm's term starts in the concatenated action vector."""
        offset = 0
        target = "left_arm_action" if key == "L" else "right_arm_action"
        for term_name in env.action_manager.active_terms:
            if term_name == target:
                return offset
            offset += env.action_manager.get_term(term_name).action_dim
        raise KeyError(f"{target} is not an active action term")
