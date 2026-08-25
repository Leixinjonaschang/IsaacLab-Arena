# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Pick and place as an explicit sequence of planned motions.

A pick is five motions, not one: approach a stand-off pose, descend to the grasp, close, retrace
the descent, and (for a place) go on to the drop. Each has different collision-world requirements
-- a descent deliberately ends in contact with something the planner would otherwise refuse to
approach -- so they cannot be collapsed into a single plan, and the muting has to be explicit.

Every method returns whether it succeeded rather than raising, so a caller stacking three objects
can report which sub-task failed instead of dying halfway through.
"""

from __future__ import annotations

import numpy as np
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab_arena_cumotion.executor import ArmExecutor
    from isaaclab_arena_cumotion.grasps import GraspProposal
    from isaaclab_arena_cumotion.planner import CumotionArmPlanner, PlanCandidate


@dataclass
class PickResult:
    """Outcome of a pick."""

    success: bool
    label: str = ""
    """Which grasp candidate was chosen."""

    descend_path: object | None = None
    """The descent, kept so the lift and any later retreat can retrace it."""

    grasp_quat_wxyz: np.ndarray | None = None
    """Orientation the grasp was taken at, reused for the place."""

    failure: str = ""
    """Why it failed, when it did."""

    trace: list[str] = field(default_factory=list)
    """One line per stage, for reporting."""


class PickAndPlace:
    """Sequences planned motions into picks and places.

    Args:
        planner: Planner for the arm being driven.
        executor: Executor driving the same arm.
        contact_obstacles: Obstacles muted for any motion that ends in contact, typically the work
            surface. They stay muted only for the duration of that motion.
    """

    def __init__(
        self,
        planner: CumotionArmPlanner,
        executor: ArmExecutor,
        contact_obstacles: Sequence[str] = (),
    ) -> None:
        self.planner = planner
        self.executor = executor
        self.contact_obstacles = tuple(contact_obstacles)

    def _set_contact_obstacles(self, enabled: bool, extra: Iterable[str] = ()) -> None:
        for name in (*self.contact_obstacles, *extra):
            self.planner.set_obstacle_enabled(name, enabled)

    def pick(
        self,
        proposals: Sequence[GraspProposal],
        pregrasp_height_m: float = 0.14,
        mute_during_descent: Iterable[str] = (),
        lift_speed: float = 0.15,
        verify_grasp=None,
        max_reach_error_m: float = 0.05,
        pregrasp_gripper_pos: float | None = None,
        max_attempts: int | None = None,
    ) -> PickResult:
        """Approach, descend, close and lift.

        The grasp candidate is chosen by planning the *whole* approach-and-descend pair for each
        proposal up front and keeping the one that is executable throughout. Choosing on the
        approach alone picks poses the arm cannot then descend from.

        Args:
            proposals: Candidate grasp poses, e.g. from :mod:`isaaclab_arena_cumotion.grasps`.
            pregrasp_height_m: Stand-off above the grasp to approach through.
            mute_during_descent: Extra obstacles muted while descending, e.g. the object itself.
            lift_speed: Playback speed for the lift; slower than the descent because the object is
                now in the fingers.
            verify_grasp: Called after the lift; return False to reject the candidate and try the
                next. This is the only trustworthy test of a grasp -- how close the tool got to the
                commanded pose says little, because a descent that stalls 14 mm short on contact
                with the object can still have it between the jaws and lift it cleanly, while one
                that arrives within 2 mm can close on nothing.
            max_reach_error_m: A loose backstop for a descent that went badly wrong, not a quality
                measure; see ``verify_grasp``.
            pregrasp_gripper_pos: How far open to hold the jaws during the approach and descent,
                defaulting to fully open. Fully open is wrong wherever the descent threads between
                neighbouring objects: the Agibot's jaws stand 104 mm apart wide open, so descending
                into a toast rack whose slices are 25 mm apart drives the far jaw through three
                slices before the near one is anywhere near the target. Set this just wider than
                the feature being pinched. The jaws still close to ``gripper_closed_pos``.
            max_attempts: How many candidates to actually execute, unlimited by default. Every
                candidate is planned against the scene as it stood before anything moved, so once
                an attempt has disturbed that scene the remaining plans describe a world that no
                longer exists. Where a failed grasp leaves things where it found them -- a bowl
                that was never touched -- working down the list is free; where it does not -- a
                slice knocked out of a rack -- only the first attempt means anything, and 1 is the
                honest setting.

        Returns:
            The outcome, including the descent path so the caller can retrace it.
        """
        result = PickResult(success=False)
        q_start = self.planner.joint_positions()

        candidates: list[tuple[str, PlanCandidate, PlanCandidate, np.ndarray]] = []
        rejected = {
            "unreachable": 0,
            "approach_unplanned": 0,
            "approach_unusable": 0,
            "descent_unplanned": 0,
            "descent_unusable": 0,
        }
        for proposal in proposals:
            if not self.planner.ik_reachable(proposal.position, proposal.quat_wxyz):
                rejected["unreachable"] += 1
                continue
            pregrasp_position = proposal.position + np.array([0.0, 0.0, pregrasp_height_m])
            approach = self.planner.plan_pose(q_start, pregrasp_position, proposal.quat_wxyz)
            if approach is None:
                rejected["approach_unplanned"] += 1
                continue
            if not approach.is_executable():
                rejected["approach_unusable"] += 1
                continue
            self._set_contact_obstacles(False, mute_during_descent)
            descend = self.planner.plan_pose(approach.q_end, proposal.position, proposal.quat_wxyz)
            self._set_contact_obstacles(True, mute_during_descent)
            if descend is None:
                rejected["descent_unplanned"] += 1
                continue
            if not descend.is_executable():
                rejected["descent_unusable"] += 1
                continue
            candidates.append((proposal.label, approach, descend, proposal.quat_wxyz, len(candidates)))

        result.trace.append(
            f"{len(candidates)}/{len(proposals)} candidates usable; rejected "
            + ", ".join(f"{k} {v}" for k, v in rejected.items() if v)
        )
        if not candidates:
            result.failure = f"no executable grasp among {len(proposals)} proposals"
            return result

        # Travel is bucketed rather than compared exactly, so that a candidate which is only
        # marginally cheaper to move to cannot outrank an earlier -- and, the way proposals are
        # generated, gentler -- approach. A steep lean aims the jaws more sideways onto the
        # feature, which reaches the pose but does not hold it.
        candidates.sort(key=lambda c: (round(c[2].max_travel_rad, 1), c[4]))

        pregrasp_open = self.planner.cfg.gripper_open_pos if pregrasp_gripper_pos is None else pregrasp_gripper_pos
        self.executor.set_gripper(pregrasp_open)
        attempts = candidates if max_attempts is None else candidates[:max_attempts]
        if len(attempts) < len(candidates):
            result.trace.append(f"executing {len(attempts)} of {len(candidates)} candidates (max_attempts)")
        for label, approach, descend, quat, _ in attempts:
            self.executor.follow(approach.path)
            self.executor.follow(descend.path)
            reached = float(np.linalg.norm(self.planner.tool_position() - self._descend_target(descend)))
            result.trace.append(
                f"grasp '{label}': descent margin {descend.limit_margin_rad:.2f} rad,"
                f" reached within {reached * 1000:.1f} mm"
            )
            if reached > max_reach_error_m:
                # Went badly wrong rather than merely stalling on contact. Back out along the way
                # in; the gripper is still open, so a retreat undoes everything.
                result.trace.append("  abandoned: descent did not arrive")
                self.executor.follow(descend.path, reverse=True)
                continue

            q_hold = self.planner.joint_positions()
            self.executor.close_gripper(hold_arm_at=q_hold)
            self.executor.follow(descend.path, reverse=True, speed=lift_speed)

            if verify_grasp is None or verify_grasp():
                result.success = True
                result.label = label
                result.descend_path = descend.path
                result.grasp_quat_wxyz = quat
                return result

            # Closed on nothing. Put the hand back where it was, open, and try the next candidate.
            result.trace.append("  abandoned: nothing came up")
            self.executor.follow(descend.path)
            self.executor.set_gripper(pregrasp_open)
            self.executor.follow(descend.path, reverse=True)

        result.failure = f"none of {len(attempts)} executed grasps held the object"
        return result

    def _descend_target(self, descend: PlanCandidate) -> np.ndarray:
        """Where the descent was meant to end, in world coordinates."""
        return self.planner.forward_kinematics(descend.q_end)

    def place(
        self,
        targets: Sequence[tuple[str, np.ndarray, np.ndarray]],
        approach_height_m: float = 0.16,
        mute_during_descent: Iterable[str] = (),
        descend_speed: float = 0.2,
        release_gripper_pos: float | None = None,
        retreat_m: float = 0.12,
        retarget=None,
        on_release=None,
    ) -> PickResult:
        """Carry to a stand-off above the drop, descend, open and retreat.

        Like :meth:`pick`, the whole carry-and-descend pair is planned for every target before any
        of it is executed, and the least-travel executable one wins.

        Args:
            targets: Candidate ``(label, release position, release orientation)`` triples. Passing
                several is how a rotationally symmetric object's spare degree of freedom is handed
                to the planner.
            approach_height_m: Stand-off above the release point to carry through.
            mute_during_descent: Obstacles muted while descending, typically the object being
                placed onto.
            descend_speed: Playback speed for the carry and descent, slower because the object is
                held.
            release_gripper_pos: Joint position to open to, instead of fully open. Letting go of a
                thin feature needs only enough travel to clear it, and opening further is not free:
                on a bowl held by its rim the fingers sweep about a rim radius each, so the inner
                one crosses the bowl and hooks the far wall, and the bowl rides back up with the
                hand instead of being left on the pile.
            retreat_m: How far to lift straight up after letting go.
            retarget: Called with no arguments once the carry has arrived at the stand-off, and
                returns the tool position the descent should actually end at. The targets were aimed
                using the object's pose in the fingers *at the grasp*, and that pose is not durable:
                a bowl carried by its rim slips tens of millimetres on the way over, so a descent
                planned before the carry lands the object somewhere the arm was never told about.
                Re-measuring on arrival and replanning the short final descent is what closes that
                gap. The orientation is kept, and the original descent is used if the replan fails.
            on_release: Called with no arguments at the two moments the object's pose is diagnostic
                -- once with the fingers still closed at the bottom of the descent, once just after
                they open -- and its return value appended to the trace. This is what separates a
                descent that ended in the wrong place from an object that never left the fingers.

        Returns:
            The outcome.
        """
        result = PickResult(success=False)
        q_start = self.planner.joint_positions()

        candidates: list[tuple[str, PlanCandidate, PlanCandidate]] = []
        rejected = {"carry_no_plan": 0, "carry_unusable": 0, "descend_no_plan": 0, "descend_unusable": 0}
        for label, position, quat_wxyz in targets:
            carry = self.planner.plan_pose(q_start, position + np.array([0.0, 0.0, approach_height_m]), quat_wxyz)
            if carry is None:
                rejected["carry_no_plan"] += 1
                continue
            if not carry.is_executable():
                rejected["carry_unusable"] += 1
                continue
            self._set_contact_obstacles(False, mute_during_descent)
            descend = self.planner.plan_pose(carry.q_end, position, quat_wxyz)
            self._set_contact_obstacles(True, mute_during_descent)
            if descend is None:
                rejected["descend_no_plan"] += 1
                continue
            if not descend.is_executable():
                rejected["descend_unusable"] += 1
                continue
            candidates.append((label, carry, descend, quat_wxyz))

        # The grasp stage reports its rejection reasons; without the same breakdown here a place
        # failure gives no clue whether the planner found nothing or found something untrackable.
        breakdown = ", ".join(f"{reason} {count}" for reason, count in rejected.items() if count)
        result.trace.append(
            f"{len(candidates)}/{len(targets)} release candidates usable"
            + (f"; rejected {breakdown}" if breakdown else "")
        )
        if not candidates:
            result.failure = "no executable carry-and-descend to any release pose"
            return result

        label, carry, descend, quat_wxyz = min(candidates, key=lambda c: c[1].max_travel_rad + c[2].max_travel_rad)
        result.label = label
        result.trace.append(f"release '{label}': carry travel {carry.max_travel_rad:.2f} rad")

        self.executor.follow(carry.path, speed=descend_speed)
        if retarget is not None:
            corrected = retarget()
            self._set_contact_obstacles(False, mute_during_descent)
            replanned = self.planner.plan_pose(self.planner.joint_positions(), corrected, quat_wxyz)
            self._set_contact_obstacles(True, mute_during_descent)
            shift = float(np.linalg.norm(corrected - self._descend_target(descend)))
            if replanned is not None and replanned.is_executable():
                descend = replanned
                result.trace.append(f"  retargeted the descent by {shift * 1000:.1f} mm")
            else:
                result.trace.append(f"  wanted to retarget by {shift * 1000:.1f} mm but could not plan it")
        self.executor.follow(descend.path, speed=descend_speed)
        # A release descent can stall on the pile the same way a grasp descent stalls on the table,
        # and it is aimed by an offset that only holds if the tool arrived. Report the miss so a
        # badly placed object is not mistaken for a badly measured grasp offset.
        reached = float(np.linalg.norm(self.planner.tool_position() - self._descend_target(descend)))
        result.trace.append(f"  descent reached within {reached * 1000:.1f} mm")
        if on_release is not None:
            result.trace.append(f"  at the bottom of the descent, still held: {on_release()}")
        q_hold = self.planner.joint_positions()
        if release_gripper_pos is None:
            self.executor.open_gripper(hold_arm_at=q_hold)
        else:
            self.executor.set_gripper(release_gripper_pos, hold_arm_at=q_hold)
        if on_release is not None:
            result.trace.append(f"  fingers open: {on_release()}")

        # Leaving is not free either. Retracing the descent backs the tool out along the lean it
        # came in on, which is a sideways move at the height of the rim it just let go of, and it
        # drags the object off the pile. Lift straight up instead, and only retrace if that will
        # not plan.
        # The full height is not always plannable from a pose that was already at the edge of the
        # arm's reach, and a short lift that clears the rim is worth far more than a long one that
        # falls back to dragging, so the height is stepped down rather than given up on.
        self._set_contact_obstacles(False, mute_during_descent)
        lift = None
        for fraction in (1.0, 0.5, 0.25):
            here = self.planner.tool_position()
            plan = self.planner.plan_pose(
                self.planner.joint_positions(), here + np.array([0.0, 0.0, retreat_m * fraction]), quat_wxyz
            )
            if plan is not None and plan.is_executable():
                lift = plan
                result.trace.append(f"  lifting straight up {retreat_m * fraction * 1000:.0f} mm to let go")
                break
        self._set_contact_obstacles(True, mute_during_descent)
        if lift is not None:
            self.executor.follow(lift.path, speed=descend_speed)
        else:
            result.trace.append("  no vertical retreat plans; retracing the descent instead")
            self.executor.follow(descend.path, reverse=True, speed=descend_speed)
        if on_release is not None:
            result.trace.append(f"  after the retreat: {on_release()}")

        result.success = True
        result.descend_path = descend.path
        return result
