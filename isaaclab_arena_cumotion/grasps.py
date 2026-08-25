# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Propose candidate grasp poses for an object, in world coordinates.

Two sources, in order of preference:

* **An annotated point.** RoboDojo's ``active.functional`` entries (a mug's ``handle``, a
  headset's ``headband_bridge``) are poses in the object's own frame. 99 of its 466 objects carry
  one; where it exists it beats anything geometric.
* **The object's shape.** For the other 367 -- bowls included -- the annotation to use is
  ``active.place``, whose ``projection_circle`` states the rim radius of a body of revolution.
  A bowl's grasp is therefore not a point but the whole rim circle, and the angle around it is a
  free parameter to hand to the planner rather than a choice to guess at.

Every proposal is emitted as a *set* of poses over the free parameters (approach lean, wrist yaw,
angle around a rim). Which one to use is not decided here: it is decided by
:meth:`~isaaclab_arena_cumotion.planner.CumotionArmPlanner.plan_best_pose`, because reachability
and executability are properties of the robot and the clutter, not of the object.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass

DOWN_FACING_ROTATION = np.diag([1.0, -1.0, -1.0])
"""180 deg about x: the tool's approach axis points straight down."""


def _rot_z(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _rot_y(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def quat_wxyz_from_matrix(rotation: np.ndarray) -> np.ndarray:
    """Rotation matrix to a cuMotion / Isaac Sim ordered quaternion."""
    import torch

    import isaaclab.utils.math as math_utils

    tensor = torch.from_numpy(np.ascontiguousarray(rotation)).float().unsqueeze(0)
    q_xyzw = math_utils.quat_from_matrix(tensor)[0].numpy()
    return np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]])


def matrix_from_quat_wxyz(quat_wxyz: np.ndarray) -> np.ndarray:
    """cuMotion / Isaac Sim ordered quaternion to a rotation matrix."""
    import torch

    import isaaclab.utils.math as math_utils

    w, x, y, z = np.asarray(quat_wxyz, dtype=np.float64)
    tensor = torch.tensor([x, y, z, w], dtype=torch.float32).unsqueeze(0)
    return math_utils.matrix_from_quat(tensor)[0].numpy().astype(np.float64)


@dataclass(frozen=True)
class GraspProposal:
    """One candidate grasp pose in world coordinates."""

    label: str
    """Human-readable description of which free parameters produced it."""

    position: np.ndarray
    """Tool-frame target position."""

    quat_wxyz: np.ndarray
    """Tool-frame target orientation."""

    def as_target(self) -> tuple[str, np.ndarray, np.ndarray]:
        """As a ``plan_best_pose`` target tuple."""
        return self.label, self.position, self.quat_wxyz


def _rot_about(axis: np.ndarray, theta: float) -> np.ndarray:
    """Rotation of ``theta`` radians about an arbitrary unit axis (Rodrigues)."""
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / np.linalg.norm(axis)
    cross = np.array([[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]], dtype=np.float64)
    return np.eye(3) + np.sin(theta) * cross + (1.0 - np.cos(theta)) * (cross @ cross)


def _object_pose(env, scene_key: str) -> tuple[np.ndarray, np.ndarray]:
    """An object's world position and rotation matrix."""
    import torch

    import isaaclab.utils.math as math_utils
    import warp as wp

    asset = env.scene[scene_key]
    position = wp.to_torch(asset.data.root_pos_w)[0].detach().cpu().numpy().astype(np.float64)
    quat_xyzw = wp.to_torch(asset.data.root_quat_w)[0].detach().cpu().numpy().astype(np.float64)
    rotation = math_utils.matrix_from_quat(torch.tensor(quat_xyzw, dtype=torch.float32).unsqueeze(0))[0].numpy()
    return position, rotation.astype(np.float64)


def annotated_point_grasps(
    env,
    scene_key: str,
    local_offset_m: tuple[float, float, float],
    tilt_deg=(10, 20, 30, 40),
    yaw_deg=(0, 15, -15, 30, -30),
    z_offset_m: float = 0.0,
) -> list[GraspProposal]:
    """Top-down grasps at an annotated point in the object's frame.

    Args:
        env: The unwrapped environment.
        scene_key: Scene key of the object.
        local_offset_m: The annotated point, in the object's own frame.
        tilt_deg: Leans of the tool away from vertical to try.
        yaw_deg: Wrist yaws to try.
        z_offset_m: Height added to the annotated point.

    Returns:
        One proposal per (tilt, yaw) combination.
    """
    position, rotation = _object_pose(env, scene_key)
    point = position + rotation @ np.asarray(local_offset_m, dtype=np.float64) + np.array([0.0, 0.0, z_offset_m])
    proposals = []
    for tilt in tilt_deg:
        for yaw in yaw_deg:
            orientation = _rot_z(np.radians(yaw)) @ _rot_y(np.radians(tilt)) @ DOWN_FACING_ROTATION
            proposals.append(
                GraspProposal(f"tilt {tilt:g} yaw {yaw:g}", point.copy(), quat_wxyz_from_matrix(orientation))
            )
    return proposals


def rim_grasps(
    env,
    scene_key: str,
    rim_radius_m: float,
    rim_height_m: float,
    num_angles: int = 24,
    tilt_deg=(0, 10, 20, 30),
    z_offset_m: float = 0.0,
    avoid_positions: list[np.ndarray] | None = None,
    avoid_radius_m: float = 0.13,
    jaw_spin_deg=(-90.0, 90.0),
) -> list[GraspProposal]:
    """Grasps distributed around the rim of a body of revolution.

    The wrist is yawed so the fingers close across the rim wall radially, which is what actually
    holds a bowl: the grasp is a pinch on the rim, and the gripper's span never has to exceed the
    object's diameter.

    Args:
        env: The unwrapped environment.
        scene_key: Scene key of the object.
        rim_radius_m: Rim radius, e.g. RoboDojo metadata's ``place.up.projection_circle.radius``.
        rim_height_m: Rim height above the object's origin.
        num_angles: How many angles around the rim to sample.
        tilt_deg: Leans of the tool away from vertical to try.
        z_offset_m: Height added to the rim.
        avoid_positions: World positions the grasp should stay clear of, e.g. the other objects.
        avoid_radius_m: How far a rim point must stay from each avoided position.
        jaw_spin_deg: Rotations about the tool's *own* approach axis, applied after the lean.
            This is what aims the jaws: for a gripper whose jaws separate along tool y, -90 puts
            that separation along the rim's radial direction so the fingers straddle the wall
            instead of sliding along it. Being a spin about the approach axis it leaves the lean,
            and therefore reachability, untouched -- unlike folding it into the yaw. Both -90 and
            +90 are offered by default because a parallel jaw grasp is unchanged by a 180 deg
            spin, so they are the same grasp reached with two different wrist configurations, and
            dropping one halves the arm's chances of being able to take it.

    Returns:
        One proposal per surviving (tilt, angle, jaw spin) combination, gentlest lean first.
    """
    position, rotation = _object_pose(env, scene_key)
    proposals = []
    # Tilt-major, so gentler approaches come first and a caller breaking ties by proposal order
    # prefers them. A steep lean aims the jaws sideways onto the rim: it reaches the pose but does
    # not hold it.
    for tilt in tilt_deg:
        for i in range(num_angles):
            theta = 2.0 * np.pi * i / num_angles
            local = np.array([rim_radius_m * np.cos(theta), rim_radius_m * np.sin(theta), rim_height_m])
            point = position + rotation @ local + np.array([0.0, 0.0, z_offset_m])
            if avoid_positions is not None and any(
                np.linalg.norm(point[:2] - np.asarray(other)[:2]) < avoid_radius_m for other in avoid_positions
            ):
                continue
            # Aim the jaws along the rim's outward normal, so they close across the wall.
            normal = (point - position)[:2]
            yaw = float(np.arctan2(normal[1], normal[0]))
            for spin in np.atleast_1d(jaw_spin_deg):
                orientation = _rot_z(yaw) @ _rot_y(np.radians(tilt)) @ DOWN_FACING_ROTATION @ _rot_z(np.radians(spin))
                proposals.append(
                    GraspProposal(
                        f"rim {np.degrees(theta):.0f} deg tilt {tilt:g} spin {spin:+g}",
                        point.copy(),
                        quat_wxyz_from_matrix(orientation),
                    )
                )
    return proposals


def slab_grasps(
    env,
    scene_key: str,
    face_normal_local: tuple[float, float, float],
    bbox_min_m: tuple[float, float, float],
    bbox_max_m: tuple[float, float, float],
    grasp_depth_m=(0.020, 0.030, 0.040),
    lateral_offset_m=(0.0, -0.030, 0.030, -0.055, 0.055),
    tilt_deg=(0, 10, 20, 30, -10, -20, -30),
    flip=(False, True),
    approach_offset_m: float = 0.0,
) -> list[GraspProposal]:
    """Pinch grasps on the exposed top edge of a flat slab standing on edge.

    A slice of bread in a toast rack is not a body of revolution, so ``rim_grasps`` does not
    describe it: there is no rim, and its ``place.projection_circle`` is just the circle
    circumscribing a square face. What holds a slab is a pinch across its *thickness*, taken on
    the part of it that sticks up above whatever it is standing in. So the jaws are aimed along
    the face normal and the grasp point is placed a little way down from the top edge.

    Which way is up is read from the object's live pose rather than passed in: only the thin axis
    is a property of the asset, and the standing orientation is a property of the scene.

    Args:
        env: The unwrapped environment.
        scene_key: Scene key of the slab.
        face_normal_local: The slab's thin axis, in its own frame, e.g. ``(0, 0, 1)``.
        bbox_min_m: The slab's bounding box minimum in its own frame.
        bbox_max_m: The slab's bounding box maximum in its own frame. Given as a box rather than as
            half-extents because an asset's origin is not necessarily at the centre of its
            geometry: RoboDojo's bread has its origin on the *bottom face*, 5.6 mm off the slab's
            mid-plane, so jaws aimed at the origin straddle it lopsidedly and one pad reaches its
            face after 3.7 mm while the other still has 14.7 mm to go. The linkage jams on the
            first contact and the pinch never closes.
        grasp_depth_m: How far below the top edge to pinch. Too shallow and the jaws hold only a
            corner; too deep and they foul whatever the slab is standing in.
        lateral_offset_m: Offsets along the slab's other in-plane axis, so the grasp does not have
            to be taken at the middle of the edge.
        tilt_deg: Leans of the approach away from straight down, taken **about the face normal**.
            That keeps the jaws exactly parallel to the two faces however far the tool leans, which
            a lean in any other plane would not; it buys reachability without spoiling the pinch.
        approach_offset_m: How much further along the approach to drive the tool frame, to put the
            finger pads -- rather than the tool frame -- at the intended grasp point. A tool frame
            is not generally between the pads: on the Agibot's right hand they sit about 17 mm
            behind it, so a pose commanded 20 mm below a slice's top edge closes the pads 3 mm
            below it, on the topmost sliver of bread. Measure it, do not assume it is zero; the
            arm reaches the commanded pose either way and only the object betrays the difference.
        flip: Whether to also offer the grasp with the jaw axis reversed. A parallel-jaw grasp is
            unchanged by a 180 deg spin about its approach axis, so the two are the same grasp
            reached with different wrist configurations, and offering both doubles the arm's
            chances of being able to take it.

    Returns:
        One proposal per (depth, offset, tilt, flip) combination, gentlest lean first.
    """
    position, rotation = _object_pose(env, scene_key)
    bbox_min = np.asarray(bbox_min_m, dtype=np.float64)
    bbox_max = np.asarray(bbox_max_m, dtype=np.float64)
    half_extents_m = 0.5 * (bbox_max - bbox_min)
    # Everything below is stated about the slab's own centre, which is where a pinch has to be
    # aimed; the asset's origin only happens to coincide with it.
    position = position + rotation @ (0.5 * (bbox_min + bbox_max))

    face_index = int(np.argmax(np.abs(np.asarray(face_normal_local, dtype=np.float64))))
    in_plane = [i for i in range(3) if i != face_index]
    # Of the slab's two in-plane axes, whichever currently stands closest to vertical is the one
    # the top edge is along; the other is the edge's own direction.
    verticality = [abs(rotation[2, i]) for i in in_plane]
    up_index = in_plane[int(np.argmax(verticality))]
    lateral_index = in_plane[1 - int(np.argmax(verticality))]

    up_world = rotation[:, up_index] * np.sign(rotation[2, up_index] or 1.0)
    lateral_world = rotation[:, lateral_index]
    normal_world = rotation @ (np.asarray(face_normal_local, dtype=np.float64) / np.linalg.norm(face_normal_local))
    # Re-orthogonalise against the measured "up": a settled slab leans, and the jaw axis has to be
    # square to the approach or the pose is not a rotation at all.
    normal_world = normal_world - np.dot(normal_world, up_world) * up_world
    normal_world /= np.linalg.norm(normal_world)

    top_edge = position + up_world * half_extents_m[up_index]

    proposals = []
    # Tilt-major, so a caller breaking ties by proposal order prefers the gentler approach.
    for tilt in tilt_deg:
        approach = _rot_about(normal_world, np.radians(tilt)) @ (-up_world)
        for depth in grasp_depth_m:
            for offset in lateral_offset_m:
                point = top_edge - up_world * depth + lateral_world * offset
                point = point + approach * approach_offset_m
                for flipped in np.atleast_1d(flip):
                    jaw = -normal_world if flipped else normal_world
                    orientation = np.column_stack([np.cross(jaw, approach), jaw, approach])
                    proposals.append(
                        GraspProposal(
                            f"depth {depth * 1000:.0f} mm offset {offset * 1000:+.0f} mm"
                            f" tilt {tilt:g} {'flipped' if flipped else 'upright'}",
                            point.copy(),
                            quat_wxyz_from_matrix(orientation),
                        )
                    )
    return proposals
