# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch

import warp as wp
from isaaclab.assets import RigidObject
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors.contact_sensor.contact_sensor import ContactSensor
from isaaclab.utils.math import matrix_from_quat

from isaaclab_arena.tasks.predicates.object_settling import get_object_initial_rest_state
from isaaclab_arena.tasks.predicates.predicate_utils import (
    get_env,
    get_root_lin_vel_w,
    get_root_pos_w,
    get_root_quat_w,
    select,
)


def object_is_above_height(
    env: ManagerBasedRLEnv,
    object_name: str,
    surface_height: float | None = None,
    use_settled_state: bool = False,
    distance: float = 1e-2,
    env_id: int | None = None,
) -> torch.Tensor:
    """Checks if an object is above a certain height.

    The reference height is either a fixed ``surface_height`` or, when ``use_settled_state`` is set, the
    object's recorded resting height (see ``objects_settled``). For envs where no settled state
    has been recorded, the result is always False.

    Returns True when ``object_name`` is at least ``distance`` m above a height reference.
    """

    assert (
        surface_height is not None
    ) != use_settled_state, "object_is_above_height requires exactly one of surface_height or use_settled_state"

    object_z = get_root_pos_w(env, object_name)[:, 2]
    if use_settled_state:
        settled_pos, has_settled = get_object_initial_rest_state(env, object_name)
        result = has_settled & (object_z > (settled_pos[:, 2] + distance))
    else:
        result = object_z > (surface_height + distance)
    return select(result, env_id)


def object_moving(
    env: ManagerBasedRLEnv,
    object_name: str,
    velocity_threshold: float = 1e-2,
    env_id: int | None = None,
) -> torch.Tensor:
    """Checks if an object is moving above a certain velocity threshold.

    Returns True when object_name's linear speed exceeds velocity_threshold (m/s).
    """

    speed = torch.linalg.vector_norm(get_root_lin_vel_w(env, object_name), dim=-1)
    result = speed > velocity_threshold
    return select(result, env_id)


def objects_in_proximity(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg,
    target_object_cfg: SceneEntityCfg,
    max_y_separation: float,
    max_x_separation: float,
    max_z_separation: float,
) -> torch.Tensor:
    """Determine if two objects are within a certain proximity of each other.

    Returns True when the object is within a certain proximity of the target object.
    """

    # Get object entities from the scene
    object: RigidObject = env.scene[object_cfg.name]
    target_object: RigidObject = env.scene[target_object_cfg.name]

    # Get positions relative to environment origin
    object_pos = wp.to_torch(object.data.root_pos_w) - env.scene.env_origins
    target_object_pos = wp.to_torch(target_object.data.root_pos_w) - env.scene.env_origins

    # object to target object
    x_separation = torch.abs(object_pos[:, 0] - target_object_pos[:, 0])
    y_separation = torch.abs(object_pos[:, 1] - target_object_pos[:, 1])
    z_separation = torch.abs(object_pos[:, 2] - target_object_pos[:, 2])

    done = x_separation < max_x_separation
    done = torch.logical_and(done, y_separation < max_y_separation)
    done = torch.logical_and(done, z_separation < max_z_separation)

    return done


def object_on_destination(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("pick_up_object"),
    contact_sensor_cfg: SceneEntityCfg = SceneEntityCfg("pick_up_object_contact_sensor"),
    force_threshold: float = 1.0,
    velocity_threshold: float = 0.5,
) -> torch.Tensor:
    """Checks if an object is in contact with it's destination location via a contact sensor.

    Returns True when the object is in contact with destination above a force threshold
    and below a velocity threshold.
    """

    unwrapped_env = get_env(env)
    object: RigidObject = unwrapped_env.scene[object_cfg.name]
    sensor: ContactSensor = unwrapped_env.scene[contact_sensor_cfg.name]

    # force_matrix_w shape is (N, B, M, 3), where N is the number of sensors, B is number of bodies in each sensor
    # and ``M`` is the number of filtered bodies.
    # We assume B = 1 and M = 1
    assert sensor.data.force_matrix_w.shape[2] == 1
    assert sensor.data.force_matrix_w.shape[1] == 1
    # NOTE(alexmillane, 2025-08-04): We expect the binary flags to have shape (N, )
    # where N is the number of envs.
    force_matrix_norm = torch.norm(wp.to_torch(sensor.data.force_matrix_w), dim=-1).reshape(-1)
    force_above_threshold = force_matrix_norm > force_threshold

    velocity_w = wp.to_torch(object.data.root_lin_vel_w)
    velocity_w_norm = torch.norm(velocity_w, dim=-1)
    velocity_below_threshold = velocity_w_norm < velocity_threshold

    condition_met = torch.logical_and(force_above_threshold, velocity_below_threshold)

    return condition_met


def objects_on_destinations(
    env: ManagerBasedRLEnv,
    object_cfg_list: list[SceneEntityCfg] = [SceneEntityCfg("pick_up_object")],
    contact_sensor_cfg_list: list[SceneEntityCfg] = [SceneEntityCfg("pick_up_object_contact_sensor")],
    force_threshold: float = 1.0,
    velocity_threshold: float = 0.5,
) -> torch.Tensor:
    """Multi-object version of `object_on_destination`.

    Returns True only when ALL objects in the list satisfy the destination condition.
    See `object_on_destination` for details on the single-object logic.
    """

    assert len(object_cfg_list) == len(contact_sensor_cfg_list), (
        "object_cfg_list and contact_sensor_cfg_list must have equal length, got "
        f"{len(object_cfg_list)} objects and {len(contact_sensor_cfg_list)} sensors"
    )

    unwrapped_env = get_env(env)
    condition_met = torch.ones((unwrapped_env.num_envs), device=unwrapped_env.device, dtype=torch.bool)
    for object_cfg, contact_sensor_cfg in zip(object_cfg_list, contact_sensor_cfg_list):
        single_condition = object_on_destination(
            env=env,
            object_cfg=object_cfg,
            contact_sensor_cfg=contact_sensor_cfg,
            force_threshold=force_threshold,
            velocity_threshold=velocity_threshold,
        )
        condition_met = torch.logical_and(condition_met, single_condition)
    return condition_met


def _object_axis_tilt(env: ManagerBasedRLEnv, object_name: str, axis: tuple[float, float, float]) -> torch.Tensor:
    """Return the angle, in radians, between an object's local ``axis`` and world +Z."""
    rot = matrix_from_quat(get_root_quat_w(env, object_name))
    local_axis = torch.tensor(axis, device=rot.device, dtype=rot.dtype)
    world_axis = rot @ local_axis
    return torch.acos(world_axis[:, 2].clamp(-1.0, 1.0))


def objects_upright(
    env: ManagerBasedRLEnv,
    object_names: list[str],
    threshold_rad: float,
    axis: tuple[float, float, float] = (0.0, 0.0, 1.0),
    env_id: int | None = None,
) -> torch.Tensor:
    """Checks that every named object is upright.

    An object counts as upright when its local ``axis``, rotated into the world, is within
    ``threshold_rad`` of world +Z.

    Args:
        env: The environment.
        object_names: Objects to check.
        threshold_rad: How far the axis may tilt away from world +Z.
        axis: The object-local axis that should point up.
        env_id: Restrict the result to a single environment, or None for all of them.

    Returns:
        True when all of object_names are upright.
    """
    result = torch.ones_like(_object_axis_tilt(env, object_names[0], axis), dtype=torch.bool)
    for name in object_names:
        result &= _object_axis_tilt(env, name, axis) < threshold_rad
    return select(result, env_id)


def lowest_object_upright(
    env: ManagerBasedRLEnv,
    object_names: list[str],
    threshold_rad: float,
    axis: tuple[float, float, float] = (0.0, 0.0, 1.0),
    env_id: int | None = None,
) -> torch.Tensor:
    """Checks that whichever of the named objects sits lowest is upright.

    Used for stacking, where the bottom item carries the pile and so has to be squarer than the
    ones resting on it.

    Args:
        env: The environment.
        object_names: Objects to check.
        threshold_rad: How far the lowest object's axis may tilt away from world +Z.
        axis: The object-local axis that should point up.
        env_id: Restrict the result to a single environment, or None for all of them.

    Returns:
        True when the lowest of object_names is upright to within threshold_rad.
    """
    heights = torch.stack([get_root_pos_w(env, name)[:, 2] for name in object_names], dim=-1)
    tilts = torch.stack([_object_axis_tilt(env, name, axis) for name in object_names], dim=-1)
    lowest = heights.argmin(dim=-1, keepdim=True)
    return select(tilts.gather(-1, lowest).squeeze(-1) < threshold_rad, env_id)


def objects_stacked(
    env: ManagerBasedRLEnv,
    object_names: list[str],
    xy_threshold: float,
    min_z_gap: float = 0.005,
    max_z_gap: float | None = None,
    env_id: int | None = None,
) -> torch.Tensor:
    """Checks that the named objects form a single pile.

    The objects are ordered by height, then every neighbouring pair must be within
    ``xy_threshold`` of each other horizontally and separated vertically by at least
    ``min_z_gap`` and, when given, at most ``max_z_gap``.

    The lower bound distinguishes a stack from objects standing side by side. The upper bound
    matters just as much in practice: without it an object held in the gripper directly above
    the pile reads as the top of it, so a run counts as solved while the last item is still in
    the air. Set it to roughly one object height.

    Args:
        env: The environment.
        object_names: Objects that should form the pile.
        xy_threshold: How far apart neighbouring objects may be horizontally.
        min_z_gap: Minimum height difference between neighbouring objects.
        max_z_gap: Maximum height difference between neighbouring objects, or None to leave the
            pile unbounded above.
        env_id: Restrict the result to a single environment, or None for all of them.

    Returns:
        True when object_names form a stack.
    """
    positions = torch.stack([get_root_pos_w(env, name) for name in object_names], dim=1)
    order = positions[:, :, 2].argsort(dim=1)
    ordered = positions.gather(1, order.unsqueeze(-1).expand(-1, -1, 3))

    lower, upper = ordered[:, :-1, :], ordered[:, 1:, :]
    xy_distance = torch.linalg.vector_norm(upper[:, :, :2] - lower[:, :, :2], dim=-1)
    z_gap = upper[:, :, 2] - lower[:, :, 2]
    ok = (xy_distance < xy_threshold) & (z_gap >= min_z_gap)
    if max_z_gap is not None:
        ok &= z_gap <= max_z_gap
    return select(ok.all(dim=1), env_id)


def objects_at_rest(
    env: ManagerBasedRLEnv,
    object_names: list[str],
    velocity_threshold: float = 0.02,
    env_id: int | None = None,
) -> torch.Tensor:
    """Checks that none of the named objects is still moving.

    Used to insist a manipulation has actually been let go of and settled, rather than being
    judged mid-motion while the robot still holds something. Calibrate the threshold against the
    contact configuration the task ends in: PhysX reports a standing velocity for bodies in a
    resting contact stack that never decays, so a value taken from a lone object is unreachable.

    Args:
        env: The environment.
        object_names: Objects to check.
        velocity_threshold: Speed below which an object counts as settled, in m/s.
        env_id: Restrict the result to a single environment, or None for all of them.

    Returns:
        True when every object in object_names is slower than velocity_threshold.
    """
    speeds = torch.stack(
        [torch.linalg.vector_norm(get_root_lin_vel_w(env, name), dim=-1) for name in object_names], dim=-1
    )
    return select((speeds < velocity_threshold).all(dim=-1), env_id)


def objects_upright_about_any_axis(
    env: ManagerBasedRLEnv,
    object_names: list[str],
    threshold_rad: float,
    axes: tuple[tuple[float, float, float], ...] = (
        (1.0, 0.0, 0.0),
        (-1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, -1.0, 0.0),
    ),
    env_id: int | None = None,
) -> torch.Tensor:
    """Checks that every named object has at least one of ``axes`` pointing up.

    Objects that are flat slabs stand on an edge rather than on a face, so no single local axis
    identifies "the right way up" -- any of the four in-plane axes will do. Defaults to those four.

    Args:
        env: The environment.
        object_names: Objects to check.
        threshold_rad: How far the best-aligned axis may tilt away from world +Z.
        axes: Object-local axes, any one of which pointing up counts.
        env_id: Restrict the result to a single environment, or None for all of them.

    Returns:
        True when every object in object_names has some axis in axes within threshold_rad of up.
    """
    result = torch.ones_like(_object_axis_tilt(env, object_names[0], axes[0]), dtype=torch.bool)
    for name in object_names:
        per_axis = torch.stack([_object_axis_tilt(env, name, axis) for axis in axes], dim=-1)
        result &= (per_axis < threshold_rad).any(dim=-1)
    return select(result, env_id)


def any_object_near_body(
    env: ManagerBasedRLEnv,
    object_names: list[str],
    body_name: str,
    max_distance_m: float,
    min_height_m: float | None = None,
    articulation_name: str = "robot",
    env_id: int | None = None,
) -> torch.Tensor:
    """Checks that at least one of the named objects sits within reach of a robot body.

    Distance is measured between the object's origin and the body's origin, so the threshold has
    to allow for both the tool frame's offset from the contact and the object's own size. With
    ``min_height_m`` set, an object only counts while its origin is above that world height --
    which separates one held at the gripper from one lying wherever the gripper happens to be.

    Args:
        env: The environment.
        object_names: Candidate objects; only one has to be near.
        body_name: Body of the articulation to measure from, e.g. a gripper's tool frame.
        max_distance_m: How far the object's origin may sit from the body's, in metres.
        min_height_m: World height the object's origin must also clear, or None for no floor.
        articulation_name: Scene key of the articulation carrying the body.
        env_id: Restrict the result to a single environment, or None for all of them.

    Returns:
        True when some object in object_names is within max_distance_m of the body.
    """
    articulation = get_env(env).scene.articulations[articulation_name]
    body_index = list(articulation.data.body_names).index(body_name)
    body_pos = wp.to_torch(articulation.data.body_pos_w)[:, body_index]
    near = []
    for name in object_names:
        position = get_root_pos_w(env, name)
        ok = torch.linalg.vector_norm(position - body_pos, dim=-1) <= max_distance_m
        if min_height_m is not None:
            ok &= position[:, 2] > min_height_m
        near.append(ok)
    return select(torch.stack(near, dim=-1).any(dim=-1), env_id)


def _objects_in_frame_box(
    env: ManagerBasedRLEnv,
    object_names: list[str],
    frame_name: str,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    z_range: tuple[float, float | None],
) -> torch.Tensor:
    """Returns, per environment and per object, whether the object's origin is inside the box.

    The box is axis-aligned in ``frame_name``'s own frame, so it follows that object as it moves.

    Returns:
        A boolean tensor of shape (num_envs, len(object_names)).
    """
    frame_rot = matrix_from_quat(get_root_quat_w(env, frame_name))
    frame_pos = get_root_pos_w(env, frame_name)
    z_min, z_max = z_range

    inside = []
    for name in object_names:
        # World offset rotated back into the frame's axes; the env origin cancels in the delta.
        local = torch.einsum("nij,nj->ni", frame_rot.transpose(1, 2), get_root_pos_w(env, name) - frame_pos)
        ok = (local[:, 0] >= x_range[0]) & (local[:, 0] <= x_range[1])
        ok &= (local[:, 1] >= y_range[0]) & (local[:, 1] <= y_range[1])
        ok &= local[:, 2] >= z_min
        if z_max is not None:
            ok &= local[:, 2] <= z_max
        inside.append(ok)
    return torch.stack(inside, dim=-1)


def any_object_in_frame_box(
    env: ManagerBasedRLEnv,
    object_names: list[str],
    frame_name: str,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    z_range: tuple[float, float | None],
    env_id: int | None = None,
) -> torch.Tensor:
    """Checks that at least one of the named objects sits inside a box fixed to another object.

    Args:
        env: The environment.
        object_names: Candidate objects; only one has to be inside.
        frame_name: Object whose frame the box is expressed in.
        x_range: Minimum and maximum along the frame's local X, in metres.
        y_range: Minimum and maximum along the frame's local Y, in metres.
        z_range: Minimum and maximum along the frame's local Z; the maximum may be None.
        env_id: Restrict the result to a single environment, or None for all of them.

    Returns:
        True when some object in object_names is inside the box.
    """
    return select(_objects_in_frame_box(env, object_names, frame_name, x_range, y_range, z_range).any(dim=-1), env_id)


def count_objects_in_frame_box(
    env: ManagerBasedRLEnv,
    object_names: list[str],
    frame_name: str,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    z_range: tuple[float, float | None],
    count: int,
    env_id: int | None = None,
) -> torch.Tensor:
    """Checks that exactly ``count`` of the named objects sit inside a box fixed to another object.

    Args:
        env: The environment.
        object_names: Objects to count.
        frame_name: Object whose frame the box is expressed in.
        x_range: Minimum and maximum along the frame's local X, in metres.
        y_range: Minimum and maximum along the frame's local Y, in metres.
        z_range: Minimum and maximum along the frame's local Z; the maximum may be None.
        count: How many objects must be inside, exactly.
        env_id: Restrict the result to a single environment, or None for all of them.

    Returns:
        True when the number of object_names inside the box equals count.
    """
    inside = _objects_in_frame_box(env, object_names, frame_name, x_range, y_range, z_range)
    return select(inside.sum(dim=-1) == count, env_id)
