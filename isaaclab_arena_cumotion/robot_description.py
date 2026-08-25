# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Generate cuMotion's XRDF robot description from Isaac Lab's Lula description.

cuMotion ships ready-made configurations for franka and ur10 only. Every other Arena embodiment
has to bring its own, but Isaac Lab already carries the same information for any robot with an
RMPFlow controller: its Lula description holds the cspace, the fixed values of the joints outside
it, and the collision spheres. This module converts one into the other so an XRDF never has to be
written by hand -- and, more to the point, never drifts from the robot the simulator loads.
"""

from __future__ import annotations

import sys
import yaml
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from isaaclab_arena_cumotion.cumotion_embodiment_cfg import CumotionEmbodimentCfg

CUMOTION_PYTHON_PATH = (
    "{isaac_sim_root}/exts/isaacsim.robot_motion.cumotion/pip_prebundle"
)
"""The native ``cumotion`` wheel is prebundled inside the extension rather than site-packages."""


def import_cumotion():
    """Import and return the native ``cumotion`` module, adding its prebundle to ``sys.path``.

    Safe to call repeatedly.
    """
    try:
        import cumotion

        return cumotion
    except ImportError:
        pass

    import isaacsim

    prebundle = Path(isaacsim.__file__).parent / "exts/isaacsim.robot_motion.cumotion/pip_prebundle"
    assert prebundle.is_dir(), f"cuMotion prebundle not found at {prebundle}"
    sys.path.insert(0, str(prebundle))
    import cumotion

    return cumotion


def xrdf_from_lula(cfg: CumotionEmbodimentCfg) -> dict[str, Any]:
    """Build an XRDF document from the embodiment's Lula robot description.

    The Lula ``cspace``/``default_q`` become the XRDF cspace, ``cspace_to_urdf_rules`` become
    ``default_joint_positions`` for every joint outside it (which is how XRDF pins the torso, head
    and the other arm), and ``collision_spheres`` become the world- and self-collision geometry.

    Args:
        cfg: Embodiment description naming the Lula file, tool frame and self-collision pairs.

    Returns:
        An XRDF document, ready to be dumped to YAML for cuMotion's loader.
    """
    lula = yaml.safe_load(Path(cfg.lula_robot_description).read_text())

    defaults: dict[str, float] = {}
    for rule in lula["cspace_to_urdf_rules"]:
        assert rule["rule"] == "fixed", f"Unsupported Lula cspace_to_urdf rule: {rule['rule']}"
        defaults[rule["name"]] = float(rule["value"])
    for joint, value in zip(lula["cspace"], lula["default_q"]):
        defaults[joint] = float(value)

    spheres: dict[str, list[dict[str, Any]]] = {}
    for entry in lula["collision_spheres"]:
        for link, link_spheres in entry.items():
            spheres[link] = [
                {"center": [float(v) for v in s["center"]], "radius": float(s["radius"])} for s in link_spheres
            ]

    geometry_name = "arena_collision_spheres"
    return {
        # 2.0 is required: world_collision is rejected under format_version 1.0.
        "format": "xrdf",
        "format_version": 2.0,
        "default_joint_positions": defaults,
        "cspace": {
            "joint_names": list(lula["cspace"]),
            "acceleration_limits": [float(v) for v in lula["acceleration_limits"]],
            "jerk_limits": [float(v) for v in lula["jerk_limits"]],
        },
        "tool_frames": [cfg.tool_frame],
        "world_collision": {"geometry": geometry_name},
        "self_collision": {"geometry": geometry_name, "ignore": cfg.self_collision_ignore},
        "geometry": {geometry_name: {"spheres": spheres}},
    }


def load_robot_description(cfg: CumotionEmbodimentCfg):
    """Load a cuMotion ``RobotDescription`` for an embodiment, generating its XRDF on the fly.

    Args:
        cfg: Embodiment description naming the Lula file and URDF.

    Returns:
        The cuMotion ``RobotDescription``.
    """
    cumotion = import_cumotion()
    xrdf_text = yaml.dump(xrdf_from_lula(cfg), sort_keys=False, default_flow_style=None)
    urdf_text = Path(cfg.robot_urdf).read_text()
    return cumotion.load_robot_from_memory(xrdf_text, urdf_text)
