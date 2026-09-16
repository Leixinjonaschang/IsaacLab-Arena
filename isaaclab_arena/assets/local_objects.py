# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Assets loaded from local (non-Nucleus) USD files on the developer's machine.

These are experimental assets that are not part of the shared object library. Their ``usd_path``
points at an absolute host path, so they are only usable on machines where that path exists.
"""

import os

from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.sim.spawners.materials.physics_materials_cfg import RigidBodyMaterialCfg

from isaaclab_arena.assets.background_library import LibraryBackground
from isaaclab_arena.assets.object_base import ObjectType
from isaaclab_arena.assets.object_library import LibraryObject
from isaaclab_arena.assets.register import register_asset

LOCAL_ASSET_DIR = os.environ.get("ARENA_LOCAL_ASSET_DIR", "/home/ubuntu/playground/objects/arena_local")
"""Host directory holding the local USD assets. Override with ``ARENA_LOCAL_ASSET_DIR``."""


@register_asset
class RoboDojoSimpleRoom(LibraryBackground):
    """The plain room RoboDojo stages every simulated task in.

    ``env_cfg/scene/default.yml`` selects ``Room: Simple_Room_nolight`` at scale 0.5 and pairs it
    with an HDRI dome (``brown_photostudio_02_4k.hdr``, intensity 1000) -- the "nolight" variant
    carries no emitters of its own, so the dome does all the lighting. Arena environments that
    want RoboDojo's look should spawn this together with the ``brown_photostudio_robolab`` HDR
    and drop the ground plane, since the room brings its own floor.
    """

    name = "robodojo_simple_room"
    tags = ["background", "robodojo"]
    # Verbatim copy of RoboDojo's Assets/Room/Simple_Room_nolight, textures included.
    usd_path = f"{LOCAL_ASSET_DIR}/simple_room_nolight/simple_room_nolight.usd"
    object_min_z = -0.05

    SCALE = (0.5, 0.5, 0.5)
    """The scale ``default.yml`` applies. The asset is authored several times life size."""


@register_asset
class RoboDojoTable(LibraryBackground):
    """The tabletop RoboDojo works on: a mahogany slab, 1.1 m deep by 1.4 m wide by 50 mm thick.

    RoboDojo builds this at run time rather than shipping it as an asset
    (``env/scene_manager/objects/table.py``): a Cube mesh scaled to ``[1.4, 1.1, 0.05]`` with the
    ``material_0122`` MDL (Mahogany_Planks) bound to it. The USD here is that slab authored once,
    with its long axis along y so an Agibot facing +x needs no rotation, and at its real size so
    the prim origin is exactly the slab centre.
    """

    name = "robodojo_table"
    tags = ["background", "robodojo"]
    usd_path = f"{LOCAL_ASSET_DIR}/robodojo_table/robodojo_table.usda"
    object_min_z = -0.05

    HALF_THICKNESS_M = 0.025
    """The work surface is ``origin_z + HALF_THICKNESS_M``, exactly -- there is no lip or inset."""

    DEPTH_M = 1.1
    WIDTH_M = 1.4

    # RoboDojo's table loader builds its PhysicsMaterial with static and dynamic friction both
    # 0.8, against Isaac Lab's default of 0.5.
    spawn_cfg_addon = {
        "physics_material": RigidBodyMaterialCfg(static_friction=0.8, dynamic_friction=0.8, restitution=0.0),
    }


@register_asset
class Bowl(LibraryObject):
    """A stackable bowl from RoboDojo's stack_bowls task."""

    name = "bowl"
    tags = ["object"]
    # RoboDojo's Assets/Object/RoboDojo/Rigid/bowl/00001, used exactly as it ships: Z-up, metres,
    # defaultPrim /root with RigidBodyAPI, collision approximated by ``convexDecomposition``.
    #
    # Use the asset unmodified. RoboDojo does the same -- its loader
    # (env/scene_manager/objects/rigid.py) sets only mass and friction and passes
    # ``collision=False`` to SingleGeometryPrim, so the collider is whatever the usdz authored.
    usd_path = f"{LOCAL_ASSET_DIR}/bowl.usdz"
    object_type = ObjectType.RIGID

    HALF_HEIGHT_M = 0.0301
    """Origin is at the geometry centre, so a bowl rests on a surface at ``surface_z + this``.

    The bowl is 110 mm across and 60 mm tall, sitting 30.1 mm below its origin."""

    # Friction values are RoboDojo's, taken from its loader rather than from the asset's
    # metadata.json: env/scene_manager/objects/rigid.py builds the PhysicsMaterial from
    # ``static_friction`` (default 0.6) and ``dynamic_friction`` (default 1.5) in the instance
    # config, and stack_bowls.yml sets neither, so the bowls run at 0.6/1.5. The metadata's
    # ``friction: 0.55`` key is never read. Dynamic above static is unusual but it is what the
    # benchmark ships, and a stack of identical bowls needs the extra grip to stay up.
    #
    # Mass is deliberately NOT set: the bowl runs at whatever its USD density gives (~0.075 kg).
    #
    # Declaring it here would be dead code that reads as if it worked. Isaac Lab's
    # ``MassPropertiesCfg`` only *modifies* an existing attribute, and this source has no
    # ``UsdPhysics.MassAPI``, so a ``mass_props`` entry is silently dropped (logged as "Could not
    # perform 'modify_mass_properties'"). Forcing metadata's 0.32 kg would need a ``.usda``
    # override layer applying ``PhysicsMassAPI`` -- and it buys nothing: both masses were measured
    # to behave identically, and RoboDojo's own articulation loader never sets mass either.
    spawn_cfg_addon = {
        "physics_material": RigidBodyMaterialCfg(static_friction=0.6, dynamic_friction=1.5, restitution=0.0),
    }


@register_asset
class Bread(LibraryObject):
    """A slice of bread from RoboDojo's make_toast task."""

    name = "bread"
    tags = ["object", "robodojo"]
    # RoboDojo's Assets/Object/RoboDojo/Rigid/bread/00000, wrapped in a ``.usda`` layer that
    # applies ``PhysicsMassAPI`` so the mass below can take effect -- the shipped usdz has none.
    usd_path = f"{LOCAL_ASSET_DIR}/bread_massed.usda"
    object_type = ObjectType.RIGID

    THICKNESS_M = 0.0118
    """The slice's thin dimension, along its local Z. Its face is 116.9 x 116.1 mm."""

    FACE_WIDTH_M = 0.1169
    FACE_HEIGHT_M = 0.1161

    # RoboDojo's rigid loader (env/scene_manager/objects/rigid.py) builds the PhysicsMaterial from
    # the instance config's ``static_friction`` (default 0.6) and ``dynamic_friction`` (default
    # 1.5); make_toast.yml sets neither, so the bread runs at the defaults. The metadata's
    # ``friction: 0.45`` key is never read -- the same pattern as the bowl above.
    #
    # Mass, unlike the bowl's, *is* set: at the USD's default density a 117 x 116 x 12 mm slab
    # weighs ~0.16 kg, over twice RoboDojo's 0.07 kg, and a slice this thin is pinched rather than
    # cradled, so the extra weight works directly against the grasp.
    spawn_cfg_addon = {
        "physics_material": RigidBodyMaterialCfg(static_friction=0.6, dynamic_friction=1.5, restitution=0.0),
    }


@register_asset
class BreadShelf(LibraryObject):
    """The four-slot toast rack the bread starts in, from RoboDojo's make_toast task."""

    name = "bread_shelf"
    tags = ["object", "robodojo"]
    # RoboDojo's Assets/Object/RoboDojo/Geometry/bread_shelf/00000, used as it ships.
    usd_path = f"{LOCAL_ASSET_DIR}/bread_shelf.usdz"
    object_type = ObjectType.RIGID

    HALF_EXTENTS_M = (0.07915, 0.041675, 0.0381)
    """Half the rack's 158.3 x 83.5 x 76.2 mm bounding box; its origin is at the box centre.

    RoboDojo's ``is_A_in_B`` tests containment against exactly this box, projected to world XY."""

    HALF_HEIGHT_M = HALF_EXTENTS_M[2]

    SLOT_X_M = (0.035, 0.010, -0.015, -0.039)
    """Where the four slots sit along the rack's local X, from its ``passive.support`` metadata."""

    SLOT_Z_M = 0.035
    """Height of every slot's support point above the rack origin, from the same metadata."""

    # The rack is a fixture, not something to be picked up. RoboDojo loads it through its
    # GeometryObject path (env/scene_manager/objects/geometry.py), which explicitly sets
    # ``rigidBodyEnabled = False`` and ``physics:kinematicEnabled`` on the referenced prim even
    # though the usdz authors a RigidBodyAPI. ``kinematic_enabled`` is Isaac Lab's equivalent.
    #
    # Friction is left at RoboDojo's GeometryObject defaults, which are 0.0/0.0 -- deliberately
    # frictionless, so a slice slides into a slot rather than catching on its lip.
    spawn_cfg_addon = {
        "rigid_props": RigidBodyPropertiesCfg(kinematic_enabled=True),
        "physics_material": RigidBodyMaterialCfg(static_friction=0.0, dynamic_friction=0.0, restitution=0.5),
    }


@register_asset
class Toaster(LibraryObject):
    """A two-slot toaster with a sliding lever, from RoboDojo's make_toast task."""

    name = "toaster"
    tags = ["object", "robodojo"]
    # RoboDojo's Assets/Object/RoboDojo/Articulation/toaster/00000, used as it ships.
    usd_path = f"{LOCAL_ASSET_DIR}/toaster.usdz"
    object_type = ObjectType.ARTICULATION

    HALF_HEIGHT_M = 0.0817
    """Origin is at the geometry centre of a 226.7 x 155.4 x 161.9 mm body."""

    LEVER_JOINT_NAME = "joint_1"
    """The lever. Its metadata tags this joint as the ``toast_botton`` affordance; the USD makes it
    prismatic with range [-0.0138, 0.0642] m, and measurement confirms that running it towards the
    upper limit carries the lever body straight down the toaster's own -Z. It rests at 0, so it
    starts 17.7% of the way along its travel, and it is a pure damper (drive stiffness 0, damping
    100) -- once pushed down it stays there rather than springing back.

    Deliberately not exposed through the ``Pressable`` affordance: ``is_pressed`` flips the
    fraction to ``1 - fraction`` for any joint whose lower limit is negative, which reverses the
    meaning of ``LEVER_PRESSED_FRACTION`` below."""

    LEVER_PRESSED_FRACTION = 0.85
    """RoboDojo's ``is_joint_position_above_ratio(toaster, percentage=0.85, tag="toast_botton")``,
    measured from the lower limit -- so the lever counts as down past 52.5 mm of its 78 mm travel."""

    SLOT_RECT_LOCAL_M = {
        "toast_slot1": ((-0.078, 0.072), (-0.041, -0.011)),
        "toast_slot2": ((-0.078, 0.072), (0.014, 0.039)),
    }
    """``(x_min, x_max), (y_min, y_max)`` of each bread slot in the toaster's own frame.

    RoboDojo stores each slot as four corner points under ``passive.functional`` and tests
    containment against the polygon they span in world XY. The corners are axis-aligned in the
    toaster frame and the toaster is only ever yawed, so an axis-aligned test in the local frame
    is the same predicate, without reconstructing the polygon every step."""

    SLOT_Z_LOCAL_M = -0.051
    """Height of both slots' corner points in the toaster frame -- the bottom of the slot."""
