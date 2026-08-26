# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import copy
import os
from collections.abc import Sequence
from dataclasses import MISSING

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
import isaaclab.utils.math as PoseUtils
from isaaclab.controllers.config.rmp_flow import AGIBOT_LEFT_ARM_RMPFLOW_CFG, AGIBOT_RIGHT_ARM_RMPFLOW_CFG
from isaaclab.envs.common import ViewerCfg
from isaaclab.envs.mdp.actions.rmpflow_actions_cfg import RMPFlowActionCfg
from isaaclab.managers import EventTermCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.sensors import CameraCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import FrameTransformerCfg, OffsetCfg
from isaaclab.utils.configclass import configclass
from isaaclab_assets.robots.agibot import AGIBOT_A2D_CFG
from isaaclab_tasks.manager_based.manipulation.pick_place.mdp import get_robot_joint_state
from isaaclab_tasks.manager_based.manipulation.stack.mdp import ee_frame_pose_in_base_frame

from isaaclab_arena.assets.register import register_asset
from isaaclab_arena.embodiments.common.arm_mode import ArmMode
from isaaclab_arena.embodiments.common.smooth_joint_actions import SmoothJointPositionActionCfg
from isaaclab_arena.embodiments.embodiment_base import EmbodimentBase
from isaaclab_arena.embodiments.franka.franka import FrankaMimicEnv
from isaaclab_arena.terms.events import reset_joint_position_and_velocity_to_defaults
from isaaclab_arena.utils.cameras import ArenaCameraCfg, get_viewer_cfg_from_robot_body
from isaaclab_arena.utils.pose import Pose

# --- Arena's default Agibot configuration ---------------------------------------------------
#
# The shipped ``AGIBOT_A2D_CFG`` is left/right asymmetric in two places, and in both the right
# side is the one whose behaviour has been measured good. Arena's default copies the right side
# onto the left wholesale rather than curating individual parameters.

MIRRORED_LEFT_WRIST_JOINT_POS = {"left_arm_joint6": 0.7, "left_arm_joint7": 0.0}
"""Left wrist angles mirroring the right arm's shipped ``-0.7`` and ``0.0``.

The shipped rest pose mirrors joints 1-5 exactly (left = -right) but not the wrists (left
joint6 +1.4725 / joint7 -0.1599 against a mirror's -1.4725 / +0.1599). ``gripper_center`` hangs
off those two joints, so the hands start 160 mm out of mirror symmetry and the left one sits
tucked over the robot's own body, outside a head-mounted view. The right arm's wrist is
mirrored onto the left -- not the other way round -- because the right arm's reach band and
grasp behaviour are the measured ones."""

_RMPFLOW_DIR = os.path.join(os.path.dirname(__file__), "rmpflow")
"""Local copies of the Agibot lula robot-description yamls, patched to the mirrored rest pose.

The rest pose is stated in three places that must agree -- ``init_state.joint_pos``, each arm
yaml's ``default_q``, and the other arm yaml's ``cspace_to_urdf_rules`` fixed values -- or an
arm's collision model and c-space attractor no longer describe the robot that is actually
there. These are copies rather than edits in place because the stock yamls are downloaded into
the Isaac asset cache on every run, where any edit would be silently overwritten."""

# Deep copies throughout: ``configclass.replace`` is ``dataclasses.replace``, i.e. shallow, so
# ``init_state`` and the actuator configs would otherwise be shared with the Isaac Lab
# module-level ``AGIBOT_A2D_CFG`` and these edits would leak into every other user of it.
AGIBOT_ARENA_A2D_CFG = copy.deepcopy(AGIBOT_A2D_CFG)
AGIBOT_ARENA_A2D_CFG.init_state.joint_pos.update(MIRRORED_LEFT_WRIST_JOINT_POS)

# The grippers ship with identical stiffness/damping but 10x/100x different drive ceilings
# (left 10 N m / 2 rad/s, support 1 N m; right 100 / 10 / 100), and only the right's behave.
# Cross-matrix measurement (right arm, bowl rim pinch + headband pinch, 5 repeats/cell):
# velocity 2 rad/s cannot grasp at all -- 0/15 lifts at effort 10/30/100, the fingers close too
# slowly and the object slips out -- while effort >= 30 is step-for-step identical to 100. The
# left gripper gets the right's values verbatim. (The arm actuators are already identical on
# both sides; the wrist rest pose above is the only arm-side asymmetry.)
AGIBOT_ARENA_A2D_CFG.actuators["left_gripper"].effort_limit_sim = {
    "left_hand_joint1": 100.0,
    "left_.*_Support_Joint": 100.0,
}
AGIBOT_ARENA_A2D_CFG.actuators["left_gripper"].velocity_limit_sim = 10.0
AGIBOT_ARENA_A2D_CFG.actuators["left_gripper_passive"].effort_limit_sim = 100.0

AGIBOT_LEFT_ARM_ARENA_RMPFLOW_CFG = copy.deepcopy(AGIBOT_LEFT_ARM_RMPFLOW_CFG)
AGIBOT_LEFT_ARM_ARENA_RMPFLOW_CFG.collision_file = os.path.join(_RMPFLOW_DIR, "agibot_left_arm_gripper.yaml")

AGIBOT_RIGHT_ARM_ARENA_RMPFLOW_CFG = copy.deepcopy(AGIBOT_RIGHT_ARM_RMPFLOW_CFG)
AGIBOT_RIGHT_ARM_ARENA_RMPFLOW_CFG.collision_file = os.path.join(_RMPFLOW_DIR, "agibot_right_arm_gripper.yaml")


@configclass
class AgibotCameraCfg(ArenaCameraCfg):
    """Camera rig for the Agibot: the head view, as a recordable sensor.

    Reproduces ``get_head_viewer_cfg``'s framing -- same eye, gaze and field of view -- so
    recorded observations match what a teleoperator saw in the viewport. Mounted under
    ``base_link`` rather than the head link because the viewer offsets are world-axis values
    (see ``HEAD_VIEW_EYE``); the head holds its default pose through the reset event, so the
    two mounts see the same thing, and the base frame is the one the offsets are stated in.
    """

    head_cam: CameraCfg = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base_link/HeadCam",
        update_period=0.0,
        height=512,
        width=512,
        data_types=["rgb"],
        # The Kit viewport's field of view, matching the teleop view and the demo recordings.
        spawn=sim_utils.PinholeCameraCfg(focal_length=18.147, horizontal_aperture=20.955, clipping_range=(0.05, 30.0)),
        offset=CameraCfg.OffsetCfg(
            # HEAD_VIEW_EYE above the measured head position (-0.157, 0, 1.263), in base frame.
            pos=(0.443, 0.0, 1.683),
            # Looking along HEAD_VIEW_LOOKAT - HEAD_VIEW_EYE = (0.66, 0, -1.05), no roll.
            rot=(0.19581, -0.19581, -0.67946, 0.67946),
            convention="opengl",
        ),
    )


@register_asset
class AgibotEmbodiment(EmbodimentBase):
    """Embodiment for the Agibot robot."""

    name = "agibot"
    default_arm_mode = ArmMode.LEFT

    HEAD_BODY_NAME = "link_pitch_head"
    """Body carrying the head, used to anchor a first-person view."""

    # World-axis offsets from the head's position, NOT head-frame coordinates. Isaac Lab's
    # ``origin_type="asset_body"`` takes only ``body_pos_w`` for the viewer origin and then adds
    # eye/lookat in world axes, so the view tracks where the head *is* but not which way it
    # faces. Measured head position with the robot at (-0.6, 0, 0): (-0.157, 0.0, 1.263).
    HEAD_VIEW_EYE = (0.0, 0.0, 0.42)
    """Viewpoint directly above the head, looking down over it.

    Sitting at the head itself is too close to frame both arms: their grippers are then ~34 deg
    off-axis and the viewport's perspective camera only spans +/-30 deg horizontally, which the
    ViewerCfg cannot widen. Standing off brings them to ~21 deg, the framing RoboDojo's head view
    has. The standoff has to go straight up, not backwards -- from behind, the head's own shell
    fills the middle of the frame."""

    HEAD_VIEW_LOOKAT = (0.66, 0.0, -0.63)
    """Gaze point: the measured offset from the head to the centre of a table-top workspace,
    straight forward and down. Symmetric in y, so both arms frame up at the edges of the view
    the way RoboDojo's head view does; an earlier value biased the gaze to the robot's right,
    which put one arm off-screen."""

    def __init__(
        self, enable_cameras: bool = False, initial_pose: Pose | None = None, arm_mode: ArmMode = ArmMode.LEFT
    ):
        super().__init__(enable_cameras, initial_pose)
        self.arm_mode = arm_mode or self.default_arm_mode
        if self.arm_mode == ArmMode.DUAL_ARM:
            self.scene_config = AgibotDualArmSceneCfg()
            self.action_config = AgibotDualArmActionsCfg()
        elif self.arm_mode == ArmMode.LEFT:
            self.scene_config = AgibotLeftArmSceneCfg()
            self.action_config = AgibotLeftArmActionsCfg()
        else:
            self.scene_config = AgibotRightArmSceneCfg()
            self.action_config = AgibotRightArmActionsCfg()
        self.observation_config = AgibotObservationsCfg()
        self.event_config = AgibotEventCfg()
        self.camera_config = AgibotCameraCfg()
        self.mimic_env = AgibotMimicEnv

    def get_head_viewer_cfg(self, lookat: tuple[float, float, float] | None = None) -> ViewerCfg:
        """Return a viewer config mounted on the head, giving a first-person view.

        Args:
            lookat: Gaze point as a world-axis offset from the head. Defaults to
                :attr:`HEAD_VIEW_LOOKAT`, which frames a table-top workspace in front of the
                robot.
        """
        return get_viewer_cfg_from_robot_body(
            body_name=self.HEAD_BODY_NAME,
            eye=self.HEAD_VIEW_EYE,
            lookat=self.HEAD_VIEW_LOOKAT if lookat is None else lookat,
        )


@configclass
class AgibotSceneCfg:
    """Scene configuration for the Agibot."""

    robot = AGIBOT_ARENA_A2D_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    ee_frame: FrameTransformerCfg = MISSING


@configclass
class AgibotLeftArmSceneCfg(AgibotSceneCfg):
    """Scene configuration for the Agibot left arm."""

    ee_frame = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base_link",
        debug_vis=False,
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Robot/gripper_center",
                name="left_end_effector",
                offset=OffsetCfg(
                    rot=(0.0, -0.7071, 0.0, 0.7071),
                ),
            ),
        ],
    )

    def __post_init__(self):
        # Add a marker to the end-effector frame
        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
        marker_cfg.prim_path = "/Visuals/FrameTransformer"
        self.ee_frame.visualizer_cfg = marker_cfg


@configclass
class AgibotRightArmSceneCfg(AgibotSceneCfg):
    """Scene configuration for the Agibot right arm."""

    ee_frame = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base_link",
        debug_vis=False,
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Robot/right_gripper_center",
                name="right_end_effector",
            ),
        ],
    )

    def __post_init__(self):
        # Add a marker to the end-effector frame
        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
        marker_cfg.prim_path = "/Visuals/FrameTransformer"
        self.ee_frame.visualizer_cfg = marker_cfg


@configclass
class AgibotDualArmSceneCfg(AgibotSceneCfg):
    """Scene configuration exposing both of the Agibot's end-effector frames."""

    ee_frame = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base_link",
        debug_vis=False,
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Robot/gripper_center",
                name="left_end_effector",
                offset=OffsetCfg(
                    rot=(0.0, -0.7071, 0.0, 0.7071),
                ),
            ),
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Robot/right_gripper_center",
                name="right_end_effector",
            ),
        ],
    )

    def __post_init__(self):
        # Add a marker to the end-effector frames
        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
        marker_cfg.prim_path = "/Visuals/FrameTransformer"
        self.ee_frame.visualizer_cfg = marker_cfg


# -90 deg about y: cancels the extra quarter turn the URDF gives the left tool frame
# (``gripper_center_joint`` rpy "0 -1.5708 -1.5708" vs the right's "0 0 -1.5708").
_LEFT_ARM_BODY_OFFSET_ROT_XYZW = (0.0, -0.7071, 0.0, 0.7071)


@configclass
class AgibotLeftArmActionsCfg:
    """Action configuration for the Agibot left arm."""

    arm_action = RMPFlowActionCfg(
        asset_name="robot",
        joint_names=["left_arm_joint.*"],
        body_name="gripper_center",
        controller=AGIBOT_LEFT_ARM_ARENA_RMPFLOW_CFG,
        scale=1.0,
        body_offset=RMPFlowActionCfg.OffsetCfg(rot=_LEFT_ARM_BODY_OFFSET_ROT_XYZW),
        use_relative_mode=True,
    )

    gripper_action = mdp.BinaryJointPositionActionCfg(
        asset_name="robot",
        joint_names=["left_hand_joint1", "left_.*_Support_Joint"],
        open_command_expr={"left_hand_joint1": 0.994, "left_.*_Support_Joint": 0.994},
        close_command_expr={"left_hand_joint1": 0.0, "left_.*_Support_Joint": 0.0},
    )


@configclass
class AgibotRightArmActionsCfg:
    """Action configuration for the Agibot right arm."""

    arm_action = RMPFlowActionCfg(
        asset_name="robot",
        joint_names=["right_arm_joint.*"],
        body_name="right_gripper_center",
        controller=AGIBOT_RIGHT_ARM_ARENA_RMPFLOW_CFG,
        scale=1.0,
        use_relative_mode=True,
    )

    gripper_action = mdp.BinaryJointPositionActionCfg(
        asset_name="robot",
        joint_names=["right_hand_joint1", "right_.*_Support_Joint"],
        open_command_expr={"right_hand_joint1": 0.994, "right_.*_Support_Joint": 0.994},
        close_command_expr={"right_hand_joint1": 0.0, "right_.*_Support_Joint": 0.0},
    )


@configclass
class AgibotDualArmActionsCfg:
    """Action configuration driving both Agibot arms at once.

    Field order is load-bearing: Isaac Lab builds the action vector from ``cfg.__dict__.items()``,
    so this lays the 14 values out as ``[left delta pose (6), left gripper (1), right delta pose
    (6), right gripper (1)]``. ``DualArmSe3Keyboard`` emits that layout.
    """

    left_arm_action = RMPFlowActionCfg(
        asset_name="robot",
        joint_names=["left_arm_joint.*"],
        body_name="gripper_center",
        controller=AGIBOT_LEFT_ARM_ARENA_RMPFLOW_CFG,
        scale=1.0,
        body_offset=RMPFlowActionCfg.OffsetCfg(rot=_LEFT_ARM_BODY_OFFSET_ROT_XYZW),
        use_relative_mode=True,
    )

    left_gripper_action = mdp.BinaryJointPositionActionCfg(
        asset_name="robot",
        joint_names=["left_hand_joint1", "left_.*_Support_Joint"],
        open_command_expr={"left_hand_joint1": 0.994, "left_.*_Support_Joint": 0.994},
        close_command_expr={"left_hand_joint1": 0.0, "left_.*_Support_Joint": 0.0},
    )

    right_arm_action = RMPFlowActionCfg(
        asset_name="robot",
        joint_names=["right_arm_joint.*"],
        body_name="right_gripper_center",
        controller=AGIBOT_RIGHT_ARM_ARENA_RMPFLOW_CFG,
        scale=1.0,
        use_relative_mode=True,
    )

    right_gripper_action = mdp.BinaryJointPositionActionCfg(
        asset_name="robot",
        joint_names=["right_hand_joint1", "right_.*_Support_Joint"],
        open_command_expr={"right_hand_joint1": 0.994, "right_.*_Support_Joint": 0.994},
        close_command_expr={"right_hand_joint1": 0.0, "right_.*_Support_Joint": 0.0},
    )


@configclass
class AgibotDualArmJointActionsCfg:
    """Action configuration driving both Agibot arms in joint space, without RMPFlow.

    Every value is an absolute joint position target. This exists for scripted demonstration
    recording: cuMotion's planned trajectories are joint paths, and playing them through the
    RMPFlow terms would re-solve -- and fight -- motions that are already solved. Driving the
    same targets through the action manager instead of writing them straight to the articulation
    is what lets Isaac Lab's recorder hooks see every step.

    The ARM terms are first-order-hold (``SmoothJointPositionActionCfg``): a zero-order hold at
    Arena's 15 Hz control rate jolts the stiff arms hard enough at each control step to work a
    pinched slab out of the gripper mid-carry, which RMPFlow's per-substep smoothing never did.
    The GRIPPER terms are deliberately plain zero-order holds: the smooth term restarts its ramp
    from the *measured* position every control step, and a gripper blocked open by the object it
    is holding then has its squeeze commanded from zero to full 15 times a second -- a pulsing
    grip that measurably walked a rim-held bowl out of the left hand mid-carry (97-138 mm of
    in-hand drift versus 13 mm under constant targets). A constant target is also what the
    binary gripper action and the direct-write executor have always applied.

    Field order is load-bearing, as in ``AgibotDualArmActionsCfg``: ``[left arm (7), left gripper
    (one per finger joint), right arm (7), right gripper]``.
    """

    left_arm_action = SmoothJointPositionActionCfg(
        asset_name="robot",
        joint_names=["left_arm_joint.*"],
        scale=1.0,
        use_default_offset=False,
    )

    left_gripper_action = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=["left_hand_joint1", "left_.*_Support_Joint"],
        scale=1.0,
        use_default_offset=False,
    )

    right_arm_action = SmoothJointPositionActionCfg(
        asset_name="robot",
        joint_names=["right_arm_joint.*"],
        scale=1.0,
        use_default_offset=False,
    )

    right_gripper_action = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=["right_hand_joint1", "right_.*_Support_Joint"],
        scale=1.0,
        use_default_offset=False,
    )


@configclass
class AgibotEventCfg:
    """Reset events for the Agibot robot."""

    reset_robot_to_default_pose = EventTermCfg(
        func=reset_joint_position_and_velocity_to_defaults,
        mode="reset",
    )
    """Restore ``init_state`` joint values and targets on every reset.

    RMPFlow expects ``joint_lift_body`` and ``joint_body_pitch`` at those default values
    through ``cspace_to_urdf_rules`` in ``agibot_left_arm_gripper.yaml``."""


@configclass
class AgibotObservationsCfg:
    """Observation configuration for the Agibot robot."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group with state values."""

        actions = ObsTerm(func=mdp.last_action)
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        # since the robot may not located at the origin of env, we get the eef pose in the base frame
        eef_pos = ObsTerm(func=ee_frame_pose_in_base_frame, params={"return_key": "pos"})
        eef_quat = ObsTerm(func=ee_frame_pose_in_base_frame, params={"return_key": "quat"})
        left_gripper_pos = ObsTerm(
            func=get_robot_joint_state, params={"joint_names": ["left_hand_joint1", "left_Right_1_Joint"]}
        )
        right_gripper_pos = ObsTerm(
            func=get_robot_joint_state,
            params={"joint_names": ["right_hand_joint1", "right_Right_1_Joint"]},
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    # observation groups
    policy: PolicyCfg = PolicyCfg()


class AgibotMimicEnv(FrankaMimicEnv):
    """Configuration for Agibot Mimic."""

    def get_object_poses(self, env_ids: Sequence[int] | None = None):
        """
        Gets the pose of each object (including rigid objects and articulated objects) in the robot base frame.
        This should be aligned with the observation configuration, to ensure all the poses are expressed in the same frame.

        Args:
            env_ids: Environment indices to get the pose for. If None, all envs are considered.

        Returns:
            A dictionary that maps object names to object pose matrix in robot base frame (4x4 torch.Tensor)
        """
        if env_ids is None:
            env_ids = slice(None)

        # Get scene state
        scene_state = self.scene.get_state(is_relative=True)
        rigid_object_states = scene_state["rigid_object"]
        articulation_states = scene_state["articulation"]

        # Get robot root pose
        robot_root_pose = articulation_states["robot"]["root_pose"]
        root_pos = robot_root_pose[env_ids, :3]
        root_quat = robot_root_pose[env_ids, 3:7]

        object_pose_matrix = dict()

        # Process rigid objects
        for obj_name, obj_state in rigid_object_states.items():
            pos_obj_base, quat_obj_base = PoseUtils.subtract_frame_transforms(
                root_pos, root_quat, obj_state["root_pose"][env_ids, :3], obj_state["root_pose"][env_ids, 3:7]
            )
            rot_obj_base = PoseUtils.matrix_from_quat(quat_obj_base)
            object_pose_matrix[obj_name] = PoseUtils.make_pose(pos_obj_base, rot_obj_base)

        # Process articulated objects (except robot)
        for art_name, art_state in articulation_states.items():
            if art_name != "robot":  # Skip robot
                pos_obj_base, quat_obj_base = PoseUtils.subtract_frame_transforms(
                    root_pos, root_quat, art_state["root_pose"][env_ids, :3], art_state["root_pose"][env_ids, 3:7]
                )
                rot_obj_base = PoseUtils.matrix_from_quat(quat_obj_base)
                object_pose_matrix[art_name] = PoseUtils.make_pose(pos_obj_base, rot_obj_base)

        return object_pose_matrix
