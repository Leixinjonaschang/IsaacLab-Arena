# Agibot dual-arm: bugs and gaps found bringing the embodiment up in Arena

**Arena version:** `ec5a9773eff5cd365bdfa0b6c246f3168d979a1b` (`ec5a9773e`, 2026-08-13, #1090), `main`.
Isaac Lab submodule `af1bab4dc173ba69b08fab779c14ead61d13fd33`. All runs at Arena's 15 Hz default.

We are porting RoboDojo `stack_bowls` onto the Agibot and need both arms usable, teleoperated from
a keyboard. **Bugs 1-3** are defects, each reproduced on a clean checkout by reverting exactly one
thing and measuring. **Gaps 4-6** are missing capability. Numbers below are from those runs.

---

## Bug 1 — `AgibotEmbodiment` never sets `event_config`

*Symptom:* after every reset the robot stands with both arms straight out to the sides, hands
~1 m from the workspace and outside any head view; no RMPFlow target tracks afterwards.

**`isaaclab_arena/embodiments/agibot/agibot.py:36-44`** sets `scene_config`, `action_config`,
`observation_config`, `mimic_env` — but never `event_config` (`grep event_config` → 0 hits). So
nothing restores `AGIBOT_A2D_CFG.init_state.joint_pos` and every joint resets to exactly `0.0`
(measured: both `gripper_center` frames at `y = ±1.12 m`). That also zeroes `joint_lift_body`
(`0.1995`) and `joint_body_pitch` (`0.6025`), which lula pins as **fixed** in
`cspace_to_urdf_rules`, so RMPFlow's model no longer describes the actual robot.

Franka has the equivalent term (`franka.py:96`). `tabletop_place_upright` only works because its
task adds its own `reset_scene_to_default` — that *masks* the defect.

**Fix:** add an `AgibotEventCfg` with
`EventTermCfg(func=mdp.reset_scene_to_default, mode="reset", params={"reset_joint_targets": True})`
and assign it in `__init__`.

---

## Bug 2 — left-arm quaternions are `(w,x,y,z)` in `(x,y,z,w)` fields

*Symptom:* the left arm is effectively unteleoperatable — under a **zero** command it drifts after
every reset (~270 mm of uncommanded runaway observed, against the ~58 mm a key press is worth) and
the hands never hold a mirrored pose. The right arm is fine.

Both `(0.7071, 0.0, -0.7071, 0.0)`:
- **`agibot.py:68`** — `FrameTransformerCfg.FrameCfg.offset.rot`
- **`agibot.py:115`** — `RMPFlowActionCfg.body_offset.rot`

Isaac Lab 3.0 uses `(x,y,z,w)` (identity `(0,0,0,1)`). The value is a -90° rotation about **y**
written in `(w,x,y,z)`; read as xyzw it is a 180° flip. The offset exists to cancel the extra
quarter-turn the URDF gives the left tool frame (`gripper_center_joint` rpy `0 -1.5708 -1.5708`
vs `right_gripper_center_joint`'s `0 0 -1.5708`), so a wrong value makes RMPFlow chase an
orientation the arm cannot hold.

Zero command, 800 control steps, right arm as unchanged control: current value → left
`gripper_center` climbs **163 mm** (mirror error z **207.0 mm**); corrected → **9.6 mm**. Right arm
lands within 1.1 mm either way, so the attribution is clean.

**Fix:** write both as `(0.0, -0.7071, 0.0, 0.7071)`.

---

## Bug 3 — the `env_cfg_callback` doc example discards the config

*Symptom:* a callback written exactly as documented crashes with
`AttributeError: 'NoneType' object has no attribute 'seed'` before the environment starts.

**`docs/pages/concepts/environment/environment_definition.rst:180-184`** mutates `env_cfg` and
returns nothing, but **`isaaclab_arena/environments/arena_env_builder.py:395`** does
`env_cfg = self.arena_env.env_cfg_callback(env_cfg)` and dereferences the result at line 398.

**Fix:** add `return env_cfg` to the doc example, or (preferably, so existing callbacks keep
working) ignore a `None` return in the builder.

---

## Gap 4 — no mobile-base support; the chassis is not actuated

*Symptom:* the robot can only be repositioned by re-spawning; nothing drives the base during an
episode, so mobile manipulation is not expressible.

`AGIBOT_A2D_CFG` (`IsaacLab/.../robots/agibot.py:83`) declares actuators for `body`
(`joint_lift_body`, `joint_body_pitch`), `head`, both arms, both grippers — **no wheel or chassis
joints**. Arena's action configs (`agibot.py:109-152`) expose only an arm delta pose plus a binary
gripper. Not Agibot-specific: of all 15 registered embodiments **none** exposes a mobile base
(Galbot's `leg_joint1..4` is a lift column; G1/GR1T2 are legged). Needs chassis joints in the asset
plus a base-velocity action term. Not implemented on our side.

---

## Gap 5 — dual-arm and keyboard teleoperation are unsupported for the Agibot

Three separate walls:

1. **No dual-arm embodiment.** `agibot.py:41-42` builds only the left or right action config;
   `ArmMode.DUAL_ARM` is unhandled (`grep DUAL_ARM agibot.py` → 0) and no scene config exposes
   both end-effector frames, though the hardware is bimanual and the enum already exists.
2. **No dual-arm device.** `device_library.py` registers `openxr`/`keyboard`/`spacemouse` only
   (lines 43, 65, 83); `Se3Keyboard` emits 7 values against the 14 a two-arm vector needs.
3. **Keyboard teleop undocumented, fails silently.** Retargeters are keyed `(device, embodiment)`
   and upstream ships 8 pairs, of which the Agibot has one (`keyboard × agibot`). All three
   `docs/.../step_2_teleoperation.rst` show only `--arena_teleop_device openxr`; nothing says
   `teleop_se3_agent.py` picks its device from its **own** `--teleop_device` while
   `--arena_teleop_device` merely populates `env_cfg.teleop_devices`. Passing only the Arena flag
   silently falls back to the built-in keyboard and dies with
   `Invalid action shape, expected: 14, received: 7`.

*We built locally, happy to upstream:* `AgibotDualArmSceneCfg` / `AgibotDualArmActionsCfg`
(layout `[left pose 6, left gripper 1, right pose 6, right gripper 1]`), a `dual_arm_keyboard`
device (Tab switches arm, per-arm gripper latch), and the matching retargeter entries.

---

## Gap 6 — in dual-arm mode an idle arm walks out of position within 0.7 s of a reset

*Symptom:* drive one arm and the other, commanded exactly zero, leaves its reset pose in ~10
control steps and never returns; the right hand exits a head-mounted viewport. **Absent in
single-arm mode**, so it is invisible upstream today — it appears the moment a second arm gets its
own RMPFlow term, which any dual-arm support will do.

Right end-effector displacement, zero command, 800 steps: single-arm (no action term) **0.2 mm**;
dual-arm **88 mm within 10 steps (0.67 s)**, settling ~64 mm; dual-arm + our workaround **0.2 mm**.

Two Isaac Lab-layer mechanisms compound. (a) The **first policy evaluation after a reset** jumps a
joint 0.4660 rad (~7 rad/s vs the config's own `joint_velocity_cap_rmp.max_velocity: 3.14`) even
though the target equals the current pose, `default_q` equals the current joints, joint velocities
are exactly `0`, the nearest limit is 1.07 rad away and lula's FK matches the sim to 0.0 mm/0.00°.
(b) `RMPFlowAction.process_actions` in relative mode rebuilds the target as *current pose + delta*,
so a zero delta means "hold where you are now" — no restoring force, and (a) is latched in
permanently (the logged target equals the measured end-effector every step and itself moves 66 mm).
We ruled out `ignore_robot_state_updates` (off → diverges to 890 mm), the self-collision
`repulsion_gain` (zeroing changes nothing digit-for-digit) and a lula/USD mismatch. Origin of the
first-step jump **not identified**.

*Workaround on our side:* hold the previous `ee_pose_des` for an arm whose delta is exactly zero,
re-latch on the first non-zero command, clear on reset → back to 0.2 mm.

---

**Also noted, outside Arena:** `RMPFlowAction.reset()`
(`IsaacLab/.../rmpflow_task_space_actions.py:186`) ignores `env_ids` and re-initialises every
environment's controller on any single reset. Read from code; no consequence measured.

**Not included:** bowl-flinging on gripper close is still open on our side and is **not**
attributed to Arena — it reproduces with stock gripper limits on both arms; diagnosis unfinished.
