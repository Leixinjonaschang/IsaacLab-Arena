#!/usr/bin/env bash
# Re-apply the franka_panda_hand_on_stand.usd asset fix after /tmp is wiped
# (e.g. after a reboot). Run from the repo root.
#
# Background: the upstream asset on the omniverse-content-staging S3 bucket has
# 23 dangling references to Isaac/IsaacLab/Robots/FrankaEmika/Props/*.usd (404
# on both staging and production buckets). The correct files live under
# .../FrankaEmika/Legacy/Props/ with identical names. PhysX/Kit tolerates the
# dangling refs as muted USD warnings; Newton's parse_usd raises a fatal
# RuntimeError, breaking --presets newton (and therefore --viz viser/newton/rerun).
#
# This script places the pre-fixed local copy into the /tmp asset mirror that
# isaaclab.utils.assets.retrieve_file_path uses. Because the file already
# exists there, the runtime skips re-downloading the broken upstream copy.
set -euo pipefail

TARGET_DIR="/tmp/Assets/Isaac/6.0/Isaac/IsaacLab/Arena/assets/robot_library"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

mkdir -p "$TARGET_DIR"
cp "$SCRIPT_DIR/franka_panda_hand_on_stand.usd" "$TARGET_DIR/franka_panda_hand_on_stand.usd"
echo "Fixed asset installed at $TARGET_DIR/franka_panda_hand_on_stand.usd"
