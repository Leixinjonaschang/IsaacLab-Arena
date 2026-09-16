#!/usr/bin/env bash
# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

# Run inside the IsaacLab-Arena container. The container uses host networking, so a GR00T
# server started on the host at 127.0.0.1:5557 is reachable without another model process.

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../.." && pwd)

SERVER_HOST=${AGIBOT_GR00T_SERVER_HOST:-127.0.0.1}
SERVER_PORT=${AGIBOT_GR00T_SERVER_PORT:-5557}
WARMUP_STEPS=${AGIBOT_GR00T_WARMUP_STEPS:-26}
ASSET_DIR=${ARENA_LOCAL_ASSET_DIR:-/tmp/isaaclab_arena_agibot_assets/arena_local}
OUTPUT_BASE_DIR=${AGIBOT_GR00T_OUTPUT_BASE_DIR:-${REPO_ROOT}/results/agibot_gr00t_n17_denoise_sweep_20ep}
HF_HOME=${HF_HOME:-/tmp/isaaclab_arena_agibot_assets/hf}
MPLCONFIGDIR=${MPLCONFIGDIR:-/tmp/matplotlib}

if [[ ! -f "${ASSET_DIR}/robodojo_table/robodojo_table.usda" ]]; then
    echo "Missing ${ASSET_DIR}/robodojo_table/robodojo_table.usda" >&2
    echo "Set ARENA_LOCAL_ASSET_DIR to the in-container arena_local asset directory." >&2
    exit 1
fi

if ! timeout 3 bash -c "</dev/tcp/${SERVER_HOST}/${SERVER_PORT}" 2>/dev/null; then
    echo "No GR00T server is listening at ${SERVER_HOST}:${SERVER_PORT}." >&2
    echo "Start isaaclab_arena_gr00t/scripts/serve_agibot_stack_bowls_n17.sh first." >&2
    exit 1
fi

mkdir -p "${OUTPUT_BASE_DIR}" "${HF_HOME}" "${MPLCONFIGDIR}"

cd "${REPO_ROOT}"
export ACCEPT_EULA=Y
export OMNI_KIT_ACCEPT_EULA=YES
export ARENA_LOCAL_ASSET_DIR="${ASSET_DIR}"
export HF_HOME
export MPLCONFIGDIR
export ISAACLAB_ARENA_GR00T_WARMUP_STEPS="${WARMUP_STEPS}"

exec ./submodules/IsaacLab/_isaac_sim/python.sh \
    isaaclab_arena/evaluation/experiment_runner.py \
    --experiment_config isaaclab_arena_environments/experiment_configs/agibot_stack_bowls_gr00t_n17_denoise_sweep_20ep.yaml \
    --headless \
    --record_camera_video \
    --output_base_dir "${OUTPUT_BASE_DIR}" \
    "shared.policy.remote_host=${SERVER_HOST}" \
    "shared.policy.remote_port=${SERVER_PORT}"
