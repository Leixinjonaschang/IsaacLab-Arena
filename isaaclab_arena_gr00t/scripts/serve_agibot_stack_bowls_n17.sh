#!/usr/bin/env bash
# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../.." && pwd)

MODEL_PATH=${AGIBOT_GR00T_MODEL_PATH:-/home/ubuntu/projects/isaaclab_arena_proj/checkpoints/gr00t_n17_sfted_stack_bowls_with_ee_pose/checkpoint-10000}
SERVER_HOST=${AGIBOT_GR00T_SERVER_HOST:-127.0.0.1}
SERVER_PORT=${AGIBOT_GR00T_SERVER_PORT:-5557}
POLICY_DEVICE=${AGIBOT_GR00T_POLICY_DEVICE:-cuda:0}

if [[ ! -d "${MODEL_PATH}" ]]; then
    echo "Checkpoint directory does not exist: ${MODEL_PATH}" >&2
    exit 1
fi

cd "${REPO_ROOT}/submodules/Isaac-GR00T"
if [[ -x .venv/bin/python ]]; then
    PYTHON=(.venv/bin/python)
else
    PYTHON=(uv run python)
fi

exec "${PYTHON[@]}" gr00t/eval/run_gr00t_server.py \
    --model-path "${MODEL_PATH}" \
    --embodiment-tag new_embodiment \
    --device "${POLICY_DEVICE}" \
    --host "${SERVER_HOST}" \
    --port "${SERVER_PORT}"
