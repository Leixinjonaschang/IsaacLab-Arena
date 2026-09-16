# Copyright (c) 2026, The Isaac Lab Arena Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for delayed policy-state history."""

import numpy as np

from isaaclab_arena_gr00t.policy.state_history import StateHistoryBuffer


def _state(first: float, second: float) -> dict[str, np.ndarray]:
    return {"pose": np.asarray([[first, first + 0.5], [second, second + 0.5]], dtype=np.float32)}


def test_state_history_returns_configured_control_step_delay():
    history = StateHistoryBuffer(delay_steps=1, num_envs=2)
    history.push(_state(0.0, 10.0))
    np.testing.assert_array_equal(history.delayed()["pose"], _state(0.0, 10.0)["pose"])

    history.push(_state(1.0, 11.0))
    np.testing.assert_array_equal(history.delayed()["pose"], _state(0.0, 10.0)["pose"])

    history.push(_state(2.0, 12.0))
    np.testing.assert_array_equal(history.delayed()["pose"], _state(1.0, 11.0)["pose"])


def test_state_history_clamps_only_reset_environments_to_their_newest_state():
    history = StateHistoryBuffer(delay_steps=1, num_envs=2)
    history.push(_state(0.0, 10.0))
    history.push(_state(1.0, 11.0))
    history.reset(np.asarray([0]))
    history.push(_state(100.0, 12.0))

    expected = np.asarray([[100.0, 100.5], [11.0, 11.5]], dtype=np.float32)
    np.testing.assert_array_equal(history.delayed()["pose"], expected)
