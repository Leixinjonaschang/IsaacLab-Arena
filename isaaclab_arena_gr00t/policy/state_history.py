# Copyright (c) 2026, The Isaac Lab Arena Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Per-environment state history for checkpoints trained with delayed proprioception."""

from __future__ import annotations

from collections import deque

import numpy as np


class StateHistoryBuffer:
    """Store policy observation groups and return a fixed number of control steps in the past.

    Until an environment has accumulated enough post-reset entries, its oldest available entry is
    repeated. This is the same reset behavior used by :class:`VideoHistoryBuffer`.
    """

    def __init__(self, delay_steps: int, num_envs: int):
        assert delay_steps >= 0, f"delay_steps must be non-negative, got {delay_steps}"
        assert num_envs > 0, f"num_envs must be positive, got {num_envs}"
        self.delay_steps = delay_steps
        self.num_envs = num_envs
        self._states: deque[dict[str, np.ndarray]] = deque(maxlen=delay_steps + 1)
        self._steps_since_reset = np.zeros(num_envs, dtype=np.int64)

    def push(self, state: dict[str, np.ndarray]) -> None:
        """Record one control step of unbatched policy-group terms."""
        assert state, "state must contain at least one term"
        values = {key: np.array(value, copy=True) for key, value in state.items()}
        for key, value in values.items():
            assert value.shape[0] == self.num_envs, (
                f"state term '{key}' expected {self.num_envs} envs, got shape {value.shape}"
            )
        if self._states:
            assert values.keys() == self._states[-1].keys(), "state term keys changed between control steps"
            for key, value in values.items():
                assert value.shape == self._states[-1][key].shape, (
                    f"state term '{key}' changed shape from {self._states[-1][key].shape} to {value.shape}"
                )
        self._states.append(values)
        self._steps_since_reset += 1

    def delayed(self) -> dict[str, np.ndarray]:
        """Return one state dict delayed by :attr:`delay_steps` independently per environment."""
        if not self._states:
            raise RuntimeError("StateHistoryBuffer.delayed() called before any state was pushed")
        delayed = {key: np.empty_like(value) for key, value in self._states[-1].items()}
        for env_index in range(self.num_envs):
            available = min(len(self._states), int(self._steps_since_reset[env_index]))
            buffer_index = -min(self.delay_steps + 1, available)
            for key in delayed:
                delayed[key][env_index] = self._states[buffer_index][key][env_index]
        return delayed

    def reset(self, env_ids: np.ndarray | slice | None = None) -> None:
        """Forget history age for selected environments without copying the shared deque."""
        if env_ids is None or isinstance(env_ids, slice):
            self._steps_since_reset[env_ids if env_ids is not None else slice(None)] = 0
        else:
            self._steps_since_reset[np.asarray(env_ids)] = 0
