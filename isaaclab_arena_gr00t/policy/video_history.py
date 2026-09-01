# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Rolling per-camera frame history for GR00T policies that ask for past video frames.

GR00T describes how much video history a checkpoint wants through its video modality config's
``delta_indices``: ``[0]`` means "current frame only", while ``[-15, 0]`` means "the frame from 15
control steps ago and the current one". N1.7 base and post-trained checkpoints can publish
different values under the same embodiment tag, so the remote policy sizes this buffer from the
server's checkpoint contract. The server hard-asserts that the request has exactly that many frames.
"""

from __future__ import annotations

import numpy as np
from collections import deque


class VideoHistoryBuffer:
    """Keep the last few resized frames per camera so any ``delta_indices`` can be served.

    Frames are stored already resized, matching GR00T's own DROID client, so the buffer costs the
    policy-sized image rather than the full render.

    Before an environment has run long enough to fill the buffer, a negative delta is clamped to the
    oldest frame that environment has, which reproduces the reference client's behaviour of reading
    a not-yet-full deque.
    """

    def __init__(self, delta_indices: list[int], num_envs: int):
        """
        Args:
            delta_indices: Video ``delta_indices`` from the GR00T modality config. Must be
                non-positive and end at 0, i.e. offsets back from the current frame.
            num_envs: Number of parallel environments sharing this buffer.
        """
        assert delta_indices, "delta_indices must not be empty"
        assert all(delta <= 0 for delta in delta_indices), f"video delta_indices must be <= 0, got {delta_indices}"
        assert delta_indices[-1] == 0, f"video delta_indices must end at the current frame, got {delta_indices}"
        self.delta_indices = list(delta_indices)
        self.num_envs = num_envs
        self.history_length = -min(self.delta_indices) + 1
        """How many frames back the buffer has to remember to serve the oldest delta."""
        self._frames: deque[list[np.ndarray]] = deque(maxlen=self.history_length)
        self._steps_since_reset = np.zeros(num_envs, dtype=np.int64)

    @property
    def horizon(self) -> int:
        """Number of frames the policy server expects per camera."""
        return len(self.delta_indices)

    def push(self, rgb_list_np: list[np.ndarray]) -> None:
        """Record one control step's resized frames.

        Args:
            rgb_list_np: One (num_envs, H, W, C) array per camera.
        """
        assert rgb_list_np, "at least one camera is required"
        for rgb_np in rgb_list_np:
            assert rgb_np.shape[0] == self.num_envs, f"expected {self.num_envs} envs, got {rgb_np.shape[0]}"
        self._frames.append([np.asarray(rgb_np) for rgb_np in rgb_list_np])
        self._steps_since_reset += 1

    def stack(self) -> list[np.ndarray]:
        """Return the frames the policy asked for, one (num_envs, horizon, H, W, C) array per camera.

        Raises:
            RuntimeError: If no frame has been pushed since the last reset.
        """
        if not self._frames:
            raise RuntimeError("VideoHistoryBuffer.stack() called before any frame was pushed")

        num_cameras = len(self._frames[-1])
        stacked: list[np.ndarray] = []
        for camera_index in range(num_cameras):
            # num_envs and horizon are both small (typically 1 and 2), so an explicit loop stays
            # cheaper to read than the equivalent gather.
            per_delta = []
            for delta in self.delta_indices:
                frame = np.empty_like(self._frames[-1][camera_index])
                for env_index in range(self.num_envs):
                    frame[env_index] = self._frames[self._buffer_index(delta, env_index)][camera_index][env_index]
                per_delta.append(frame)
            stacked.append(np.stack(per_delta, axis=1))
        return stacked

    def _buffer_index(self, delta: int, env_index: int) -> int:
        """Return the buffer slot holding ``delta`` steps ago for one environment.

        Clamped both by how many frames the buffer holds and by how many steps this environment has
        run since its last reset, so a fresh environment reads its own oldest frame.
        """
        available = min(len(self._frames), int(self._steps_since_reset[env_index]))
        return -min(-delta + 1, available)

    def reset(self, env_ids: np.ndarray | slice | None = None) -> None:
        """Forget the history of the given environments so they restart from their next frame."""
        if env_ids is None or isinstance(env_ids, slice):
            self._steps_since_reset[env_ids if env_ids is not None else slice(None)] = 0
        else:
            self._steps_since_reset[np.asarray(env_ids)] = 0
