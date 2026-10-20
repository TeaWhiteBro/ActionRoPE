"""Clip geometry shared by all backbones.

Conventions
-----------
* A clip has 81 frames at 16 fps and 832x480 pixels (5 s).
* Screen axes: x to the right, y downwards, in pixels of the 832x480 frame.
* ``frame_offset_px[i] = (dx, dy)`` is the on-screen displacement of the player (= of the camera, which follows the
  player) at frame i relative to frame 0, so ``frame_offset_px[0] == (0, 0)``. The static background moves by
  ``-frame_offset_px``.
* Every backbone maps the 81 per-frame offsets to its own latent frames (see ``frames_to_cells`` for the Wan-family
  causal VAE; LTX-2.5 and MiniMax-H3 have their own grouping in their model files).

Actions
-------
In a planar-camera game an action moves the camera by a known amount. We describe an action sequence as segments
``<direction>[*<speed>]:<n_frames>``, e.g. ``"right:40,left:40"`` (walk right for 2.5 s, then back). ``speed`` is a
multiplier of the nominal walking speed of the training data. Segments start at frame 1; frames not covered by any
segment are idle.
"""
from __future__ import annotations

import math

import numpy as np

FRAMES = 81
FPS = 16
WIDTH, HEIGHT = 832, 480

# Nominal walking speed of the player in the training data, in px/s at 832x480 and 16 fps (x, y). It is the median
# speed over straight-walk training clips; the evaluation commands are multiples of the horizontal value.
WALK_SPEED_PX_S = (69.24106392992304, 46.012630371093756)

_S = math.sqrt(0.5)
DIRECTIONS = {
    "idle": (0.0, 0.0),
    "up": (0.0, -1.0),
    "up-left": (-_S, -_S),
    "left": (-1.0, 0.0),
    "down-left": (-_S, _S),
    "down": (0.0, 1.0),
    "down-right": (_S, _S),
    "right": (1.0, 0.0),
    "up-right": (_S, -_S),
}
_ALIASES = {"u": "up", "d": "down", "l": "left", "r": "right", "ul": "up-left", "ur": "up-right",
            "dl": "down-left", "dr": "down-right", "stop": "idle", "none": "idle"}


def parse_actions(spec: str, n_frames: int = FRAMES) -> np.ndarray:
    """Action string -> per-frame velocity [n_frames, 2] in px/frame (frame 0 has no velocity).

    ``spec`` is a comma-separated list of ``<direction>[*<speed>]:<n_frames>`` segments, e.g. ``"right:40,left:40"``
    or ``"up-right*1.5:24,idle:8,left:48"``. Directions: idle, up, down, left, right, up-left, up-right, down-left,
    down-right (or u, d, l, r, ul, ur, dl, dr). The segments may cover at most ``n_frames - 1`` frames.
    """
    vel = np.zeros((n_frames, 2), np.float64)
    frame = 1
    for seg in filter(None, (s.strip() for s in spec.split(","))):
        try:
            head, count = seg.rsplit(":", 1)
            name, _, speed = head.partition("*")
            name = _ALIASES.get(name.strip().lower(), name.strip().lower())
            d = np.array(DIRECTIONS[name], np.float64)
            mult, count = float(speed) if speed else 1.0, int(count)
        except (KeyError, ValueError) as e:
            raise ValueError(f"cannot parse action segment {seg!r} (expected <direction>[*<speed>]:<n_frames>)") from e
        if count < 0 or frame + count > n_frames:
            raise ValueError(f"action segments cover more than {n_frames - 1} frames: {spec!r}")
        vel[frame:frame + count] = d * np.array(WALK_SPEED_PX_S) * mult / FPS
        frame += count
    return vel


def offsets_from_velocity(vel: np.ndarray) -> np.ndarray:
    """Per-frame velocity [F, 2] (px/frame) -> per-frame camera offset [F, 2] with offset[0] = 0."""
    vel = np.asarray(vel, np.float64).copy()
    vel[0] = 0.0
    return np.cumsum(vel, axis=0)


def offsets_from_actions(spec: str, n_frames: int = FRAMES) -> np.ndarray:
    """Action string -> per-frame camera offset ``frame_offset_px`` [n_frames, 2]."""
    return offsets_from_velocity(parse_actions(spec, n_frames))


def cell_frames(k: int) -> list[int]:
    """Frames covered by latent frame (cell) k of the Wan-family causal VAE: cell 0 = frame 0, cell k = frames
    4k-3 .. 4k."""
    return [0] if k == 0 else list(range(4 * k - 3, 4 * k + 1))


def frames_to_cells(frame_offset_px: np.ndarray) -> np.ndarray:
    """Per-frame offsets [81, 2] -> per-cell offsets [21, 2]: the mean over the frames of each cell."""
    off = np.asarray(frame_offset_px, np.float64)
    if off.shape != (FRAMES, 2):
        raise ValueError(f"frame_offset_px must have shape ({FRAMES}, 2), got {off.shape}")
    return np.stack([off[cell_frames(k)].mean(0) for k in range((FRAMES - 1) // 4 + 1)], 0)
