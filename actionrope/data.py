"""Dataset index shared by all backbones.

The dataset is described by one JSON-lines annotation file, one row per 81-frame clip:

    {"clip": "000123",                      # clip id = file stem of the video and of every latent file
     "video": "clips/000123.mp4",           # path relative to the annotation file
     "split": "train",                      # "train" or "val"
     "kind": "plain",                       # "plain", or "burn" / "hardcut" for scene-transition clips
     "valid": true,                         # false if the camera track of the clip is unreliable
     "rt_ratio": 0.98,                      # real-time ratio of the recording (plain clips)
     "sources": ["000045", "000981"],       # transition clips: the two clips they were composed from
     "frame_offset_px": [[0.0, 0.0], ...],  # 81 x (dx, dy) camera offsets in px (see actionrope.geometry)
     "prompts": ["In a 2.5D top down view, the player is in ...", ...]}   # 21 per-cell scene prompts (no actions)

The 21 prompts follow the Wan-family latent frames (``geometry.cell_frames``); they only differ inside transition
clips. Latents are pre-computed per backbone with ``scripts/encode_latents.py`` into ``<latent_dir>/<clip>.pt``.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass

import numpy as np
import torch

TRANSITION_KINDS = ("burn", "hardcut")


@dataclass
class Clip:
    clip: str
    split: str
    kind: str
    valid: bool
    rt_ratio: float | None
    sources: tuple[str, ...]
    frame_offset_px: np.ndarray       # float64 [81, 2]
    prompts: list[str]
    video: str | None = None


def load_annotations(path: str) -> list[Clip]:
    root = os.path.dirname(os.path.abspath(path))
    clips = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            r = json.loads(line)
            clips.append(Clip(
                clip=r["clip"], split=r["split"], kind=r.get("kind", "plain"), valid=bool(r.get("valid", True)),
                rt_ratio=r.get("rt_ratio"), sources=tuple(r.get("sources") or ()),
                frame_offset_px=np.asarray(r["frame_offset_px"], np.float64), prompts=list(r["prompts"]),
                video=os.path.join(root, r["video"]) if r.get("video") else None))
    return clips


def training_clips(clips: list[Clip], min_rt_ratio: float = 0.8, exclude_val_sources: bool = True,
                   verbose: bool = True) -> list[Clip]:
    """Training split after the filters of the paper:

    * drop clips whose camera track is invalid;
    * drop transition clips composed from a validation clip (their frames would leak into training);
    * drop plain clips recorded below ``min_rt_ratio`` of real time (the game stuttered, so motion per frame drops).
    """
    val = {c.clip for c in clips if c.split == "val"}
    out, n_invalid, n_leak, n_slow = [], 0, 0, 0
    for c in clips:
        if c.split != "train":
            continue
        if not c.valid:
            n_invalid += 1
        elif c.kind in TRANSITION_KINDS:
            if exclude_val_sources and any(s in val for s in c.sources):
                n_leak += 1
            else:
                out.append(c)
        elif min_rt_ratio > 0 and float(1.0 if c.rt_ratio is None else c.rt_ratio) < min_rt_ratio:
            n_slow += 1
        else:
            out.append(c)
    if verbose:
        n_train = sum(c.split == "train" for c in clips)
        print(f"[data] train clips: {n_train} - invalid {n_invalid} - leaking transitions {n_leak} - "
              f"rt_ratio < {min_rt_ratio} {n_slow} = {len(out)}", flush=True)
    return out


def validation_clips(clips: list[Clip], min_rt_ratio: float = 0.9, n: int | None = 16, seed: int = 0) -> list[Clip]:
    """Validation clips used to monitor the training loss (``n`` random clips with rt_ratio >= ``min_rt_ratio``)."""
    import random
    pool = [c for c in clips if c.split == "val" and c.valid and c.kind not in TRANSITION_KINDS
            and float(1.0 if c.rt_ratio is None else c.rt_ratio) >= min_rt_ratio]
    pool.sort(key=lambda c: c.clip)
    if n is not None and n < len(pool):
        random.Random(seed).shuffle(pool)
        pool = sorted(pool[:n], key=lambda c: c.clip)
    return pool


def load_text_table(path: str) -> dict[str, torch.Tensor]:
    """{prompt: embedding} written by ``scripts/build_text_table.py``."""
    d = torch.load(path, map_location="cpu", weights_only=False)
    return d["table"] if isinstance(d, dict) and "table" in d and isinstance(d["table"], dict) else d
