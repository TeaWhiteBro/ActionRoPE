r"""Pre-compute the training latents of one backbone: <out_dir>/<clip>.pt for every clip of the annotation file.

    python scripts/encode_latents.py --model wan --model_dir models/Wan2.2-TI2V-5B \
        --annotations data/annotations.jsonl --out_dir data/latents/wan [--split train,val] [--shard 0/8]

Run one process per GPU with ``--shard i/N`` to split the work; existing files are skipped.
"""
from __future__ import annotations

import argparse
import importlib
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

from actionrope.data import load_annotations  # noqa: E402
from actionrope.geometry import FRAMES  # noqa: E402
from actionrope.media import read_frames  # noqa: E402
from actionrope.models import MODELS  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", choices=MODELS, required=True)
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--annotations", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--split", default="train,val")
    ap.add_argument("--shard", default="0/1", help="i/N: process every N-th clip starting at i")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    i, n = (int(x) for x in args.shard.split("/"))
    splits = set(args.split.split(","))
    clips = [c for c in load_annotations(args.annotations) if c.split in splits][i::n]
    os.makedirs(args.out_dir, exist_ok=True)
    todo = [c for c in clips if not os.path.exists(os.path.join(args.out_dir, c.clip + ".pt"))]
    print(f"[encode] {len(clips)} clips in shard {args.shard}, {len(todo)} to do", flush=True)
    encoder = importlib.import_module(f"actionrope.models.{args.model}").LatentEncoder(args.model_dir, args.device)
    t0 = time.time()
    for k, clip in enumerate(todo):
        frames = read_frames(clip.video, FRAMES)
        if len(frames) < FRAMES:
            raise ValueError(f"{clip.video}: {len(frames)} frames, expected {FRAMES}")
        out = encoder(frames, clip)
        path = os.path.join(args.out_dir, clip.clip + ".pt")
        torch.save(out, path + ".tmp")
        os.replace(path + ".tmp", path)
        if k % 50 == 0:
            print(f"[encode] {k + 1}/{len(todo)} {clip.clip} {(time.time() - t0) / (k + 1):.1f}s/clip", flush=True)


if __name__ == "__main__":
    main()
