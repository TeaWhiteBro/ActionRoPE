r"""Generate a clip from a first frame and an action sequence.

    python scripts/sample.py --model wan --ckpt checkpoints/wan2.2-ti2v-5b/actionrope_cns.safetensors \
        --model_dir models/Wan2.2-TI2V-5B --first_frame assets/first_frame.png --actions "right:40,left:40" --out clip.mp4

``--actions`` lists ``<direction>[*<speed>]:<n_frames>`` segments over the 80 frames after the first one (16 fps), e.g.
``"right:40,left:40"`` walks right for 2.5 s and back. ``--offsets`` takes per-frame camera offsets instead (a JSON list
of 81 [dx, dy] pairs in pixels). The sampling defaults of each backbone are the settings used in the paper.
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from actionrope.geometry import offsets_from_actions  # noqa: E402
from actionrope.media import load_first_frame, write_video  # noqa: E402
from actionrope.models import MODELS  # noqa: E402
from actionrope.prompts import NULL_PROMPT  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", choices=MODELS, required=True)
    ap.add_argument("--ckpt", required=True, help="ActionRoPE checkpoint (.safetensors)")
    ap.add_argument("--model_dir", required=True, help="directory with the base model files (see README)")
    ap.add_argument("--first_frame", required=True, help="image or video; resized (centre crop) to 832x480")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--actions", default="right:40,left:40")
    g.add_argument("--offsets", help="JSON file with 81 per-frame [dx, dy] camera offsets in pixels")
    ap.add_argument("--prompt", default=NULL_PROMPT, help="scene prompt (the action never goes into the text)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=None, help="default: the paper setting of the backbone")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                    help="extra backbone options, e.g. text_table=... (Generator) or cfg=... (sampling); values are YAML")
    ap.add_argument("--out", default="clip.mp4")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    mod = importlib.import_module(f"actionrope.models.{args.model}")
    if args.offsets:
        with open(args.offsets, encoding="utf-8") as fh:
            offsets = np.asarray(json.load(fh), np.float64)
    else:
        offsets = offsets_from_actions(args.actions)
    import inspect
    import yaml
    extra = {}
    for item in args.set:
        key, sep, value = item.partition("=")
        if not sep:
            raise SystemExit(f"--set expects key=value, got {item!r}")
        extra[key.strip()] = yaml.safe_load(value)
    init_keys = set(inspect.signature(mod.Generator.__init__).parameters)
    init_kw = {k: v for k, v in extra.items() if k in init_keys}
    kw = {k: v for k, v in extra.items() if k not in init_keys}
    if args.steps is not None:
        kw["steps"] = args.steps
    gen = mod.Generator(ckpt=args.ckpt, model_dir=args.model_dir, device=args.device, **init_kw)
    frames, fps = gen(load_first_frame(args.first_frame), offsets, prompt=args.prompt, seed=args.seed, **kw)
    write_video(args.out, frames, fps=fps, **mod.WRITE_KW)
    print(f"wrote {args.out}: {len(frames)} frames at {fps:g} fps")


if __name__ == "__main__":
    main()
