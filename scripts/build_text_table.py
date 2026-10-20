r"""Encode every distinct scene prompt of the dataset (plus NULL_PROMPT) once: {prompt: embedding}.

    python scripts/build_text_table.py --model wan --model_dir models/Wan2.2-TI2V-5B \
        --annotations data/annotations.jsonl --out data/text_table_umt5.pt

Wan2.2 and SkyReels-V2 share the UMT5 table; LTX-2.5 needs its own (``--model ltx``). MiniMax-H3 encodes its text
together with the first frame, so its embeddings are stored in the latent files instead.
"""
from __future__ import annotations

import argparse
import importlib
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

from actionrope.data import load_annotations  # noqa: E402
from actionrope.prompts import NULL_PROMPT  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", choices=("wan", "skyreels", "ltx"), required=True)
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--annotations", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    prompts = sorted({p for c in load_annotations(args.annotations) for p in c.prompts} | {NULL_PROMPT})
    print(f"[text] {len(prompts)} distinct prompts", flush=True)
    mod = importlib.import_module(f"actionrope.models.{args.model}")
    table = mod.build_text_table(prompts, args.model_dir, args.device)
    torch.save({"table": table}, args.out)
    print(f"[text] wrote {args.out}")


if __name__ == "__main__":
    main()
