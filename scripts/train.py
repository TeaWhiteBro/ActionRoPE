r"""Fine-tune a backbone with ActionRoPE + CNS.

    accelerate launch --config_file configs/accelerate/zero2_ga2.yaml scripts/train.py --config configs/train/wan.yaml \
        --set annotations=data/annotations.jsonl latent_dir=data/latents/wan output=runs/wan

The YAML file holds the recipe of the released checkpoint; ``--set key=value`` overrides single entries (values are
parsed as YAML). Use the accelerate config whose gradient accumulation matches ``grad_accum`` in the training config.
"""
from __future__ import annotations

import argparse
import importlib
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml  # noqa: E402

from actionrope.trainer import train  # noqa: E402


def load_config(path: str, overrides: list[str]) -> dict:
    with open(path, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    for item in overrides:
        key, sep, value = item.partition("=")
        if not sep:
            raise SystemExit(f"--set expects key=value, got {item!r}")
        cfg[key.strip()] = yaml.safe_load(value)
    missing = [k for k in ("model", "model_dir", "annotations", "latent_dir", "output") if not cfg.get(k)]
    if missing:
        raise SystemExit(f"missing config entries: {missing} (set them in the YAML file or with --set)")
    return cfg


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    args = ap.parse_args()
    cfg = load_config(args.config, args.set)
    mod = importlib.import_module(f"actionrope.models.{cfg['model']}")
    train(cfg, mod.TrainTask(cfg))


if __name__ == "__main__":
    main()
