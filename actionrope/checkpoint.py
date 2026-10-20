"""safetensors checkpoints with a small, uniform metadata header.

Every ActionRoPE checkpoint stores (as strings):

    model            "wan2.2-ti2v-5b" | "skyreels-v2-df-1.3b" | "ltx-2.5-22b" | "minimax-h3-33b"
    format           "full" (complete DiT state dict) or "lora" (LoRA factors + new modules only)
    mask_channel     "true" / "false"   known/new input channel
    cns              "true" / "false"   Coordinate-Conditioned Noise Scheduling
    cns_shift        base shift S of the sampling schedule
    cns_shift_known  shift S_known of known tokens
    cns_alpha        exponent alpha (1 in the paper)
    step             training step
    lora_rank, lora_alpha, lora_targets   (format = "lora" only)
"""
from __future__ import annotations

import json
import os

import torch


def save_checkpoint(path: str, state_dict: dict[str, torch.Tensor], metadata: dict) -> None:
    from safetensors.torch import save_file
    meta = {k: (json.dumps(v) if isinstance(v, (list, dict)) else str(v).lower() if isinstance(v, bool) else str(v))
            for k, v in metadata.items()}
    save_file({k: v.detach().contiguous().cpu() for k, v in state_dict.items()}, path, metadata=meta)
    os.chmod(path, 0o644)      # safetensors follows the umask, which may make the file private


def read_metadata(path: str) -> dict[str, str]:
    from safetensors import safe_open
    with safe_open(path, "pt") as f:
        return dict(f.metadata() or {})


def meta_bool(meta: dict, key: str, default: bool = False) -> bool:
    v = meta.get(key)
    return default if v is None else str(v).strip().lower() in ("1", "true", "yes")


def meta_float(meta: dict, key: str, default: float | None = None) -> float | None:
    v = meta.get(key)
    return default if v is None or v == "" else float(v)
