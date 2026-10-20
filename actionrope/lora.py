"""Minimal LoRA for nn.Linear layers (used for LTX-2.5 and MiniMax-H3).

``LoRALinear`` computes base(x) + B(A x) * alpha / rank with A Kaiming-uniform and B zero-initialised, so the wrapped
model is unchanged at installation time. Checkpoints are written either as LoRA factors (``lora_state_dict``) or with
the factors folded into the base weights (``merged_state_dict``).
"""
from __future__ import annotations

import math
from typing import Callable, Iterable

import torch
import torch.nn as nn

_WRAPPERS = {"_fsdp_wrapped_module", "_checkpoint_wrapped_module"}


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int, alpha: float):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        d_out, d_in = base.weight.shape
        self.rank, self.scale = int(rank), float(alpha) / int(rank)
        kw = dict(dtype=base.weight.dtype, device=base.weight.device)
        self.lora_A = nn.Parameter(torch.empty(self.rank, d_in, **kw))
        self.lora_B = nn.Parameter(torch.zeros(d_out, self.rank, **kw))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x):
        return self.base(x) + nn.functional.linear(nn.functional.linear(x, self.lora_A), self.lora_B) * self.scale


def install_lora(root: nn.Module, target: Callable[[str], bool], rank: int, alpha: float | None = None,
                 keep_trainable: Iterable[str] = ()) -> int:
    """Wrap every nn.Linear whose module name satisfies ``target`` and freeze all other parameters, except those whose
    name ends with one of ``keep_trainable``. Returns the number of wrapped layers."""
    alpha = rank if alpha is None else alpha
    hits = [(n, m) for n, m in root.named_modules() if isinstance(m, nn.Linear) and target(n)]
    if not hits:
        raise ValueError("install_lora: no nn.Linear matched the target rule")
    for name, mod in hits:
        parent_name, _, child = name.rpartition(".")
        setattr(root.get_submodule(parent_name) if parent_name else root, child, LoRALinear(mod, rank, alpha))
    keep = tuple(keep_trainable)
    for n, p in root.named_parameters():
        p.requires_grad_(n.endswith(("lora_A", "lora_B")) or (bool(keep) and n.endswith(keep)))
    return len(hits)


def _clean(name: str) -> str:
    return ".".join(p for p in name.split(".") if p not in _WRAPPERS)


@torch.no_grad()
def merged_state_dict(root: nn.Module, state_dict: dict | None = None) -> dict:
    """State dict without LoRA wrappers: every ``<layer>.base.weight`` becomes ``<layer>.weight`` with B A * scale
    folded in (in fp32, then cast back). ``state_dict`` may be a gathered copy whose keys carry an extra prefix."""
    sd = dict(root.state_dict() if state_dict is None else state_dict)
    scales = {_clean(n): m.scale for n, m in root.named_modules() if isinstance(m, LoRALinear)}

    def scale_of(owner):
        for cut in range(owner.count(".") + 1):
            cand = owner.split(".", cut)[-1] if cut else owner
            if cand in scales:
                return scales[cand]
        return None

    out, merged = {}, 0
    for k, v in sd.items():
        if k.endswith((".lora_A", ".lora_B")):
            continue
        if ".base." in k:
            owner, tail = k.split(".base.", 1)
            s = scale_of(owner)
            if tail == "weight" and s is not None:
                A, B = sd[owner + ".lora_A"], sd[owner + ".lora_B"]
                v = (v.float() + (B.float() @ A.float()) * s).to(v.dtype)
                merged += 1
            k = owner + "." + tail
        out[k] = v
    if scales and merged == 0:
        raise RuntimeError("merged_state_dict: no LoRA layer was merged (key names do not match the module tree)")
    return out


def lora_state_dict(root: nn.Module, state_dict: dict | None = None, extra: Iterable[str] = ()) -> dict:
    """Only the LoRA factors (``<layer>.lora_A`` / ``<layer>.lora_B``) plus parameters whose name ends with one of
    ``extra``; keys are cleaned of FSDP/checkpoint wrappers."""
    sd = root.state_dict() if state_dict is None else state_dict
    extra = tuple(extra)
    return {_clean(k): v for k, v in sd.items() if k.endswith((".lora_A", ".lora_B")) or (extra and k.endswith(extra))}
