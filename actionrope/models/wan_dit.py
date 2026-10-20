"""ActionRoPE forward pass for the Wan-family video DiT (``diffsynth.models.wan_video_dit.WanModel``).

Used by Wan2.2-TI2V-5B (``wan.py``) and SkyReels-V2-DF-1.3B (``skyreels.py``), which share the DiT architecture.

ActionRoPE (Sec. 3.1): token (i, j) of latent frame k gets the rotary position (k, i + dy_k / s, j + dx_k / s) instead
of its screen position (k, i, j), where (dx_k, dy_k) is the camera offset of latent frame k in pixels and s the number
of pixels per token (32 for Wan2.2, 16 for SkyReels-V2). Tokens that see the same world location therefore share the
same positional address across time. The rotary frequencies are the DiT's own; only the positions change, so no
parameters are added on the control path.

``arope_forward`` re-implements the DiT forward of DiffSynth's ``model_fn_wan_video`` for image-to-video with a
clean first frame (latent frame 0 at timestep 0) and adds:
  * world-coordinate rotary positions (``offset_px``),
  * the optional known/new input channel (``mask_input``, a zero-initialised Conv3d added to the patch embedding),
  * per-token timesteps (``timestep_map``) for CNS,
  * per-latent-frame text (context [b, f, L, C]: every latent frame cross-attends only to its own prompt).
With ``offset_px=None``, no mask, no timestep map and one prompt it equals the original forward.
"""
from __future__ import annotations

import functools

import torch
import torch.nn as nn
from einops import rearrange

from diffsynth.core.gradient import gradient_checkpoint_forward
from diffsynth.models.wan_video_dit import WanModel, modulate, sinusoidal_embedding_1d

ROPE_THETA = 10000.0
MASK_EMBEDDING = "arope_mask_embedding"


# ----------------------------------------------------------------------------- world-coordinate RoPE
def _inv_freq(dim: int, device=None) -> torch.Tensor:
    """Inverse frequencies of the DiT's 1-D RoPE (identical to DiffSynth's ``precompute_freqs_cis(dim)``)."""
    return 1.0 / (ROPE_THETA ** (torch.arange(0, dim, 2, device=device)[: (dim // 2)].double() / dim))


def _axis_freqs(pos: torch.Tensor, dim: int) -> torch.Tensor:
    """1-D RoPE at arbitrary (fractional) positions, complex128 [..., dim // 2]; equals the table at integers."""
    angle = pos.double().unsqueeze(-1) * _inv_freq(dim, pos.device)
    return torch.polar(torch.ones_like(angle), angle)


def build_world_freqs(dit: WanModel, f: int, h: int, w: int, offset_tok: torch.Tensor | None, device=None) -> torch.Tensor:
    """Rotary frequencies of all f*h*w tokens at positions (k, i + dy_k, j + dx_k).

    offset_tok: [b, f, 2] = (dx, dy) in token units, or None for the standard screen grid.
    Returns [f*h*w, 1, d] (standard) or [b, f*h*w, 1, d] (world coordinates), complex like ``dit.freqs``.
    """
    f_table, h_table, w_table = dit.freqs
    if device is None:
        device = offset_tok.device if offset_tok is not None else f_table.device
    d_h, d_w = h_table.shape[1] * 2, w_table.shape[1] * 2
    if offset_tok is None:
        return torch.cat([
            f_table[:f].view(f, 1, 1, -1).expand(f, h, w, -1),
            h_table[:h].view(1, h, 1, -1).expand(f, h, w, -1),
            w_table[:w].view(1, 1, w, -1).expand(f, h, w, -1),
        ], dim=-1).reshape(f * h * w, 1, -1).to(device)
    offset_tok = offset_tok.to(device=device, dtype=torch.float64)
    b = offset_tok.shape[0]
    assert offset_tok.shape == (b, f, 2), f"offset_tok must be [b, {f}, 2], got {tuple(offset_tok.shape)}"
    dx, dy = offset_tok[..., 0], offset_tok[..., 1]
    pos_h = torch.arange(h, device=device, dtype=torch.float64).view(1, 1, h) + dy.view(b, f, 1)
    pos_w = torch.arange(w, device=device, dtype=torch.float64).view(1, 1, w) + dx.view(b, f, 1)
    fh, fw = _axis_freqs(pos_h, d_h), _axis_freqs(pos_w, d_w)
    freqs = torch.cat([
        f_table[:f].to(device).view(1, f, 1, 1, -1).expand(b, f, h, w, -1),
        fh.view(b, f, h, 1, -1).expand(b, f, h, w, -1),
        fw.view(b, f, 1, w, -1).expand(b, f, h, w, -1),
    ], dim=-1)
    return freqs.reshape(b, f * h * w, 1, -1).to(f_table.dtype)


# ----------------------------------------------------------------------------- known/new input channel
def install_mask_channel(dit: WanModel) -> WanModel:
    """Add the zero-initialised known/new input channel ``dit.arope_mask_embedding`` = Conv3d(1, dim, patch, patch).
    The model output is unchanged at installation time."""
    if not hasattr(dit, MASK_EMBEDDING):
        ref = dit.patch_embedding
        conv = nn.Conv3d(1, dit.dim, kernel_size=dit.patch_size, stride=dit.patch_size)
        nn.init.zeros_(conv.weight)
        nn.init.zeros_(conv.bias)
        dit.add_module(MASK_EMBEDDING, conv.to(device=ref.weight.device, dtype=ref.weight.dtype))
    return dit


# ----------------------------------------------------------------------------- forward
def _per_token_timestep(latents, timestep, n_tok: int, first_frame_cond: bool, timestep_map=None) -> torch.Tensor:
    """[b, f * n_tok] timesteps (0..1000); latent frame 0 is the clean condition frame (t = 0)."""
    b, f = latents.shape[0], latents.shape[2]
    if timestep_map is not None:
        t_tok = timestep_map.to(device=latents.device, dtype=latents.dtype).reshape(b, f, n_tok).clone()
    else:
        timestep = torch.as_tensor(timestep, device=latents.device).to(latents.dtype).reshape(-1)
        timestep = timestep.expand(b) if timestep.numel() == 1 else timestep
        t_tok = timestep.view(b, 1, 1).expand(b, f, n_tok).clone()
    if first_frame_cond:
        t_tok[:, 0] = 0
    return t_tok.reshape(b, f * n_tok)


def _embed_context(dit: WanModel, context: torch.Tensor, context_ids=None) -> torch.Tensor:
    """[b, L, C] -> [b, L, dim]; per-frame context [b, f, L, C] -> [(b f), L, dim]. Frames with equal
    ``context_ids`` [b, f] share a prompt and are embedded once."""
    if context.dim() == 3:
        return dit.text_embedding(context)
    b, f, L, c = context.shape
    rows = context.reshape(b * f, L, c)
    if context_ids is None:
        return dit.text_embedding(rows)
    ids = torch.as_tensor(context_ids).reshape(-1).tolist()
    first = {}
    for i, k in enumerate(ids):
        first.setdefault(k, i)
    uniq = list(first)
    emb = dit.text_embedding(rows[torch.tensor([first[k] for k in uniq], device=rows.device)])
    return emb[torch.tensor([uniq.index(k) for k in ids], device=rows.device)]


def _block_forward(block, x, context, t_mod, freqs, *, f: int):
    """``DiTBlock.forward`` with per-frame text cross-attention when context is [(b f), L, dim]."""
    has_seq = len(t_mod.shape) == 4
    chunk_dim = 2 if has_seq else 1
    shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
        block.modulation.to(dtype=t_mod.dtype, device=t_mod.device) + t_mod).chunk(6, dim=chunk_dim)
    if has_seq:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            shift_msa.squeeze(2), scale_msa.squeeze(2), gate_msa.squeeze(2),
            shift_mlp.squeeze(2), scale_mlp.squeeze(2), gate_mlp.squeeze(2))
    x = block.gate(x, gate_msa, block.self_attn(modulate(block.norm1(x), shift_msa, scale_msa), freqs))
    q_in = block.norm3(x)
    if context.shape[0] == x.shape[0]:
        x = x + block.cross_attn(q_in, context)
    else:
        q_in = rearrange(q_in, "b (f n) c -> (b f) n c", f=f)
        x = x + rearrange(block.cross_attn(q_in, context), "(b f) n c -> b (f n) c", f=f)
    return block.gate(x, gate_mlp, block.ffn(modulate(block.norm2(x), shift_mlp, scale_mlp)))


def arope_forward(dit: WanModel, latents: torch.Tensor, timestep, context: torch.Tensor, *,
                  offset_px: torch.Tensor | None = None, px_per_token: float = 32, mask_input: torch.Tensor | None = None,
                  timestep_map: torch.Tensor | None = None, context_ids=None, first_frame_cond: bool = True,
                  t_mod_fn=None, use_gradient_checkpointing: bool = False) -> torch.Tensor:
    """Velocity prediction for latents [b, c, f, H, W].

    offset_px     [b, f, 2] camera offset of every latent frame in pixels (None: standard RoPE)
    px_per_token  pixels per DiT token (VAE stride x patch size)
    mask_input    [b, 1, f, H, W] known = 1 / new = 0 (requires ``install_mask_channel``)
    timestep_map  [b, f, H / 2, W / 2] per-token timesteps for CNS (None: ``timestep`` for every token)
    context       [b, L, C] (one prompt) or [b, f, L, C] (one prompt per latent frame, see ``context_ids``)
    t_mod_fn      optional callable applied to the timestep modulation [b, n, 6, dim]
    """
    b, _, f, lat_h, lat_w = latents.shape
    _, p_h, p_w = dit.patch_size
    n_tok = (lat_h // p_h) * (lat_w // p_w)

    t_tok = _per_token_timestep(latents, timestep, n_tok, first_frame_cond, timestep_map)
    t = dit.time_embedding(sinusoidal_embedding_1d(dit.freq_dim, t_tok.reshape(-1)).view(b, t_tok.shape[1], -1))
    t_mod = dit.time_projection(t).unflatten(2, (6, dit.dim))
    if t_mod_fn is not None:
        t_mod = t_mod_fn(t_mod)
    context = _embed_context(dit, context, context_ids)

    x = dit.patch_embedding(latents)
    if mask_input is not None:
        if not hasattr(dit, MASK_EMBEDDING):
            raise ValueError("mask_input given but the known/new input channel is not installed")
        x = x + getattr(dit, MASK_EMBEDDING)(mask_input.to(device=x.device, dtype=x.dtype))
    f, h, w = x.shape[2:]
    x = rearrange(x, "b c f h w -> b (f h w) c").contiguous()

    offset_tok = None if offset_px is None else offset_px.to(torch.float64) / px_per_token
    freqs = build_world_freqs(dit, f, h, w, offset_tok, device=x.device)
    for block in dit.blocks:
        x = gradient_checkpoint_forward(functools.partial(_block_forward, block, f=f),
                                        use_gradient_checkpointing, False, x, context, t_mod, freqs)
    x = dit.head(x, t)
    return dit.unpatchify(x, (f, h, w))
