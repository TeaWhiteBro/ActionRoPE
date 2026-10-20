"""ActionRoPE + CNS for the MiniMax-H3 audio-video DiT (``diffsynth.models.minimax_h3_dit.MiniMaxH3DiT``, FL2VA task).

H3 packs one sequence per sample, ``[text | first-frame condition | audio | video | padding]``, and gives every token a
float position (t, h, w) that its RoPE multiplies by the inverse frequencies directly. ActionRoPE therefore only has to
edit these positions; the DiT itself is used unmodified.

* Space. The spatial grid of the packed sequence is spaced by 64 / sqrt(lat_h * lat_w) per token, i.e. 1.6204 per
  32-px token at 832x480. Token (i, j) of video latent frame k moves to (h_i + dy_k * u, w_j + dx_k * u) with
  u = spacing / 32 px, which is the world coordinate (i + dy_k / 32, j + dx_k / 32) of Sec. 3.1 on H3's grid.
* Time. The video VAE groups pixel frames as (1, 4, 4, 4, 4), repeating (17n + 5 frames -> 5n + 2 latent frames), and
  the temporal position of latent frame k is (5/3) x its first pixel frame. The camera offset of latent frame k is the
  per-frame offset interpolated linearly at the centre of the frames it covers.
* First frame. FL2VA encodes the first frame as 390 separate condition tokens at the positions of video latent frame
  0 with an almost clean timestep. They are the origin of the world coordinates (offset 0, kappa = 1) and are dropped
  from the output; video latent frame 0 itself is generated (and supervised) like every other latent frame.
* Per-token timesteps are native in H3 (one AdaLN row per distinct timestep), so CNS only replaces the timestep of
  every video token by 1 - sigma_tok (the DiT's convention is t = 1 - sigma).
* kappa is evaluated on the 15 x 26 token grid (32 px per token) and is binary. It drives CNS, the 1 / ``new_weight``
  loss weights and the known/new input channel, a zero-initialised ``nn.Linear(1, hidden)`` whose output is added to
  the video patch projection (installing it leaves the model output unchanged).
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn

from ..coords import known_mask

# 832x480 frames: the video VAE downsamples space by 16 and the DiT patchifies 2x2, so one token covers 32 px.
LAT_H, LAT_W = 30, 52
TOK_H, TOK_W = LAT_H // 2, LAT_W // 2
TOKENS_PER_FRAME = TOK_H * TOK_W        # 390
PX_PER_TOKEN = 32
LATENT_C = 24
AUDIO_C, AUDIO_DIM = 2, 32
FRAME_GROUP = (1, 4, 4, 4, 4)
COND_NOISE_AUG = 0.999                  # timestep (t = 1 - sigma) of the first-frame condition tokens
MASK_EMBED = "arope_mask_embed"
MASK_KEYS = (f"{MASK_EMBED}.weight", f"{MASK_EMBED}.bias")
LORA_TARGETS = "blocks.*.attn.qkv_proj|attn.out_proj|mlp.fc1|mlp.fc2"


def lora_target(name: str) -> bool:
    """LoRA layers: attention qkv/out projections and both MLP layers of every DiT block."""
    return name.startswith("blocks.") and name.endswith(("attn.qkv_proj", "attn.out_proj", "mlp.fc1", "mlp.fc2"))


# ----------------------------------------------------------------------------- frames and latent frames
def latent_frames(num_frames: int) -> int:
    if (num_frames - 5) % 17 != 0:
        raise ValueError(f"MiniMax-H3 needs 17n + 5 frames, got {num_frames}")
    return (num_frames - 5) // 17 * 5 + 2


def audio_frames(num_frames: int) -> int:
    """Length of the audio latent for a clip of ``num_frames`` video frames (24 fps video, 40 audio latents / s)."""
    return round(num_frames / 24.0 * 40.0)


def latent_frame_spans(latent_t: int) -> list[tuple[int, int]]:
    """(first pixel frame, number of pixel frames) of every latent frame."""
    out, start = [], 0
    for k in range(latent_t):
        n = FRAME_GROUP[k % len(FRAME_GROUP)]
        out.append((start, n))
        start += n
    return out


def latent_frame_offsets(frame_offset_px: torch.Tensor, latent_t: int) -> torch.Tensor:
    """Per-frame offsets [b, F, 2] -> per-latent-frame offsets [b, latent_t, 2] (float64, pixels), interpolated
    linearly at the centre of the pixel frames of each latent frame."""
    off = frame_offset_px.to(torch.float64)
    nf = off.shape[1]
    c = torch.tensor([s + (n - 1) / 2.0 for s, n in latent_frame_spans(latent_t)], dtype=torch.float64)
    c = c.clamp(0, nf - 1).to(off.device)
    i0 = c.floor().long()
    i1 = (i0 + 1).clamp(max=nf - 1)
    w = c - i0.to(torch.float64)
    return off[:, i0] * (1.0 - w)[None, :, None] + off[:, i1] * w[None, :, None]


def token_kappa(offset_latent_px: torch.Tensor) -> torch.Tensor:
    """Per-latent-frame offsets [b, f, 2] (pixels) -> kappa float32 [b, f * 390] in the DiT's (t, h, w) row order."""
    known = known_mask(offset_latent_px, TOK_H, TOK_W, PX_PER_TOKEN)
    return known.to(torch.float32).reshape(offset_latent_px.shape[0], -1)


# ----------------------------------------------------------------------------- packed sequence and positions
def build_packed(text_len: int, text_tags: torch.Tensor, latent_t: int, audio_t: int, device) -> dict:
    """DiffSynth's FL2VA sequence layout for one first frame (text | condition | audio | video | padding)."""
    from diffsynth.pipelines.minimax_h3_audio_video import MiniMaxH3Unit_PackedSequenceBuilder
    packed = MiniMaxH3Unit_PackedSequenceBuilder()._build_packed_fl2va(int(text_len), latent_t, LAT_H, LAT_W, audio_t, [0])
    packed["token_tags"][packed["text_pos"]] = text_tags.cpu()
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in packed.items()}


def unit_per_px(positions: torch.Tensor, img_pos: torch.Tensor, n_cond: int) -> float:
    """Position units per pixel, read from the packed grid (spacing of neighbouring video tokens / 32 px)."""
    g, vid = positions[0], img_pos[n_cond:]
    step_w = float(g[vid[1], 2] - g[vid[0], 2])
    step_h = float(g[vid[TOK_W], 1] - g[vid[0], 1])
    expect = 64.0 / math.sqrt(LAT_H * LAT_W)
    if abs(step_w - expect) > 1e-6 or abs(step_h - expect) > 1e-6:
        raise RuntimeError(f"unexpected H3 token grid spacing ({step_h}, {step_w}), expected {expect}")
    return step_w / PX_PER_TOKEN


def world_positions(packed: dict, n_cond: int, latent_t: int, frame_offset_px: torch.Tensor):
    """ActionRoPE positions: video tokens of latent frame k are shifted by its camera offset (h += dy, w += dx).

    frame_offset_px: [F, 2] per-frame offsets of the generated frames (frame 0 = 0). Text, audio and the first-frame
    condition tokens keep their positions. Returns (positions [1, S, 3] float64, latent-frame offsets [latent_t, 2]).
    """
    g = packed["img_position_ids"].clone()
    u = unit_per_px(g, packed["img_pos"], n_cond)
    off = latent_frame_offsets(frame_offset_px.to(torch.float64).unsqueeze(0), latent_t)[0].to(g.device)
    vid = packed["img_pos"][n_cond:].view(latent_t, TOKENS_PER_FRAME)
    g[0, vid, 1] += (off[:, 1] * u).unsqueeze(1).to(g.dtype)
    g[0, vid, 2] += (off[:, 0] * u).unsqueeze(1).to(g.dtype)
    return g, off


def kappa_rows(kappa_video: torch.Tensor, n_cond: int) -> torch.Tensor:
    """Input of the mask channel for all image rows [n_cond + n_video, 1]: condition rows are known (1)."""
    k = kappa_video.reshape(-1).to(torch.float32)
    return torch.cat([torch.ones(n_cond, dtype=torch.float32, device=k.device), k]).unsqueeze(1)


def noise_cond_rows(clean: torch.Tensor, noise: torch.Tensor, t: float = COND_NOISE_AUG) -> torch.Tensor:
    """Noise augmentation of the first-frame condition tokens: t * clean + (1 - t) * noise (computed in fp32)."""
    return (t * clean.float() + (1.0 - t) * noise.float()).to(clean.dtype)


# ----------------------------------------------------------------------------- known/new input channel
def install_mask_channel(dit: nn.Module) -> nn.Linear:
    """Add the zero-initialised known/new channel ``dit.arope_mask_embed`` (Linear(1, hidden)) on top of
    ``dit.video_patch_proj``. Its input is set per forward pass by ``forward``."""
    if hasattr(dit, MASK_EMBED):
        return getattr(dit, MASK_EMBED)
    base = dit.video_patch_proj
    emb = nn.Linear(1, base.out_features)
    nn.init.zeros_(emb.weight)
    nn.init.zeros_(emb.bias)
    dit.add_module(MASK_EMBED, emb.to(device=base.weight.device, dtype=base.weight.dtype))
    dit._arope_kappa_rows = None

    def hook(module, inputs, out):
        k = dit._arope_kappa_rows
        if k is None or k.shape[0] != out.shape[0]:
            raise RuntimeError("the known/new channel needs kappa for every image row of this forward pass")
        return out + getattr(dit, MASK_EMBED)(k.to(device=out.device, dtype=out.dtype))

    base.register_forward_hook(hook)
    return getattr(dit, MASK_EMBED)


def has_mask_channel(dit: nn.Module) -> bool:
    return hasattr(dit, MASK_EMBED)


# ----------------------------------------------------------------------------- forward
def forward(dit, video_latents, audio_latents, packed, prompt_embeds, timestep_video, timestep_audio, cond_rows,
            positions, sigma_tok=None, kappa=None, use_gradient_checkpointing=False):
    """One DiT evaluation; DiffSynth's ``model_fn_minimax_h3`` (FL2VA, no retake / reference / ControlNet inputs) with
    ActionRoPE positions, per-token video timesteps (CNS) and the known/new channel.

    timestep_video / timestep_audio: scalar timesteps in [0, 1000] (sigma * 1000). sigma_tok: per-video-token sigma
    [f * 390] or None. kappa: per-video-token kappa [f * 390] (needed when the mask channel is installed).
    Returns the velocities (video [1, 24, f, 30, 52], audio [2, 32, T]) in the scheduler's convention.
    """
    from diffsynth.models.minimax_h3_dit import pack_audio, patchify_video, unpack_audio, unpatchify_video
    t_video = 1.0 - float(timestep_video) / 1000
    t_audio = 1.0 - float(timestep_audio) / 1000
    dtype, device = video_latents.dtype, video_latents.device
    f, h, w = video_latents.shape[2:]
    audio_channel, audio_t = audio_latents.shape[0], audio_latents.shape[-1]
    video_rows = patchify_video(video_latents)
    audio_rows = pack_audio(audio_latents)

    img_pos, audio_pos, text_pos = packed["img_pos"], packed["audio_pos"], packed["text_pos"]
    cu, seq_len, text_len = packed["cu_seqlens"], packed["seq_len"], text_pos.shape[0]
    n_cond = cond_rows.shape[0]
    x = torch.zeros(1, seq_len, 96, dtype=dtype, device=device)
    x[0].index_copy_(0, img_pos[n_cond:], video_rows)
    x[0].index_copy_(0, img_pos[:n_cond], cond_rows)
    audio_x = torch.zeros(1, seq_len, 32, dtype=dtype, device=device)
    audio_x[0].index_copy_(0, audio_pos, audio_rows)

    timesteps = torch.full((seq_len,), t_video, dtype=torch.float32, device=device)
    timesteps[audio_pos] = t_audio
    if sigma_tok is not None:
        st = sigma_tok.to(device=device, dtype=torch.float32).view(-1)
        if st.shape[0] != video_rows.shape[0]:
            raise ValueError(f"sigma_tok has {st.shape[0]} entries for {video_rows.shape[0]} video tokens")
        timesteps[img_pos[n_cond:]] = 1.0 - st
    timesteps[img_pos[:n_cond]] = max(t_video, COND_NOISE_AUG)
    unique_timesteps, inverse_indices = torch.unique(timesteps, sorted=True, return_inverse=True)
    refiner_cu = torch.tensor([0, text_len, text_len], dtype=torch.int32, device=device)

    if has_mask_channel(dit):
        if kappa is None:
            raise ValueError("the model has a known/new channel: pass kappa")
        dit._arope_kappa_rows = kappa_rows(kappa, n_cond)
    try:
        v_video_rows, v_audio_rows = dit(
            x=x, audio_x=audio_x, img_position_ids=positions,
            unique_timesteps=unique_timesteps, inverse_indices=inverse_indices, token_tags=packed["token_tags"],
            prompt_embeds=prompt_embeds,
            img_pos_info={"position_ids": img_pos}, audio_pos_info={"position_ids": audio_pos},
            text_pos_info={"position_ids": text_pos}, img_pos_for_infer_output_info={"position_ids": img_pos},
            packed_seq_params={"cu_seqlens_q": cu, "max_seqlen_q": int(cu[1])},
            refiner_packed_seq_params={"cu_seqlens_q": refiner_cu, "max_seqlen_q": text_len},
            update_mask=None, skip_mask_out_condition=True,
            use_gradient_checkpointing=use_gradient_checkpointing, use_gradient_checkpointing_offload=False,
            control_hints=None)
    finally:
        if has_mask_channel(dit):
            dit._arope_kappa_rows = None
    v_video = unpatchify_video(v_video_rows[n_cond:], f, h, w)
    v_audio = unpack_audio(v_audio_rows, audio_channel, audio_t)
    return -v_video, -v_audio


# ----------------------------------------------------------------------------- retiming
def retime_offsets(frame_offset_px: np.ndarray, n_out: int) -> np.ndarray:
    """Compress a per-frame offset plan [F, 2] to ``n_out`` frames over the same duration (linear interpolation).

    H3 generates 17n + 5 frames, so the 81-frame plan is played over 73 frames: the total displacement is unchanged
    and the clip covers the same commanded motion with fewer, more widely spaced frames."""
    off = np.asarray(frame_offset_px, np.float64)
    n_in = off.shape[0]
    t = np.arange(n_out, dtype=np.float64) * (n_in - 1) / (n_out - 1)
    i0 = np.floor(t).astype(int)
    i1 = np.minimum(i0 + 1, n_in - 1)
    w = (t - i0)[:, None]
    return off[i0] * (1 - w) + off[i1] * w


def retime_frames(frames: np.ndarray, n_out: int) -> np.ndarray:
    """Nearest-neighbour resampling of a clip to ``n_out`` frames (first and last frame kept)."""
    n_in = frames.shape[0]
    idx = np.rint(np.arange(n_out, dtype=np.float64) * (n_in - 1) / (n_out - 1)).astype(int)
    return frames[idx]
