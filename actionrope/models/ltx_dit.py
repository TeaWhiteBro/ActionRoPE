"""ActionRoPE, the known/new input channel and CNS for the LTX-2.5 DiT (``diffsynth.models.ltx2_dit.LTXModel``).

Unlike the Wan family, LTX-2.5 receives the position of every token explicitly: DiffSynth passes
``video_positions`` [b, 3, n_tokens, 2] with the [start, end) interval of each token along (time in seconds, height
in pixels, width in pixels), and the DiT turns the interval midpoints into rotary embeddings (normalised by
``max_pos = [20 s, 2048 px, 2048 px]``). One token is one latent cell of 32x32 pixels (VAE downsampling 32x,
patch size 1). ActionRoPE (Sec. 3.1) therefore only adds the camera offset of each latent frame to the height and
width intervals,

    token (k, i, j):  height [32 i + dy_k, 32 i + 32 + dy_k),  width [32 j + dx_k, 32 j + 32 + dx_k)  (pixels),

i.e. the world coordinate (i + dy_k / 32, j + dx_k / 32) in pixel units; the DiT itself is unchanged.

LTX-2.5 has a causal VAE with 8x temporal downsampling: latent frame 0 encodes pixel frame 0 and latent frame k >= 1
encodes pixel frames 8k-7 .. 8k. The offset of latent frame k is the offset of the pixel frame at the temporal
midpoint of its interval, 8k-3 (frame 0 for k = 0); it is point-sampled, not averaged.

This module also holds the pieces that hook into DiffSynth's LTX-2.5 pipeline without editing it. Every hook is
installed explicitly, is idempotent and is bound to one model / pipeline instance:

* ``install_mask_channel``    known/new input channel: zero-initialised Linear(1, dim) added to ``patchify_proj``.
* ``install_per_frame_text``  per-latent-frame text cross-attention on every transformer block.
* ``CNSTimesteps``            CNS: replaces the per-token video timesteps the DiT receives.
* ``install_cns_step``        CNS: per-token Euler step of the pipeline's scheduler.
* ``install_world_positions`` / ``install_repeated_prompt``  the pipeline's position and prompt units.
"""
from __future__ import annotations

import math
import os
import types

import numpy as np
import torch
import torch.nn as nn

from ..cns import cns_sigma
from ..geometry import FPS, FRAMES, HEIGHT, WIDTH

PX_PER_TOKEN = 32                     # one LTX-2.5 token = one latent cell = 32x32 pixels
LATENT_CHANNELS = 128
LATENT_FRAMES, LATENT_H, LATENT_W = 11, 15, 26
LATENT_FRAME_PIXELS = (0, 5, 13, 21, 29, 37, 45, 53, 61, 69, 77)   # pixel frame that gives each latent frame its offset
# Base shift of the LTX-2 schedule at our token count: sigma = e^mu u / (1 + (e^mu - 1) u) with mu = 2.05 (followed
# by DiffSynth's terminal stretch to sigma_min = 0.1). CNS inverts the shift with this constant.
SHIFT = math.exp(2.05)
MASK_EMBEDDING = "arope_mask_embedding"
DTYPE = torch.bfloat16

# Base model files. ``model_dir`` is a snapshot of Lightricks/LTX-2.5; the tokenizer and the text-encoder post
# modules come from DiffSynth-Studio/LTX-2.5-Repackage (``repack_dir``).
DIT_FILE = "diffusion_models/ltx-2.5-22b-dev-transformer-bf16.safetensors"
GEMMA_FILE = "text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors"
VIDEO_VAE_FILE = "vae/ltx-2.5-video-vae-bf16.safetensors"
AUDIO_VAE_FILE = "vae/ltx-2.5-audio-vae-bf16.safetensors"
POST_MODULES_FILE = "text_encoder_post_modules.safetensors"
TOKENIZER_DIR = "tokenizer"
REPACK_NAME = "LTX-2.5-Repackage"


# ----------------------------------------------------------------------------- base model loading
def resolve_repack_dir(model_dir: str, repack_dir: str | None = None) -> str:
    """The LTX-2.5-Repackage snapshot: ``repack_dir`` if given, else ``<model_dir>/LTX-2.5-Repackage`` or a sibling
    directory ``LTX-2.5-Repackage`` next to ``model_dir``."""
    cands = [repack_dir] if repack_dir else [os.path.join(model_dir, REPACK_NAME),
                                             os.path.join(os.path.dirname(os.path.abspath(model_dir)), REPACK_NAME)]
    for c in cands:
        if c and os.path.isdir(os.path.join(c, TOKENIZER_DIR)):
            return c
    raise FileNotFoundError(f"LTX-2.5-Repackage (tokenizer + text-encoder post modules) not found in {cands}")


def load_pipeline(model_dir: str, components: tuple[str, ...], device: str = "cuda", repack_dir: str | None = None,
                  vram_limit: float | None = None):
    """DiffSynth's LTX2AudioVideoPipeline with the requested components ("text", "dit", "video_vae", "audio_vae").

    Every model is loaded in bf16 with DiffSynth's VRAM management (stored on the CPU, moved to ``device`` when a
    pipeline stage needs it); with all devices set to ``device`` the wrappers are pass-through."""
    from diffsynth.pipelines.ltx2_audio_video import LTX2AudioVideoPipeline, ModelConfig
    repack = resolve_repack_dir(model_dir, repack_dir)
    vram = dict(offload_dtype=DTYPE, offload_device="cpu", onload_dtype=DTYPE, onload_device=device,
                preparing_dtype=DTYPE, preparing_device=device, computation_dtype=DTYPE, computation_device=device)
    files = {"text": [os.path.join(model_dir, GEMMA_FILE), os.path.join(repack, POST_MODULES_FILE)],
             "dit": [os.path.join(model_dir, DIT_FILE)],
             "video_vae": [os.path.join(model_dir, VIDEO_VAE_FILE)],
             "audio_vae": [os.path.join(model_dir, AUDIO_VAE_FILE)]}
    configs = [ModelConfig(path=p, **vram) for c in ("text", "dit", "video_vae", "audio_vae") if c in components
               for p in files[c]]
    return LTX2AudioVideoPipeline.from_pretrained(
        torch_dtype=DTYPE, device=device, vram_limit=vram_limit,
        tokenizer_config=ModelConfig(path=os.path.join(repack, TOKENIZER_DIR)), model_configs=configs)


# ----------------------------------------------------------------------------- token grid and world positions
def latent_grid(device="cpu"):
    """Token positions of the 832x480x81 clip exactly as DiffSynth's pipeline computes them.

    Returns (pos [1, 3, n, 2] float32 with time in seconds, patch_frame_idx [n] long, latent_frame_pixel_idx [11]
    long): the pixel frame whose offset each token / latent frame takes (the rounded temporal midpoint)."""
    from diffsynth.models.ltx2_common import VIDEO_SCALE_FACTORS, VideoLatentShape, VideoPixelShape, get_pixel_coords
    from diffsynth.models.ltx2_video_vae import VideoLatentPatchifier
    shape = VideoLatentShape.from_pixel_shape(
        shape=VideoPixelShape(batch=1, frames=FRAMES, width=WIDTH, height=HEIGHT, fps=FPS), latent_channels=LATENT_CHANNELS)
    assert (shape.frames, shape.height, shape.width) == (LATENT_FRAMES, LATENT_H, LATENT_W), shape
    bounds = VideoLatentPatchifier(patch_size=1).get_patch_grid_bounds(shape, device=device)
    pos = get_pixel_coords(bounds, VIDEO_SCALE_FACTORS, True).float()
    pos[:, 0] = pos[:, 0] / FPS
    idx = ((pos[0, 0, :, 0] + pos[0, 0, :, 1]) * 0.5 * FPS).round().long().clamp(0, FRAMES - 1)
    per_frame = idx.view(LATENT_FRAMES, LATENT_H * LATENT_W)
    lf_idx = per_frame[:, 0].clone()
    assert torch.equal(per_frame, lf_idx.view(-1, 1).expand_as(per_frame)), "tokens of a latent frame must share a pixel frame"
    assert tuple(lf_idx.tolist()) == LATENT_FRAME_PIXELS, lf_idx
    return pos, idx, lf_idx


def latent_frame_offsets(frame_offset_px) -> torch.Tensor:
    """Per-frame camera offsets [..., 81, 2] -> offsets of the 11 latent frames [..., 11, 2] (pixel frames 8k-3)."""
    off = torch.as_tensor(frame_offset_px)
    return off[..., list(LATENT_FRAME_PIXELS), :]


def world_positions(pos: torch.Tensor, patch_frame_idx: torch.Tensor, frame_offset_px: torch.Tensor) -> torch.Tensor:
    """Training: screen positions [1, 3, n, 2] + per-frame offsets [b, 81, 2] -> world positions [b, 3, n, 2] float32.

    The sum is formed in float64 and rounded once to float32 (the positions stay float32 as in DiffSynth's sampler:
    in bf16 the pixel coordinates would only have a resolution of 4-8 px)."""
    off = frame_offset_px.to(torch.float64)
    idx = patch_frame_idx.to(off.device)
    vp = pos.to(torch.float64).expand(off.shape[0], -1, -1, -1).clone()
    vp[:, 1] += off[:, idx, 1].unsqueeze(-1)       # height axis (y, down)
    vp[:, 2] += off[:, idx, 0].unsqueeze(-1)       # width axis (x, right)
    return vp.to(pos.dtype)


def add_offsets_to_positions(vp: torch.Tensor, frame_offset_px: torch.Tensor, frame_rate: float) -> torch.Tensor:
    """Sampling: add per-frame offsets [81, 2] to positions [b, 3, n, 2] computed by the pipeline (time in seconds).

    The pixel frame of each token is recomputed from its temporal midpoint; the offsets are cast to the dtype of the
    positions (float32) before the addition, as in the sampler of the paper."""
    t = vp[:, 0].to(torch.float64)
    idx = ((t[..., 0] + t[..., 1]) * 0.5 * frame_rate).round().long().clamp(0, frame_offset_px.shape[0] - 1)
    off = frame_offset_px.to(torch.float64)
    dx, dy = off[idx.cpu(), 0].to(vp.device), off[idx.cpu(), 1].to(vp.device)
    out = vp.clone()
    out[:, 1] = out[:, 1] + dy.unsqueeze(-1).to(vp.dtype)
    out[:, 2] = out[:, 2] + dx.unsqueeze(-1).to(vp.dtype)
    return out


# ----------------------------------------------------------------------------- known/new input channel
def install_mask_channel(dit: nn.Module) -> nn.Module:
    """Add the zero-initialised known/new input channel ``dit.arope_mask_embedding`` = Linear(1, dim) and a forward
    hook on ``dit.patchify_proj`` that adds ``arope_mask_embedding(dit._arope_mask_patch)`` to the patch embedding.

    ``dit._arope_mask_patch`` is the patchified known mask [b, n_tokens, 1] (known = 1, new = 0) of the current
    forward pass, set by the caller; None disables the channel. The model output is unchanged at installation."""
    if not hasattr(dit, MASK_EMBEDDING):
        embed = nn.Linear(1, dit.inner_dim, bias=True)
        nn.init.zeros_(embed.weight)
        nn.init.zeros_(embed.bias)
        ref = dit.patchify_proj.weight
        dit.add_module(MASK_EMBEDDING, embed.to(device=ref.device, dtype=ref.dtype))
    if not getattr(dit.patchify_proj, "_arope_mask_hook", False):
        def hook(module, inputs, output):
            mask = getattr(dit, "_arope_mask_patch", None)
            if mask is None:
                return output
            return output + getattr(dit, MASK_EMBEDDING)(mask.to(dtype=output.dtype, device=output.device))
        dit.patchify_proj.register_forward_hook(hook)
        dit.patchify_proj._arope_mask_hook = True
    if not hasattr(dit, "_arope_mask_patch"):
        dit._arope_mask_patch = None
    return dit


# ----------------------------------------------------------------------------- per-latent-frame text
def _per_frame_text_cross_attention(self, x, context, attn, scale_shift_table, prompt_scale_shift_table, timestep,
                                    prompt_timestep, context_mask, cross_attention_adaln=False):
    """BasicAVTransformerBlock._apply_text_cross_attention where latent frame k attends only to the k-th prompt.

    The video context is [b, f * L, D] (prompt k in rows k*L .. (k+1)*L); video tokens are regrouped as (b f) rows of
    one latent frame each and paired with their own prompt, including the AdaLN modulation of queries and keys
    (``cross_attention_adaln``). The audio branch and the off state (f = None) use the original method."""
    orig = type(self)._apply_text_cross_attention
    f = self._arope_text_frames["f"]
    if f is None or attn is not getattr(self, "attn2", None):
        return orig(self, x, context, attn, scale_shift_table, prompt_scale_shift_table, timestep, prompt_timestep,
                    context_mask, cross_attention_adaln)
    from diffsynth.models import ltx2_dit as M
    b, N, D = x.shape
    assert N % f == 0 and context.shape[1] % f == 0, (N, context.shape, f)
    assert context_mask is None, "per-frame text does not support a context mask"
    n, L = N // f, context.shape[1] // f
    x_ = x.reshape(b * f, n, D)
    ctx_ = context.reshape(b * f, L, context.shape[-1])

    def per_token(t):          # query-side modulation [b, T, C]: T = 1 -> repeat per frame, T = N -> split per frame
        if t.shape[1] == 1:
            return t.repeat_interleave(f, dim=0)
        assert t.shape[1] == N, (t.shape, N)
        return t.reshape(b * f, n, t.shape[-1])

    if cross_attention_adaln:
        shift_q, scale_q, gate = self.get_ada_values(scale_shift_table, b, timestep, slice(6, 9))
        pt = prompt_timestep
        if pt.shape[1] == 1:
            pt_ = pt.repeat_interleave(f, dim=0)
        else:
            assert pt.shape[1] == context.shape[1], (pt.shape, context.shape)
            pt_ = pt.reshape(b * f, L, pt.shape[-1])
        y = M.apply_cross_attention_adaln(x_, ctx_, attn, per_token(shift_q), per_token(scale_q), per_token(gate),
                                          prompt_scale_shift_table, pt_, None, self.norm_eps)
    else:
        y = attn(M.rms_norm(x_, eps=self.norm_eps), context=ctx_, mask=None)
    return y.reshape(b, N, D)


def install_per_frame_text(dit: nn.Module) -> None:
    """Bind the per-frame text cross-attention to every transformer block of ``dit`` (instance attributes, so other
    models in the process are unaffected). Off until ``set_text_frames(dit, f)``; with identical prompts for all
    frames it equals the global cross-attention up to floating-point summation order."""
    from diffsynth.models.ltx2_dit import BasicAVTransformerBlock
    state = dit.__dict__.setdefault("_arope_text_frames", {"f": None})
    for m in dit.modules():
        if isinstance(m, BasicAVTransformerBlock) and "_apply_text_cross_attention" not in m.__dict__:
            m.__dict__["_arope_text_frames"] = state
            m._apply_text_cross_attention = types.MethodType(_per_frame_text_cross_attention, m)


def set_text_frames(dit: nn.Module, f: int | None) -> None:
    """f latent frames -> the video context must be [b, f * L, D]; None -> one prompt for the whole clip."""
    dit.__dict__["_arope_text_frames"]["f"] = f


# ----------------------------------------------------------------------------- CNS
class CNSTimesteps:
    """Stands in for the DiT inside DiffSynth's ``model_fn_ltx2`` and replaces the per-token video timesteps sigma
    by sigma_t = cns_sigma(sigma, kappa) (Sec. 3.3). The clean first frame keeps sigma = 0 (cns_sigma(0) = 0).

    kappa_tokens: knownness of the video tokens [b, n_tokens, 1] float32 (patchified, known = 1)."""

    def __init__(self, dit, kappa_tokens: torch.Tensor, shift: float, shift_known: float, alpha: float):
        self.dit, self.kappa = dit, kappa_tokens
        self.shift, self.shift_known, self.alpha = float(shift), float(shift_known), float(alpha)

    def __call__(self, video_timesteps=None, **kw):
        n = self.kappa.shape[1]
        kap = self.kappa.to(device=video_timesteps.device, dtype=torch.float32)
        t = cns_sigma(video_timesteps[:, :n].float(), kap, self.shift, self.shift_known, self.alpha).to(video_timesteps.dtype)
        if video_timesteps.shape[1] > n:                  # appended reference tokens (unused here) keep their timestep
            t = torch.cat([t, video_timesteps[:, n:]], dim=1)
        return self.dit(video_timesteps=t, **kw)


def install_cns_step(scheduler, state: dict, shift: float, shift_known: float, alpha: float) -> None:
    """Replace ``scheduler.step`` (this instance only) by the per-token Euler step
    x <- x + v * (sigma_t(sigma_next) - sigma_t(sigma)) with the knownness ``state["kappa"]`` [b, 1, f, h, w].

    The first latent frame uses kappa = 0 (the base schedule; the pipeline steers it back to the clean latent), and
    tensors that are not video latents of that shape (the audio latents) use the original step."""
    if "_arope_orig_step" in scheduler.__dict__:
        scheduler.__dict__["_arope_cns"] = (state, float(shift), float(shift_known), float(alpha))
        return
    orig = scheduler.step
    scheduler.__dict__["_arope_orig_step"] = orig
    scheduler.__dict__["_arope_cns"] = (state, float(shift), float(shift_known), float(alpha))

    def step(model_output, timestep, sample, to_final=False, **kw):
        st, S, S_known, a = scheduler.__dict__["_arope_cns"]
        kap = st.get("kappa")
        if kap is None or sample.dim() != 5 or tuple(sample.shape[2:]) != tuple(kap.shape[2:]):
            return orig(model_output, timestep, sample, to_final=to_final, **kw)
        t = timestep.cpu() if isinstance(timestep, torch.Tensor) else timestep
        tid = int(torch.argmin((scheduler.timesteps - t).abs()))
        s_i = float(scheduler.sigmas[tid])
        s_n = 0.0 if (to_final or tid + 1 >= len(scheduler.timesteps)) else float(scheduler.sigmas[tid + 1])
        k = kap.to(device=sample.device, dtype=torch.float32).clone()
        k[:, :, 0] = 0.0
        d = cns_sigma(s_n, k, S, S_known, a) - cns_sigma(s_i, k, S, S_known, a)
        return sample + model_output * d.to(sample.dtype)

    scheduler.step = step


# ----------------------------------------------------------------------------- pipeline units
def _unit(pipe, cls_name: str):
    hits = [u for u in pipe.units if type(u).__name__ == cls_name]
    assert len(hits) == 1, (cls_name, [type(u).__name__ for u in pipe.units])
    return hits[0]


def install_world_positions(pipe, state: dict) -> None:
    """Add ``state["frame_offset_px"]`` ([81, 2] float64 or None) to the video positions computed by the pipeline's
    noise-initialiser unit (this pipeline instance only)."""
    unit = _unit(pipe, "LTX2AudioVideoUnit_NoiseInitializer")
    if "_arope_orig_process" in unit.__dict__:
        return
    orig = unit.process
    unit.__dict__["_arope_orig_process"] = orig

    def process(pipe_, **kw):
        out = orig(pipe_, **kw)
        off = state.get("frame_offset_px")
        if off is not None and out.get("video_positions") is not None:
            out["video_positions"] = add_offsets_to_positions(out["video_positions"], off, kw.get("frame_rate", FPS))
        return out

    unit.process = process


def install_repeated_prompt(pipe, state: dict) -> None:
    """Repeat the video context of the encoded prompt ``state["text_frames"]`` times ([1, L, D] -> [1, f * L, D]) for
    the per-frame text cross-attention; positive and negative prompts alike (this pipeline instance only)."""
    unit = _unit(pipe, "LTX2AudioVideoUnit_PromptEmbedder")
    if "_arope_orig_process" in unit.__dict__:
        return
    orig = unit.process
    unit.__dict__["_arope_orig_process"] = orig

    def process(pipe_, **kw):
        out = orig(pipe_, **kw)
        f = state.get("text_frames")
        if f:
            out["video_context"] = out["video_context"].repeat(1, f, 1)
        return out

    unit.process = process


# ----------------------------------------------------------------------------- checkpoint keys
def clean_key(name: str) -> str:
    """DiT parameter name without DiffSynth's VRAM-management wrapper segments (``transformer_blocks.0.module.attn1``
    -> ``transformer_blocks.0.attn1``); the result equals the parameter name in the LTX-2.5 base checkpoint."""
    return ".".join(p for p in name.split(".") if p != "module")


def clean_to_model_keys(dit: nn.Module) -> dict[str, str]:
    """{clean name: name in ``dit.state_dict()``}; raises if two parameters collapse to one clean name."""
    out = {}
    for k in dit.state_dict():
        c = clean_key(k)
        if c in out:
            raise RuntimeError(f"ambiguous clean key {c}: {out[c]} / {k}")
        out[c] = k
    return out


def is_lora_target(name: str) -> bool:
    """LoRA targets: the video self-attention, video text cross-attention and video FFN linears of the 48 blocks
    (480 layers). Audio, audio-video cross-attention, gates, AdaLN and projections stay frozen."""
    parts = clean_key(name).split(".")
    if len(parts) < 3 or parts[0] != "transformer_blocks":
        return False
    return ".".join(parts[2:]) in {"attn1.to_q", "attn1.to_k", "attn1.to_v", "attn1.to_out.0",
                                   "attn2.to_q", "attn2.to_k", "attn2.to_v", "attn2.to_out.0",
                                   "ff.net.0.proj", "ff.net.2"}


def offsets_tensor(frame_offset_px) -> torch.Tensor:
    off = torch.as_tensor(np.asarray(frame_offset_px, np.float64))
    if tuple(off.shape) != (FRAMES, 2):
        raise ValueError(f"frame_offset_px must have shape ({FRAMES}, 2), got {tuple(off.shape)}")
    return off
