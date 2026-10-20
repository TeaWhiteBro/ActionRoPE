"""Wan2.2-TI2V-5B with ActionRoPE + CNS (the main backbone of the paper).

Geometry: the Wan2.2 VAE compresses 16x in space and 4x in time (latent frame 0 = video frame 0, latent frame k =
frames 4k-3 .. 4k), so an 81-frame 832x480 clip becomes 21 latent frames of 48 x 30 x 52. The DiT patch is (1, 2, 2):
one token covers 32 x 32 pixels, one latent cell 16 x 16 pixels.

* Camera offsets. The offset of a latent frame is the mean of the per-frame offsets of the frames it covers
  (``geometry.frames_to_cells``); the token positions are shifted by offset / 32 (ActionRoPE, Sec. 3.1), the known/new
  regions are evaluated on the 30 x 52 latent grid (offset / 16).
* Known/new input channel. A zero-initialised Conv3d(1, 3072) adds the known map to the patch embedding (15,360
  parameters).
* Training (Sec. 4.1). Full fine-tuning of the DiT with the flow-matching loss of Wan2.2: one timestep index per
  sample from the 1000-step training schedule (shift 5), loss weighted by DiffSynth's Wan timestep weighting. Latent
  frame 0 is the clean first frame (timestep 0, no loss); newly revealed latent cells have loss weight ``new_weight``.
  With CNS every latent cell is noised to its own level sigma_tok(sigma, kappa) and the DiT receives 1000 sigma_tok
  as per-token timestep (kappa average-pooled to the 2 x 2 token grid). Each latent frame cross-attends to its own
  scene prompt (prompts change inside scene-transition clips); with probability ``scene_dropout`` the whole clip
  uses NULL_PROMPT.
* Sampling. 50 Euler steps of the shift-5 schedule, no classifier-free guidance, NULL_PROMPT. Latent frame 0 is
  written back to the encoded first frame after every step. With CNS each latent cell is integrated along its own
  sigma_tok axis.
"""
from __future__ import annotations

import os
import random

import numpy as np
import torch
import torch.nn as nn

from ..checkpoint import meta_bool, meta_float, read_metadata
from ..cns import cns_sigma
from ..coords import knownness, loss_weight_map
from ..data import load_annotations, load_text_table, training_clips, validation_clips
from ..geometry import FPS, frames_to_cells
from ..prompts import NULL_PROMPT
from .wan_dit import MASK_EMBEDDING, arope_forward, install_mask_channel

MODEL_NAME = "wan2.2-ti2v-5b"
DIT_FILES = [f"diffusion_pytorch_model-0000{i}-of-00003.safetensors" for i in (1, 2, 3)]
VAE_FILE = "Wan2.2_VAE.pth"

LATENT_SHAPE = (48, 21, 30, 52)          # (channels, latent frames, height, width) of an 81-frame 832x480 clip
LAT_H, LAT_W = LATENT_SHAPE[2], LATENT_SHAPE[3]
PX_PER_LATENT = 16                        # VAE stride
PX_PER_TOKEN = 32                         # VAE stride x DiT patch size
SHIFT = 5.0                               # Wan2.2 flow-matching shift (training and sampling)
STEPS = 50
VAE_TILE = dict(tile_size=(30, 52), tile_stride=(15, 26))   # only used if tiled=True; frames are encoded whole

# Evaluation videos of the paper were written with imageio's quality scale 9 (libx264, crf 5).
WRITE_KW = {"quality": 9}
NULL_PROMPT_FILE = "umt5_null_prompt.pt"      # {NULL_PROMPT: UMT5 embedding}, shipped with the checkpoint


# ----------------------------------------------------------------------------- helpers
def cell_offsets(frame_offset_px: np.ndarray) -> torch.Tensor:
    """Per-frame camera offsets [81, 2] -> per-latent-frame offsets float32 [21, 2] (mean over the frames of a cell)."""
    return torch.from_numpy(frames_to_cells(frame_offset_px)).to(torch.float32)


def known_input(offset_px: torch.Tensor) -> torch.Tensor:
    """Input of the known/new channel: float32 [b, 1, 21, 30, 52], 1 = known, 0 = newly revealed."""
    return knownness(offset_px, LAT_H, LAT_W, PX_PER_LATENT)


def cns_kappa(offset_px: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Knownness on the latent grid [b, 1, f, 30, 52] (noise levels) and on the token grid [b, f, 15, 26]
    (timestep embedding; the mean over the 2 x 2 latent cells of a token)."""
    kappa = knownness(offset_px, LAT_H, LAT_W, PX_PER_LATENT)
    return kappa, torch.nn.functional.avg_pool2d(kappa[:, 0], kernel_size=2)


def load_pipeline(model_dir: str, device: str = "cuda", vae: bool = True, torch_dtype=torch.bfloat16):
    """DiffSynth WanVideoPipeline with the DiT (and the VAE); no text encoder."""
    from diffsynth.pipelines.wan_video import ModelConfig, WanVideoPipeline
    configs = [ModelConfig(path=[os.path.join(model_dir, f) for f in DIT_FILES])]
    if vae:
        configs.append(ModelConfig(path=os.path.join(model_dir, VAE_FILE)))
    return WanVideoPipeline.from_pretrained(torch_dtype=torch_dtype, device=device, model_configs=configs,
                                            tokenizer_config=None, redirect_common_files=False)


def load_checkpoint(dit: nn.Module, path: str) -> dict:
    """Load a full ActionRoPE checkpoint into ``dit`` (strict); installs the known/new channel if the checkpoint has
    one. Returns the checkpoint metadata."""
    from safetensors.torch import load_file
    meta = read_metadata(path)
    sd = load_file(path)
    has_mask = any(k.startswith(MASK_EMBEDDING + ".") for k in sd)
    if meta.get("mask_channel") is not None and meta_bool(meta, "mask_channel") != has_mask:
        raise ValueError(f"{path}: metadata mask_channel={meta['mask_channel']} but the tensors say {has_mask}")
    if has_mask:
        install_mask_channel(dit)
    dit.load_state_dict(sd, strict=True)
    return meta


def build_text_table(prompts: list[str], model_dir: str, device: str = "cuda") -> dict[str, torch.Tensor]:
    """{prompt: bf16 [512, 4096]} UMT5 embeddings (the encoder of the Wan family)."""
    from .umt5 import build_text_table as _build
    return _build(prompts, model_dir, device)


class PromptEmbeddings:
    """Prompt -> UMT5 embedding. Looks the prompt up in an optional embedding file ({prompt: tensor}, e.g. the text
    table of ``scripts/build_text_table.py``) and otherwise encodes it with UMT5 from ``model_dir`` (loaded lazily)."""

    def __init__(self, model_dir: str, device: str, table: str | dict | None = None):
        self.model_dir, self.device = model_dir, device
        self.table = dict(load_text_table(table) if isinstance(table, str) else (table or {}))
        self.encoder = None

    def __call__(self, prompt: str) -> torch.Tensor:
        if prompt not in self.table:
            if self.encoder is None:
                from .umt5 import UMT5Encoder
                self.encoder = UMT5Encoder(self.model_dir, self.device)
            self.table[prompt] = self.encoder([prompt])[0]
        return self.table[prompt].unsqueeze(0).to(device=self.device, dtype=torch.bfloat16)   # [1, 512, 4096]


# ----------------------------------------------------------------------------- VAE
class _vae_fp32:
    """Run the VAE in fp32 inside the block."""

    def __init__(self, pipe):
        self.pipe = pipe

    def __enter__(self):
        self.pipe.vae.model.float()

    def __exit__(self, *exc):
        self.pipe.vae.model.to(self.pipe.torch_dtype)


@torch.no_grad()
def encode_first_frame(pipe, img: np.ndarray) -> torch.Tensor:
    """uint8 [480, 832, 3] -> latent [1, 48, 1, 30, 52]: the pipeline's (bf16-loaded) VAE run in fp32, not tiled, as in
    the paper's evaluation. ``LatentEncoder`` loads the VAE weights in full fp32 instead (closer to the training data)."""
    from PIL import Image
    x = pipe.preprocess_image(Image.fromarray(img), torch_dtype=torch.float32).transpose(0, 1)   # [3, 1, H, W]
    with _vae_fp32(pipe):
        z = pipe.vae.encode([x], device=pipe.device, tiled=False, **VAE_TILE)
    return z.to(dtype=pipe.torch_dtype, device=pipe.device)


@torch.no_grad()
def decode(pipe, latents: torch.Tensor) -> np.ndarray:
    """latent [1, 48, 21, 30, 52] -> uint8 frames [81, 480, 832, 3] (not tiled)."""
    video = pipe.vae.decode(latents, device=pipe.device, tiled=False, **VAE_TILE)
    return np.stack([np.array(f) for f in pipe.vae_output_to_video(video)], 0)


# ----------------------------------------------------------------------------- sampling
@torch.no_grad()
def sample(dit, z0: torch.Tensor, context: torch.Tensor, offset_px: torch.Tensor, mask_input: torch.Tensor | None, *,
           steps: int = STEPS, shift: float = SHIFT, seed: int = 0, cns: dict | None = None) -> torch.Tensor:
    """Flow-matching Euler sampler (no guidance). z0 [1, 48, 1, 30, 52] first-frame latent, context [1, L, 4096],
    offset_px [1, 21, 2]. ``cns`` = {"shift_known", "alpha"} enables CNS with base shift ``shift``."""
    from diffsynth.diffusion.flow_match import FlowMatchScheduler
    sched = FlowMatchScheduler("Wan")
    sched.set_timesteps(steps, shift=shift)
    gen = torch.Generator("cpu").manual_seed(seed)
    latents = torch.randn((1, z0.shape[1], LATENT_SHAPE[1], LAT_H, LAT_W), generator=gen, dtype=torch.float32)
    latents = latents.to(dtype=z0.dtype, device=z0.device)
    latents[:, :, 0:1] = z0
    if cns is not None:
        kappa, kappa_tok = cns_kappa(offset_px)
        kappa, kappa_tok = kappa.to(latents.device), kappa_tok.to(latents.device)

        def sigma_tok(s, k):
            return cns_sigma(s, k, shift, cns["shift_known"], cns["alpha"])
    for i, t in enumerate(sched.timesteps):
        timestep = t.unsqueeze(0).to(dtype=latents.dtype, device=latents.device)
        tmap = None
        if cns is not None:
            tmap = (1000.0 * sigma_tok(float(sched.sigmas[i]), kappa_tok)).to(latents.dtype)
        pred = arope_forward(dit, latents, timestep, context, offset_px=offset_px, px_per_token=PX_PER_TOKEN,
                             mask_input=mask_input, timestep_map=tmap, first_frame_cond=True)
        if cns is not None:
            s_i = float(sched.sigmas[i])
            s_n = float(sched.sigmas[i + 1]) if i + 1 < steps else 0.0
            latents = latents + pred * (sigma_tok(s_n, kappa) - sigma_tok(s_i, kappa)).to(latents.dtype)
        else:
            latents = sched.step(pred, sched.timesteps[i], latents)
        latents[:, :, 0:1] = z0
    return latents


class Generator:
    """First frame + per-frame camera offsets -> 81-frame clip with a full ActionRoPE checkpoint.

    Prompt embeddings are looked up in ``text_table`` (a {prompt: embedding} file such as the training text table;
    default: ``umt5_null_prompt.pt`` next to the checkpoint, which holds NULL_PROMPT) and otherwise computed with the
    UMT5 encoder of ``model_dir`` (loaded on first use). The lookup is the path of the paper's evaluation: it avoids
    the 11 GB encoder, and the evaluation clips are reproduced bit-exactly only if no other model has run on the GPU
    in the same process before the first sampling call (e.g. running UMT5 first changes cuDNN's attention kernel
    choice, a bf16-rounding-level difference).
    """

    def __init__(self, ckpt: str, model_dir: str, device: str = "cuda", text_table: str | dict | None = None, **kw):
        self.device = device
        self.pipe = load_pipeline(model_dir, device, vae=True)
        self.dit = self.pipe.dit
        self.meta = load_checkpoint(self.dit, ckpt)
        self.dit.eval().requires_grad_(False)
        self.mask_channel = hasattr(self.dit, MASK_EMBEDDING)
        self.shift = meta_float(self.meta, "cns_shift", SHIFT)
        self.cns = ({"shift_known": meta_float(self.meta, "cns_shift_known"), "alpha": meta_float(self.meta, "cns_alpha", 1.0)}
                    if meta_bool(self.meta, "cns") else None)
        if text_table is None:
            shipped = os.path.join(os.path.dirname(os.path.abspath(ckpt)), NULL_PROMPT_FILE)
            text_table = shipped if os.path.isfile(shipped) else None
        self.prompts = PromptEmbeddings(model_dir, device, text_table)

    @torch.no_grad()
    def __call__(self, first_frame: np.ndarray, frame_offset_px: np.ndarray, prompt: str = NULL_PROMPT, seed: int = 0,
                 steps: int = STEPS, shift: float | None = None, cns: bool | None = None) -> tuple[np.ndarray, float]:
        z0 = encode_first_frame(self.pipe, first_frame)
        context = self.prompts(prompt)
        offset = cell_offsets(frame_offset_px).unsqueeze(0).to(self.device)
        mask = known_input(offset) if self.mask_channel else None
        use_cns = self.cns is not None if cns is None else bool(cns)
        if use_cns and self.cns is None:
            raise ValueError("cns=True needs a checkpoint trained with CNS (no cns_shift_known in the metadata)")
        latents = sample(self.dit, z0, context, offset, mask, steps=steps, shift=self.shift if shift is None else shift,
                         seed=seed, cns=self.cns if use_cns else None)
        return decode(self.pipe, latents), float(FPS)

    def eval_frames(self, frames: np.ndarray) -> np.ndarray:
        return frames


# ----------------------------------------------------------------------------- latents
class LatentEncoder:
    """81 uint8 frames -> {"latent": bf16 [48, 21, 30, 52]} with the Wan2.2 VAE in fp32 (not tiled)."""

    def __init__(self, model_dir: str, device: str = "cuda"):
        from diffsynth.pipelines.wan_video import ModelConfig, WanVideoPipeline
        self.device = device
        self.pipe = WanVideoPipeline.from_pretrained(torch_dtype=torch.float32, device=device, tokenizer_config=None,
                                                     model_configs=[ModelConfig(path=os.path.join(model_dir, VAE_FILE))],
                                                     redirect_common_files=False)

    @torch.no_grad()
    def __call__(self, frames: np.ndarray, clip=None) -> dict:
        x = torch.from_numpy(np.array(frames, dtype=np.uint8)).permute(3, 0, 1, 2).float().div_(127.5).sub_(1.0)
        z = self.pipe.vae.encode([x], device=self.device, tiled=False, **VAE_TILE)[0]
        if tuple(z.shape) != LATENT_SHAPE:
            raise ValueError(f"latent shape {tuple(z.shape)}, expected {LATENT_SHAPE} (frames must be 81 x 480 x 832)")
        return {"latent": z.to(torch.bfloat16).cpu()}


# ----------------------------------------------------------------------------- training
class LatentDataset(torch.utils.data.Dataset):
    """Pre-computed latents + per-latent-frame scene prompts (looked up in the text table) + camera offsets."""

    def __init__(self, clips, latent_dir: str, table: dict, scene_dropout: float = 0.0):
        self.clips, self.latent_dir, self.table = clips, latent_dir, table
        self.scene_dropout = float(scene_dropout)
        # prompt identity -> id: latent frames with equal ids share one text embedding in the forward pass
        self.prompt_id = {k: i for i, k in enumerate(sorted(table))}
        missing = sorted({p for c in clips for p in c.prompts} - set(table))
        if missing:
            raise KeyError(f"{len(missing)} prompts are missing from the text table, e.g. {missing[0]!r}")
        if self.scene_dropout > 0 and NULL_PROMPT not in table:
            raise KeyError(f"scene dropout needs {NULL_PROMPT!r} in the text table")

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, i):
        c = self.clips[i]
        latent = torch.load(os.path.join(self.latent_dir, c.clip + ".pt"), map_location="cpu", weights_only=False)["latent"]
        if tuple(latent.shape) != LATENT_SHAPE:
            raise ValueError(f"{c.clip}: latent shape {tuple(latent.shape)}, expected {LATENT_SHAPE}")
        drop = self.scene_dropout > 0 and random.random() < self.scene_dropout   # scene dropout: whole clip
        prompts = [NULL_PROMPT] * len(c.prompts) if drop else c.prompts
        return {"clip": c.clip,
                "latent": latent.to(torch.bfloat16).contiguous(),
                "context": torch.stack([self.table[p] for p in prompts], 0).to(torch.bfloat16),
                "context_ids": torch.tensor([self.prompt_id[p] for p in prompts], dtype=torch.int64),
                "offset_px": cell_offsets(c.frame_offset_px)}


def collate(samples: list[dict]) -> dict:
    return {k: torch.stack([s[k] for s in samples], 0) if torch.is_tensor(samples[0][k]) else [s[k] for s in samples]
            for k in samples[0]}


class TrainModule(nn.Module):
    """DiT + ActionRoPE/CNS flow-matching loss; ``forward`` returns (loss, stats) as required by DeepSpeed."""

    def __init__(self, dit, mask_channel: bool = True, new_weight: float = 2.0, cns: bool = True, cns_shift: float = SHIFT,
                 cns_shift_known: float = 2.0, cns_alpha: float = 1.0, gradient_checkpointing: bool = True):
        super().__init__()
        from diffsynth.diffusion.flow_match import FlowMatchScheduler
        self.dit = dit
        self.mask_channel, self.new_weight = bool(mask_channel), float(new_weight)
        self.cns, self.cns_shift = bool(cns), float(cns_shift)
        self.cns_shift_known, self.cns_alpha = float(cns_shift_known), float(cns_alpha)
        self.gradient_checkpointing = bool(gradient_checkpointing)
        self.scheduler = FlowMatchScheduler("Wan")             # sigma = 5 u / (1 + 4 u) over 1000 training steps
        self.scheduler.set_timesteps(1000, training=True)

    def forward(self, batch: dict, timestep_id: int, generator: torch.Generator | None = None, train: bool = True):
        x0 = batch["latent"]                                                    # bf16 [b, 48, 21, 30, 52]
        b, c = x0.shape[:2]
        offset_px = batch["offset_px"].to(x0.device)                           # float32 [b, 21, 2]
        timestep = self.scheduler.timesteps[timestep_id].view(1)
        noise = (torch.randn_like(x0) if generator is None
                 else torch.randn(x0.shape, generator=generator, device=x0.device, dtype=x0.dtype))
        timestep_map = None
        if self.cns:
            # CNS: every latent cell is noised to sigma_tok(sigma, kappa); the target stays eps - x0
            sigma = float(self.scheduler.sigmas[timestep_id])
            kappa, kappa_tok = cns_kappa(offset_px)
            sig = cns_sigma(sigma, kappa, self.cns_shift, self.cns_shift_known, self.cns_alpha).to(x0.dtype)
            x_t = (1.0 - sig) * x0 + sig * noise
            timestep_map = (1000.0 * cns_sigma(sigma, kappa_tok, self.cns_shift, self.cns_shift_known,
                                               self.cns_alpha)).to(x0.dtype)
        else:
            x_t = self.scheduler.add_noise(x0, noise, timestep)
        target = self.scheduler.training_target(x0, noise, timestep)
        x_t[:, :, 0:1] = x0[:, :, 0:1]                                          # latent frame 0 = clean first frame
        weight = loss_weight_map(offset_px, LAT_H, LAT_W, PX_PER_LATENT, self.new_weight)
        mask = known_input(offset_px) if self.mask_channel else None
        pred = arope_forward(self.dit, x_t, timestep.to(device=x0.device, dtype=x0.dtype), batch["context"],
                             offset_px=offset_px, px_per_token=PX_PER_TOKEN, mask_input=mask, timestep_map=timestep_map,
                             context_ids=batch["context_ids"], first_frame_cond=True,
                             use_gradient_checkpointing=self.gradient_checkpointing and train)
        pred, target, weight = pred[:, :, 1:], target[:, :, 1:], weight[:, :, 1:]  # no loss on the first frame
        se = (pred.float() - target.float()) ** 2
        loss_w = (se * weight).sum() / (weight.sum() * c)                      # weight broadcasts over channels
        loss = loss_w * self.scheduler.training_weight(timestep).to(loss_w.device)
        return loss, {"loss_w": loss_w.detach(), "mse": se.mean().detach(), "timestep": float(timestep)}


class TrainTask:
    """Training task for ``actionrope.trainer.train`` (see ``configs/train/wan.yaml``)."""

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self._table = None
        from diffsynth.diffusion.flow_match import FlowMatchScheduler
        self._sched = FlowMatchScheduler("Wan")
        self._sched.set_timesteps(1000, training=True)

    def _method(self) -> dict:
        c = self.cfg
        return dict(mask_channel=bool(c.get("mask_channel", True)), new_weight=float(c.get("new_weight", 2.0)),
                    cns=bool(c.get("cns", True)), cns_shift=float(c.get("cns_shift", SHIFT)),
                    cns_shift_known=float(c.get("cns_shift_known", 2.0)), cns_alpha=float(c.get("cns_alpha", 1.0)))

    def build_module(self) -> nn.Module:
        m = self._method()
        dit = load_pipeline(self.cfg["model_dir"], device="cpu", vae=False).dit
        if m["mask_channel"]:
            install_mask_channel(dit)
        dit.train().requires_grad_(True)
        return TrainModule(dit, gradient_checkpointing=bool(self.cfg.get("gradient_checkpointing", True)), **m)

    def datasets(self):
        c = self.cfg
        if self._table is None:
            self._table = load_text_table(c["text_table"])
        clips = load_annotations(c["annotations"])
        train = training_clips(clips, min_rt_ratio=float(c.get("min_rt_ratio", 0.8)))
        train_ds = LatentDataset(train, c["latent_dir"], self._table, scene_dropout=float(c.get("scene_dropout", 0.1)))
        val_ds = None
        if int(c.get("val_every", 0)) > 0 and int(c.get("val_clips", 16)) > 0:
            val = validation_clips(clips, min_rt_ratio=float(c.get("val_min_rt_ratio", 0.9)), n=int(c.get("val_clips", 16)),
                                   seed=int(c["seed"]))
            val_ds = LatentDataset(val, c["latent_dir"], self._table, scene_dropout=0.0)
        return train_ds, val_ds

    collate = staticmethod(collate)

    def on_prepared(self, module, device):
        # the RoPE tables are plain tensors (not buffers): move them once instead of on every step
        module.dit.freqs = tuple(t.to(device) for t in module.dit.freqs)

    def export(self, state_dict: dict, step: int):
        tensors = {k[len("dit."):]: v for k, v in state_dict.items() if k.startswith("dit.")}
        m = self._method()
        meta = {"model": MODEL_NAME, "format": "full", "mask_channel": m["mask_channel"], "cns": m["cns"],
                "cns_shift": m["cns_shift"], "cns_shift_known": m["cns_shift_known"], "cns_alpha": m["cns_alpha"]}
        return tensors, meta

    def timestep_value(self, tid: int) -> float:
        return float(self._sched.timesteps[int(tid)])
