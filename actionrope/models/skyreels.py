"""SkyReels-V2-DF-1.3B-540P with ActionRoPE + CNS.

Backbone
--------
SkyReels-V2-DF-1.3B-540P (the diffusion-forcing model of SkyReels-V2) is a Wan2.1 video DiT: DiffSynth's ``WanModel``
with dim 1536, 30 layers, 12 heads, patch (1, 2, 2), 16 latent channels and a UMT5-XXL text encoder, plus one
SkyReels-specific branch, the frame-rate conditioning ``fps_embedding`` / ``fps_projection`` (``inject_sample_info`` in
the official config), which adds a learned vector to the timestep modulation of every token. The DiT is built
directly (DiffSynth's model loader does not know the frame-rate branch) and the official fp32 weights are loaded
strictly. The diffusion-forcing ability of the model (a noise level per frame) is used only through per-token
timesteps: latent frame 0 is the clean first frame (t = 0) and CNS gives every token its own noise level. All 21 latent
frames are denoised jointly with bidirectional attention; there is no autoregressive rollout.

Geometry
--------
The Wan2.1 VAE compresses 8x in space and 4x in time: 81 frames of 832x480 become latents [16, 21, 60, 104], and the
patch (1, 2, 2) gives 30 x 52 tokens per latent frame. ActionRoPE therefore uses s = 16 pixels per token, and the
known/new map (mask channel, loss weights, CNS knownness) lives on the 60 x 104 latent grid with 8 pixels per cell.
Per-frame camera offsets are mapped to latent frames with ``geometry.frames_to_cells`` (mean over the 4 frames of a
latent frame).

Training (``TrainTask``, ``configs/train/skyreels.yaml``)
--------------------------------------------------------
Full fine-tuning of the DiT (every parameter, including the mask channel and the frame-rate branch) from the official
weights. Flow matching on DiffSynth's Wan schedule (shift 5, 1000 training timesteps, one uniformly drawn index per
micro-batch, DiffSynth's timestep loss weight) with the velocity target noise - x0. With CNS the sample is noised per
latent cell, x_t = (1 - sigma_t) x0 + sigma_t noise with sigma_t = cns_sigma(sigma, kappa), and the DiT receives the
per-token timesteps 1000 * cns_sigma(sigma, kappa_tok), where kappa_tok is kappa average-pooled to the token grid.
Latent frame 0 is clean and excluded from the loss; newly revealed cells weigh ``new_weight`` = 2. Every latent frame
cross-attends to its own scene prompt (pre-computed UMT5 embeddings from the text table); with probability
``scene_dropout`` = 0.1 the whole clip uses NULL_PROMPT instead.

Frame-rate branch
-----------------
The released checkpoint was trained with the frame-rate branch active: index 0, which the official SkyReels pipeline
uses for 16 fps (the frame rate of our data; every other rate maps to index 1), added to the timestep modulation of
every token. The evaluation of the paper sampled WITHOUT this term. ``Generator`` therefore defaults to
``fps_conditioning=False``, which reproduces the paper's videos; ``fps_conditioning=True`` adds the term as in training.

Sampling (``Generator``)
------------------------
Paper settings: 50 Euler steps on the Wan flow-matching schedule with shift 5, no classifier-free guidance,
NULL_PROMPT, initial noise drawn on the CPU (``torch.Generator().manual_seed(seed)``, [1, 16, 21, 60, 104], cast to
bf16), the encoded first frame written into latent frame 0 after every step, bf16 DiT. With CNS (read from the
checkpoint metadata) every latent cell follows its own noise level, x <- x + v * (sigma_t(next) - sigma_t(now)),
computed in fp32. The evaluation videos were written with libx264 at CRF 16 (``WRITE_KW``). Prompt embeddings are
looked up in ``umt5_null_prompt.pt`` next to the checkpoint (NULL_PROMPT, the text-table entry the evaluation used) or
in ``text_table=``; other prompts are encoded with UMT5 (``models/umt5.py``), which reproduces the table entries exactly.

Model files (``model_dir`` = a local copy of the SkyReels-V2-DF-1.3B-540P release): ``model.safetensors`` (training
only), ``Wan2.1_VAE.pth``, ``models_t5_umt5-xxl-enc-bf16.pth`` and ``google/umt5-xxl`` (prompt encoder; the same files as
in the Wan2.2-TI2V-5B release).
"""
from __future__ import annotations

import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..checkpoint import meta_bool, meta_float, read_metadata
from ..cns import cns_sigma
from ..coords import knownness, loss_weight_map
from ..data import load_annotations, load_text_table, training_clips, validation_clips
from ..geometry import FPS, FRAMES, HEIGHT, WIDTH, frames_to_cells
from ..prompts import NULL_PROMPT
from .wan_dit import MASK_EMBEDDING, arope_forward, install_mask_channel

MODEL_NAME = "skyreels-v2-df-1.3b"
WRITE_KW = {"crf": 16}

DIT_FILE = "model.safetensors"
VAE_FILE = "Wan2.1_VAE.pth"
NULL_PROMPT_FILE = "umt5_null_prompt.pt"      # {NULL_PROMPT: UMT5 embedding}, shipped with the checkpoint

# WanModel configuration of SkyReels-V2-DF-1.3B-540P (its config.json; text-to-video DiT without image input).
DIT_CONFIG = dict(dim=1536, in_dim=16, ffn_dim=8960, out_dim=16, text_dim=4096, freq_dim=256, eps=1e-6,
                  patch_size=(1, 2, 2), num_heads=12, num_layers=30, has_image_input=False)
FPS_KEYS = ("fps_embedding.weight", "fps_projection.0.weight", "fps_projection.0.bias",
            "fps_projection.2.weight", "fps_projection.2.bias")
MASK_KEYS = (f"{MASK_EMBEDDING}.weight", f"{MASK_EMBEDDING}.bias")

LATENT_SHAPE = (16, 21, 60, 104)     # channels, latent frames, height, width of an 81-frame 832x480 clip
PX_PER_LATENT = 8                    # Wan2.1 VAE spatial stride
PX_PER_TOKEN = 16                    # VAE stride x DiT patch
SHIFT = 5.0                          # shift of the Wan flow-matching schedule (training and sampling)
FPS_INDEX = 0                        # frame-rate bucket of 16 fps
STEPS = 50


# ----------------------------------------------------------------------------- model
def install_fps_branch(dit: nn.Module) -> nn.Module:
    """SkyReels' frame-rate conditioning: ``fps_embedding`` (2 buckets) and ``fps_projection`` (-> 6 x dim), with the
    key names of the official checkpoint."""
    if not hasattr(dit, "fps_embedding"):
        dev = dit.patch_embedding.weight.device
        dit.add_module("fps_embedding", nn.Embedding(2, dit.dim, device=dev))
        dit.add_module("fps_projection", nn.Sequential(nn.Linear(dit.dim, dit.dim, device=dev), nn.SiLU(),
                                                       nn.Linear(dit.dim, dit.dim * 6, device=dev)))
    return dit


def build_dit(mask_channel: bool = True, device="cpu", **config) -> nn.Module:
    """WanModel of SkyReels-V2-DF-1.3B with the frame-rate branch and (optionally) the known/new input channel.
    ``config`` overrides entries of ``DIT_CONFIG`` (used by the tests to build a small model)."""
    from diffsynth.models.wan_video_dit import WanModel
    with torch.device(device):
        dit = WanModel(**{**DIT_CONFIG, **config})
    if mask_channel:
        install_mask_channel(dit)
    return install_fps_branch(dit)


def _load_state_dict(dit: nn.Module, sd: dict, allowed_missing=(), source: str = "") -> None:
    missing, unexpected = dit.load_state_dict(sd, strict=False)
    bad = [k for k in missing if k not in allowed_missing]
    if bad or unexpected:
        raise RuntimeError(f"{source}: state dict does not match the model (missing {bad[:5]} ({len(bad)}), "
                           f"unexpected {list(unexpected)[:5]} ({len(unexpected)}))")


def load_base_weights(dit: nn.Module, model_dir: str) -> None:
    """Official fp32 weights; only the new known/new input channel may be missing (it stays zero-initialised)."""
    from safetensors.torch import load_file
    path = os.path.join(model_dir, DIT_FILE)
    _load_state_dict(dit, load_file(path), MASK_KEYS, path)


def load_checkpoint(dit: nn.Module, path: str) -> dict:
    """ActionRoPE checkpoint (``format`` = full: the complete DiT state dict, loaded strictly) -> its metadata."""
    from safetensors.torch import load_file
    meta = read_metadata(path)
    if meta.get("model", MODEL_NAME) != MODEL_NAME or meta.get("format", "full") != "full":
        raise ValueError(f"{path}: expected a full {MODEL_NAME} checkpoint, got model={meta.get('model')} "
                         f"format={meta.get('format')}")
    _load_state_dict(dit, load_file(path), (), path)
    return meta


def fps_modulation(dit: nn.Module, fps_index: int = FPS_INDEX):
    """``t_mod_fn`` for ``arope_forward``: adds the frame-rate vector to the timestep modulation of every token, as
    the official SkyReels forward does (e0 + fps_projection(fps_embedding(fps)))."""
    def t_mod_fn(t_mod: torch.Tensor) -> torch.Tensor:
        idx = torch.tensor([int(fps_index)], device=t_mod.device, dtype=torch.long)
        mod = dit.fps_projection(dit.fps_embedding(idx)).to(dtype=t_mod.dtype)     # [1, 6 * dim]
        return t_mod + mod.unflatten(1, (6, dit.dim)).unsqueeze(1)                  # broadcast over batch and tokens
    return t_mod_fn


def cell_offsets(frame_offset_px: np.ndarray) -> torch.Tensor:
    """Per-frame camera offsets [81, 2] -> per-latent-frame offsets, float32 [21, 2]."""
    return torch.tensor(frames_to_cells(frame_offset_px), dtype=torch.float32)


def load_vae(model_dir: str, device: str):
    """Wan2.1 VAE of the SkyReels release (bf16)."""
    from diffsynth.pipelines.wan_video import ModelConfig, WanVideoPipeline
    pipe = WanVideoPipeline.from_pretrained(torch_dtype=torch.bfloat16, device=device, redirect_common_files=False,
                                            model_configs=[ModelConfig(path=os.path.join(model_dir, VAE_FILE))],
                                            tokenizer_config=None)
    return pipe.vae


def _single(x: torch.Tensor) -> torch.Tensor:
    x = x[0] if isinstance(x, (list, tuple)) else x
    return x.squeeze(0) if x.dim() == 5 else x


# ----------------------------------------------------------------------------- training
class SkyReelsTrainModule(nn.Module):
    """``module(batch, timestep_id, generator=None, train=True) -> (loss, stats)`` for ``actionrope.trainer``."""

    def __init__(self, dit: nn.Module, mask_channel: bool = True, new_weight: float = 2.0, shift: float = SHIFT,
                 cns: bool = True, cns_shift_known: float = 2.0, cns_alpha: float = 1.0, fps_index: int | None = FPS_INDEX,
                 gradient_checkpointing: bool = True):
        super().__init__()
        from diffsynth.diffusion.flow_match import FlowMatchScheduler
        self.dit = dit
        self.mask_channel, self.new_weight = bool(mask_channel), float(new_weight)
        self.shift, self.cns = float(shift), bool(cns)
        self.cns_shift_known, self.cns_alpha = float(cns_shift_known), float(cns_alpha)
        self.gradient_checkpointing = bool(gradient_checkpointing)
        self.t_mod_fn = None if fps_index is None else fps_modulation(dit, fps_index)
        self.scheduler = FlowMatchScheduler("Wan")
        self.scheduler.set_timesteps(1000, training=True, shift=self.shift)

    def forward(self, batch: dict, timestep_id: int, generator: torch.Generator | None = None, train: bool = True):
        x0 = batch["latent"]                                  # bf16 [b, 16, 21, 60, 104]
        off = batch["offset_px"]                              # float32 [b, 21, 2], latent frame 0 = (0, 0)
        b, c, _, h, w = x0.shape
        timestep = self.scheduler.timesteps[timestep_id].view(1)
        if generator is None:
            noise = torch.randn_like(x0)
        else:
            noise = torch.randn(x0.shape, generator=generator, device=x0.device, dtype=x0.dtype)
        timestep_map = None
        if self.cns:
            sigma = float(self.scheduler.sigmas[timestep_id])
            kappa = knownness(off, h, w, PX_PER_LATENT)                                   # [b, 1, 21, 60, 104]
            sig = cns_sigma(sigma, kappa, self.shift, self.cns_shift_known, self.cns_alpha).to(x0.dtype)
            x_t = (1.0 - sig) * x0 + sig * noise
            kappa_tok = F.avg_pool2d(kappa[:, 0], kernel_size=2)                         # [b, 21, 30, 52]
            timestep_map = (1000.0 * cns_sigma(sigma, kappa_tok, self.shift, self.cns_shift_known,
                                               self.cns_alpha)).to(x0.dtype)
        else:
            x_t = self.scheduler.add_noise(x0, noise, timestep)
        target = self.scheduler.training_target(x0, noise, timestep)
        x_t[:, :, 0:1] = x0[:, :, 0:1]                         # clean first frame (its timestep is 0 in the DiT)

        weight = loss_weight_map(off, h, w, PX_PER_LATENT, self.new_weight)
        mask_input = knownness(off, h, w, PX_PER_LATENT) if self.mask_channel else None
        pred = arope_forward(self.dit, x_t, timestep.to(device=x0.device, dtype=x0.dtype), batch["context"],
                             offset_px=off, px_per_token=PX_PER_TOKEN, mask_input=mask_input,
                             timestep_map=timestep_map, context_ids=batch.get("context_ids"), first_frame_cond=True,
                             t_mod_fn=self.t_mod_fn, use_gradient_checkpointing=self.gradient_checkpointing and train)
        pred, target, weight = pred[:, :, 1:], target[:, :, 1:], weight[:, :, 1:]   # frame 0 is not a target
        se = (pred.float() - target.float()) ** 2
        loss_w = (se * weight).sum() / (weight.sum() * c)
        loss = loss_w * self.scheduler.training_weight(timestep).to(loss_w.device)
        return loss, {"loss_w": loss_w.detach(), "mse": se.mean().detach()}


class LatentDataset(torch.utils.data.Dataset):
    """Pre-computed latents + per-latent-frame offsets + per-latent-frame scene prompt embeddings."""

    def __init__(self, clips, latent_dir: str, text_table: dict, scene_dropout: float = 0.0):
        self.clips, self.latent_dir, self.table = list(clips), latent_dir, text_table
        self.scene_dropout = float(scene_dropout)
        self.key_to_id = {k: i for i, k in enumerate(sorted(text_table))}
        missing = sorted({p for c in self.clips for p in c.prompts} - set(text_table))
        if missing:
            raise KeyError(f"{len(missing)} prompts are not in the text table, e.g. {missing[0]!r}")
        self.offsets = [cell_offsets(c.frame_offset_px) for c in self.clips]

    def __len__(self) -> int:
        return len(self.clips)

    def __getitem__(self, i: int) -> dict:
        clip = self.clips[i]
        latent = torch.load(os.path.join(self.latent_dir, clip.clip + ".pt"), map_location="cpu",
                            weights_only=True)["latent"]
        if tuple(latent.shape) != LATENT_SHAPE:
            raise ValueError(f"{clip.clip}: latent shape {tuple(latent.shape)}, expected {LATENT_SHAPE}")
        drop = self.scene_dropout > 0 and random.random() < self.scene_dropout
        prompts = [NULL_PROMPT] * LATENT_SHAPE[1] if drop else list(clip.prompts)
        return {"clip": clip.clip,
                "latent": latent.to(torch.bfloat16).contiguous(),
                "context": torch.stack([self.table[p] for p in prompts], 0).to(torch.bfloat16),
                "context_ids": torch.tensor([self.key_to_id[p] for p in prompts], dtype=torch.int64),
                "offset_px": self.offsets[i].clone()}


class TrainTask:
    """Training task for ``actionrope.trainer.train`` (configuration: ``configs/train/skyreels.yaml``)."""

    def __init__(self, cfg: dict):
        from diffsynth.diffusion.flow_match import FlowMatchScheduler
        self.cfg = dict(cfg)
        self._schedule = FlowMatchScheduler("Wan")
        self._schedule.set_timesteps(1000, training=True, shift=float(self.cfg.get("shift", SHIFT)))

    def build_module(self) -> nn.Module:
        cfg = self.cfg
        dit = build_dit(mask_channel=bool(cfg.get("mask_channel", True)))
        load_base_weights(dit, cfg["model_dir"])
        dit.train().requires_grad_(True)
        fps_index = cfg.get("fps_index", FPS_INDEX)
        return SkyReelsTrainModule(
            dit, mask_channel=bool(cfg.get("mask_channel", True)), new_weight=float(cfg.get("new_weight", 2.0)),
            shift=float(cfg.get("shift", SHIFT)), cns=bool(cfg.get("cns", True)),
            cns_shift_known=float(cfg.get("cns_shift_known", 2.0)), cns_alpha=float(cfg.get("cns_alpha", 1.0)),
            fps_index=None if fps_index is None else int(fps_index),
            gradient_checkpointing=bool(cfg.get("gradient_checkpointing", True)))

    def datasets(self):
        cfg = self.cfg
        clips = load_annotations(cfg["annotations"])
        table = load_text_table(cfg["text_table"])
        train = LatentDataset(training_clips(clips, min_rt_ratio=float(cfg.get("min_rt_ratio", 0.8))),
                              cfg["latent_dir"], table, float(cfg.get("scene_dropout", 0.1)))
        val = None
        if int(cfg.get("val_every", 0)) > 0 and int(cfg.get("val_clips", 16)) > 0:
            val_clips = validation_clips(clips, min_rt_ratio=float(cfg.get("val_min_rt_ratio", 0.9)),
                                         n=int(cfg.get("val_clips", 16)), seed=int(cfg.get("seed", 0)))
            val = LatentDataset(val_clips, cfg.get("val_latent_dir") or cfg["latent_dir"], table, 0.0)
        return train, val

    @staticmethod
    def collate(samples: list[dict]) -> dict:
        return {k: torch.stack([s[k] for s in samples], 0) if torch.is_tensor(samples[0][k]) else [s[k] for s in samples]
                for k in samples[0]}

    @staticmethod
    def on_prepared(module: nn.Module, device) -> None:
        # the RoPE tables are plain tensors (not buffers): move them once instead of on every forward
        module.dit.freqs = tuple(t.to(device) for t in module.dit.freqs)

    def export(self, state_dict: dict, step: int):
        cfg = self.cfg
        tensors = {k[len("dit."):]: v.to(torch.bfloat16) for k, v in state_dict.items() if k.startswith("dit.")}
        cns = bool(cfg.get("cns", True))
        shift = float(cfg.get("shift", SHIFT))
        meta = {"model": MODEL_NAME, "format": "full", "mask_channel": bool(cfg.get("mask_channel", True)),
                "cns": cns, "cns_shift": shift, "cns_shift_known": float(cfg.get("cns_shift_known", 2.0)) if cns else shift,
                "cns_alpha": float(cfg.get("cns_alpha", 1.0)) if cns else 0.0}
        if cfg.get("fps_index", FPS_INDEX) is not None:
            meta["fps_index"] = int(cfg.get("fps_index", FPS_INDEX))
        return tensors, meta

    def timestep_value(self, tid: int) -> float:
        return float(self._schedule.timesteps[int(tid)])


# ----------------------------------------------------------------------------- inference
class Generator:
    """First frame + per-frame camera offsets -> 81-frame clip.

    ckpt              ActionRoPE checkpoint (CNS and mask-channel settings are read from its metadata)
    model_dir         SkyReels-V2-DF-1.3B-540P files (the VAE; and the prompt encoder unless ``text_encoder_dir``)
    text_encoder_dir  directory with ``models_t5_umt5-xxl-enc-bf16.pth`` and ``google/umt5-xxl`` (default: model_dir)
    text_table        {prompt: embedding} file or dict whose prompts are looked up instead of being encoded
                      (default: ``umt5_null_prompt.pt`` next to the checkpoint, if present)
    fps_conditioning  add the frame-rate term to the timestep modulation (False = the paper's evaluation)
    """

    def __init__(self, ckpt: str, model_dir: str, device: str = "cuda", text_encoder_dir: str | None = None,
                 text_table: str | dict | None = None, fps_conditioning: bool = False):
        self.device = device
        dit = build_dit(mask_channel=meta_bool(read_metadata(ckpt), "mask_channel", True))
        meta = load_checkpoint(dit, ckpt)
        self.mask_channel = hasattr(dit, MASK_EMBEDDING)
        self.cns = meta_bool(meta, "cns", False)
        self.shift = meta_float(meta, "cns_shift", SHIFT)
        self.shift_known = meta_float(meta, "cns_shift_known", self.shift)
        self.alpha = meta_float(meta, "cns_alpha", 1.0)
        self.fps_index = int(meta_float(meta, "fps_index", FPS_INDEX))
        self.fps_conditioning = bool(fps_conditioning)
        self.dit = dit.to(device=device, dtype=torch.bfloat16).eval().requires_grad_(False)
        self.dit.freqs = tuple(t.to(device) for t in self.dit.freqs)
        self.vae = load_vae(model_dir, device)
        self.text_encoder_dir = text_encoder_dir or model_dir
        if text_table is None:
            shipped = os.path.join(os.path.dirname(os.path.abspath(ckpt)), NULL_PROMPT_FILE)
            text_table = shipped if os.path.isfile(shipped) else None
        self._table = dict(load_text_table(text_table) if isinstance(text_table, str) else (text_table or {}))
        self._encoder = None
        self._cache: dict[str, torch.Tensor] = {}
        self._warm_up()

    @torch.no_grad()
    def _warm_up(self) -> None:
        """One DiT forward at the sampling shapes before the text encoder can run: cuBLAS/cuDNN choose their kernels
        depending on earlier GPU work in the process, so UMT5 running first changes the DiT's kernels and its bf16
        rounding. The paper's evaluation never ran the encoder; with this warm-up the encoder path reproduces its
        videos byte for byte (measured on H200)."""
        dev, (_, f, h, w) = self.device, LATENT_SHAPE
        arope_forward(self.dit, torch.zeros(1, *LATENT_SHAPE, device=dev, dtype=torch.bfloat16),
                      torch.full((1,), 999.0, device=dev, dtype=torch.bfloat16),
                      torch.zeros(1, 512, self.dit.text_embedding[0].in_features, device=dev, dtype=torch.bfloat16),
                      offset_px=torch.zeros(1, f, 2, device=dev), px_per_token=PX_PER_TOKEN,
                      mask_input=torch.ones(1, 1, f, h, w, device=dev) if self.mask_channel else None,
                      timestep_map=torch.full((1, f, h // 2, w // 2), 999.0, device=dev, dtype=torch.bfloat16)
                      if self.cns else None)

    def prompt_embedding(self, prompt: str) -> torch.Tensor:
        """bf16 [512, 4096] UMT5 embedding of ``prompt`` (text table entry if available, else encoded)."""
        if prompt in self._table:
            return self._table[prompt]
        if prompt not in self._cache:
            if self._encoder is None:
                from .umt5 import UMT5Encoder
                self._encoder = UMT5Encoder(self.text_encoder_dir, self.device)
            self._cache[prompt] = self._encoder([prompt])[0]
        return self._cache[prompt]

    @torch.no_grad()
    def encode_first_frame(self, first_frame: np.ndarray) -> torch.Tensor:
        """uint8 [480, 832, 3] -> clean latent of latent frame 0, bf16 [16, 1, 60, 104]."""
        x = torch.from_numpy(np.ascontiguousarray(first_frame)).permute(2, 0, 1).float() / 127.5 - 1.0
        x = x.unsqueeze(1).unsqueeze(0).to(self.device, torch.bfloat16)          # [1, 3, 1, H, W]
        return _single(self.vae.encode(x, device=self.device))

    @torch.no_grad()
    def __call__(self, first_frame: np.ndarray, frame_offset_px: np.ndarray, prompt: str = NULL_PROMPT, seed: int = 0,
                 steps: int = STEPS, shift: float | None = None, cns: bool | None = None,
                 fps_conditioning: bool | None = None) -> tuple[np.ndarray, float]:
        from diffsynth.diffusion.flow_match import FlowMatchScheduler
        if tuple(first_frame.shape) != (HEIGHT, WIDTH, 3):
            raise ValueError(f"first_frame must be uint8 [{HEIGHT}, {WIDTH}, 3] (see media.load_first_frame)")
        dev = self.device
        shift = self.shift if shift is None else float(shift)
        cns = self.cns if cns is None else bool(cns)
        fps = self.fps_conditioning if fps_conditioning is None else bool(fps_conditioning)
        _, n_lat, h, w = LATENT_SHAPE

        context = self.prompt_embedding(prompt).unsqueeze(0).to(dev, torch.bfloat16)  # one prompt for the clip
        z0 = self.encode_first_frame(first_frame).unsqueeze(0)                     # [1, 16, 1, 60, 104]
        off = cell_offsets(frame_offset_px).to(dev).unsqueeze(0)                   # [1, 21, 2]
        mask_input = knownness(off, h, w, PX_PER_LATENT) if self.mask_channel else None
        kappa = knownness(off, h, w, PX_PER_LATENT) if cns else None
        kappa_tok = F.avg_pool2d(kappa[:, 0], kernel_size=2) if cns else None
        t_mod_fn = fps_modulation(self.dit, self.fps_index) if fps else None

        def sigma_t(s, k):
            return cns_sigma(s, k, shift, self.shift_known, self.alpha)

        sched = FlowMatchScheduler("Wan")
        sched.set_timesteps(int(steps), training=False, shift=shift)
        g = torch.Generator("cpu").manual_seed(int(seed))
        lat = torch.randn(1, *LATENT_SHAPE, generator=g).to(dev, torch.bfloat16)
        lat[:, :, 0:1] = z0
        n = len(sched.timesteps)
        for i, t in enumerate(sched.timesteps):
            s_now = float(sched.sigmas[i])
            s_next = float(sched.sigmas[i + 1]) if i + 1 < n else 0.0
            tmap = (1000.0 * sigma_t(s_now, kappa_tok)).to(torch.bfloat16) if cns else None
            v = arope_forward(self.dit, lat, torch.full((1,), float(t), device=dev, dtype=torch.bfloat16), context,
                              offset_px=off, px_per_token=PX_PER_TOKEN, mask_input=mask_input, timestep_map=tmap,
                              first_frame_cond=True, t_mod_fn=t_mod_fn)
            if cns:
                d_sigma = sigma_t(s_next, kappa) - sigma_t(s_now, kappa)
                lat = (lat.float() + v.float() * d_sigma.float()).to(torch.bfloat16)
            else:
                lat = sched.step(v.float(), t, lat.float()).to(torch.bfloat16)
            lat[:, :, 0:1] = z0
        video = _single(self.vae.decode(lat, device=dev))                          # [3, 81, H, W] in [-1, 1]
        frames = ((video.permute(1, 2, 3, 0).float().clamp(-1, 1) + 1) * 127.5).to(torch.uint8).cpu().numpy()
        return frames, FPS

    def eval_frames(self, frames: np.ndarray) -> np.ndarray:
        return frames


# ----------------------------------------------------------------------------- data preparation
class LatentEncoder:
    """81 RGB frames -> {"latent": bf16 [16, 21, 60, 104]} with the Wan2.1 VAE."""

    def __init__(self, model_dir: str, device: str = "cuda"):
        self.device = device
        self.vae = load_vae(model_dir, device)

    @torch.no_grad()
    def __call__(self, frames: np.ndarray, clip=None) -> dict:
        from PIL import Image
        frames = [np.asarray(Image.fromarray(f).convert("RGB").resize((WIDTH, HEIGHT))) for f in frames[:FRAMES]]
        x = torch.stack([torch.from_numpy(f).permute(2, 0, 1) for f in frames]).float() / 127.5 - 1.0
        x = x.permute(1, 0, 2, 3).unsqueeze(0).to(self.device, torch.bfloat16)      # [1, 3, 81, H, W]
        return {"latent": _single(self.vae.encode(x, device=self.device)).to(torch.bfloat16).cpu()}


def build_text_table(prompts: list[str], model_dir: str, device: str = "cuda") -> dict:
    """UMT5 table shared with Wan2.2 (SkyReels-V2 ships the same encoder files)."""
    from .umt5 import build_text_table as _build
    return _build(prompts, model_dir, device)
