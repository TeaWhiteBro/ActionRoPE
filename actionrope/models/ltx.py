"""LTX-2.5 (22B) with ActionRoPE + CNS: sampling, latent encoding, text table and the training task.

Backbone: Lightricks LTX-2.5 (``ltx-2.5-22b-dev``, joint audio-video DiT, Gemma text encoder), run through
DiffSynth-Studio's ``LTX2AudioVideoPipeline``. ``model_dir`` is a snapshot of Lightricks/LTX-2.5; the tokenizer and
text-encoder post modules come from DiffSynth-Studio/LTX-2.5-Repackage, found in ``<model_dir>/LTX-2.5-Repackage``,
next to ``model_dir`` or given as ``repack_dir``.

Method (see ``ltx_dit.py`` for the DiT side):
  * ActionRoPE: the camera offset of each latent frame is added to the explicit pixel positions of its tokens.
  * Known/new input channel: zero-initialised Linear(1, 4096) on the patch embedding (8,192 parameters).
  * CNS: per-token noise levels sigma_t = cns_sigma(sigma, kappa) with S = e^2.05 (the LTX-2 base shift) and
    S_known = 3.1 (S_known / S = 0.4 as on Wan2.2), for training and sampling.
  * Text: scene prompts only, one per latent frame (latent frame k uses the prompt of Wan-family cell 2k); every
    latent frame cross-attends only to its own prompt. Scene dropout 0.1 during training; NULL_PROMPT at evaluation.
  * First frame: LTX's own image-to-video conditioning (clean latent frame 0 at timestep 0 plus the LTX-2.5 keyframe
    embedding); latent frame 0 is excluded from the loss; newly revealed tokens get loss weight 2.
  * Fine-tuning: LoRA rank 128 / alpha 128 on the 480 video attention and FFN linears, plus the mask channel.

Checkpoints (``actionrope.checkpoint``): ``format = full`` stores the complete DiT (4,093 bf16 tensors, LoRA merged)
under the parameter names of the LTX-2.5 base checkpoint (without its ``model.diffusion_model.`` prefix) plus
``arope_mask_embedding.{weight,bias}``; ``format = lora`` stores only the LoRA factors (``<layer>.lora_A`` /
``<layer>.lora_B``) and the mask channel, merged into the base weights at load time.
"""
from __future__ import annotations

import os
import random

import numpy as np
import torch
import torch.nn as nn

from ..checkpoint import meta_bool, meta_float, read_metadata
from ..cns import cns_sigma
from ..coords import known_mask, knownness, loss_weight_map
from ..data import load_annotations, load_text_table, training_clips, validation_clips
from ..geometry import FPS, FRAMES, HEIGHT, WIDTH
from ..lora import install_lora, lora_state_dict, merged_state_dict
from ..prompts import NULL_PROMPT
from .ltx_dit import (DTYPE, LATENT_CHANNELS, LATENT_FRAMES, LATENT_H, LATENT_W, MASK_EMBEDDING, PX_PER_TOKEN, SHIFT,
                      TOKENIZER_DIR, CNSTimesteps, clean_key, clean_to_model_keys, install_cns_step,
                      install_mask_channel, install_per_frame_text, install_repeated_prompt, install_world_positions,
                      is_lora_target, latent_frame_offsets, latent_grid, load_pipeline, offsets_tensor,
                      resolve_repack_dir, set_text_frames, world_positions)

MODEL_NAME = "ltx-2.5-22b"
WRITE_KW = {"crf": 16}
# Wan-family cell whose prompt each latent frame uses (cell 2k covers the pixel frames of latent frame k).
LATENT_FRAME_CELLS = tuple(range(0, 21, 2))
MASK_KEYS = (f"{MASK_EMBEDDING}.weight", f"{MASK_EMBEDDING}.bias")
LORA_TARGETS = ["attn1.to_q", "attn1.to_k", "attn1.to_v", "attn1.to_out.0", "attn2.to_q", "attn2.to_k",
                "attn2.to_v", "attn2.to_out.0", "ff.net.0.proj", "ff.net.2"]

# Paper sampling settings.
STEPS = 30
CFG_SCALE = 3.0


def _negative_prompt(pipe) -> str:
    return pipe.default_negative_prompt["LTX-2.5"]


# ----------------------------------------------------------------------------- checkpoint I/O
@torch.no_grad()
def load_checkpoint_into(dit: nn.Module, path: str) -> dict:
    """Load an ActionRoPE checkpoint into a DiffSynth-built DiT (mask channel already installed). ``format = full``
    replaces every tensor; ``format = lora`` merges B A * alpha / rank into the base weights (fp32, on the CPU, as
    the full export does). Asserts that no key is missing or unexpected. Returns the metadata."""
    from safetensors.torch import load_file
    meta = read_metadata(path)
    sd = load_file(path)
    names = clean_to_model_keys(dit)
    if meta.get("format", "full") == "lora":
        own = dit.state_dict()
        rank, alpha = int(meta["lora_rank"]), float(meta.get("lora_alpha", meta["lora_rank"]))
        layers = sorted({k.rsplit(".", 1)[0] for k in sd if k.endswith((".lora_A", ".lora_B"))})
        rest = [k for k in sd if not k.endswith((".lora_A", ".lora_B"))]
        bad = [k for k in rest if k not in MASK_KEYS or k not in names]
        bad += [k for k in MASK_KEYS if k in names and k not in sd]
        bad += [f"{layer}.weight" for layer in layers if f"{layer}.weight" not in names]
        if bad:
            raise RuntimeError(f"{path}: unexpected or missing keys {bad[:5]} ({len(bad)})")
        new = {}
        for layer in layers:
            key = names[layer + ".weight"]
            w = own[key].detach().cpu()
            A, B = sd[layer + ".lora_A"], sd[layer + ".lora_B"]
            new[key] = (w.float() + (B.float() @ A.float()) * (alpha / rank)).to(w.dtype)
        new.update({names[k]: sd[k] for k in rest})
        dit.load_state_dict(new, strict=False)
    else:
        missing, unexpected = sorted(set(names) - set(sd)), sorted(set(sd) - set(names))
        if missing or unexpected:
            raise RuntimeError(f"{path}: missing {missing[:5]} ({len(missing)}), unexpected {unexpected[:5]} ({len(unexpected)})")
        dit.load_state_dict({names[k]: v for k, v in sd.items()}, strict=True)
    return meta


# ----------------------------------------------------------------------------- sampling
class Generator:
    """First frame + per-frame camera offsets -> 81-frame clip (832x480, 16 fps).

    Defaults are the paper's settings: 30 steps, CFG 3.0 with DiffSynth's default LTX-2.5 negative prompt,
    NULL_PROMPT for all 11 latent frames, single stage (no upsampler), tiled VAE decoding, and the joint audio-video
    DiT as in the evaluation (only the video is returned). The CNS settings come from the checkpoint metadata.
    ``vram_limit`` (GB) streams the DiT layer by layer when the GPU is shared (DiffSynth VRAM management)."""

    def __init__(self, ckpt: str, model_dir: str, device: str = "cuda", repack_dir: str | None = None,
                 vram_limit: float | None = None):
        meta = read_metadata(ckpt)
        if meta.get("model", MODEL_NAME) != MODEL_NAME:
            raise ValueError(f"{ckpt} is a {meta.get('model')} checkpoint, not {MODEL_NAME}")
        self.device = device
        self.pipe = pipe = load_pipeline(model_dir, ("text", "dit", "video_vae", "audio_vae"), device, repack_dir, vram_limit)
        self.mask_channel = meta_bool(meta, "mask_channel", True)
        self.cns = meta_bool(meta, "cns", False)
        self.cns_kw = dict(shift=meta_float(meta, "cns_shift", SHIFT), shift_known=meta_float(meta, "cns_shift_known"),
                           alpha=meta_float(meta, "cns_alpha", 1.0))
        if self.mask_channel:
            install_mask_channel(pipe.dit)
        load_checkpoint_into(pipe.dit, ckpt)
        if vram_limit is None:
            pipe.dit.to(device=pipe.device, dtype=pipe.torch_dtype)
        elif self.mask_channel:
            getattr(pipe.dit, MASK_EMBEDDING).to(device=pipe.device, dtype=pipe.torch_dtype)
        _, _, self.lf_idx = latent_grid()
        self.state = {"frame_offset_px": None, "kappa": None, "text_frames": LATENT_FRAMES}
        install_per_frame_text(pipe.dit)
        set_text_frames(pipe.dit, LATENT_FRAMES)
        install_repeated_prompt(pipe, self.state)
        install_world_positions(pipe, self.state)
        if self.cns:
            from diffsynth.pipelines.ltx2_audio_video import model_fn_ltx2
            state, kw = self.state, self.cns_kw

            def model_fn(dit=None, **inputs):
                kap = state["kappa"]
                if kap is not None:
                    dit = CNSTimesteps(dit, pipe.video_patchifier.patchify(kap.to(torch.float32)), **kw)
                return model_fn_ltx2(dit=dit, **inputs)

            pipe.model_fn = model_fn
            install_cns_step(pipe.scheduler, self.state, **kw)

    @torch.no_grad()
    def __call__(self, first_frame: np.ndarray, frame_offset_px: np.ndarray, prompt: str = NULL_PROMPT, seed: int = 0,
                 steps: int = STEPS, cfg_scale: float = CFG_SCALE, negative_prompt: str | None = None,
                 tiled: bool = True) -> tuple[np.ndarray, float]:
        from PIL import Image
        pipe = self.pipe
        off = offsets_tensor(frame_offset_px)
        off_lat = latent_frame_offsets(off).unsqueeze(0)                      # [1, 11, 2] px
        self.state["frame_offset_px"] = off
        if self.mask_channel:
            known = known_mask(off_lat, LATENT_H, LATENT_W, PX_PER_TOKEN)
            pipe.dit._arope_mask_patch = known.to(torch.float32).reshape(1, -1, 1).to(device=pipe.device, dtype=pipe.torch_dtype)
        if self.cns:
            self.state["kappa"] = knownness(off_lat, LATENT_H, LATENT_W, PX_PER_TOKEN).to(pipe.device)
        try:
            video, _ = pipe(prompt=prompt, negative_prompt=_negative_prompt(pipe) if negative_prompt is None else negative_prompt,
                            seed=seed, height=HEIGHT, width=WIDTH, num_frames=FRAMES, frame_rate=FPS,
                            cfg_scale=cfg_scale, num_inference_steps=steps, tiled=tiled,
                            input_images=[Image.fromarray(np.asarray(first_frame, np.uint8))],
                            input_images_indexes=[0], input_images_strength=1.0)
        finally:
            self.state.update(frame_offset_px=None, kappa=None)
            pipe.dit._arope_mask_patch = None
        return np.stack([np.asarray(f) for f in video], 0), FPS

    def eval_frames(self, frames: np.ndarray) -> np.ndarray:
        return frames


# ----------------------------------------------------------------------------- latents and text
class LatentEncoder:
    """81 RGB frames -> {"latent": bf16 [128, 11, 15, 26]} with the LTX-2.5 video VAE encoder (untiled)."""

    def __init__(self, model_dir: str, device: str = "cuda", repack_dir: str | None = None):
        self.pipe = load_pipeline(model_dir, ("video_vae",), device, repack_dir)
        self.pipe.load_models_to_device(["video_vae_encoder"])

    @torch.no_grad()
    def __call__(self, frames: np.ndarray, clip=None) -> dict:
        from PIL import Image
        imgs = [Image.fromarray(np.asarray(f, np.uint8)).convert("RGB").resize((WIDTH, HEIGHT)) for f in frames[:FRAMES]]
        x = self.pipe.preprocess_video(imgs)
        lat = self.pipe.video_vae_encoder.encode(x, False, 512, 128)[0].to(torch.bfloat16).cpu()
        assert tuple(lat.shape) == (LATENT_CHANNELS, LATENT_FRAMES, LATENT_H, LATENT_W), lat.shape
        return {"latent": lat}


@torch.no_grad()
def build_text_table(prompts: list[str], model_dir: str, device: str = "cuda", repack_dir: str | None = None) -> dict:
    """{prompt: fp16 [1024, 4096]} video-branch prompt embeddings (Gemma encoder + LTX-2.5 post modules).

    Prompts are tokenised with DiffSynth's LTX-2 Gemma tokenizer wrapper, which does not prepend a BOS token. This is
    what DiffSynth's text-only LTX pipeline uses, and the embeddings the released checkpoint was trained with were
    computed this way. (The sampling pipeline, which also loads the DiT, uses the LTX-2.5 wrapper that prepends BOS;
    the Generator keeps that behaviour of the paper's evaluation.)"""
    from diffsynth.models.ltx2_text_encoder import LTXVGemmaTokenizer
    from diffsynth.pipelines.ltx2_audio_video import LTX2AudioVideoUnit_PromptEmbedder
    pipe = load_pipeline(model_dir, ("text",), device, repack_dir)
    pipe.tokenizer = LTXVGemmaTokenizer(os.path.join(resolve_repack_dir(model_dir, repack_dir), TOKENIZER_DIR))
    pipe.load_models_to_device(("text_encoder", "text_encoder_post_modules"))
    unit = LTX2AudioVideoUnit_PromptEmbedder()
    table = {}
    for i, p in enumerate(sorted(set(prompts))):
        v, _, _ = unit.encode_prompt(pipe, p)
        table[p] = v[0].to(torch.float16).cpu()
        if (i + 1) % 50 == 0:
            print(f"[text] {i + 1} prompts", flush=True)
    return table


# ----------------------------------------------------------------------------- training
class LTXDataset(torch.utils.data.Dataset):
    """Training / validation samples: latent, 11 per-latent-frame scene prompt embeddings, camera offsets."""

    def __init__(self, clips, latent_dir: str, text_table: dict, scene_dropout: float = 0.0):
        self.clips, self.latent_dir, self.table, self.scene_dropout = list(clips), latent_dir, text_table, float(scene_dropout)
        missing = [c for c in self.clips for p in (c.prompts[k] for k in LATENT_FRAME_CELLS) if p not in text_table]
        if missing or NULL_PROMPT not in text_table:
            raise KeyError(f"text table lacks prompts of {len(missing)} clips (e.g. {missing[0].clip if missing else ''}) "
                           "or NULL_PROMPT; rebuild it with scripts/build_text_table.py --model ltx")

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, i):
        c = self.clips[i]
        d = torch.load(os.path.join(self.latent_dir, c.clip + ".pt"), map_location="cpu", weights_only=False)
        latent = d["latent"]
        if tuple(latent.shape) != (LATENT_CHANNELS, LATENT_FRAMES, LATENT_H, LATENT_W):
            raise ValueError(f"{c.clip}: latent shape {tuple(latent.shape)}")
        # scene dropout: the whole clip switches to NULL_PROMPT
        drop = self.scene_dropout > 0 and random.random() < self.scene_dropout
        keys = [NULL_PROMPT if drop else c.prompts[k] for k in LATENT_FRAME_CELLS]
        return {"clip": c.clip, "latent": latent.to(torch.bfloat16).contiguous(),
                "context": torch.stack([self.table[k] for k in keys], 0).to(torch.bfloat16),   # [11, L, D]
                "frame_offset_px": torch.as_tensor(c.frame_offset_px, dtype=torch.float64)}


def collate(samples: list[dict]) -> dict:
    if len({s["context"].shape[1] for s in samples}) != 1:
        raise ValueError("prompt embeddings of different length cannot be batched")
    out = {k: torch.stack([s[k] for s in samples], 0) for k in ("latent", "context", "frame_offset_px")}
    out["clip"] = [s["clip"] for s in samples]
    return out


class LTXTrainModule(nn.Module):
    """DiT + flow-matching loss: ``module(batch, timestep_id)`` -> (loss, stats).

    Follows DiffSynth's ``model_fn_ltx2`` (patchify -> first-frame conditioning -> DiT -> unpatchify) with world
    positions, the mask channel, per-frame text and CNS-noised inputs; target = noise - x0, loss weighted per token
    (known 1, newly revealed ``new_weight``) over latent frames 1..10 and by the LTX-2 timestep weighting."""

    def __init__(self, dit, mask_channel=True, new_weight=2.0, cns=True, cns_shift_known=3.1, cns_alpha=1.0,
                 gradient_checkpointing=True):
        super().__init__()
        from diffsynth.diffusion.flow_match import FlowMatchScheduler
        from diffsynth.models.ltx2_video_vae import VideoLatentPatchifier
        self.dit = dit
        self.patchifier = VideoLatentPatchifier(patch_size=1)
        pos, patch_idx, lf_idx = latent_grid()
        self.register_buffer("pos", pos, persistent=False)
        self.register_buffer("patch_frame_idx", patch_idx, persistent=False)
        self.register_buffer("latent_frame_pixel_idx", lf_idx, persistent=False)
        self.mask_channel, self.new_weight = bool(mask_channel), float(new_weight)
        self.cns, self.cns_kw = bool(cns), dict(shift=SHIFT, shift_known=float(cns_shift_known), alpha=float(cns_alpha))
        self.gradient_checkpointing = bool(gradient_checkpointing)
        self.scheduler = FlowMatchScheduler("LTX-2")
        self.scheduler.set_timesteps(1000, training=True)

    def forward(self, batch, timestep_id: int, generator=None, train: bool = True, noise=None):
        from diffsynth.pipelines.ltx2_audio_video import model_fn_ltx2
        x0 = batch["latent"]                                                   # bf16 [b, 128, 11, 15, 26]
        b, c, f, h, w = x0.shape
        dev = x0.device
        timestep = self.scheduler.timesteps[timestep_id].view(1)
        if noise is None:
            noise = torch.randn_like(x0) if generator is None else \
                torch.randn(x0.shape, generator=generator, device=generator.device, dtype=x0.dtype).to(dev)
        off = batch["frame_offset_px"].to(dev)                                 # [b, 81, 2] px
        off_lat = off[:, self.latent_frame_pixel_idx.to(dev)]                  # [b, 11, 2] px
        kappa = knownness(off_lat, h, w, PX_PER_TOKEN).to(dev)                 # [b, 1, f, h, w], known = 1
        if self.cns:
            sigma = float(self.scheduler.sigmas[timestep_id])
            sig_tok = cns_sigma(sigma, kappa, **self.cns_kw).to(x0.dtype)
            noisy = (1.0 - sig_tok) * x0 + sig_tok * noise
        else:
            noisy = self.scheduler.add_noise(x0, noise, timestep)
        target = self.scheduler.training_target(x0, noise, timestep)
        positions = world_positions(self.pos.to(dev), self.patch_frame_idx.to(dev), off).float()
        weight = loss_weight_map(off_lat, h, w, PX_PER_TOKEN, self.new_weight).to(dev)
        self.dit._arope_mask_patch = self.patchifier.patchify(kappa.to(x0.dtype)) if self.mask_channel else None
        keyframes = torch.zeros(b, 1, f, h, w, dtype=torch.float32, device=dev)
        keyframes[:, :, 0] = 1.0                                               # LTX-2.5 keyframe embedding on frame 0
        denoise_mask = torch.ones(b, 1, f, h, w, dtype=x0.dtype, device=dev)
        denoise_mask[:, :, 0] = 0.0                                            # frame 0 = clean first-frame latent
        ctx = batch["context"]                                                 # [b, f, L, D]
        set_text_frames(self.dit, f)          # resident: gradient checkpointing re-runs the blocks in backward
        dit = CNSTimesteps(self.dit, self.patchifier.patchify(kappa), **self.cns_kw) if self.cns else self.dit
        pred, _ = model_fn_ltx2(
            dit=dit, video_latents=noisy, video_context=ctx.reshape(b, f * ctx.shape[2], ctx.shape[3]),
            video_positions=positions, video_patchifier=self.patchifier, timestep=timestep.to(device=dev, dtype=x0.dtype),
            video_keyframes_mask=keyframes, input_latents_video=x0, denoise_mask_video=denoise_mask,
            use_gradient_checkpointing=self.gradient_checkpointing and train)
        se = (pred.float() - target.float()) ** 2
        se_w, w_eff = se[:, :, 1:], weight[:, :, 1:]                            # the first frame is not in the loss
        loss_w = (se_w * w_eff).sum() / (w_eff.sum() * c)
        loss = loss_w * self.scheduler.training_weight(timestep).to(loss_w.device)
        return loss, {"loss_w": loss_w.detach(), "mse": se_w.mean().detach()}


class TrainTask:
    """Training task for ``actionrope.trainer`` (configs/train/ltx.yaml)."""

    def __init__(self, cfg: dict):
        from diffsynth.diffusion.flow_match import FlowMatchScheduler
        self.cfg = cfg
        self.scheduler = FlowMatchScheduler("LTX-2")
        self.scheduler.set_timesteps(1000, training=True)
        self.module = None
        self.n_keys = None

    def datasets(self):
        cfg = self.cfg
        clips = load_annotations(cfg["annotations"])
        table = load_text_table(cfg["text_table"])
        train = training_clips(clips, min_rt_ratio=float(cfg.get("min_rt_ratio", 0.8)))
        val = validation_clips(clips, min_rt_ratio=float(cfg.get("val_min_rt_ratio", 0.9)),
                               n=int(cfg.get("val_clips", 16)), seed=int(cfg["seed"]))
        train_ds = LTXDataset(train, cfg["latent_dir"], table, float(cfg.get("scene_dropout", 0.1)))
        val_ds = LTXDataset(val, cfg["latent_dir"], table, 0.0) if val and int(cfg.get("val_every", 0)) > 0 else None
        return train_ds, val_ds

    def collate(self, samples):
        return collate(samples)

    def build_module(self) -> LTXTrainModule:
        cfg = self.cfg
        dit = load_pipeline(cfg["model_dir"], ("dit",), "cpu", cfg.get("repack_dir")).dit
        if cfg.get("mask_channel", True):
            install_mask_channel(dit)
        install_per_frame_text(dit)
        dit.train()
        self.n_keys = len(dit.state_dict())
        rank = int(cfg.get("lora_rank", 128))
        if rank > 0:
            n = install_lora(dit, is_lora_target, rank, float(cfg.get("lora_alpha", rank)),
                             keep_trainable=MASK_KEYS if cfg.get("mask_channel", True) else ())
            print(f"[lora] rank {rank}: {n} linears", flush=True)
        else:
            dit.requires_grad_(True)
        self.module = LTXTrainModule(dit, mask_channel=cfg.get("mask_channel", True),
                                     new_weight=float(cfg.get("new_weight", 2.0)), cns=cfg.get("cns", True),
                                     cns_shift_known=float(cfg.get("cns_shift_known", 3.1)),
                                     cns_alpha=float(cfg.get("cns_alpha", 1.0)),
                                     gradient_checkpointing=cfg.get("gradient_checkpointing", True))
        return self.module

    def timestep_value(self, tid: int) -> float:
        return float(self.scheduler.timesteps[tid])

    def _metadata(self) -> dict:
        cfg = self.cfg
        cns = bool(cfg.get("cns", True))
        meta = {"model": MODEL_NAME, "mask_channel": bool(cfg.get("mask_channel", True)), "cns": cns}
        if cns:
            meta.update(cns_shift=repr(SHIFT), cns_shift_known=float(cfg.get("cns_shift_known", 3.1)),
                        cns_alpha=float(cfg.get("cns_alpha", 1.0)))
        rank = int(cfg.get("lora_rank", 128))
        if rank > 0:
            meta.update(lora_rank=rank, lora_alpha=int(float(cfg.get("lora_alpha", rank))))
        return meta

    def export(self, state_dict: dict, step: int):
        """Gathered state dict of the training module -> (tensors, metadata). ``export_format: full`` (default) writes
        the complete DiT with the LoRA merged (fp32 on the CPU, then bf16); ``lora`` only the LoRA factors and the
        mask channel. Keys are the clean LTX-2.5 parameter names (see ``ltx_dit.clean_key``)."""
        dit = self.module.dit
        sd = {k: v.detach().cpu() for k, v in state_dict.items()}
        meta = self._metadata()
        lora = int(self.cfg.get("lora_rank", 128)) > 0
        if self.cfg.get("export_format", "full") == "lora":
            if not lora:
                raise ValueError("export_format: lora needs lora_rank > 0")
            sd = lora_state_dict(dit, sd, extra=MASK_KEYS)
            meta.update(format="lora", lora_targets=LORA_TARGETS)
        else:
            if lora:
                sd = merged_state_dict(dit, sd)
            meta["format"] = "full"
        out = {}
        for k, v in sd.items():
            if not k.startswith("dit."):
                continue
            ck = clean_key(k[len("dit."):])
            if ck in out:
                raise RuntimeError(f"duplicate key {ck}")
            out[ck] = v.to(DTYPE)
        if meta["format"] == "full" and len(out) != self.n_keys:
            raise RuntimeError(f"export has {len(out)} tensors, the DiT has {self.n_keys}")
        return out, meta


__all__ = ["WRITE_KW", "Generator", "LatentEncoder", "TrainTask", "build_text_table", "LTXTrainModule",
           "LTXDataset", "collate", "load_checkpoint_into"]
