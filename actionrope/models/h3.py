"""MiniMax-H3 33B (image-to-audio-video, FL2VA) with ActionRoPE + CNS: inference, latent encoding and training.

The released model is a LoRA (rank 128 on the attention and MLP layers of the 50 DiT blocks) plus the known/new
input channel; the base weights stay frozen. The ActionRoPE / CNS math on H3's token layout is in ``h3_dit.py``.

Paper settings
--------------
* Frames. H3 generates 17n + 5 frames at a fixed 24 fps clock. Training uses the first 73 frames of every 81-frame
  clip as they are (ActionRoPE expresses motion in pixels, not in seconds). For sampling, the 81-frame action plan is
  compressed in time to 73 frames (same total displacement) and the 73 generated frames span the commanded 5 s
  (14.4 fps); the metrics are computed on these frames resampled to 81 by nearest neighbour (``eval_frames``).
* Text. Qwen3-VL sees the first frame ("<Picture 1>") followed by the prompt. Training uses the scene prompt of the
  clip (cell 0) and NULL_PROMPT with probability 0.1; evaluation always uses NULL_PROMPT.
* Audio. H3 always denoises audio and video jointly. Our clips are silent, so training feeds the latent of a silent
  waveform noised to the same timestep and gives it no loss; sampling denoises the audio from noise with H3's
  audio schedule and discards it.
* Training: shift-2.22 timestep table and loss weighting of DiffSynth's H3 recipe, target noise - x0, loss over all
  22 latent frames with weight 1 / 2 for known / newly revealed tokens, CNS with S = 12 (the sampling shift),
  S_known = 4.8, alpha = 1. Sampling: 30 Euler steps, flow shift 12 (audio 3), no CFG (H3 is guidance-distilled).

Base model files: the FL2VA partition of the MiniMax-H3 release (``text_encoder/``, ``transformer/``,
``video_vae/source/``, ``audio_vae/``, ``processor/``) in ``model_dir``. The text encoder (64 GB) and the DiT (67 GB)
do not fit on one 80-141 GB GPU together; ``Generator`` swaps them unless ``text_device`` names a second GPU.
"""
from __future__ import annotations

import gc
import glob
import hashlib
import os
import random

import numpy as np
import torch
import torch.nn as nn

from ..checkpoint import meta_bool, meta_float, read_metadata
from ..cns import cns_sigma
from ..data import load_annotations, training_clips, validation_clips
from ..geometry import FPS, FRAMES, HEIGHT, WIDTH
from ..lora import install_lora, lora_state_dict
from ..prompts import NULL_PROMPT
from .h3_dit import (AUDIO_C, AUDIO_DIM, LAT_H, LAT_W, LATENT_C, LORA_TARGETS, MASK_EMBED, MASK_KEYS, audio_frames,
                     build_packed, forward, has_mask_channel, install_mask_channel, latent_frame_offsets, latent_frames,
                     lora_target, noise_cond_rows, retime_frames, retime_offsets, token_kappa, world_positions)

MODEL_NAME = "minimax-h3-33b"
WRITE_KW = {"crf": 16}
NUM_FRAMES = 73                 # 17 * 4 + 5
SAMPLE_STEPS = 30
SAMPLE_SHIFT = 12.0             # video flow shift of H3 (also the base shift S of CNS)
AUDIO_SHIFT = 3.0
TRAIN_SHIFT = 2.22              # shift of the 1000-step training timestep table
CNS_SHIFT_KNOWN = 4.8           # = S * 0.4, the S_known / S ratio found on Wan2.2


# ----------------------------------------------------------------------------- base model
def load_pipeline(model_dir: str, components, device: str):
    """DiffSynth's MiniMax-H3 pipeline with only ``components`` (text_encoder, dit, video_vae, audio_vae) loaded in
    bf16 on ``device``."""
    os.environ.setdefault("DIFFSYNTH_SKIP_DOWNLOAD", "True")
    from diffsynth.pipelines.minimax_h3_audio_video import MiniMaxH3Pipeline, ModelConfig
    files = {
        "text_encoder": sorted(glob.glob(os.path.join(model_dir, "text_encoder", "model-*.safetensors"))),
        "dit": sorted(glob.glob(os.path.join(model_dir, "transformer", "model-*.safetensors"))),
        "video_vae": [os.path.join(model_dir, "video_vae", "source", "model.safetensors")],
        "audio_vae": [os.path.join(model_dir, "audio_vae", "model.safetensors")],
    }
    configs = []
    for name in components:
        paths = files[name]
        if not paths or not all(os.path.isfile(p) for p in paths):
            raise FileNotFoundError(f"MiniMax-H3 {name} weights not found under {model_dir}")
        configs.append(ModelConfig(path=paths if len(paths) > 1 else paths[0]))
    return MiniMaxH3Pipeline.from_pretrained(torch_dtype=torch.bfloat16, device=device, model_configs=configs,
                                             processor_config=ModelConfig(path=os.path.join(model_dir, "processor")))


@torch.no_grad()
def encode_prompt(pipe, prompt: str, first_frame: np.ndarray):
    """FL2VA text embedding: Qwen3-VL reads the first frame and then the prompt.
    Returns (embeddings [L, 5120] bf16 on CPU, token tags [L] long: 0 = image tokens, 1 = text tokens)."""
    from PIL import Image
    from diffsynth.models.minimax_h3_text_encoder import image_token_counts, presentation_fl2va
    img = Image.fromarray(np.asarray(first_frame)).convert("RGB").resize((WIDTH, HEIGHT))
    pixel_values, image_grid_thw, counts = image_token_counts(pipe.processor, [img])
    input_ids, tags = presentation_fl2va(pipe.tokenizer, prompt, counts)
    ids = input_ids.unsqueeze(0).to(pipe.device)
    hidden = pipe.text_encoder(input_ids=ids, attention_mask=torch.ones_like(ids),
                               pixel_values=pixel_values.to(pipe.device, pipe.torch_dtype),
                               image_grid_thw=image_grid_thw.to(pipe.device, torch.long))
    return hidden.to(torch.bfloat16).cpu(), tags.view(-1).to(torch.long)


@torch.no_grad()
def encode_first_frame(pipe, first_frame: np.ndarray) -> torch.Tensor:
    """First frame -> clean condition tokens [390, 96] bf16 (single-image VAE encoding, patchified)."""
    from PIL import Image
    from diffsynth.models.minimax_h3_dit import patchify_video
    img = Image.fromarray(np.asarray(first_frame)).convert("RGB").resize((WIDTH, HEIGHT), Image.LANCZOS)
    x = pipe.preprocess_image(img, torch_dtype=torch.float32, min_value=0)
    z = pipe.video_vae.encode_video(x, dtype=pipe.torch_dtype, process_image=True)       # [1, 24, 1, 30, 52]
    return patchify_video(z).to(torch.bfloat16)


@torch.no_grad()
def silent_audio_latent(model_dir: str, num_frames: int, device: str) -> torch.Tensor:
    """Audio latent [2, 32, T] (bf16) of a silent stereo waveform as long as ``num_frames`` video frames."""
    pipe = load_pipeline(model_dir, ("audio_vae",), device)
    n = int(round(num_frames / 24.0 * pipe.audio_vae.sample_rate))
    lat = pipe.audio_vae.encode_audio(torch.zeros(AUDIO_C, n, dtype=torch.float32, device=pipe.device),
                                      dtype=pipe.torch_dtype).to(torch.bfloat16)
    want = audio_frames(num_frames)
    if lat.shape[-1] > want:
        lat = lat[..., :want]
    elif lat.shape[-1] < want:
        lat = torch.cat([lat, lat[..., -1:].repeat(1, 1, want - lat.shape[-1])], -1)
    del pipe
    return lat.contiguous()


def load_lora(dit: nn.Module, path: str) -> dict:
    """Install the known/new channel (if the checkpoint has one) and the LoRA layers, then load the checkpoint."""
    from safetensors.torch import load_file
    meta = read_metadata(path)
    if meta.get("model") != MODEL_NAME or meta.get("format") != "lora":
        raise ValueError(f"{path} is not a {MODEL_NAME} LoRA checkpoint (metadata: {meta})")
    if meta_bool(meta, "mask_channel"):
        install_mask_channel(dit)
    install_lora(dit, lora_target, rank=int(meta["lora_rank"]), alpha=float(meta["lora_alpha"]))
    sd = load_file(path)
    _, unexpected = dit.load_state_dict(sd, strict=False)
    expected = {k for k in dit.state_dict() if k.endswith((".lora_A", ".lora_B")) or k.startswith(MASK_EMBED + ".")}
    if unexpected or expected != set(sd):
        raise ValueError(f"{path}: checkpoint keys do not match the model (unexpected {sorted(unexpected)[:3]}, "
                         f"missing {sorted(expected - set(sd))[:3]})")
    return meta


# ----------------------------------------------------------------------------- sampling
@torch.no_grad()
def sample(pipe, prompt_embeds, text_tags, cond_rows, frame_offset_px, *, seed: int, steps: int = SAMPLE_STEPS,
           flow_shift: float = SAMPLE_SHIFT, audio_shift: float = AUDIO_SHIFT, cns: bool = True,
           cns_shift: float = SAMPLE_SHIFT, cns_shift_known: float = CNS_SHIFT_KNOWN, cns_alpha: float = 1.0):
    """Euler sampling with ActionRoPE positions and per-token CNS steps. Returns video latents [1, 24, f, 30, 52].

    frame_offset_px: [F, 2] offsets of the F = 17n + 5 generated frames (frame 0 = 0). cond_rows: [390, 96] clean
    first-frame tokens. With CNS every video token integrates its own schedule sigma_tok(sigma, kappa); audio follows
    H3's audio schedule and is not returned.
    """
    from diffsynth.models.minimax_h3_dit import patchify_video, unpatchify_video
    device, dtype = pipe.device, pipe.torch_dtype
    num_frames = int(frame_offset_px.shape[0])
    latent_t, audio_t = latent_frames(num_frames), audio_frames(num_frames)
    pipe.scheduler.set_timesteps(steps, shift=flow_shift)
    pipe.scheduler_audio.set_timesteps(steps, shift=audio_shift)
    video = pipe.generate_noise((1, LATENT_C, latent_t, LAT_H, LAT_W), seed=seed, rand_device="cpu", rand_torch_dtype=dtype)
    audio = pipe.generate_noise((AUDIO_C, AUDIO_DIM, audio_t), seed=seed, rand_device="cpu", rand_torch_dtype=dtype)
    # noise of the condition tokens: frame 0 of a second draw with the same seed (as DiffSynth's keyframe encoder)
    noise_c = pipe.generate_noise((1, LATENT_C, latent_t + 1, LAT_H, LAT_W), seed, "cpu", dtype)[:, :, :1]
    cond = noise_cond_rows(cond_rows.to(device, dtype), patchify_video(noise_c).to(device, dtype)).contiguous()

    pe = prompt_embeds.to(device, dtype)
    packed = build_packed(pe.shape[0], text_tags, latent_t, audio_t, device)
    positions, off_lat = world_positions(packed, cond.shape[0], latent_t, torch.as_tensor(frame_offset_px, dtype=torch.float64))
    kappa = token_kappa(off_lat.unsqueeze(0))[0].to(device)
    mask_kappa = kappa if has_mask_channel(pipe.dit) else None

    sig = pipe.scheduler.sigmas
    x_rows = patchify_video(video)
    for i in range(len(pipe.scheduler.timesteps)):
        t_v = pipe.scheduler.timesteps[i].view(1).to(torch.float32)
        t_a = pipe.scheduler_audio.timesteps[i].view(1).to(torch.float32)
        s_i = float(sig[i])
        s_n = float(sig[i + 1]) if i + 1 < len(sig) else 0.0
        st_i = cns_sigma(s_i, kappa, cns_shift, cns_shift_known, cns_alpha) if cns else None
        v_video, v_audio = forward(pipe.dit, unpatchify_video(x_rows, latent_t, LAT_H, LAT_W), audio, packed, pe,
                                   t_v, t_a, cond, positions, sigma_tok=st_i, kappa=mask_kappa)
        v_rows = patchify_video(v_video)
        if cns:
            st_n = cns_sigma(s_n, kappa, cns_shift, cns_shift_known, cns_alpha)
            x_rows = x_rows + v_rows * (st_n - st_i).to(v_rows.dtype).unsqueeze(1)
        else:
            x_rows = x_rows + v_rows * (s_n - s_i)
        audio = pipe.scheduler_audio.step(v_audio, pipe.scheduler_audio.timesteps[i], audio)
    return unpatchify_video(x_rows, latent_t, LAT_H, LAT_W)


@torch.no_grad()
def decode(pipe, latents: torch.Tensor) -> np.ndarray:
    """Video latents -> uint8 frames [F, 480, 832, 3]."""
    out = pipe.video_vae.decode_video(latents, dtype=pipe.torch_dtype, tiled=True, tile_size=256, tile_overlap=64)
    video = pipe.vae_output_to_video(out, min_value=0, max_value=1)
    n = (latents.shape[2] - 2) // 5 * 17 + 5
    return np.stack([np.asarray(im) for im in video])[:n]


class Generator:
    """First frame + per-frame camera offsets -> clip.

    The DiT is loaded on the first call. The text encoder runs on ``text_device`` if given (and stays there);
    otherwise it is loaded on ``device`` while the DiT waits on the CPU. Text embeddings are cached per
    (first frame, prompt); call ``encode_prompts`` first to embed many first frames in one text-encoder session.
    """

    def __init__(self, ckpt: str, model_dir: str, device: str = "cuda", text_device: str | None = None):
        meta = read_metadata(ckpt)
        if meta.get("model") != MODEL_NAME or meta.get("format") != "lora":
            raise ValueError(f"{ckpt} is not a {MODEL_NAME} LoRA checkpoint")
        self.ckpt, self.model_dir, self.device = ckpt, model_dir, device
        self.text_device = text_device if text_device not in (None, device) else None
        self.num_frames = int(meta.get("num_frames", NUM_FRAMES))
        latent_frames(self.num_frames)
        self.cns = meta_bool(meta, "cns")
        self.cns_shift = meta_float(meta, "cns_shift", SAMPLE_SHIFT)
        self.cns_shift_known = meta_float(meta, "cns_shift_known", CNS_SHIFT_KNOWN)
        self.cns_alpha = meta_float(meta, "cns_alpha", 1.0)
        self.pipe = None
        self._dit_on_cpu = False
        self._text_pipe = None
        self._text_cache: dict = {}

    def _gpu_dit(self):
        if self.pipe is None:
            self.pipe = load_pipeline(self.model_dir, ("dit", "video_vae"), self.device)
            load_lora(self.pipe.dit, self.ckpt)
            self.pipe.dit.eval()
        elif self._dit_on_cpu:
            self.pipe.dit.to(self.device)
            self._dit_on_cpu = False
        return self.pipe

    @staticmethod
    def _key(first_frame: np.ndarray, prompt: str):
        return prompt, hashlib.sha1(np.ascontiguousarray(first_frame).tobytes()).hexdigest()

    def encode_prompts(self, first_frames, prompt: str = NULL_PROMPT) -> None:
        todo = [f for f in first_frames if self._key(f, prompt) not in self._text_cache]
        if not todo:
            return
        if self.text_device is None:
            if self.pipe is not None and not self._dit_on_cpu:
                self.pipe.dit.to("cpu")
                self._dit_on_cpu = True
                torch.cuda.empty_cache()
            text_pipe = load_pipeline(self.model_dir, ("text_encoder",), self.device)
        else:
            if self._text_pipe is None:
                self._text_pipe = load_pipeline(self.model_dir, ("text_encoder",), self.text_device)
            text_pipe = self._text_pipe
        for f in todo:
            self._text_cache[self._key(f, prompt)] = encode_prompt(text_pipe, prompt, f)
        if self.text_device is None:
            del text_pipe
            gc.collect()
            torch.cuda.empty_cache()

    def __call__(self, first_frame: np.ndarray, frame_offset_px: np.ndarray, prompt: str = NULL_PROMPT, seed: int = 0,
                 steps: int = SAMPLE_STEPS, flow_shift: float = SAMPLE_SHIFT, audio_shift: float = AUDIO_SHIFT):
        off = np.asarray(frame_offset_px, np.float64)
        if off.shape != (FRAMES, 2):
            raise ValueError(f"frame_offset_px must have shape ({FRAMES}, 2), got {off.shape}")
        first_frame = np.asarray(first_frame, np.uint8)
        if first_frame.shape != (HEIGHT, WIDTH, 3):
            raise ValueError(f"first_frame must be uint8 [{HEIGHT}, {WIDTH}, 3], got {first_frame.shape}")
        self.encode_prompts([first_frame], prompt)
        text, tags = self._text_cache[self._key(first_frame, prompt)]
        pipe = self._gpu_dit()
        cond = encode_first_frame(pipe, first_frame)
        latents = sample(pipe, text, tags, cond, retime_offsets(off, self.num_frames), seed=seed, steps=steps,
                         flow_shift=flow_shift, audio_shift=audio_shift, cns=self.cns, cns_shift=self.cns_shift,
                         cns_shift_known=self.cns_shift_known, cns_alpha=self.cns_alpha)
        frames = decode(pipe, latents)
        # the generated frames span the 5 s of the 81-frame plan
        return frames, FPS * (self.num_frames - 1) / (FRAMES - 1)

    def eval_frames(self, frames: np.ndarray) -> np.ndarray:
        """Generated frames -> the 81 frames of the evaluation protocol (nearest neighbour in time)."""
        return retime_frames(np.asarray(frames), FRAMES)


# ----------------------------------------------------------------------------- latents
class LatentEncoder:
    """Clip -> {"latent", "cond_rows", "text_scene", "tags_scene", "text_null", "tags_null", "scene"}.

    The video latent covers the first 73 frames, the condition tokens encode frame 0, and the two text embeddings
    (Qwen3-VL sees frame 0) belong to the clip's scene prompt (cell 0) and to NULL_PROMPT (scene dropout)."""

    def __init__(self, model_dir: str, device: str = "cuda", num_frames: int = NUM_FRAMES):
        latent_frames(num_frames)
        self.num_frames = num_frames
        self.pipe = load_pipeline(model_dir, ("text_encoder", "video_vae"), device)

    @torch.no_grad()
    def __call__(self, frames: np.ndarray, clip) -> dict:
        pipe = self.pipe
        frames = np.ascontiguousarray(frames[:self.num_frames])
        video = torch.from_numpy(frames).to(pipe.device, torch.float32).permute(3, 0, 1, 2).unsqueeze(0) / 255.0
        latent = pipe.video_vae.encode_video(video, dtype=pipe.torch_dtype, tiled=False)[0].to(torch.bfloat16).cpu()
        scene = clip.prompts[0]
        text_scene, tags_scene = encode_prompt(pipe, scene, frames[0])
        text_null, tags_null = encode_prompt(pipe, NULL_PROMPT, frames[0])
        return {"latent": latent.contiguous(), "cond_rows": encode_first_frame(pipe, frames[0]).cpu().contiguous(),
                "text_scene": text_scene.contiguous(), "tags_scene": tags_scene,
                "text_null": text_null.contiguous(), "tags_null": tags_null, "scene": scene}


# ----------------------------------------------------------------------------- training
class H3Dataset(torch.utils.data.Dataset):
    def __init__(self, clips, latent_dir: str, num_frames: int = NUM_FRAMES, scene_dropout: float = 0.0):
        self.clips, self.latent_dir = list(clips), latent_dir
        self.num_frames, self.latent_t = num_frames, latent_frames(num_frames)
        self.scene_dropout = float(scene_dropout)
        missing = [c.clip for c in self.clips if not os.path.isfile(self._path(c))]
        if missing:
            raise FileNotFoundError(f"{len(missing)} latent files missing in {latent_dir} (e.g. {missing[0]}.pt); "
                                    "run scripts/encode_latents.py --model h3")

    def _path(self, c):
        return os.path.join(self.latent_dir, c.clip + ".pt")

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, i):
        c = self.clips[i]
        d = torch.load(self._path(c), map_location="cpu", weights_only=True)
        if tuple(d["latent"].shape) != (LATENT_C, self.latent_t, LAT_H, LAT_W):
            raise ValueError(f"{c.clip}: latent shape {tuple(d['latent'].shape)}, expected {self.num_frames} frames")
        if d.get("scene", c.prompts[0]) != c.prompts[0]:
            raise ValueError(f"{c.clip}: the latent file was encoded for another scene prompt")
        null = self.scene_dropout > 0 and random.random() < self.scene_dropout
        text, tags = (d["text_null"], d["tags_null"]) if null else (d["text_scene"], d["tags_scene"])
        return {"clip": c.clip, "latent": d["latent"].to(torch.bfloat16), "cond_rows": d["cond_rows"].to(torch.bfloat16),
                "text": text.to(torch.bfloat16), "tags": tags.to(torch.long),
                "frame_offset_px": torch.from_numpy(np.ascontiguousarray(c.frame_offset_px[:self.num_frames]))}


class H3TrainModule(nn.Module):
    """Flow-matching loss of one clip at one timestep index (ActionRoPE positions, CNS noising, weighted loss)."""

    def __init__(self, dit: nn.Module, num_frames: int = NUM_FRAMES, new_weight: float = 2.0, cns: bool = True,
                 cns_shift: float = SAMPLE_SHIFT, cns_shift_known: float = CNS_SHIFT_KNOWN, cns_alpha: float = 1.0,
                 train_shift: float = TRAIN_SHIFT, gradient_checkpointing: bool = True):
        super().__init__()
        from diffsynth.diffusion.flow_match import FlowMatchScheduler
        self.dit = dit
        self.latent_t, self.audio_t = latent_frames(num_frames), audio_frames(num_frames)
        self.new_weight, self.gradient_checkpointing = float(new_weight), bool(gradient_checkpointing)
        self.cns, self.cns_shift, self.cns_shift_known, self.cns_alpha = bool(cns), float(cns_shift), float(cns_shift_known), float(cns_alpha)
        self.scheduler = FlowMatchScheduler("MiniMax-H3")
        self.scheduler.set_timesteps(1000, training=True, shift=train_shift)
        self.scheduler_audio = FlowMatchScheduler("MiniMax-H3")
        self.scheduler_audio.set_timesteps(1000, training=True, shift=train_shift)
        self.audio_silence = None     # [2, 32, T] bf16, set by TrainTask.on_prepared

    def forward(self, batch: dict, timestep_id: int, generator: torch.Generator | None = None, train: bool = True):
        from diffsynth.models.minimax_h3_dit import patchify_video, unpatchify_video

        def randn(like):
            if generator is None:
                return torch.randn_like(like)
            return torch.randn(like.shape, generator=generator, device=like.device, dtype=like.dtype)

        x0 = batch["latent"]                                              # [1, 24, f, 30, 52]
        device = x0.device
        _, _, f, h, w = x0.shape
        if f != self.latent_t:
            raise ValueError(f"latent has {f} latent frames, expected {self.latent_t}")
        t_v = self.scheduler.timesteps[timestep_id].view(1).to(torch.float32)
        t_a = self.scheduler_audio.timesteps[timestep_id].view(1).to(torch.float32)
        sigma = float(self.scheduler.sigmas[timestep_id])
        noise = randn(x0)
        off = batch["frame_offset_px"].to(device)                         # [1, 73, 2] px
        kappa = token_kappa(latent_frame_offsets(off, f))[0].to(device)  # [f * 390]
        x0_rows, n_rows = patchify_video(x0), patchify_video(noise)
        sigma_tok = None
        if self.cns:
            sigma_tok = cns_sigma(sigma, kappa, self.cns_shift, self.cns_shift_known, self.cns_alpha)
            noisy_rows = (1.0 - sigma_tok).unsqueeze(1).to(x0.dtype) * x0_rows + sigma_tok.unsqueeze(1).to(x0.dtype) * n_rows
        else:
            noisy_rows = (1.0 - sigma) * x0_rows + sigma * n_rows
        target_rows = n_rows.float() - x0_rows.float()

        # silent audio at the same timestep: forward only, no loss
        a0 = self.audio_silence.to(device)
        audio = self.scheduler_audio.add_noise(a0, randn(a0), t_a)
        cond = noise_cond_rows(batch["cond_rows"].to(device), randn(batch["cond_rows"].to(device)))
        text, tags = batch["text"].to(device), batch["tags"]
        packed = build_packed(text.shape[0], tags, f, a0.shape[-1], device)
        positions, _ = world_positions(packed, cond.shape[0], f, off[0])
        pred, _ = forward(self.dit, unpatchify_video(noisy_rows, f, h, w), audio, packed, text, t_v, t_a, cond, positions,
                          sigma_tok=sigma_tok, kappa=kappa, use_gradient_checkpointing=self.gradient_checkpointing and train)
        se = ((patchify_video(pred).float() - target_rows) ** 2).mean(-1)      # per token
        weight = 1.0 + (self.new_weight - 1.0) * (1.0 - kappa)                  # known 1, newly revealed new_weight
        loss_w = (se * weight).sum() / weight.sum()
        loss = loss_w * self.scheduler.training_weight(t_v).to(device)
        return loss, {"loss_w": loss_w.detach(), "mse": se.mean().detach()}


def checkpoint_metadata(cfg: dict) -> dict:
    cns = float(cfg.get("cns_alpha", 1.0)) > 0
    return {"model": MODEL_NAME, "format": "lora", "mask_channel": bool(cfg.get("mask_channel", True)), "cns": cns,
            "cns_shift": float(cfg.get("cns_shift", SAMPLE_SHIFT)),
            "cns_shift_known": float(cfg.get("cns_shift_known", CNS_SHIFT_KNOWN)),
            "cns_alpha": float(cfg.get("cns_alpha", 1.0)),
            "lora_rank": int(cfg.get("lora_rank", 128)), "lora_alpha": int(cfg.get("lora_alpha") or cfg.get("lora_rank", 128)),
            "lora_targets": LORA_TARGETS, "num_frames": int(cfg.get("num_frames", NUM_FRAMES))}


class TrainTask:
    """Training task for ``actionrope.trainer`` (see configs/train/h3.yaml)."""

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.num_frames = int(cfg.get("num_frames", NUM_FRAMES))
        self.module = None
        self._timesteps = None

    def build_module(self) -> nn.Module:
        cfg = self.cfg
        dit = load_pipeline(cfg["model_dir"], ("dit",), "cpu").dit
        dit.train()
        mask = bool(cfg.get("mask_channel", True))
        if mask:
            install_mask_channel(dit)
        rank = int(cfg.get("lora_rank", 128))
        install_lora(dit, lora_target, rank=rank, alpha=float(cfg.get("lora_alpha") or rank),
                     keep_trainable=MASK_KEYS if mask else ())
        self.module = H3TrainModule(dit, self.num_frames, new_weight=float(cfg.get("new_weight", 2.0)),
                                    cns=float(cfg.get("cns_alpha", 1.0)) > 0,
                                    cns_shift=float(cfg.get("cns_shift", SAMPLE_SHIFT)),
                                    cns_shift_known=float(cfg.get("cns_shift_known", CNS_SHIFT_KNOWN)),
                                    cns_alpha=float(cfg.get("cns_alpha", 1.0)),
                                    train_shift=float(cfg.get("train_shift", TRAIN_SHIFT)),
                                    gradient_checkpointing=bool(cfg.get("gradient_checkpointing", True)))
        return self.module

    def datasets(self):
        cfg = self.cfg
        clips = load_annotations(cfg["annotations"])
        train = H3Dataset(training_clips(clips, min_rt_ratio=float(cfg.get("min_rt_ratio", 0.8))), cfg["latent_dir"],
                          self.num_frames, scene_dropout=float(cfg.get("scene_dropout", 0.1)))
        val = None
        if int(cfg.get("val_every", 0)) > 0:
            val = H3Dataset(validation_clips(clips, min_rt_ratio=float(cfg.get("val_min_rt_ratio", 0.9)),
                                             n=int(cfg.get("val_clips", 16)), seed=int(cfg.get("seed", 0))),
                            cfg["latent_dir"], self.num_frames)
        return train, val

    def collate(self, samples):
        if len(samples) != 1:
            raise ValueError("MiniMax-H3 training uses one clip per micro-batch (text lengths differ)")
        s = samples[0]
        return {"clip": [s["clip"]], "latent": s["latent"].unsqueeze(0), "cond_rows": s["cond_rows"], "text": s["text"],
                "tags": s["tags"], "frame_offset_px": s["frame_offset_px"].unsqueeze(0)}

    def on_prepared(self, module, device):
        path = self.cfg.get("audio_silence")
        if path:
            lat = torch.load(path, map_location="cpu", weights_only=True)
            lat = lat[self.num_frames] if isinstance(lat, dict) else lat
        else:
            lat = silent_audio_latent(self.cfg["model_dir"], self.num_frames, str(device))
            torch.cuda.empty_cache()
        module.audio_silence = lat.to(device=device, dtype=torch.bfloat16)

    def export(self, state_dict: dict, step: int):
        sd = lora_state_dict(self.module, state_dict, extra=MASK_KEYS)
        tensors = {(k[len("dit."):] if k.startswith("dit.") else k): v.to(torch.bfloat16) for k, v in sd.items()}
        return tensors, checkpoint_metadata(self.cfg)

    def timestep_value(self, tid: int) -> float:
        if self._timesteps is None:
            from diffsynth.diffusion.flow_match import FlowMatchScheduler
            s = FlowMatchScheduler("MiniMax-H3")
            s.set_timesteps(1000, training=True, shift=float(self.cfg.get("train_shift", TRAIN_SHIFT)))
            self._timesteps = s.timesteps
        return float(self._timesteps[tid])
