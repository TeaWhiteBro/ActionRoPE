"""CPU tests of the LTX-2.5 integration on a tiny randomly initialised LTXModel (no model files needed).

    PYTHONPATH=.:<DiffSynth-Studio> DIFFSYNTH_SKIP_DOWNLOAD=True python -m pytest tests/test_ltx.py
"""
from __future__ import annotations

import os

import numpy as np
import pytest
import torch

os.environ.setdefault("DIFFSYNTH_SKIP_DOWNLOAD", "True")
pytest.importorskip("diffsynth")

from diffsynth.diffusion.flow_match import FlowMatchScheduler  # noqa: E402
from diffsynth.models.ltx2_dit import LTXModel, LTXModelType  # noqa: E402
from diffsynth.models.ltx2_video_vae import VideoLatentPatchifier  # noqa: E402
from diffsynth.pipelines.ltx2_audio_video import model_fn_ltx2  # noqa: E402

from actionrope.checkpoint import save_checkpoint  # noqa: E402
from actionrope.cns import cns_sigma  # noqa: E402
from actionrope.coords import known_mask  # noqa: E402
from actionrope.data import Clip  # noqa: E402
from actionrope.geometry import FRAMES, offsets_from_actions  # noqa: E402
from actionrope.lora import install_lora  # noqa: E402
from actionrope.models import ltx, ltx_dit  # noqa: E402
from actionrope.prompts import NULL_PROMPT  # noqa: E402

CAPTION = 32


def tiny_dit(seed: int = 0, layers: int = 2) -> LTXModel:
    """Video-only LTX DiT with the LTX-2.5 cross-attention AdaLN and keyframe embedding, tiny width."""
    dit = LTXModel(model_type=LTXModelType.VideoOnly, num_attention_heads=2, attention_head_dim=8, in_channels=128,
                   out_channels=128, num_layers=layers, cross_attention_dim=16, caption_channels=CAPTION,
                   positional_embedding_max_pos=[20, 2048, 2048], cross_attention_adaln=True,
                   use_keyframes_abs_pos_embedding=True)
    g = torch.Generator().manual_seed(seed)       # some LTX tables are allocated uninitialised: fill everything
    with torch.no_grad():
        for p in dit.parameters():
            p.copy_(torch.randn(p.shape, generator=g) * 0.1)
    return dit


def tiny_inputs(seed: int = 0, frames: int = 2, h: int = 2, w: int = 3, L: int = 5):
    from diffsynth.models.ltx2_common import VIDEO_SCALE_FACTORS, VideoLatentShape, get_pixel_coords
    g = torch.Generator().manual_seed(seed)
    shape = VideoLatentShape(batch=1, channels=128, frames=frames, height=h, width=w)
    pos = get_pixel_coords(VideoLatentPatchifier(1).get_patch_grid_bounds(shape), VIDEO_SCALE_FACTORS, True).float()
    pos[:, 0] /= 16
    kf = torch.zeros(1, 1, frames, h, w)
    kf[:, :, 0] = 1
    return dict(video_latents=torch.randn(1, 128, frames, h, w, generator=g), video_positions=pos,
                video_patchifier=VideoLatentPatchifier(1), timestep=torch.tensor([600.0]), video_keyframes_mask=kf), \
        torch.randn(1, L, CAPTION, generator=g)


def test_latent_grid_and_frame_sampling():
    pos, idx, lf = ltx_dit.latent_grid()
    assert pos.shape == (1, 3, 11 * 15 * 26, 2) and pos.dtype == torch.float32
    assert tuple(lf.tolist()) == ltx_dit.LATENT_FRAME_PIXELS == (0,) + tuple(8 * k - 3 for k in range(1, 11))
    assert float(pos[0, 1].max()) == 480 and float(pos[0, 2].max()) == 832       # 32 px per token
    off = torch.as_tensor(offsets_from_actions("right:40,left:40"))
    assert torch.equal(ltx_dit.latent_frame_offsets(off), off[lf])


def test_world_positions_shift_height_and_width_only():
    pos, idx, lf = ltx_dit.latent_grid()
    off = torch.zeros(2, FRAMES, 2, dtype=torch.float64)
    off[0, :, 0] = torch.linspace(0, 300, FRAMES)
    off[1, :, 1] = torch.linspace(0, -150, FRAMES)
    vp = ltx_dit.world_positions(pos, idx, off)
    n = 15 * 26
    for b in range(2):
        assert torch.equal(vp[b, 0], pos[0, 0])
        for k in range(11):
            sl = slice(k * n, (k + 1) * n)
            dx, dy = off[b, int(lf[k])].float()
            assert torch.allclose(vp[b, 2, sl] - pos[0, 2, sl], dx.expand(n, 2), atol=1e-3)
            assert torch.allclose(vp[b, 1, sl] - pos[0, 1, sl], dy.expand(n, 2), atol=1e-3)
    # the sampler path gives the same positions for float32-representable offsets
    off32 = off.float().double()
    for b in range(2):
        assert torch.equal(ltx_dit.add_offsets_to_positions(pos, off32[b], 16), ltx_dit.world_positions(pos, idx, off32[b:b + 1]))


@torch.no_grad()
def test_mask_channel_is_zero_at_installation():
    dit = tiny_dit()
    kw, ctx = tiny_inputs()
    ref, _ = model_fn_ltx2(dit=dit, video_context=ctx, **kw)
    ltx_dit.install_mask_channel(dit)
    ltx_dit.install_mask_channel(dit)                                          # idempotent
    dit._arope_mask_patch = torch.rand(1, 12, 1) * 10
    out, _ = model_fn_ltx2(dit=dit, video_context=ctx, **kw)
    assert torch.equal(out, ref)
    assert sum(p.numel() for p in getattr(dit, ltx_dit.MASK_EMBEDDING).parameters()) == 2 * dit.inner_dim


@torch.no_grad()
def test_per_frame_text_equals_global_and_is_frame_local():
    dit = tiny_dit(layers=1).eval()
    kw, ctx = tiny_inputs(frames=3)
    ref, _ = model_fn_ltx2(dit=dit, video_context=ctx, **kw)
    ltx_dit.install_per_frame_text(dit)
    ltx_dit.install_per_frame_text(dit)                                        # idempotent
    off, _ = model_fn_ltx2(dit=dit, video_context=ctx, **kw)
    assert torch.equal(off, ref)                                               # off by default
    ltx_dit.set_text_frames(dit, 3)
    same, _ = model_fn_ltx2(dit=dit, video_context=ctx.repeat(1, 3, 1), **kw)
    ctx2 = ctx.repeat(1, 3, 1).clone()
    ctx2[:, 10:15] = torch.randn(1, 5, CAPTION)                                # prompt of latent frame 2 only
    diff, _ = model_fn_ltx2(dit=dit, video_context=ctx2, **kw)
    ltx_dit.set_text_frames(dit, None)
    assert torch.allclose(same, ref, atol=1e-5, rtol=1e-4)
    delta = (diff - ref).abs().amax(dim=(0, 1, 3, 4))
    assert float(delta[2]) > 1e-5 and float(delta[:2].max()) < 1e-7


@torch.no_grad()
def test_cns_timesteps_wrapper():
    """The DiT receives sigma_t = cns_sigma(sigma, kappa) per token; the clean first frame keeps 0."""
    seen = {}

    def fake_dit(video_timesteps=None, **kw):
        seen["t"] = video_timesteps
        return kw["video_latents"], None

    kw, ctx = tiny_inputs(frames=2)
    kap = torch.zeros(1, 1, 2, 2, 3)
    kap[..., 0] = 1.0                                                          # first token column known
    patch = VideoLatentPatchifier(1)
    dm = torch.ones(1, 1, 2, 2, 3)
    dm[:, :, 0] = 0
    wrapped = ltx_dit.CNSTimesteps(fake_dit, patch.patchify(kap), ltx_dit.SHIFT, 3.1, 1.0)
    model_fn_ltx2(dit=wrapped, video_context=ctx, input_latents_video=kw["video_latents"], denoise_mask_video=dm, **kw)
    t = seen["t"].view(2, 2, 3)
    s = 0.6
    assert torch.equal(t[0], torch.zeros(2, 3))
    assert torch.allclose(t[1, :, 0], cns_sigma(torch.tensor(s), torch.tensor(1.0), ltx_dit.SHIFT, 3.1))
    assert torch.allclose(t[1, :, 1:], torch.full((2, 2), s))                 # newly revealed: base schedule
    assert float(t[1, 0, 0]) < s


def test_cns_step_matches_base_step_for_new_tokens():
    sch = FlowMatchScheduler("LTX-2")
    sch.set_timesteps(30)
    base = FlowMatchScheduler("LTX-2")
    base.set_timesteps(30)
    state = {"kappa": torch.zeros(1, 1, 2, 2, 3)}
    ltx_dit.install_cns_step(sch, state, ltx_dit.SHIFT, 3.1, 1.0)
    ltx_dit.install_cns_step(sch, state, ltx_dit.SHIFT, 3.1, 1.0)            # idempotent
    x, v = torch.randn(1, 128, 2, 2, 3), torch.randn(1, 128, 2, 2, 3)
    t = sch.timesteps[5]
    assert torch.allclose(sch.step(v, t, x), base.step(v, t, x), atol=1e-6)
    state["kappa"] = torch.ones(1, 1, 2, 2, 3)
    y = sch.step(v, t, x)
    assert torch.allclose(y[:, :, 0], base.step(v, t, x)[:, :, 0], atol=1e-6)   # first frame: base schedule
    assert not torch.allclose(y[:, :, 1], base.step(v, t, x)[:, :, 1])
    a = torch.randn(1, 8, 4, 16)                                                # audio latents: original step
    assert torch.equal(sch.step(a, t, a), base.step(a, t, a))


def test_lora_targets_and_clean_keys():
    dit = tiny_dit(layers=2)
    names = [n for n, m in dit.named_modules() if isinstance(m, torch.nn.Linear) and ltx_dit.is_lora_target(n)]
    assert len(names) == 2 * 10
    assert ltx_dit.clean_key("transformer_blocks.3.module.attn1.k_norm.module.weight") == "transformer_blocks.3.attn1.k_norm.weight"
    assert ltx_dit.is_lora_target("transformer_blocks.3.module.ff.net.2")
    assert not ltx_dit.is_lora_target("transformer_blocks.3.module.audio_attn1.to_q")


def _toy_task(tmp_path, fmt):
    dit = tiny_dit(layers=2).to(torch.bfloat16)                                # base weights in bf16 as in LTX-2.5
    ltx_dit.install_mask_channel(dit)
    ltx_dit.install_per_frame_text(dit)
    task = ltx.TrainTask.__new__(ltx.TrainTask)
    task.cfg = {"lora_rank": 4, "lora_alpha": 4, "export_format": fmt, "mask_channel": True, "cns": True}
    task.n_keys = len(dit.state_dict())
    torch.manual_seed(0)
    install_lora(dit, ltx_dit.is_lora_target, 4, 4, keep_trainable=ltx.MASK_KEYS)
    task.module = ltx.LTXTrainModule(dit)
    with torch.no_grad():
        for n, p in dit.named_parameters():
            if n.endswith("lora_B") or ltx_dit.MASK_EMBEDDING in n:
                p.normal_()
    tensors, meta = task.export(task.module.state_dict(), 7)
    path = str(tmp_path / f"{fmt}.safetensors")
    save_checkpoint(path, tensors, {**meta, "step": 7})
    return path, tensors, meta


def test_export_full_and_lora_load_to_the_same_weights(tmp_path):
    full, t_full, meta = _toy_task(tmp_path, "full")
    lora, t_lora, meta_l = _toy_task(tmp_path, "lora")
    assert meta["format"] == "full" and meta_l["format"] == "lora" and meta["cns_shift"] == repr(ltx_dit.SHIFT)
    assert all(".module." not in k and "lora" not in k for k in t_full)
    assert len(t_lora) == 2 * 20 + 2
    loaded = {}
    for path in (full, lora):
        d = tiny_dit(layers=2).to(torch.bfloat16)
        ltx_dit.install_mask_channel(d)
        ltx.load_checkpoint_into(d, path)
        loaded[path] = d.state_dict()
    for k in loaded[full]:
        assert torch.equal(loaded[full][k], loaded[lora][k]), k


def test_train_module_forward_backward_on_toy_dit():
    dit = tiny_dit(layers=1).to(torch.bfloat16)
    ltx_dit.install_mask_channel(dit)
    ltx_dit.install_per_frame_text(dit)
    install_lora(dit, ltx_dit.is_lora_target, 4, 4, keep_trainable=ltx.MASK_KEYS)
    module = ltx.LTXTrainModule(dit, gradient_checkpointing=False)
    off = torch.as_tensor(offsets_from_actions("right:80"))[None]
    batch = {"latent": torch.randn(1, 128, 11, 15, 26).to(torch.bfloat16),
             "context": torch.randn(1, 11, 3, CAPTION).to(torch.bfloat16), "frame_offset_px": off}
    loss, stats = module(batch, 500)
    loss.backward()
    assert torch.isfinite(loss) and float(stats["loss_w"]) > 0
    assert getattr(dit, ltx_dit.MASK_EMBEDDING).weight.grad is not None
    kap = known_mask(ltx_dit.latent_frame_offsets(off), 15, 26, 32)
    assert bool(kap[0, 0].all()) and not bool(kap[0, -1].all())


def test_dataset_prompts_per_latent_frame(tmp_path):
    prompts = [f"In a 2.5D top down view, the player is in scene {k // 10}." for k in range(21)]
    table = {p: torch.full((3, 4), float(i), dtype=torch.float16) for i, p in enumerate(sorted(set(prompts)))}
    table[NULL_PROMPT] = torch.zeros(3, 4, dtype=torch.float16)
    torch.save({"latent": torch.zeros(128, 11, 15, 26, dtype=torch.bfloat16)}, tmp_path / "c0.pt")
    clip = Clip(clip="c0", split="train", kind="burn", valid=True, rt_ratio=None, sources=(),
                frame_offset_px=np.zeros((FRAMES, 2)), prompts=prompts)
    s = ltx.LTXDataset([clip], str(tmp_path), table)[0]
    assert s["context"].shape == (11, 3, 4) and s["context"].dtype == torch.bfloat16
    expect = [table[prompts[c]][0, 0].item() for c in ltx.LATENT_FRAME_CELLS]
    assert s["context"][:, 0, 0].tolist() == expect
    s = ltx.LTXDataset([clip], str(tmp_path), table, scene_dropout=1.0)[0]
    assert float(s["context"].abs().sum()) == 0
    b = ltx.collate([s])
    assert b["latent"].shape == (1, 128, 11, 15, 26) and b["clip"] == ["c0"]
