"""CPU tests of the MiniMax-H3 integration (no model weights; a two-layer DiT stands in for the 33B model)."""
import math

import numpy as np
import pytest
import torch

from actionrope.models import h3
from actionrope.models.h3_dit import (LAT_H, LAT_W, TOKENS_PER_FRAME, audio_frames, install_mask_channel,
                                      latent_frame_offsets, latent_frame_spans, latent_frames, lora_target,
                                      retime_frames, retime_offsets, token_kappa)


def test_latent_layout():
    assert latent_frames(73) == 22 and latent_frames(90) == 27
    with pytest.raises(ValueError):
        latent_frames(81)
    spans = latent_frame_spans(22)
    assert spans[:3] == [(0, 1), (1, 4), (5, 4)] and spans[5] == (17, 1) and sum(n for _, n in spans) == 73
    assert audio_frames(73) == 122


def test_latent_frame_offsets_interpolate_at_span_centres():
    off = torch.zeros(1, 73, 2, dtype=torch.float64)
    off[0, :, 0] = torch.arange(73, dtype=torch.float64)          # 1 px per frame to the right
    lat = latent_frame_offsets(off, 22)[0, :, 0]
    centres = [s + (n - 1) / 2 for s, n in latent_frame_spans(22)]
    assert torch.allclose(lat, torch.tensor(centres, dtype=torch.float64))
    assert lat[0] == 0 and lat[1] == 2.5 and lat[-1] == 70.5


def test_token_kappa():
    off = torch.zeros(1, 22, 2, dtype=torch.float64)
    assert token_kappa(off).sum() == 22 * TOKENS_PER_FRAME
    off[0, 5:, 0] = 832.0                                         # a full screen to the right: everything is new
    k = token_kappa(off)[0].view(22, TOKENS_PER_FRAME)
    assert k[:5].sum() == 5 * TOKENS_PER_FRAME and k[5:].sum() == 0
    off[0, 5:, 0] = 32.0                                          # one token column revealed
    k = token_kappa(off)[0].view(22, 15, 26)
    assert k[5:, :, -1].sum() == 0 and k[5:, :, :-1].min() == 1


def test_retime():
    plan = np.stack([np.linspace(0, 80, 81), np.zeros(81)], 1)
    r = retime_offsets(plan, 73)
    assert r.shape == (73, 2) and r[0, 0] == 0 and abs(r[-1, 0] - 80) < 1e-12
    assert np.allclose(np.diff(r[:, 0]), 80 / 72)
    frames = np.arange(73)[:, None, None, None] * np.ones((1, 2, 2, 3), np.uint8)
    out = retime_frames(frames, 81)
    assert out.shape[0] == 81 and out[0, 0, 0, 0] == 0 and out[-1, 0, 0, 0] == 72
    assert np.array_equal(out[:, 0, 0, 0], np.rint(np.arange(81) * 72 / 80).astype(int))


# ----------------------------------------------------------------------------- with DiffSynth (tiny DiT)
def _tiny_dit():
    pytest.importorskip("diffsynth.models.minimax_h3_dit")
    from diffsynth.models.minimax_h3_dit import MiniMaxH3DiT
    torch.manual_seed(0)
    hs = 256
    dit = MiniMaxH3DiT(num_layers=2, token_refiner_num_layers=1, hidden_size=hs, num_attention_heads=2,
                       attention_head_dim=128, ffn_hidden_size=512, latents_dim=24, audio_latents_dim=32, text_dim=5120,
                       timestep_input_dim=256, time_embed_hidden_size=hs, time_embed_dim=128,
                       adaln_out_features=6 * hs * 3, final_adaln_out_features=2 * hs, rope_inv_freq_len=16)
    return dit.to(torch.bfloat16)


def _batch(text_len=40):
    torch.manual_seed(1)
    off = torch.zeros(1, 73, 2, dtype=torch.float64)
    off[0, :, 0] = torch.linspace(0, 200, 73, dtype=torch.float64)
    return {"clip": ["x"], "latent": torch.randn(1, 24, 22, LAT_H, LAT_W).to(torch.bfloat16),
            "cond_rows": torch.randn(390, 96).to(torch.bfloat16), "text": torch.randn(text_len, 5120).to(torch.bfloat16),
            "tags": torch.ones(text_len, dtype=torch.long), "frame_offset_px": off}


def test_world_positions():
    pytest.importorskip("diffsynth.pipelines.minimax_h3_audio_video")
    from actionrope.models.h3_dit import build_packed, unit_per_px, world_positions
    packed = build_packed(10, torch.ones(10, dtype=torch.long), 22, 122, "cpu")
    off = torch.zeros(73, 2, dtype=torch.float64)
    off[:, 0] = torch.arange(73, dtype=torch.float64) * 4.0       # right
    off[:, 1] = -torch.arange(73, dtype=torch.float64) * 2.0      # up
    pos, off_lat = world_positions(packed, 390, 22, off)
    base, ip = packed["img_position_ids"], packed["img_pos"]
    u = unit_per_px(base, ip, 390)
    assert abs(u - 64 / math.sqrt(LAT_H * LAT_W) / 32) < 1e-12
    vid = ip[390:].view(22, TOKENS_PER_FRAME)
    k = 7
    d = pos[0, vid[k]] - base[0, vid[k]]
    assert torch.allclose(d[:, 1], torch.full_like(d[:, 1], float(off_lat[k, 1]) * u))
    assert torch.allclose(d[:, 2], torch.full_like(d[:, 2], float(off_lat[k, 0]) * u))
    assert torch.equal(d[:, 0], torch.zeros_like(d[:, 0]))                       # time is untouched
    others = torch.ones(pos.shape[1], dtype=torch.bool)
    others[ip[390:]] = False
    assert torch.equal(pos[0, others], base[0, others])                          # text, audio, first frame


def _module(dit):
    m = h3.H3TrainModule(dit, gradient_checkpointing=False)
    m.audio_silence = torch.zeros(2, 32, audio_frames(73), dtype=torch.bfloat16)
    return m


def test_mask_channel_starts_as_identity():
    dit = _tiny_dit()
    m = _module(dit)
    batch = _batch()
    torch.manual_seed(2)
    loss0, _ = m(batch, 500, train=False)
    install_mask_channel(dit)
    torch.manual_seed(2)
    loss1, _ = m(batch, 500, train=False)
    assert torch.equal(loss0, loss1)


def test_train_step_and_export_roundtrip(tmp_path):
    from actionrope.checkpoint import save_checkpoint
    from actionrope.lora import install_lora
    dit = _tiny_dit()
    install_mask_channel(dit)
    n = install_lora(dit, lora_target, rank=4, alpha=4, keep_trainable=("arope_mask_embed.weight", "arope_mask_embed.bias"))
    assert n == 2 * 4
    m = _module(dit)
    with torch.no_grad():                                         # validation path: noise from a generator
        v1, _ = m(_batch(), 500, generator=torch.Generator().manual_seed(3), train=False)
        v2, _ = m(_batch(), 500, generator=torch.Generator().manual_seed(3), train=False)
    assert torch.equal(v1, v2)
    loss, stats = m(_batch(), 500)
    loss.backward()
    assert torch.isfinite(loss) and float(stats["loss_w"]) > 0
    assert dit.blocks[0].attn.qkv_proj.lora_B.grad.abs().max() > 0
    assert dit.arope_mask_embed.weight.grad.abs().max() > 0
    assert dit.blocks[0].attn.qkv_proj.base.weight.grad is None

    task = h3.TrainTask({"lora_rank": 4, "lora_alpha": 4})
    task.module = m
    with torch.no_grad():
        dit.blocks[0].attn.qkv_proj.lora_B.add_(0.5)
    tensors, meta = task.export(m.state_dict(), 7)
    assert set(tensors) == {k for k in dit.state_dict() if k.endswith((".lora_A", ".lora_B")) or k.startswith("arope_mask_embed.")}
    assert meta["model"] == "minimax-h3-33b" and meta["format"] == "lora" and meta["num_frames"] == 73
    path = str(tmp_path / "lora.safetensors")
    save_checkpoint(path, tensors, {**meta, "step": 7})

    fresh = _tiny_dit()
    got = h3.load_lora(fresh, path)
    assert got["lora_rank"] == "4" and got["cns_shift_known"] == "4.8"
    sd = fresh.state_dict()
    assert all(torch.equal(sd[k], v) for k, v in tensors.items())


class _FakePipe:
    def __init__(self, dit):
        from diffsynth.diffusion.flow_match import FlowMatchScheduler
        self.device, self.torch_dtype, self.dit = "cpu", torch.bfloat16, dit
        self.scheduler, self.scheduler_audio = FlowMatchScheduler("MiniMax-H3"), FlowMatchScheduler("MiniMax-H3")

    def generate_noise(self, shape, seed=None, rand_device="cpu", rand_torch_dtype=torch.float32):
        g = torch.Generator(rand_device).manual_seed(seed)
        return torch.randn(shape, generator=g, device=rand_device, dtype=rand_torch_dtype).to(self.torch_dtype)


def test_sampling_loop():
    dit = _tiny_dit()
    install_mask_channel(dit)
    pipe = _FakePipe(dit)
    b = _batch()
    off = retime_offsets(np.stack([np.linspace(0, 300, 81), np.zeros(81)], 1), 73)
    lat = h3.sample(pipe, b["text"], b["tags"], b["cond_rows"], off, seed=0, steps=2)
    assert tuple(lat.shape) == (1, 24, 22, LAT_H, LAT_W) and torch.isfinite(lat.float()).all()
    again = h3.sample(pipe, b["text"], b["tags"], b["cond_rows"], off, seed=0, steps=2)
    assert torch.equal(lat, again)
    plain = h3.sample(pipe, b["text"], b["tags"], b["cond_rows"], off, seed=0, steps=2, cns=False)
    assert not torch.equal(lat, plain)
