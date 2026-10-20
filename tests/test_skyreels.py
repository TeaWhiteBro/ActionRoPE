"""CPU tests of the SkyReels-V2-DF-1.3B integration (small random DiT, no model files needed).

    PYTHONPATH=.:<DiffSynth> python -m pytest tests/test_skyreels.py
"""
from __future__ import annotations

import os

import numpy as np
import pytest
import torch
import yaml

from actionrope.checkpoint import read_metadata, save_checkpoint
from actionrope.coords import known_mask
from actionrope.geometry import FRAMES, HEIGHT, WIDTH, frames_to_cells
from actionrope.models import skyreels as S
from actionrope.models.wan_dit import arope_forward, build_world_freqs

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# head_dim 12 splits into 4 / 4 / 4 rotary dims for (frame, h, w)
TOY = dict(dim=24, ffn_dim=48, text_dim=32, freq_dim=16, num_heads=2, num_layers=2)


def toy_dit(seed: int = 0):
    torch.manual_seed(seed)
    dit = S.build_dit(mask_channel=True, **TOY)
    for p in dit.parameters():          # give the zero-initialised mask channel a non-trivial value
        if p.abs().sum() == 0:
            torch.nn.init.normal_(p, std=0.02)
    return dit


def test_geometry():
    c, f, h, w = S.LATENT_SHAPE
    assert (h * S.PX_PER_LATENT, w * S.PX_PER_LATENT) == (HEIGHT, WIDTH)
    assert S.PX_PER_TOKEN == S.PX_PER_LATENT * S.DIT_CONFIG["patch_size"][1]
    assert frames_to_cells(np.zeros((FRAMES, 2))).shape == (f, 2) and c == S.DIT_CONFIG["in_dim"]


def test_model_keys():
    """Key set = WanModel + mask channel + the frame-rate branch of the official checkpoint."""
    dit = S.build_dit(mask_channel=True, device="meta")
    keys = set(dit.state_dict())
    assert set(S.FPS_KEYS) <= keys and set(S.MASK_KEYS) <= keys and len(keys) == 832
    assert tuple(dit.fps_projection[2].weight.shape) == (6 * 1536, 1536)


def test_world_rope_integer_shift():
    """An offset of whole tokens (16 px) moves every token to the rotary position of its neighbour."""
    dit = toy_dit()
    f, h, w = 3, 6, 8
    off_px = torch.tensor([[[0.0, 0.0], [32.0, 16.0], [-48.0, 0.0]]])
    world = build_world_freqs(dit, f, h, w, off_px.double() / S.PX_PER_TOKEN)[0].view(f, h, w, -1)
    big = build_world_freqs(dit, f, h + 4, w + 8, None).view(f, h + 4, w + 8, -1)
    d_f = dit.freqs[0].shape[1]
    for k, (dx, dy) in enumerate([(0, 0), (2, 1), (-3, 0)]):
        for i in range(h):
            for j in range(w):
                if 0 <= i + dy < h + 4 and 0 <= j + dx < w + 8:
                    ref = big[k, i + dy, j + dx]
                    assert torch.allclose(world[k, i, j, d_f:], ref[d_f:], atol=1e-12)


def test_known_mask_latent_grid():
    """Known cells on the 60 x 104 latent grid (8 px per cell) vs. a brute-force loop."""
    torch.manual_seed(1)
    off = (torch.rand(1, 4, 2) - 0.5) * 2 * 48
    got = known_mask(off, 60, 104, S.PX_PER_LATENT)[0]
    for k in range(4):
        dx, dy = (off[0, k] / S.PX_PER_LATENT).tolist()
        ii = torch.arange(60, dtype=torch.float64).view(-1, 1) + dy
        jj = torch.arange(104, dtype=torch.float64).view(1, -1) + dx
        ref = (ii >= -0.5) & (ii <= 59.5) & (jj >= -0.5) & (jj <= 103.5)
        assert torch.equal(got[k], ref)
    assert 0 < int((~got).sum()) < got.numel()


def test_fps_modulation():
    dit = toy_dit()
    with torch.no_grad():
        zero = torch.zeros(2, 5, 6, dit.dim)
        d0, d1 = S.fps_modulation(dit, 0)(zero), S.fps_modulation(dit, 1)(zero)
        t_mod = torch.randn(2, 5, 6, dit.dim)
        shifted = S.fps_modulation(dit, 0)(t_mod)
    assert torch.equal(d0, d0[:1, :1].expand_as(d0)) and not torch.allclose(d0, d1)   # same vector for every token
    assert torch.allclose(shifted - t_mod, d0, atol=1e-6)


def test_mask_channel_zero_init():
    torch.manual_seed(0)
    dit = S.build_dit(mask_channel=True, **TOY).eval()
    x = torch.randn(1, 16, 3, 8, 12)
    ctx = torch.randn(1, 7, TOY["text_dim"])
    off = torch.tensor([[[0.0, 0.0], [20.0, 0.0], [40.0, -8.0]]])
    mask = torch.rand(1, 1, 3, 8, 12).round()
    with torch.no_grad():
        a = arope_forward(dit, x, torch.tensor([500.0]), ctx, offset_px=off, px_per_token=S.PX_PER_TOKEN)
        b = arope_forward(dit, x, torch.tensor([500.0]), ctx, offset_px=off, px_per_token=S.PX_PER_TOKEN, mask_input=mask)
    assert torch.equal(a, b)


def _toy_batch(f=5, h=8, w=12):
    off = torch.zeros(1, f, 2)
    off[0, :, 0] = torch.linspace(0, 40, f)
    ctx = torch.randn(1, f, 7, TOY["text_dim"])
    ids = torch.tensor([[0, 0, 1, 1, 1]])
    ctx[0, 1] = ctx[0, 0]
    ctx[0, 3:] = ctx[0, 2]
    return {"clip": ["toy"], "latent": torch.randn(1, 16, f, h, w), "context": ctx, "context_ids": ids, "offset_px": off}


@pytest.mark.parametrize("cns", [True, False])
def test_train_module_step(cns):
    dit = toy_dit()
    module = S.SkyReelsTrainModule(dit, cns=cns, fps_index=0, gradient_checkpointing=True)
    batch = _toy_batch()
    loss, stats = module(batch, 400)
    loss.backward()
    assert torch.isfinite(loss) and float(stats["loss_w"]) > 0
    assert dit.arope_mask_embedding.weight.grad.abs().sum() > 0
    assert dit.fps_projection[2].weight.grad.abs().sum() > 0
    # fixed noise through a generator is reproducible (validation path)
    g1, g2 = torch.Generator().manual_seed(3), torch.Generator().manual_seed(3)
    with torch.no_grad():
        assert torch.equal(module(batch, 400, generator=g1, train=False)[0], module(batch, 400, generator=g2, train=False)[0])


def test_frame0_not_in_loss(monkeypatch):
    """Latent frame 0 enters the DiT clean and is not part of the loss: with a zero prediction the loss equals the
    weighted velocity error of frames 1..20 only."""
    dit = toy_dit()
    module = S.SkyReelsTrainModule(dit, cns=True, fps_index=0, gradient_checkpointing=False)
    batch = _toy_batch()
    captured = {}

    def fake_forward(dit_, x_t, timestep, context, **kw):
        captured["x_t"] = x_t
        return torch.zeros_like(x_t)

    monkeypatch.setattr(S, "arope_forward", fake_forward)
    loss, _ = module(batch, 400, generator=torch.Generator().manual_seed(0))
    assert torch.equal(captured["x_t"][:, :, 0], batch["latent"][:, :, 0])
    noise = torch.randn(batch["latent"].shape, generator=torch.Generator().manual_seed(0))
    target = noise - batch["latent"]
    w = S.loss_weight_map(batch["offset_px"], 8, 12, S.PX_PER_LATENT)[:, :, 1:]
    ref = ((target[:, :, 1:] ** 2) * w).sum() / (w.sum() * 16) * module.scheduler.training_weight(module.scheduler.timesteps[400].view(1))
    assert torch.allclose(loss, ref.float(), rtol=1e-5)


def test_export_roundtrip(tmp_path):
    cfg = yaml.safe_load(open(os.path.join(ROOT, "configs", "train", "skyreels.yaml"), encoding="utf-8"))
    task = S.TrainTask(cfg)
    dit = toy_dit()
    module = S.SkyReelsTrainModule(dit)
    tensors, meta = task.export(module.state_dict(), 3)
    assert set(tensors) == set(dit.state_dict()) and all(v.dtype == torch.bfloat16 for v in tensors.values())
    path = str(tmp_path / "ckpt.safetensors")
    save_checkpoint(path, tensors, {**meta, "step": 3})
    m = read_metadata(path)
    assert (m["model"], m["format"], m["mask_channel"], m["cns"]) == (S.MODEL_NAME, "full", "true", "true")
    assert (float(m["cns_shift"]), float(m["cns_shift_known"]), float(m["cns_alpha"]), m["fps_index"]) == (5.0, 2.0, 1.0, "0")
    fresh = S.build_dit(mask_channel=True, **TOY)
    S.load_checkpoint(fresh, path)
    for k, v in fresh.state_dict().items():
        assert torch.equal(v.to(torch.bfloat16), tensors[k])


def test_config_recipe():
    cfg = yaml.safe_load(open(os.path.join(ROOT, "configs", "train", "skyreels.yaml"), encoding="utf-8"))
    assert cfg["model"] == "skyreels" and cfg["grad_accum"] == 2 and cfg["warmup_steps"] == 200
    assert (cfg["lr"], cfg["max_steps"], cfg["new_weight"], cfg["scene_dropout"]) == (1e-5, 5000, 2.0, 0.1)
    assert (cfg["shift"], cfg["cns_shift_known"], cfg["cns_alpha"], cfg["fps_index"]) == (5.0, 2.0, 1.0, 0)
    task = S.TrainTask(cfg)
    assert [round(task.timestep_value(t)) for t in cfg["val_timesteps"]] == [201, 501, 800]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
