"""CPU tests of the Wan2.2-TI2V-5B integration on a tiny randomly initialised WanModel (no model files needed).

    PYTHONPATH=.:<DiffSynth-Studio> DIFFSYNTH_SKIP_DOWNLOAD=True python -m pytest tests/test_wan.py
"""
from __future__ import annotations

import os

import numpy as np
import pytest
import torch

os.environ.setdefault("DIFFSYNTH_SKIP_DOWNLOAD", "True")
pytest.importorskip("diffsynth")

from diffsynth.models.wan_video_dit import WanModel  # noqa: E402

from actionrope.cns import cns_sigma  # noqa: E402
from actionrope.coords import loss_weight_map  # noqa: E402
from actionrope.data import Clip  # noqa: E402
from actionrope.geometry import offsets_from_actions  # noqa: E402
from actionrope.models import wan, wan_dit  # noqa: E402
from actionrope.prompts import NULL_PROMPT  # noqa: E402

C, TEXT_DIM = 4, 32


def tiny_dit(seed: int = 0, layers: int = 2) -> WanModel:
    """Same block structure as Wan2.2-TI2V-5B (head dim 128 -> the same 44/42/42 RoPE split), tiny width."""
    torch.manual_seed(seed)
    return WanModel(dim=128, in_dim=C, ffn_dim=256, out_dim=C, text_dim=TEXT_DIM, freq_dim=256, eps=1e-6,
                    patch_size=(1, 2, 2), num_heads=1, num_layers=layers, has_image_input=False,
                    seperated_timestep=True, require_vae_embedding=False, require_clip_embedding=False,
                    fuse_vae_embedding_in_latents=True).eval()


def walk_offsets(spec: str = "right:40,left:40") -> torch.Tensor:
    return wan.cell_offsets(offsets_from_actions(spec)).unsqueeze(0)          # [1, 21, 2]


@torch.no_grad()
def test_forward_equals_upstream_without_actionrope():
    """No offsets, no mask, one prompt: identical to DiffSynth's TI2V forward (first frame at t = 0)."""
    from diffsynth.pipelines.wan_video import model_fn_wan_video
    dit = tiny_dit()
    x, ctx, t = torch.randn(1, C, 3, 6, 8), torch.randn(1, 16, TEXT_DIM), torch.tensor([700.0])
    ref = model_fn_wan_video(dit=dit, latents=x, timestep=t, context=ctx, fuse_vae_embedding_in_latents=True)
    assert torch.equal(wan_dit.arope_forward(dit, x, t, ctx), ref)


@torch.no_grad()
def test_world_freqs_integer_offsets_shift_the_table():
    """Integer token offsets move the h/w rotary positions by exactly that many table rows."""
    dit = tiny_dit()
    f, h, w = 3, 4, 5
    off = torch.tensor([[[0.0, 0.0], [2.0, 1.0], [3.0, 4.0]]])               # (dx, dy) in tokens
    fr = wan_dit.build_world_freqs(dit, f, h, w, off).view(f, h, w, -1)
    f_tab, h_tab, w_tab = dit.freqs
    nf, nh = f_tab.shape[1], h_tab.shape[1]
    for k in range(f):
        dx, dy = int(off[0, k, 0]), int(off[0, k, 1])
        assert torch.equal(fr[k, :, :, :nf], f_tab[k].expand(h, w, -1))
        assert torch.equal(fr[k, :, 0, nf:nf + nh], h_tab[dy:dy + h])
        assert torch.equal(fr[k, 0, :, nf + nh:], w_tab[dx:dx + w])


@torch.no_grad()
def test_zero_offsets_and_zero_init_mask_channel_keep_the_output():
    dit = tiny_dit()
    x, ctx, t = torch.randn(1, C, 21, 30, 52), torch.randn(1, 16, TEXT_DIM), torch.tensor([500.0])
    ref = wan_dit.arope_forward(dit, x, t, ctx)
    assert torch.equal(wan_dit.arope_forward(dit, x, t, ctx, offset_px=torch.zeros(1, 21, 2)), ref)
    wan_dit.install_mask_channel(dit)
    off = walk_offsets()
    out = wan_dit.arope_forward(dit, x, t, ctx, offset_px=off, px_per_token=wan.PX_PER_TOKEN, mask_input=wan.known_input(off))
    assert torch.equal(out, wan_dit.arope_forward(dit, x, t, ctx, offset_px=off, px_per_token=wan.PX_PER_TOKEN))
    assert not torch.equal(out, ref)                                          # the offsets do change the output


@torch.no_grad()
def test_per_frame_context():
    """Equal prompts per latent frame == one prompt; context_ids group the frames by prompt identity."""
    dit = tiny_dit()
    x, t = torch.randn(1, C, 4, 6, 8), torch.tensor([300.0])
    rows = torch.randn(2, 16, TEXT_DIM)
    single = wan_dit.arope_forward(dit, x, t, rows[:1])
    per_frame = rows[[0, 0, 0, 0]].unsqueeze(0)
    assert torch.equal(wan_dit.arope_forward(dit, x, t, per_frame, context_ids=torch.zeros(1, 4, dtype=torch.long)), single)
    ids = torch.tensor([[0, 0, 1, 1]])
    mixed = rows[ids[0]].unsqueeze(0)
    out = wan_dit.arope_forward(dit, x, t, mixed, context_ids=ids)
    assert torch.equal(out, wan_dit.arope_forward(dit, x, t, mixed, context_ids=None))
    assert not torch.equal(out, single)


def test_known_maps_and_cns_levels():
    off = walk_offsets("right*1.5:80")
    kappa, kappa_tok = wan.cns_kappa(off)
    assert kappa.shape == (1, 1, 21, 30, 52) and kappa_tok.shape == (1, 21, 15, 26)
    assert torch.equal(kappa, wan.known_input(off))
    assert kappa[0, 0, 0].min() == 1                                          # the first frame is known
    assert 0 < kappa[0, 0, 20].mean() < 1                                     # walking right reveals new cells
    assert kappa[0, 0, 20, :, -1].max() == 0 and kappa[0, 0, 20, :, 0].min() == 1
    sig = torch.tensor(0.6)
    assert torch.allclose(cns_sigma(sig, torch.zeros(()), 5.0, 2.0), sig)   # newly revealed: base schedule
    assert cns_sigma(sig, torch.ones(()), 5.0, 2.0) < sig                   # known: lower noise
    w = loss_weight_map(off, wan.LAT_H, wan.LAT_W, wan.PX_PER_LATENT, 2.0)
    assert set(w.unique().tolist()) == {1.0, 2.0} and torch.equal(w == 1, kappa == 1)


def test_cell_offsets_are_the_mean_of_the_cell_frames():
    fo = offsets_from_actions("right:80")
    off = wan.cell_offsets(fo)
    assert off.shape == (21, 2) and off.dtype == torch.float32 and off[0].abs().max() == 0
    assert torch.allclose(off[1].double(), torch.from_numpy(fo[1:5].mean(0)), atol=1e-5)


def test_train_module_loss_and_mask_gradient():
    dit = tiny_dit(layers=1)
    wan_dit.install_mask_channel(dit)
    dit.train().requires_grad_(True)
    module = wan.TrainModule(dit, gradient_checkpointing=False)
    batch = {"latent": torch.randn(1, C, 21, 30, 52), "context": torch.randn(1, 21, 16, TEXT_DIM),
             "context_ids": torch.zeros(1, 21, dtype=torch.long), "offset_px": walk_offsets()}
    loss, stats = module(batch, 500, generator=torch.Generator().manual_seed(0))
    assert torch.isfinite(loss) and stats["loss_w"] > 0
    loss.backward()
    assert dit.arope_mask_embedding.weight.grad.abs().sum() > 0              # the known/new channel is wired in
    loss2, _ = module(batch, 500, generator=torch.Generator().manual_seed(0))
    assert torch.equal(loss.detach(), loss2.detach())                        # fixed generator -> fixed noise


@torch.no_grad()
def test_sampler_keeps_the_first_frame():
    dit = tiny_dit(layers=1)
    z0 = torch.randn(1, C, 1, 30, 52)
    off = walk_offsets()
    for cns in (None, {"shift_known": 2.0, "alpha": 1.0}):
        out = wan.sample(dit, z0, torch.randn(1, 16, TEXT_DIM), off, None, steps=2, seed=0, cns=cns)
        assert out.shape == (1, C, 21, 30, 52) and torch.equal(out[:, :, :1], z0) and torch.isfinite(out).all()


def test_dataset_prompts_ids_and_scene_dropout(tmp_path):
    a = "In a 2.5D top down view, the player is in a dark cave lit by blue crystal glow."
    b = "In a 2.5D top down view, the player is in a dark cave lit by blue crystal glow burning away into a warm farmland hamlet."
    table = {p: torch.full((16, TEXT_DIM), float(i)) for i, p in enumerate([NULL_PROMPT, a, b])}
    torch.save({"latent": torch.zeros(wan.LATENT_SHAPE, dtype=torch.bfloat16)}, tmp_path / "x.pt")
    clip = Clip(clip="x", split="train", kind="burn", valid=True, rt_ratio=None, sources=(),
                frame_offset_px=offsets_from_actions("right:80"), prompts=[a] * 10 + [b] * 11)
    s = wan.LatentDataset([clip], str(tmp_path), table)[0]
    assert s["context"].shape == (21, 16, TEXT_DIM) and s["offset_px"].shape == (21, 2)
    ids = s["context_ids"].tolist()
    assert len(set(ids[:10])) == 1 and len(set(ids[10:])) == 1 and ids[0] != ids[10]
    d = wan.LatentDataset([clip], str(tmp_path), table, scene_dropout=1.0)[0]
    assert torch.equal(d["context"], table[NULL_PROMPT].expand(21, -1, -1).to(torch.bfloat16))
    batch = wan.collate([s, s])
    assert batch["latent"].shape == (2,) + wan.LATENT_SHAPE and batch["clip"] == ["x", "x"]


def test_export_format():
    task = wan.TrainTask({"seed": 0})
    tensors, meta = task.export({"dit.blocks.0.weight": torch.zeros(1), "dit.arope_mask_embedding.bias": torch.zeros(1)}, 3)
    assert set(tensors) == {"blocks.0.weight", "arope_mask_embedding.bias"}
    assert meta["model"] == wan.MODEL_NAME and meta["format"] == "full" and meta["cns"] is True
    assert (meta["cns_shift"], meta["cns_shift_known"], meta["cns_alpha"]) == (5.0, 2.0, 1.0)
    assert np.isclose(task.timestep_value(833), 500.6, atol=0.01)
