"""CPU tests of the backbone-agnostic core: python -m pytest tests/test_core.py"""
import json
import math

import numpy as np
import pytest
import torch

from actionrope import geometry as G
from actionrope.cns import cns_sigma
from actionrope.coords import known_mask, knownness, loss_weight_map
from actionrope.data import load_annotations, training_clips, validation_clips
from actionrope.lora import LoRALinear, install_lora, lora_state_dict, merged_state_dict
from actionrope.prompts import NULL_PROMPT, strip_action


def test_actions_integrate_to_offsets():
    off = G.offsets_from_actions("right:40,left:40")
    v = G.WALK_SPEED_PX_S[0] / G.FPS
    assert off.shape == (G.FRAMES, 2)
    assert np.allclose(off[0], 0) and np.allclose(off[40], [40 * v, 0]) and np.allclose(off[80], 0, atol=1e-9)
    diag = G.offsets_from_actions("up-left*2:10")
    assert np.allclose(diag[10], np.array([-1, -1]) * math.sqrt(0.5) * np.array(G.WALK_SPEED_PX_S) * 2 * 10 / G.FPS)
    assert np.allclose(diag[11:], diag[10])                       # idle after the last segment
    with pytest.raises(ValueError):
        G.offsets_from_actions("right:81")
    with pytest.raises(ValueError):
        G.offsets_from_actions("sideways:10")


def test_frames_to_cells_groups_causal_vae_frames():
    off = np.stack([np.arange(G.FRAMES), -np.arange(G.FRAMES)], 1).astype(float)
    cells = G.frames_to_cells(off)
    assert cells.shape == (21, 2)
    assert cells[0, 0] == 0 and cells[1, 0] == 2.5 and cells[20, 0] == 78.5


def test_known_mask_geometry():
    h, w, p = 30, 52, 16
    off = torch.zeros(1, 2, 2)
    off[0, 1] = torch.tensor([3 * p, 0.0])                       # camera moved 3 cells to the right
    m = known_mask(off, h, w, p)
    assert m[0, 0].all()                                          # frame 0 is the footprint itself
    assert m[0, 1, :, : w - 3].all() and not m[0, 1, :, w - 3:].any()   # the right-most 3 columns are new
    off[0, 1] = torch.tensor([0.5 * p, 0.0])                      # half a cell: boundary is closed -> still known
    assert known_mask(off, h, w, p)[0, 1].all()
    wmap = loss_weight_map(off, h, w, p, new_weight=2.0)
    assert wmap.shape == (1, 1, 2, h, w) and set(wmap.unique().tolist()) <= {1.0, 2.0}
    assert knownness(off, h, w, p).dtype == torch.float32


def test_cns_sigma_endpoints_and_order():
    kappa = torch.tensor([0.0, 0.5, 1.0])
    assert torch.allclose(cns_sigma(1.0, kappa, 5.0, 2.0), torch.ones(3))
    assert torch.allclose(cns_sigma(0.0, kappa, 5.0, 2.0), torch.zeros(3))
    s = cns_sigma(0.6, kappa, 5.0, 2.0)
    assert abs(float(s[0]) - 0.6) < 1e-6                          # newly revealed tokens keep the base schedule
    assert s[2] < s[1] < s[0]                                     # known tokens are less noisy


def test_lora_zero_init_and_merge():
    torch.manual_seed(0)
    net = torch.nn.Sequential(torch.nn.Linear(8, 8), torch.nn.ReLU(), torch.nn.Linear(8, 4))
    x = torch.randn(3, 8)
    ref = net(x)
    n = install_lora(net, lambda name: name in ("0", "2"), rank=2, alpha=4)
    assert n == 2 and torch.equal(net(x), ref)                    # B = 0: unchanged at installation
    for m in net.modules():
        if isinstance(m, LoRALinear):
            torch.nn.init.normal_(m.lora_B)
    out = net(x)
    plain = torch.nn.Sequential(torch.nn.Linear(8, 8), torch.nn.ReLU(), torch.nn.Linear(8, 4))
    plain.load_state_dict(merged_state_dict(net))
    assert torch.allclose(plain(x), out, atol=1e-6)
    assert set(lora_state_dict(net)) == {"0.lora_A", "0.lora_B", "2.lora_A", "2.lora_B"}


def test_prompts():
    p = "In a 2.5D top down view, in a dark cave lit by blue crystal glow, the player is moving right."
    assert strip_action(p) == "In a 2.5D top down view, the player is in a dark cave lit by blue crystal glow."
    assert strip_action("In a 2.5D top down view, the player is standing idle.") == NULL_PROMPT


def test_training_filters(tmp_path):
    rows = [
        dict(clip="a", split="train", rt_ratio=1.0),
        dict(clip="b", split="train", rt_ratio=0.5),                              # too slow
        dict(clip="c", split="train", valid=False),                               # invalid camera track
        dict(clip="d", split="train", kind="burn", sources=["a", "v"]),           # composed from a val clip
        dict(clip="e", split="train", kind="hardcut", sources=["a", "b"]),
        dict(clip="v", split="val", rt_ratio=0.95),
    ]
    path = tmp_path / "ann.jsonl"
    with open(path, "w") as fh:
        for r in rows:
            fh.write(json.dumps({**r, "frame_offset_px": [[0, 0]] * G.FRAMES, "prompts": [NULL_PROMPT] * 21}) + "\n")
    clips = load_annotations(str(path))
    assert [c.clip for c in training_clips(clips, verbose=False)] == ["a", "e"]
    assert [c.clip for c in validation_clips(clips)] == ["v"]
