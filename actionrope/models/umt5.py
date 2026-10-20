"""UMT5-XXL prompt encoder of the Wan family (Wan2.2-TI2V-5B and SkyReels-V2 use the same encoder).

Prompts are tokenised to 512 tokens and the embeddings after the true sequence length are zeroed, exactly as
DiffSynth's ``WanVideoUnit_PromptEmbedder`` does. Training reads pre-computed embeddings from a text table
(``scripts/build_text_table.py``) so the 11 GB encoder does not have to be loaded.
"""
from __future__ import annotations

import os

import torch

T5_WEIGHTS = "models_t5_umt5-xxl-enc-bf16.pth"
T5_TOKENIZER = os.path.join("google", "umt5-xxl")


class UMT5Encoder:
    def __init__(self, model_dir: str, device: str = "cuda"):
        from diffsynth.pipelines.wan_video import ModelConfig, WanVideoPipeline
        self.device = device
        self.pipe = WanVideoPipeline.from_pretrained(
            torch_dtype=torch.bfloat16, device=device, redirect_common_files=False,
            model_configs=[ModelConfig(path=os.path.join(model_dir, T5_WEIGHTS))],
            tokenizer_config=ModelConfig(path=os.path.join(model_dir, T5_TOKENIZER)))
        self.pipe.load_models_to_device(["text_encoder"])

    @torch.no_grad()
    def __call__(self, prompts: list[str]) -> list[torch.Tensor]:
        """-> one bf16 [512, 4096] CPU tensor per prompt."""
        ids, mask = self.pipe.tokenizer(prompts, return_mask=True, add_special_tokens=True)
        ids, mask = ids.to(self.device), mask.to(self.device)
        seq_lens = mask.gt(0).sum(dim=1).long()
        emb = self.pipe.text_encoder(ids, mask)
        for j, n in enumerate(seq_lens):
            emb[j, n:] = 0
        return [e.to(torch.bfloat16).cpu().clone() for e in emb]


def build_text_table(prompts: list[str], model_dir: str, device: str = "cuda", batch_size: int = 32) -> dict:
    """{prompt: bf16 [512, 4096]} for the sorted, de-duplicated prompts (batches of ``batch_size``)."""
    enc = UMT5Encoder(model_dir, device)
    keys = sorted(set(prompts))
    table = {}
    for i in range(0, len(keys), batch_size):
        chunk = keys[i:i + batch_size]
        table.update(zip(chunk, enc(chunk)))
    return table
