"""Training loop shared by all backbones (Accelerate + DeepSpeed ZeRO-2, bf16).

A backbone provides a *task* object with

    task.build_module()            -> nn.Module on CPU; ``module(batch, timestep_id, generator=None, train=True)``
                                      returns ``(loss, stats)`` (stats: dict of floats / 0-d tensors)
    task.datasets()                -> (train_dataset, val_dataset or None)
    task.collate(samples)          -> batch
    task.on_prepared(module, dev)  -> called once after ``accelerator.prepare`` (optional)
    task.export(state_dict, step)  -> (tensors, metadata) to write for the gathered ``state_dict``
    task.timestep_value(tid)       -> timestep of training index ``tid`` (for logging)

and the loop below does the rest: AdamW, linear warm-up, gradient clipping, gradient accumulation, one random
timestep index per micro-batch, periodic validation at fixed timesteps with fixed noise, and checkpoints.
"""
from __future__ import annotations

import csv
import json
import math
import os
import time
import zlib

import torch
from accelerate import Accelerator
from accelerate.utils import DistributedType, set_seed

from .checkpoint import save_checkpoint


def to_device(batch, device):
    if torch.is_tensor(batch):
        return batch.to(device, non_blocking=True)
    if isinstance(batch, dict):
        return {k: to_device(v, device) for k, v in batch.items()}
    if isinstance(batch, (list, tuple)) and batch and torch.is_tensor(batch[0]):
        return type(batch)(to_device(v, device) for v in batch)
    return batch


class CsvLog:
    def __init__(self, path: str, fields: list[str]):
        self.fh = open(path, "a", newline="", encoding="utf-8")
        self.w = csv.DictWriter(self.fh, fieldnames=fields, extrasaction="ignore")
        if self.fh.tell() == 0:
            self.w.writeheader()

    def write(self, row: dict):
        self.w.writerow(row)
        self.fh.flush()


@torch.no_grad()
def validate(accelerator, model, task, val_ds, tids, seed):
    """Mean weighted loss over the validation clips for each timestep index in ``tids`` (fixed noise per clip)."""
    model.eval()
    rank, world, device = accelerator.process_index, accelerator.num_processes, accelerator.device
    acc = torch.zeros(len(tids), 2, device=device, dtype=torch.float64)
    for i in range(rank, len(val_ds), world):
        sample = val_ds[i]
        batch = to_device(task.collate([sample]), device)
        for j, tid in enumerate(tids):
            g = torch.Generator(device=device)
            g.manual_seed((zlib.crc32(str(sample.get("clip", i)).encode()) * 1009 + tid * 7 + seed) % (2 ** 31))
            _, stats = model(batch, tid, generator=g, train=False)
            acc[j, 0] += float(stats["loss_w"])
            acc[j, 1] += 1
    acc = accelerator.reduce(acc, reduction="sum")
    model.train()
    return {tid: (acc[j, 0] / acc[j, 1].clamp(min=1)).item() for j, tid in enumerate(tids)}


def train(cfg: dict, task) -> None:
    """``cfg`` keys: output, lr, weight_decay, warmup_steps, max_steps, grad_accum, grad_clip, save_every,
    val_every, val_timesteps, num_workers, seed, log_every."""
    accelerator = Accelerator(gradient_accumulation_steps=int(cfg["grad_accum"]))
    is_main = accelerator.is_main_process
    set_seed(int(cfg["seed"]), device_specific=True)
    if accelerator.distributed_type == DistributedType.DEEPSPEED:
        ds = accelerator.state.deepspeed_plugin.deepspeed_config
        ds["gradient_clipping"] = float(cfg["grad_clip"])   # DeepSpeed clips inside engine.step()
        if ds.get("gradient_accumulation_steps") not in ("auto", int(cfg["grad_accum"])):
            raise SystemExit(f"grad_accum={cfg['grad_accum']} does not match the accelerate config "
                             f"(gradient_accumulation_steps={ds.get('gradient_accumulation_steps')})")
        ds["gradient_accumulation_steps"] = int(cfg["grad_accum"])
    out = cfg["output"]
    if is_main:
        os.makedirs(out, exist_ok=True)
        with open(os.path.join(out, "config.json"), "w", encoding="utf-8") as fh:
            json.dump(cfg, fh, indent=2, default=str)

    train_ds, val_ds = task.datasets()
    loader = torch.utils.data.DataLoader(train_ds, batch_size=1, shuffle=True, drop_last=True, collate_fn=task.collate,
                                         num_workers=int(cfg["num_workers"]), pin_memory=True,
                                         persistent_workers=int(cfg["num_workers"]) > 0)
    t0 = time.time()
    module = task.build_module()
    params = [p for p in module.parameters() if p.requires_grad]
    if is_main:
        n_all = sum(p.numel() for p in module.parameters())
        print(f"[model] built in {time.time() - t0:.0f}s: {sum(p.numel() for p in params):,} trainable / {n_all:,} parameters",
              flush=True)
    optimizer = torch.optim.AdamW(params, lr=float(cfg["lr"]), weight_decay=float(cfg["weight_decay"]))
    warm = max(int(cfg["warmup_steps"]), 0)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: min(1.0, (s + 1) / warm) if warm > 0 else 1.0)
    model, optimizer, loader = accelerator.prepare(module, optimizer, loader)
    if hasattr(task, "on_prepared"):
        task.on_prepared(module, accelerator.device)

    log = CsvLog(os.path.join(out, "train_log.csv"), ["step", "loss", "loss_w", "timestep", "lr", "grad_norm", "sec"]) if is_main else None
    vlog = CsvLog(os.path.join(out, "val_log.csv"), ["step", "timestep", "loss_w"]) if is_main else None
    val_tids = [int(t) for t in cfg.get("val_timesteps", [])]
    # one timestep index per micro-batch; the generator differs per rank so the GPUs see different noise levels
    gen = torch.Generator().manual_seed(int(cfg["seed"]) * 1000 + accelerator.process_index)

    def save(step):
        accelerator.wait_for_everyone()
        sd = accelerator.get_state_dict(model)
        if is_main:
            tensors, meta = task.export(sd, step)
            path = os.path.join(out, f"step-{step}.safetensors")
            save_checkpoint(path, tensors, {**meta, "step": step})
            print(f"[save] {path} ({len(tensors)} tensors)", flush=True)
        del sd
        accelerator.wait_for_everyone()

    if is_main:
        print(f"[train] {len(train_ds)} clips, {accelerator.num_processes} processes, grad_accum {cfg['grad_accum']}, "
              f"{cfg['max_steps']} steps", flush=True)
    model.train()
    step, t_acc, last_saved = 0, 0.0, 0
    while step < int(cfg["max_steps"]):
        for batch in loader:
            if step >= int(cfg["max_steps"]):
                break
            t_start = time.time()
            batch = to_device(batch, accelerator.device)
            tid = int(torch.randint(0, 1000, (1,), generator=gen))
            lr_now = scheduler.get_last_lr()[0]
            with accelerator.accumulate(model):
                loss, stats = model(batch, tid)
                accelerator.backward(loss)
                grad_norm = accelerator.clip_grad_norm_(params, float(cfg["grad_clip"])) if accelerator.sync_gradients else None
                optimizer.step()
                if accelerator.sync_gradients:       # step the warm-up per optimiser step, not per micro-batch
                    scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            t_acc += time.time() - t_start
            if not accelerator.sync_gradients:
                continue
            step += 1
            loss_mean = accelerator.reduce(loss.detach().float(), reduction="mean").item()
            if is_main and (step % int(cfg.get("log_every", 10)) == 0 or step <= 5):
                gn = float(grad_norm) if grad_norm is not None and math.isfinite(float(grad_norm)) else float("nan")
                row = {"step": step, "loss": loss_mean, "loss_w": float(stats["loss_w"]),
                       "timestep": float(task.timestep_value(tid)), "lr": lr_now, "grad_norm": gn, "sec": t_acc}
                log.write(row)
                print(f"step {step}/{cfg['max_steps']} loss {loss_mean:.4f} (w {row['loss_w']:.4f}, t {row['timestep']:.0f}) "
                      f"lr {lr_now:.2e} gnorm {gn:.3f} {t_acc:.2f}s", flush=True)
            t_acc = 0.0
            if val_ds is not None and int(cfg.get("val_every", 0)) > 0 and step % int(cfg["val_every"]) == 0:
                res = validate(accelerator, model, task, val_ds, val_tids, int(cfg["seed"]))
                if is_main:
                    for tid_v, lw in res.items():
                        vlog.write({"step": step, "timestep": float(task.timestep_value(tid_v)), "loss_w": lw})
                    print(f"[val] step {step} " + " ".join(f"t={task.timestep_value(k):.0f}: {v:.4f}" for k, v in res.items()), flush=True)
            if int(cfg["save_every"]) > 0 and step % int(cfg["save_every"]) == 0:
                save(step)
                last_saved = step
    if last_saved != step:
        save(step)
    accelerator.wait_for_everyone()
    accelerator.end_training()
