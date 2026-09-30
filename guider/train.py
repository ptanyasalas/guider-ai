from __future__ import annotations

import contextlib
import math
import os
from pathlib import Path
import random
import time

import numpy as np
import torch

from .config import load_config
from .data import get_batch, open_memmap, prepare_data
from .model import GuiderLM
from .tokenizer import ByteBPETokenizer


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def set_learning_rate(optimizer, step, cfg):
    tc = cfg["training"]
    if tc["warmup_steps"] > 0 and step < tc["warmup_steps"]:
        lr = tc["learning_rate"] * (step + 1) / tc["warmup_steps"]
    else:
        progress = min(1.0, max(0.0, (step - tc["warmup_steps"]) /
                                max(1, tc["max_steps"] - tc["warmup_steps"])))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        lr = tc["min_learning_rate"] + (tc["learning_rate"] - tc["min_learning_rate"]) * cosine
    for group in optimizer.param_groups:
        group["lr"] = lr
    return lr


def amp_context(enabled):
    return torch.autocast(device_type="cuda", dtype=torch.float16) if enabled else contextlib.nullcontext()


def estimate_loss(model, train_data, val_data, cfg, device, train_rng, val_rng, amp_enabled):
    model.eval()
    result = {}
    tc, block_size = cfg["training"], cfg["model"]["block_size"]
    with torch.no_grad():
        for name, data, rng in (("train", train_data, train_rng), ("val", val_data, val_rng)):
            losses = []
            for _ in range(tc["eval_steps"]):
                x, y = get_batch(data, tc["batch_size"], block_size, device, rng)
                with amp_context(amp_enabled):
                    _, loss = model(x, y)
                losses.append(loss.item())
            result[name] = float(np.mean(losses))
    model.train()
    return result


def atomic_save(state, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, temporary)
    os.replace(temporary, path)


def raw_model(model):
    return getattr(model, "_orig_mod", model)


def make_checkpoint(model, optimizer, scaler, step, best_val, wait_count, cfg, tokenizer, train_rng, val_rng):
    state = {
        "step": step,
        "best_val": best_val,
        "wait_count": wait_count,
        "model_state": raw_model(model).state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scaler_state": scaler.state_dict(),
        "config": cfg,
        "tokenizer": tokenizer.to_dict(),
        "python_rng_state": random.getstate(),
        "numpy_rng_state": np.random.get_state(),
        "torch_rng_state": torch.get_rng_state(),
        "train_rng_state": train_rng.bit_generator.state,
        "val_rng_state": val_rng.bit_generator.state,
    }
    if torch.cuda.is_available():
        state["cuda_rng_state"] = torch.cuda.get_rng_state_all()
    return state


def load_checkpoint(path, model, optimizer, scaler, device, train_rng, val_rng):
    state = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model_state"])
    optimizer.load_state_dict(state["optimizer_state"])
    for opt_state in optimizer.state.values():
        for key, value in opt_state.items():
            if torch.is_tensor(value):
                opt_state[key] = value.to(device)
    if state.get("scaler_state"):
        scaler.load_state_dict(state["scaler_state"])
    random.setstate(state["python_rng_state"])
    np.random.set_state(state["numpy_rng_state"])
    torch.set_rng_state(state["torch_rng_state"])
    if device.type == "cuda" and state.get("cuda_rng_state") is not None:
        torch.cuda.set_rng_state_all(state["cuda_rng_state"])
    if state.get("train_rng_state"):
        train_rng.bit_generator.state = state["train_rng_state"]
    if state.get("val_rng_state"):
        val_rng.bit_generator.state = state["val_rng_state"]
    return int(state.get("step", 0)), float(state.get("best_val", float("inf"))), int(state.get("wait_count", 0))


def checkpoint_directory(cfg):
    drive_root = Path("/content/drive/MyDrive")
    if drive_root.exists():
        return Path(cfg["checkpoint"]["drive_dir"])
    local = Path(cfg["checkpoint"]["local_dir"])
    print(f"Drive is not mounted; checkpoints will be local: {local}")
    return local


def main():
    cfg = load_config()
    set_seed(cfg["training"]["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_enabled = device.type == "cuda" and cfg["training"]["amp"]

    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")

    dc = cfg["data"]
    local_dir = Path(dc["local_dir"])
    persistent_dir = Path(dc["persistent_dir"]) if dc.get("persistent_dir") else None
    prepare_data(local_dir, persistent_dir, [dc["train_file"], dc["val_file"], dc["tokenizer_file"]])
    tokenizer = ByteBPETokenizer.from_file(local_dir / dc["tokenizer_file"], cfg["tokenizer"]["eos_token"])
    if tokenizer.vocab_size > np.iinfo(np.uint16).max + 1:
        raise ValueError("Tokenizer vocabulary does not fit in uint16.")

    train_data = open_memmap(local_dir / dc["train_file"])
    val_data = open_memmap(local_dir / dc["val_file"])
    mc = cfg["model"]
    model = GuiderLM(tokenizer.vocab_size, mc["block_size"], mc["n_layer"], mc["n_head"], mc["n_embd"], mc["dropout"]).to(device)
    try:
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=cfg["training"]["learning_rate"],
            weight_decay=cfg["training"]["weight_decay"], fused=(device.type == "cuda"),
        )
    except TypeError:
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=cfg["training"]["learning_rate"],
            weight_decay=cfg["training"]["weight_decay"],
        )
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    train_rng = np.random.default_rng(cfg["training"]["seed"] + 1)
    val_rng = np.random.default_rng(cfg["training"]["seed"] + 2)
    ckpt_dir = checkpoint_directory(cfg)
    latest_path = ckpt_dir / cfg["checkpoint"]["filename"]
    best_path = ckpt_dir / cfg["checkpoint"]["best_filename"]
    start_step, best_val, wait_count = 0, float("inf"), 0

    if cfg["checkpoint"]["resume"] and latest_path.exists():
        print(f"Resuming from {latest_path}")
        start_step, best_val, wait_count = load_checkpoint(
            latest_path, model, optimizer, scaler, device, train_rng, val_rng
        )

    if device.type == "cuda" and cfg["training"]["compile"]:
        try:
            model = torch.compile(model)
            print("torch.compile enabled (first step may take longer).")
        except Exception as exc:
            print(f"torch.compile unavailable; continuing without it: {exc}")

    params = sum(p.numel() for p in raw_model(model).parameters())
    print(
        f"Guider v0.3 | device={device} | AMP={amp_enabled} | vocab={tokenizer.vocab_size:,} | "
        f"train={train_data.size:,} tokens | val={val_data.size:,} tokens | params={params:,}"
    )
    if start_step >= cfg["training"]["max_steps"]:
        print("Checkpoint has already reached max_steps.")
        return

    started = time.perf_counter()
    completed_step = start_step
    step = start_step
    for step in range(start_step, cfg["training"]["max_steps"]):
        if step % cfg["training"]["eval_interval"] == 0:
            losses = estimate_loss(model, train_data, val_data, cfg, device, train_rng, val_rng, amp_enabled)
            print(
                f"step {step:>6} | train {losses['train']:.4f} | val {losses['val']:.4f} | "
                f"elapsed {(time.perf_counter() - started) / 60:.1f} min"
            )
            if losses["val"] < best_val - cfg["early_stopping"]["min_delta"]:
                best_val, wait_count = losses["val"], 0
                atomic_save(make_checkpoint(model, optimizer, scaler, step, best_val, wait_count, cfg, tokenizer, train_rng, val_rng), best_path)
                print(f"New best checkpoint: {best_path}")
            else:
                wait_count += 1
            if cfg["early_stopping"]["enabled"] and wait_count >= cfg["early_stopping"]["patience"]:
                print("Early stopping: validation loss has stopped improving.")
                break

        set_learning_rate(optimizer, step, cfg)
        optimizer.zero_grad(set_to_none=True)
        x, y = get_batch(train_data, cfg["training"]["batch_size"], mc["block_size"], device, train_rng)
        with amp_context(amp_enabled):
            _, loss = model(x, y)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(raw_model(model).parameters(), cfg["training"]["grad_clip"])
        scaler.step(optimizer)
        scaler.update()

        completed_step = step + 1
        if completed_step % cfg["checkpoint"]["save_interval"] == 0:
            atomic_save(make_checkpoint(model, optimizer, scaler, completed_step, best_val, wait_count, cfg, tokenizer, train_rng, val_rng), latest_path)
            print(f"Checkpoint saved: {latest_path}")

    atomic_save(make_checkpoint(model, optimizer, scaler, completed_step, best_val, wait_count, cfg, tokenizer, train_rng, val_rng), latest_path)
    print(f"Training finished. Latest: {latest_path}; best: {best_path}")


if __name__ == "__main__":
    main()
