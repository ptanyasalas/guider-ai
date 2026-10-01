from __future__ import annotations

import argparse
import contextlib
import math
import os
from pathlib import Path
import random
import time

import numpy as np
import torch

from .config import load_config
from .data import get_batch, open_memmap
from .model import GuiderLM
from .tokenizer import ByteBPETokenizer


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def amp_context(enabled):
    return torch.autocast(device_type="cuda", dtype=torch.float16) if enabled else contextlib.nullcontext()


def learning_rate(optimizer, step, cfg):
    tc = cfg["training"]
    if tc["warmup_steps"] and step < tc["warmup_steps"]:
        lr = tc["learning_rate"] * (step + 1) / tc["warmup_steps"]
    else:
        progress = min(1.0, max(0.0, (step - tc["warmup_steps"]) / max(1, tc["max_steps"] - tc["warmup_steps"])))
        cosine = 0.5 * (1 + math.cos(math.pi * progress))
        lr = tc["min_learning_rate"] + (tc["learning_rate"] - tc["min_learning_rate"]) * cosine
    for group in optimizer.param_groups:
        group["lr"] = lr
    return lr


def raw_model(model):
    return getattr(model, "_orig_mod", model)


def atomic_save(state, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, tmp)
    os.replace(tmp, path)


def checkpoint_state(model, optimizer, scaler, step, best_val, wait, cfg, tokenizer, rng):
    state = {
        "format_version": 3,
        "architecture": raw_model(model).ARCHITECTURE,
        "step": step,
        "best_val": best_val,
        "wait_count": wait,
        "model_state": raw_model(model).state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scaler_state": scaler.state_dict(),
        "config": cfg,
        "tokenizer": tokenizer.to_dict(),
        "python_rng_state": random.getstate(),
        "numpy_rng_state": np.random.get_state(),
        "torch_rng_state": torch.get_rng_state(),
        "train_rng_state": rng.bit_generator.state,
    }
    if torch.cuda.is_available():
        state["cuda_rng_state"] = torch.cuda.get_rng_state_all()
    return state


def load_initial(path, model, tokenizer):
    state = torch.load(path, map_location="cpu", weights_only=False)
    if state.get("architecture") != raw_model(model).ARCHITECTURE:
        raise ValueError(
            f"Initial checkpoint architecture {state.get('architecture')!r} does not match "
            f"{raw_model(model).ARCHITECTURE!r}. Guider 0.3/0.4 checkpoints cannot initialize 0.4.1."
        )
    old_tok = ByteBPETokenizer.from_dict(state["tokenizer"])
    if old_tok.to_dict()["tokenizer_json"] != tokenizer.to_dict()["tokenizer_json"]:
        raise ValueError("Initial checkpoint tokenizer differs from current tokenizer.")
    model.load_state_dict(state["model_state"])
    print(f"Initialized weights from {path}; source step={state.get('step')}")


def restore_resume(path, model, optimizer, scaler, device, rng):
    state = torch.load(path, map_location=device, weights_only=False)
    if state.get("architecture") != raw_model(model).ARCHITECTURE:
        raise ValueError(
            f"Resume checkpoint architecture {state.get('architecture')!r} does not match "
            f"{raw_model(model).ARCHITECTURE!r}. Move old-version checkpoints aside before training."
        )
    model.load_state_dict(state["model_state"])
    optimizer.load_state_dict(state["optimizer_state"])
    for opt in optimizer.state.values():
        for key, value in opt.items():
            if torch.is_tensor(value):
                opt[key] = value.to(device)
    if state.get("scaler_state"):
        scaler.load_state_dict(state["scaler_state"])
    random.setstate(state["python_rng_state"])
    np.random.set_state(state["numpy_rng_state"])
    torch.set_rng_state(state["torch_rng_state"])
    if device.type == "cuda" and state.get("cuda_rng_state") is not None:
        torch.cuda.set_rng_state_all(state["cuda_rng_state"])
    if state.get("train_rng_state"):
        rng.bit_generator.state = state["train_rng_state"]
    return int(state.get("step", 0)), float(state.get("best_val", float("inf"))), int(state.get("wait_count", 0))


def evaluate(model, train_data, val_data, cfg, device, amp_enabled, step):
    model.eval()
    tc = cfg["training"]
    result = {}
    seed = int(tc["seed"]) + 100000 + step * 17
    with torch.no_grad():
        for name, data, offset in (("train", train_data, 1), ("val", val_data, 2)):
            rng = np.random.default_rng(seed + offset)
            losses = []
            for _ in range(int(tc["eval_steps"])):
                x, y = get_batch(data, int(tc.get("eval_batch_size", tc["batch_size"])), cfg["model"]["block_size"], device, rng)
                with amp_context(amp_enabled):
                    _, loss = model(x, y)
                losses.append(float(loss.item()))
            result[name] = float(np.mean(losses))
    model.train()
    return result


def init_wandb(cfg):
    enabled = os.environ.get("WANDB_MODE", "online").lower() != "disabled"
    if not enabled:
        print("Weights & Biases disabled via WANDB_MODE=disabled.")
        return None
    try:
        import wandb
    except ImportError:
        print("wandb is not installed; continuing without experiment tracking.")
        return None
    try:
        return wandb.init(
            project=os.environ.get("WANDB_PROJECT", "guider-ai"),
            entity=os.environ.get("WANDB_ENTITY") or None,
            name=os.environ.get("WANDB_RUN_NAME", cfg.get("run_name", "Guider 0.4.1")),
            config=cfg,
            resume="allow",
            save_code=True,
        )
    except Exception as exc:
        print(f"W&B initialization failed; training continues without remote tracking: {exc}")
        return None


def log_checkpoint_artifact(run, path, step, kind):
    if run is None:
        return
    try:
        import wandb
        artifact = wandb.Artifact(
            name=f"{run.project}-{kind}",
            type="model",
            metadata={"step": int(step), "checkpoint_file": path.name},
        )
        artifact.add_file(str(path), name=path.name)
        run.log_artifact(artifact, aliases=["latest", f"step-{step}"] if kind == "latest" else ["best", f"step-{step}"])
        print(f"Uploaded {kind} checkpoint to W&B Artifacts at step {step}.")
    except Exception as exc:
        print(f"Warning: W&B checkpoint upload failed at step {step}: {exc}")


def main():
    parser = argparse.ArgumentParser(description="Train Guider 0.4.1.")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    set_seed(int(cfg["training"]["seed"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_enabled = device.type == "cuda" and bool(cfg["training"]["amp"])
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")

    dc = cfg["data"]
    local_dir = Path(dc["local_dir"])
    required = [local_dir / dc["tokenizer_file"], local_dir / dc["train_file"], local_dir / dc["val_file"]]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Preprocessed files are missing:\n  - " + "\n  - ".join(missing)
            + "\nRun preprocesar.py with the same --config first."
        )
    tokenizer = ByteBPETokenizer.from_file(required[0], cfg["tokenizer"]["eos_token"])
    train_data = open_memmap(required[1])
    val_data = open_memmap(required[2])
    if train_data.size <= cfg["model"]["block_size"] + 1 or val_data.size <= cfg["model"]["block_size"] + 1:
        raise ValueError("Train and validation token files must each contain more than block_size + 1 tokens.")

    model = GuiderLM.from_config(tokenizer.vocab_size, cfg["model"]).to(device)
    try:
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["training"]["learning_rate"], weight_decay=cfg["training"]["weight_decay"], fused=device.type == "cuda")
    except TypeError:
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["training"]["learning_rate"], weight_decay=cfg["training"]["weight_decay"])
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    rng = np.random.default_rng(int(cfg["training"]["seed"]) + 1)
    ckpt_cfg = cfg["checkpoint"]
    ckpt_dir = Path(ckpt_cfg["local_dir"])
    latest = ckpt_dir / ckpt_cfg["filename"]
    best = ckpt_dir / ckpt_cfg["best_filename"]
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    if cfg["training"].get("init_from"):
        init_path = Path(cfg["training"]["init_from"])
        if not init_path.exists():
            raise FileNotFoundError(f"training.init_from not found: {init_path}")
        load_initial(init_path, model, tokenizer)

    start, best_val, wait = 0, float("inf"), 0
    if ckpt_cfg.get("resume", True) and latest.exists():
        start, best_val, wait = restore_resume(latest, model, optimizer, scaler, device, rng)
        print(f"Resumed {latest} at step {start}")

    if device.type == "cuda" and cfg["training"].get("compile", False):
        try:
            model = torch.compile(model)
        except Exception as exc:
            print(f"torch.compile disabled after error: {exc}")

    params = sum(p.numel() for p in raw_model(model).parameters())
    accum = int(cfg["training"].get("grad_accum_steps", 1))
    print(f"{cfg.get('run_name', 'Guider 0.4.1')} | device={device} AMP={amp_enabled} vocab={tokenizer.vocab_size:,} params={params:,} effective_batch={cfg['training']['batch_size'] * accum:,}")
    print(f"Train tokens={train_data.size:,}; validation tokens={val_data.size:,}")
    if start >= cfg["training"]["max_steps"]:
        print("Checkpoint already reached max_steps.")
        return

    run = init_wandb(cfg)
    artifact_interval = int(ckpt_cfg.get("artifact_interval", 1000))
    if artifact_interval < 1:
        artifact_interval = 1000
    completed = start
    began = time.perf_counter()
    model.train()
    optimizer.zero_grad(set_to_none=True)
    try:
        for step in range(start, int(cfg["training"]["max_steps"])):
            if step % int(cfg["training"]["eval_interval"]) == 0:
                losses = evaluate(model, train_data, val_data, cfg, device, amp_enabled, step)
                elapsed = (time.perf_counter() - began) / 60
                print(f"step {step:>6} | train {losses['train']:.4f} | val {losses['val']:.4f} | {elapsed:.1f} min")
                if run is not None:
                    run.log({"step": step, "train/loss": losses["train"], "val/loss": losses["val"], "elapsed_minutes": elapsed}, step=step)
                if losses["val"] < best_val - float(cfg["early_stopping"]["min_delta"]):
                    best_val, wait = losses["val"], 0
                    atomic_save(checkpoint_state(model, optimizer, scaler, step, best_val, wait, cfg, tokenizer, rng), best)
                    print(f"New best: {best}")
                    log_checkpoint_artifact(run, best, step, "best")
                else:
                    wait += 1
                if cfg["early_stopping"]["enabled"] and wait >= int(cfg["early_stopping"]["patience"]):
                    print("Early stopping: validation loss stopped improving.")
                    break

            lr = learning_rate(optimizer, step, cfg)
            for _ in range(accum):
                x, y = get_batch(train_data, int(cfg["training"]["batch_size"]), cfg["model"]["block_size"], device, rng)
                with amp_context(amp_enabled):
                    _, loss = model(x, y)
                    loss = loss / accum
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Non-finite loss at step {step}: {loss.item()}")
                scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(raw_model(model).parameters(), float(cfg["training"]["grad_clip"]))
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            completed = step + 1

            if run is not None and completed % int(cfg["training"].get("log_interval", 25)) == 0:
                run.log({"step": completed, "learning_rate": lr, "train/step": completed}, step=completed)

            if completed % int(ckpt_cfg["save_interval"]) == 0:
                atomic_save(checkpoint_state(model, optimizer, scaler, completed, best_val, wait, cfg, tokenizer, rng), latest)
                print(f"Saved local recovery checkpoint: {latest}")
                if completed % artifact_interval == 0:
                    log_checkpoint_artifact(run, latest, completed, "latest")

        atomic_save(checkpoint_state(model, optimizer, scaler, completed, best_val, wait, cfg, tokenizer, rng), latest)
        print(f"Training stopped at step {completed}; latest={latest}; best={best}")
        log_checkpoint_artifact(run, latest, completed, "latest")
    finally:
        if run is not None:
            run.finish()


if __name__ == "__main__":
    main()
