from pathlib import Path
import argparse

import torch

from .config import load_config
from .model import GuiderLM
from .tokenizer import ByteBPETokenizer


def resolve_checkpoint(cfg, requested="auto"):
    drive_value = cfg["checkpoint"].get("drive_dir")
    drive_dir = Path(drive_value) if drive_value else None
    local_dir = Path(cfg["checkpoint"]["local_dir"])
    candidates = {
        "best": ([drive_dir / cfg["checkpoint"]["best_filename"]] if drive_dir else [])
                + [local_dir / cfg["checkpoint"]["best_filename"]],
        "latest": ([drive_dir / cfg["checkpoint"]["filename"]] if drive_dir else [])
                  + [local_dir / cfg["checkpoint"]["filename"]],
    }
    if requested not in {"auto", "best", "latest"}:
        path = Path(requested)
        if path.exists():
            return path
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    search = candidates["best"] + candidates["latest"] if requested == "auto" else candidates[requested]
    for path in search:
        if path.exists():
            return path
    raise FileNotFoundError("No checkpoint found for this config.")


def trim_at_eos(ids, eos_id):
    try:
        return ids[:ids.index(int(eos_id))]
    except ValueError:
        return ids


def main():
    parser = argparse.ArgumentParser(description="Generate with Guider 0.4.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--checkpoint", default="auto", help="auto, best, latest, or checkpoint path")
    parser.add_argument("--prompt", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    path = resolve_checkpoint(cfg, args.checkpoint)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(path, map_location=device, weights_only=False)

    if checkpoint.get("architecture") != GuiderLM.ARCHITECTURE:
        raise ValueError("This is not a Guider 0.4 checkpoint. Use Guider 0.3 code for 0.3 weights.")

    tokenizer = ByteBPETokenizer.from_dict(checkpoint["tokenizer"])
    model = GuiderLM.from_config(tokenizer.vocab_size, checkpoint["config"]["model"]).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    gc = checkpoint["config"]["generation"]
    eos_id = tokenizer.eos_token_id
    print(f"{checkpoint['config'].get('run_name', 'Guider')} | step={checkpoint.get('step', '?')}")
    print(f"Checkpoint: {path}")
    if checkpoint.get("best_val") is not None:
        print(f"Best validation loss: {checkpoint['best_val']:.4f}")

    single_prompt = args.prompt
    while True:
        prompt = single_prompt if single_prompt is not None else input("> ").strip()
        if not prompt:
            print("Please enter a non-empty prompt.")
            if single_prompt is not None:
                return
            continue
        ids = tokenizer.encode(prompt)
        x = torch.tensor([ids], dtype=torch.long, device=device)
        with torch.inference_mode():
            generated = model.generate(
                x,
                max_new_tokens=int(gc["max_new_tokens"]),
                temperature=float(gc["temperature"]),
                top_k=gc.get("top_k"),
                top_p=float(gc.get("top_p", 1.0)),
                repetition_penalty=float(gc.get("repetition_penalty", 1.0)),
                eos_token_id=eos_id,
            )
        print(tokenizer.decode(trim_at_eos(generated[0].tolist(), eos_id)))
        print()
        if single_prompt is not None:
            return


if __name__ == "__main__":
    main()
