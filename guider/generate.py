from pathlib import Path
import argparse

import torch

from .config import load_config
from .model import GuiderLM
from .tokenizer import ByteBPETokenizer


def resolve_checkpoint(cfg, requested=None):
    drive_value = cfg["checkpoint"].get("drive_dir")
    drive_dir = Path(drive_value) if drive_value else None
    local_dir = Path(cfg["checkpoint"]["local_dir"])

    candidates = {
        "best": [
            *( [drive_dir / cfg["checkpoint"]["best_filename"]] if drive_dir is not None else [] ),
            local_dir / cfg["checkpoint"]["best_filename"],
        ],
        "latest": [
            *( [drive_dir / cfg["checkpoint"]["filename"]] if drive_dir is not None else [] ),
            local_dir / cfg["checkpoint"]["filename"],
        ],
    }

    if requested and requested not in {"auto", "best", "latest"}:
        candidate = Path(requested)
        if candidate.exists():
            return candidate
        raise FileNotFoundError(f"Checkpoint not found: {candidate}")

    if requested == "best":
        search = candidates["best"]
    elif requested == "latest":
        search = candidates["latest"]
    else:
        search = candidates["best"] + candidates["latest"]

    for candidate in search:
        if candidate.exists():
            return candidate

    raise FileNotFoundError("No checkpoint found. Run python -m guider.train first.")


def main():
    parser = argparse.ArgumentParser(description="Generate text with a Guider checkpoint.")
    parser.add_argument(
        "--checkpoint",
        default="auto",
        help="auto, best, latest, or a direct checkpoint path",
    )
    args = parser.parse_args()

    cfg = load_config()
    checkpoint_path = resolve_checkpoint(cfg, args.checkpoint)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    tokenizer = ByteBPETokenizer.from_dict(checkpoint["tokenizer"])
    mc = checkpoint["config"]["model"]
    model = GuiderLM(
        tokenizer.vocab_size, mc["block_size"], mc["n_layer"],
        mc["n_head"], mc["n_embd"], mc["dropout"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    step = checkpoint.get("step", "?")
    best_val = checkpoint.get("best_val")
    print("Guider v0.3 | byte-level BPE")
    print(f"Checkpoint: {checkpoint_path} | step={step}")
    if best_val is not None:
        print(f"Best validation loss stored: {best_val:.4f}")
    print("Type a prompt. Ctrl+C to exit.\n")

    while True:
        prompt = input("> ").strip()
        if not prompt:
            print("Please enter a non-empty prompt.\n")
            continue
        ids = tokenizer.encode(prompt)
        x = torch.tensor([ids], dtype=torch.long, device=device)
        with torch.inference_mode():
            generated = model.generate(
                x,
                max_new_tokens=checkpoint["config"]["generation"]["max_new_tokens"],
                temperature=checkpoint["config"]["generation"]["temperature"],
                top_k=checkpoint["config"]["generation"]["top_k"],
            )
        print(tokenizer.decode(generated[0].tolist()))
        print()


if __name__ == "__main__":
    main()
