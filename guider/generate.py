from pathlib import Path

import torch

from .model import GuiderLM
from .tokenizer import ByteBPETokenizer


CHECKPOINT = Path("checkpoints/guider.pt")


def main():
    if not CHECKPOINT.exists():
        raise FileNotFoundError("No checkpoint found. Run: python -m guider.train")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint = torch.load(CHECKPOINT, map_location=device, weights_only=False)

    cfg = checkpoint["config"]
    tokenizer = ByteBPETokenizer.from_dict(checkpoint["tokenizer"])

    model_cfg = cfg["model"]
    model = GuiderLM(
        vocab_size=tokenizer.vocab_size,
        block_size=model_cfg["block_size"],
        n_layer=model_cfg["n_layer"],
        n_head=model_cfg["n_head"],
        n_embd=model_cfg["n_embd"],
        dropout=model_cfg["dropout"],
    ).to(device)

    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    print("Guider v0.2 | byte-level BPE tokenizer")
    print("Type a prompt. Ctrl+C to exit.\n")

    while True:
        prompt = input("> ")
        ids = tokenizer.encode(prompt)

        if not ids:
            print("Please enter a non-empty prompt.\n")
            continue

        x = torch.tensor([ids], dtype=torch.long, device=device)
        generated = model.generate(
            x,
            max_new_tokens=cfg["generation"]["max_new_tokens"],
            temperature=cfg["generation"]["temperature"],
            top_k=cfg["generation"]["top_k"],
        )

        print(tokenizer.decode(generated[0].tolist()))
        print()


if __name__ == "__main__":
    main()
