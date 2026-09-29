from pathlib import Path
import torch

from .data import CharTokenizer
from .model import GuiderLM


CHECKPOINT = Path("checkpoints/guider.pt")


def main():
    if not CHECKPOINT.exists():
        raise FileNotFoundError("No checkpoint found. Run: python -m guider.train")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint = torch.load(CHECKPOINT, map_location=device, weights_only=False)
    cfg = checkpoint["config"]
    tokenizer = CharTokenizer("")
    tokenizer.chars = checkpoint["vocab"]
    tokenizer.stoi = {ch: i for i, ch in enumerate(tokenizer.chars)}
    tokenizer.itos = {i: ch for i, ch in enumerate(tokenizer.chars)}

    m = cfg["model"]
    model = GuiderLM(
        vocab_size=len(tokenizer.chars),
        block_size=m["block_size"],
        n_layer=m["n_layer"],
        n_head=m["n_head"],
        n_embd=m["n_embd"],
        dropout=m["dropout"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    print("Guider v0.1")
    print("Type a prompt. Ctrl+C to exit.\n")

    while True:
        prompt = input("> ")
        ids = tokenizer.encode(prompt)
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
