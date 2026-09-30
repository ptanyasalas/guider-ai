from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from guider.model import GuiderLM
from guider.model_v03 import GuiderLMv03
from guider.tokenizer import ByteBPETokenizer


def trim_at_eos(ids, eos_id):
    try:
        return ids[:ids.index(int(eos_id))]
    except ValueError:
        return ids


def load_model(path, device):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    tokenizer = ByteBPETokenizer.from_dict(checkpoint["tokenizer"])
    if checkpoint.get("architecture") == GuiderLM.ARCHITECTURE:
        model = GuiderLM.from_config(tokenizer.vocab_size, checkpoint["config"]["model"]).to(device)
        version = checkpoint["config"].get("run_name", "Guider 0.4")
    else:
        cfg = checkpoint["config"]["model"]
        model = GuiderLMv03(
            tokenizer.vocab_size,
            cfg["block_size"],
            cfg["n_layer"],
            cfg["n_head"],
            cfg["n_embd"],
            cfg["dropout"],
        ).to(device)
        version = "Guider 0.3"
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, tokenizer, checkpoint, version


def generate_one(model, tokenizer, checkpoint, prompt, device, seed):
    gc = checkpoint["config"]["generation"]
    torch.manual_seed(seed)
    ids = tokenizer.encode(prompt)
    x = torch.tensor([ids], dtype=torch.long, device=device)
    with torch.inference_mode():
        if checkpoint.get("architecture") == GuiderLM.ARCHITECTURE:
            output = model.generate(
                x,
                max_new_tokens=int(gc.get("max_new_tokens", 200)),
                temperature=float(gc.get("temperature", 0.8)),
                top_k=gc.get("top_k"),
                top_p=float(gc.get("top_p", 1.0)),
                repetition_penalty=float(gc.get("repetition_penalty", 1.0)),
                eos_token_id=tokenizer.eos_token_id,
            )
        else:
            output = model.generate(
                x,
                max_new_tokens=int(gc.get("max_new_tokens", 200)),
                temperature=float(gc.get("temperature", 0.8)),
                top_k=gc.get("top_k"),
            )
    return tokenizer.decode(trim_at_eos(output[0].tolist(), tokenizer.eos_token_id))


def main():
    parser = argparse.ArgumentParser(description="Evaluate Guider checkpoints on a fixed prompt set.")
    parser.add_argument("--checkpoint", required=True, help="Path to a Guider .pt checkpoint")
    parser.add_argument("--label", default=None, help="Human-readable checkpoint label")
    parser.add_argument("--prompts", default="eval_prompts.json")
    parser.add_argument("--output", default="evaluation_results.md")
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)
    prompts = json.loads(Path(args.prompts).read_text(encoding="utf-8"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer, checkpoint, version = load_model(checkpoint_path, device)
    label = args.label or version
    lines = [
        "# Guider fixed-prompt evaluation",
        "",
        f"- Label: {label}",
        f"- Checkpoint: {checkpoint_path}",
        f"- Step: {checkpoint.get('step', '?')}",
        f"- Best validation loss stored: {checkpoint.get('best_val', 'unknown')}",
        f"- Device: {device}",
        "",
        "This is a qualitative fixed-prompt comparison, not a benchmark score.",
        "",
    ]
    for index, prompt in enumerate(prompts):
        result = generate_one(model, tokenizer, checkpoint, prompt, device, args.seed + index)
        lines.extend([
            f"## Prompt {index + 1}",
            "",
            f"**Prompt:** {prompt}",
            "",
            "**Output:**",
            "",
            result,
            "",
        ])
    Path(args.output).write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved {len(prompts)} prompt results to {args.output}")


if __name__ == "__main__":
    main()
