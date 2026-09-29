# Guider AI

Guider is an experimental language model built from scratch with AI-assisted development.

The first milestone is intentionally small: a character-level decoder-only Transformer that can be trained locally from a plain text file and used from a CLI.

## Requirements

- Python 3.10+
- PyTorch
- A CPU works for the first tests; a CUDA GPU is strongly recommended for larger runs.

## Quick start

### 1. Create an environment

```bash
python -m venv .venv
source .venv/bin/activate
```

Windows PowerShell:

```powershell
.venv\\Scripts\\Activate.ps1
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Add training data

Put plain text in:

```
data/train.txt
```

For the first experiment, a few MB is enough to verify the pipeline. This first implementation uses a character-level tokenizer, so the training text should contain the languages and characters you want the model to learn.

### 4. Train

```bash
python -m guider.train
```

Checkpoints are written to `checkpoints/`.

### 5. Generate text

```bash
python -m guider.generate
```

## Project structure

```
guider-ai/
├── data/
│   └── train.txt
├── guider/
│   ├── __init__.py
│   ├── config.py
│   ├── model.py
│   ├── data.py
│   ├── train.py
│   └── generate.py
├── checkpoints/
├── config.yaml
├── requirements.txt
└── README.md
```

## Current scope

This is deliberately **not** a production LLM yet.

The first milestone is:

1. Load text.
2. Build a character vocabulary.
3. Train a decoder-only Transformer with next-token prediction.
4. Save a checkpoint.
5. Generate text from the checkpoint.

Later milestones can replace the character tokenizer with BPE, scale the model, improve the dataset, add instruction tuning, and build the Guider CLI/application.

## Important

Do not judge model quality from the tiny starter dataset. The purpose of v0.1 is to prove that the complete training -> checkpoint -> generation pipeline works end to end.
