# Guider AI

Guider is an experimental decoder-only Transformer language model designed for Google Colab GPU experiments. Colab's local disk is temporary, so training checkpoints are also uploaded periodically to Weights & Biases (W&B) Artifacts when W&B is configured.

## Guider 0.4.1

- 10-layer, 384-wide Transformer; 512-token context; byte-level BPE tokenizer.
- RoPE, RMSNorm and SwiGLU blocks.
- Streaming Hugging Face datasets, with preflight checks before expensive tokenization.
- Atomic token-file and checkpoint writes, gradient accumulation, FP16 AMP, deterministic validation sampling, gradient clipping, and resume support.
- W&B logs training/validation loss and learning rate; checkpoint artifacts are uploaded periodically.

**Compatibility:** Guider 0.4.1 uses a distinct architecture identifier. Guider 0.3 and earlier 0.4 checkpoints are not valid resume/initialization checkpoints for 0.4.1. Keep older files for comparison; do not overwrite them.

## 1. Colab setup

Choose a GPU runtime (T4 if available), then run:

```python
!git clone https://github.com/ptanyasalas/guider-ai.git
%cd guider-ai
!pip install -r requirements.txt
!nvidia-smi
```

If you already cloned the repository in this runtime, use `%cd guider-ai` and `!git pull` instead of cloning it again.

## 2. Set up W&B

1. Create/sign in to your W&B account and create an API key in your account settings.
2. In Colab, open the **Secrets** panel (key icon), add a secret named `WANDB_API_KEY`, paste the key there, and enable notebook access to that secret.
3. Run:

```python
from google.colab import userdata
import os

os.environ["WANDB_API_KEY"] = userdata.get("WANDB_API_KEY")
os.environ["WANDB_PROJECT"] = "guider-ai"
```

Do not paste the API key into repository files or share it in notebook output. If you prefer to run without remote tracking, set `os.environ["WANDB_MODE"] = "disabled"` before training.

W&B dashboard: https://wandb.ai/ — choose project `guider-ai`. Training curves appear in the run; checkpoint copies appear under the run's **Artifacts** section.

## 3. Preprocess Guider 0.4.1

The first-run config deliberately uses a smaller token cap so you can verify the full pipeline before spending a long GPU session. It uses the documented `sample-10BT` FineWeb-Edu subset and TinyStories.

```python
!python preprocesar.py --config configs/guider_0_4_1.yaml
```

The script now checks each dataset source before doing the main work and writes token files atomically. If it fails, copy the full traceback, including the first error line and the final lines. Do not start training unless preprocessing reports completion.

Local files are written to `data/guider_0_4_1/`. The starter config caps training at 10 million tokens and validation at 500,000 tokens. These are initial test settings, not a claim that the model is fully trained.

## 4. Train

After preprocessing completes:

```python
!python -m guider.train --config configs/guider_0_4_1.yaml
```

The script writes a local recovery checkpoint every 100 steps. It uploads checkpoint artifacts to W&B periodically (every 500 steps by default, and at selected best-checkpoint points). If the Colab runtime resets, the local files may disappear, but completed uploads remain in W&B.

To download a checkpoint to your computer manually from Colab:

```python
from google.colab import files
files.download("checkpoints/guider_0_4_1/guider_0_4_1_best.pt")
```

To download a checkpoint stored in W&B later, open the run's **Artifacts** section and download the desired artifact, or use W&B's documented artifact download API. A notebook cannot reliably write directly into a folder on your computer without a browser download action; W&B Artifacts are the automatic off-runtime backup.

## 5. Generate

```python
!python -m guider.generate --config configs/guider_0_4_1.yaml --checkpoint best --prompt "Once upon a time"
```

## Troubleshooting

- **Dataset loading / split error:** ensure the runtime has internet access, `datasets` installed, and use the exact full traceback to diagnose the source name/config/split.
- **Missing tokenizer or .bin files:** rerun preprocessing with the same config.
- **Architecture mismatch:** do not resume an old checkpoint from Guider 0.3 or 0.4a with Guider 0.4.1.
- **W&B not logging:** verify the Colab Secret is named exactly `WANDB_API_KEY`, notebook access is enabled, and the run has internet access. The training script continues without W&B if initialization fails, printing a warning.
- **Disk pressure:** token files and checkpoints use Colab's temporary disk; monitor available disk space before increasing token caps.

## Dataset and evaluation notes

Review the dataset cards, licenses and terms for [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) and [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) before redistributing derived data. Validation loss is useful for tracking optimization but does not alone measure answer quality.
