# Guider AI

Guider is an experimental decoder-only Transformer language model designed for Google Colab T4 experiments without requiring Google Drive.

## Guider 0.4 milestones

### 0.4a — larger-corpus pretraining
- 10-layer, 384-wide Transformer; 512-token context and 8,192-token BPE vocabulary.
- RoPE, RMSNorm and SwiGLU blocks.
- Streaming mixture dominated by HuggingFaceFW/fineweb-edu with a small TinyStories replay component.
- Gradient accumulation, FP16 AMP, deterministic validation sampling, gradient clipping and periodic atomic checkpoints.

### 0.4b — instruction tuning
- Same architecture and exact tokenizer as 0.4a.
- Starts from the 0.4a best checkpoint.
- Uses UltraChat conversations with small TinyStories and FineWeb-Edu replay components.
- Lower learning rate intended to adapt the model without replacing all its previous language-model training.

### Generation fix
Guider 0.4 generation now stops as soon as EOS is sampled, preventing the next unrelated story from being appended. Top-k, top-p and repetition-penalty sampling controls are configurable.

Guider 0.4 is a new architecture and **is not compatible with Guider 0.3 checkpoint weights**. Keep your 0.3 Best and Latest files for comparison.

## Google Colab setup

Choose a T4 GPU runtime if available:

```python
!nvidia-smi
!git clone https://github.com/ptanyasalas/guider-ai.git
%cd guider-ai
!pip install -r requirements.txt
```

## Run Guider 0.4a

Preprocess a large streaming corpus mixture (this can take a long time and creates large local files):

```python
!python preprocesar.py --config configs/guider_0_4a.yaml
```

Train:

```python
!python -m guider.train --config configs/guider_0_4a.yaml
```

Generate from the best checkpoint:

```python
!python -m guider.generate --config configs/guider_0_4a.yaml --checkpoint best --prompt "Once upon a time"
```

## Run Guider 0.4b

Run this after 0.4a has produced its best checkpoint. First prepare the conversation corpus:

```python
!python preprocesar.py --config configs/guider_0_4b.yaml
```

Then train from the 0.4a weights:

```python
!python -m guider.train --config configs/guider_0_4b.yaml
```

Generate:

```python
!python -m guider.generate --config configs/guider_0_4b.yaml --checkpoint best --prompt "Explain why the sky looks blue."
```

## Dataset scale and realistic expectations

- 0.4a is capped at 2 billion training tokens and 30 million validation tokens.
- 0.4b is capped at 300 million training tokens and 10 million validation tokens.
- The source corpus is much larger than TinyStories, but the caps are the maximum preprocessed token counts, not a promise that Colab will finish training on all of them.
- A T4 session may time out. Checkpoints are saved periodically; download them before the runtime ends.
- Streaming avoids downloading the full original dataset archive, but preprocessing still writes large token files to the runtime disk.

The FineWeb-Edu and UltraChat datasets are hosted on Hugging Face. Review their dataset cards, licenses, and terms before using or redistributing derived data.

## Checkpoints and temporary storage

Colab local storage is temporary. Download the 0.4a best checkpoint:

```python
from google.colab import files
files.download("checkpoints/guider_0_4a/guider_0_4a_best.pt")
```

For 0.4b:

```python
from google.colab import files
files.download("checkpoints/guider_0_4b/guider_0_4b_best.pt")
```

After a runtime reset, re-upload the checkpoint and rerun preprocessing to recreate the matching token files. Do not mix data/tokenizer files between the 0.4a and 0.4b folders.

## Evaluation plan

Use the same fixed prompts and decoding settings to compare Guider 0.3 Best, 0.3 Latest, 0.4a Best and 0.4b Best. Track validation loss separately from human evaluation of coherence, instruction following, repetition and EOS stopping. Lower validation loss alone does not guarantee better answers.


## Fixed-prompt evaluation across versions

The evaluation script supports both saved Guider 0.3 checkpoints and new Guider 0.4 checkpoints. It runs the same prompts from `eval_prompts.json` and writes a Markdown report.

~~~python
!python avaluar.py --checkpoint checkpoints/guider_best.pt --label "Guider 0.3 Best" --output eval_03_best.md
!python avaluar.py --checkpoint checkpoints/guider_latest.pt --label "Guider 0.3 Latest" --output eval_03_latest.md
!python avaluar.py --checkpoint checkpoints/guider_0_4a/guider_0_4a_best.pt --label "Guider 0.4a Best" --output eval_04a.md
!python avaluar.py --checkpoint checkpoints/guider_0_4b/guider_0_4b_best.pt --label "Guider 0.4b Best" --output eval_04b.md
~~~

Compare coherence, instruction following, repetition and EOS stopping. This is a qualitative fixed-prompt comparison, not a benchmark score. Lower validation loss alone does not guarantee better answers.
