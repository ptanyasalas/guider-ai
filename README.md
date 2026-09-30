# Guider AI v0.3

Guider is an experimental decoder-only Transformer language model. It can train in Google Colab on a GPU **without mounting Google Drive**.

## Google Colab quick start (no Drive)

1. Open a new Colab notebook and choose **Runtime > Change runtime type > T4 GPU**, if available.
2. Check the GPU:

    ```python
    !nvidia-smi
    ```

3. Clone the repository and install dependencies:

    ```python
    !git clone https://github.com/ptanyasalas/guider-ai.git
    %cd guider-ai
    !pip install -r requirements.txt
    ```

4. Preprocess TinyStories (this downloads/streams the dataset and may take a while):

    ```python
    !python preprocesar.py
    ```

5. Start training:

    ```python
    !python -m guider.train
    ```

6. Generate text after training:

    ```python
    !python -m guider.generate
    ```

## Important: Colab storage without Drive

Data and checkpoints are saved in the Colab runtime's local disk. **They are temporary** and can disappear when the runtime disconnects or resets. If you want to keep a trained checkpoint, download it before ending the session:

    ```python
    from google.colab import files
    files.download("checkpoints/guider_best.pt")
    ```

If that file does not exist yet, try `checkpoints/guider_latest.pt`. To resume training in the same live runtime, rerun `!python -m guider.train`; the latest local checkpoint is resumed automatically. After a runtime reset, upload your saved checkpoint into the `checkpoints/` folder before restarting training. The processed data files must also be recreated after a reset by rerunning `!python preprocesar.py`.

## Changes in v0.3

- Fast byte-level BPE using the Rust-backed tokenizers library.
- One-time preprocessing to uint16 binary files: `data/train.bin` and `data/val.bin`.
- Memory-mapped data access with numpy.memmap.
- CUDA training with FP16 automatic mixed precision (AMP).
- Fused AdamW when supported and optimized causal attention.
- Warmup plus cosine learning-rate decay.
- Dropout, weight decay, validation checks, and early stopping.
- Local checkpoints that work without Google Drive; Drive persistence is optional.

## Dataset and storage

The preprocessor streams the `roneneldan/TinyStories` dataset from Hugging Face. It trains the tokenizer on a sample from the training split only, then creates `train.bin` and `val.bin` using uint16 token IDs. The default caps are 500,000,000 training tokens and 10,000,000 validation tokens. A 500-million-token uint16 file is about 1 GB before filesystem overhead.

Tokenized files and checkpoints are intentionally not committed to Git.

## Important notes

No setting can guarantee zero overfitting. Dropout, weight decay, validation loss monitoring, and early stopping reduce the risk and help stop when validation performance stops improving.

The v0.2 tokenizer/checkpoints are not compatible with the v0.3 tokenizer. Run preprocessing to create a matching tokenizer and binary data files before training.
