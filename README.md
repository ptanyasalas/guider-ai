# Guider AI v0.3

Guider is an experimental decoder-only Transformer language model. This version is designed for larger text datasets and GPU training in Google Colab.

## Changes in v0.3

- Fast byte-level BPE using the Rust-backed tokenizers library.
- One-time preprocessing to uint16 binary files: data/train.bin and data/val.bin.
- Memory-mapped data access with numpy.memmap; the full token dataset is not loaded into RAM.
- CUDA training with FP16 automatic mixed precision (AMP).
- Fused AdamW when supported, optimized causal attention, and optional torch.compile.
- Warmup plus cosine learning-rate decay.
- Dropout, weight decay, validation checks, and early stopping.
- Periodic Google Drive checkpoints, including optimizer and AMP scaler state, so training can resume.

## Google Colab quick start

1. Open a new Colab notebook.
2. Choose Runtime > Change runtime type > T4 GPU if that option is available.
3. Run the following cell to verify the assigned GPU:

    !nvidia-smi

4. Clone the repository and install dependencies:

    !git clone https://github.com/ptanyasalas/guider-ai.git
    %cd guider-ai
    !pip install -r requirements.txt

5. Mount Google Drive:

    from google.colab import drive
    drive.mount('/content/drive')

6. Preprocess TinyStories once:

    !python preprocesar.py

7. Start training:

    !python -m guider.train

8. Generate text from the best/latest available checkpoint:

    !python -m guider.generate

The default config uses /content/drive/MyDrive/Guider/data and /content/drive/MyDrive/Guider/checkpoints. If the runtime disconnects, reconnect, mount Drive again, clone/install if needed, and rerun the training command. The latest checkpoint resumes automatically.

## Dataset and storage

The preprocessor streams the roneneldan/TinyStories dataset from Hugging Face. It trains the tokenizer on a sample from the training split only, then creates train.bin and val.bin using uint16 token IDs. The default caps are 500,000,000 training tokens and 10,000,000 validation tokens. A 500-million-token uint16 file is about 1 GB before filesystem overhead.

Tokenized files and checkpoints are intentionally not committed to Git.

## Important notes

No setting can guarantee zero overfitting. Dropout, weight decay, validation loss monitoring, and early stopping reduce the risk and help stop when validation performance stops improving.

The v0.2 tokenizer/checkpoints are not compatible with the v0.3 tokenizer. Run preprocessing to create a matching tokenizer and binary data files before training.
