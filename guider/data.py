from pathlib import Path
import shutil

import numpy as np
import torch


TOKEN_DTYPE = np.uint16


def prepare_data(local_dir, persistent_dir, filenames):
    local_dir = Path(local_dir)
    persistent_dir = Path(persistent_dir) if persistent_dir else None
    local_dir.mkdir(parents=True, exist_ok=True)

    for filename in filenames:
        local_path = local_dir / filename
        if local_path.exists():
            continue
        persistent_path = persistent_dir / filename if persistent_dir else None
        if persistent_path is None or not persistent_path.exists():
            raise FileNotFoundError(
                f"Missing {filename}; expected it in {local_path}"
                + (f" or {persistent_path}" if persistent_path else "")
            )
        print(f"Copying {filename} from persistent storage to local disk...")
        shutil.copy2(persistent_path, local_path)


def open_memmap(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Token file not found: {path}")
    if path.stat().st_size % np.dtype(TOKEN_DTYPE).itemsize:
        raise ValueError(f"Invalid uint16 file size: {path}")
    return np.memmap(path, dtype=TOKEN_DTYPE, mode="r")


def get_batch(data, batch_size, block_size, device, rng, pin_memory=True):
    if data.size <= block_size + 1:
        raise ValueError(f"Not enough tokens for block_size={block_size}: {data.size}")
    starts = rng.integers(0, data.size - block_size, size=batch_size, dtype=np.int64)
    offsets = np.arange(block_size + 1, dtype=np.int64)
    batch = np.asarray(data[starts[:, None] + offsets], dtype=np.int64)
    tokens = torch.from_numpy(batch)
    if pin_memory and device.type == "cuda":
        tokens = tokens.pin_memory()
    return (
        tokens[:, :-1].to(device, non_blocking=True),
        tokens[:, 1:].to(device, non_blocking=True),
    )
