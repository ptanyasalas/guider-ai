from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel as ByteLevelPreTokenizer
from tokenizers.trainers import BpeTrainer


class ByteBPETokenizer:
    """Fast byte-level BPE tokenizer backed by the Rust tokenizers library."""

    TYPE = "byte_bpe_fast"

    def __init__(self, tokenizer: Tokenizer, eos_token: str = "<|endoftext|>"):
        self._tokenizer = tokenizer
        self.eos_token = eos_token
        self._eos_id = tokenizer.token_to_id(eos_token)

    @classmethod
    def train(
        cls,
        texts: Iterable[str],
        vocab_size: int = 2048,
        min_pair_frequency: int = 2,
        eos_token: str = "<|endoftext|>",
    ) -> "ByteBPETokenizer":
        if vocab_size < 256:
            raise ValueError("vocab_size must be at least 256.")
        if min_pair_frequency < 1:
            raise ValueError("min_pair_frequency must be at least 1.")

        tokenizer = Tokenizer(BPE(unk_token=None))
        tokenizer.pre_tokenizer = ByteLevelPreTokenizer(add_prefix_space=False)
        tokenizer.decoder = ByteLevelDecoder()
        trainer = BpeTrainer(
            vocab_size=vocab_size,
            min_frequency=min_pair_frequency,
            special_tokens=[eos_token],
            initial_alphabet=ByteLevelPreTokenizer.alphabet(),
        )
        tokenizer.train_from_iterator(texts, trainer=trainer)
        result = cls(tokenizer, eos_token=eos_token)
        if result._eos_id is None:
            raise RuntimeError("Tokenizer failed to create the EOS token.")
        return result

    @classmethod
    def from_file(cls, path: str | Path, eos_token: str = "<|endoftext|>"):
        return cls(Tokenizer.from_file(str(path)), eos_token=eos_token)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._tokenizer.save(str(path))

    def encode(self, text: str, add_eos: bool = False) -> list[int]:
        ids = self._tokenizer.encode(text).ids
        if add_eos:
            ids.append(self.eos_token_id)
        return ids

    def encode_batch(self, texts: list[str], add_eos: bool = False) -> list[list[int]]:
        result = [encoding.ids for encoding in self._tokenizer.encode_batch(texts)]
        if add_eos:
            eos_id = self.eos_token_id
            return [ids + [eos_id] for ids in result]
        return result

    def decode(self, ids) -> str:
        return self._tokenizer.decode([int(i) for i in ids], skip_special_tokens=False)

    @property
    def vocab_size(self) -> int:
        return self._tokenizer.get_vocab_size(with_added_tokens=True)

    @property
    def eos_token_id(self) -> int:
        if self._eos_id is None:
            raise RuntimeError("EOS token is missing from tokenizer.")
        return int(self._eos_id)

    @property
    def merge_count(self) -> int:
        state = json.loads(self._tokenizer.to_str())
        return len(state.get("model", {}).get("merges", []))

    def to_dict(self) -> dict:
        return {
            "type": self.TYPE,
            "eos_token": self.eos_token,
            "tokenizer_json": self._tokenizer.to_str(),
        }

    @classmethod
    def from_dict(cls, state: dict) -> "ByteBPETokenizer":
        if not isinstance(state, dict):
            raise ValueError("Tokenizer state must be a dictionary.")
        if state.get("type") == cls.TYPE and state.get("tokenizer_json"):
            return cls(
                Tokenizer.from_str(state["tokenizer_json"]),
                eos_token=state.get("eos_token", "<|endoftext|>"),
            )
        if state.get("type") == "byte_bpe":
            raise ValueError(
                "This is a Guider v0.2 tokenizer. Run v0.3 preprocessing to create a matching tokenizer."
            )
        raise ValueError("Unsupported tokenizer type.")
