class ByteBPETokenizer:
    """Small byte-level BPE tokenizer implemented from scratch."""

    BASE_VOCAB_SIZE = 256

    def __init__(self, merges=None):
        self.merges = [tuple(map(int, pair)) for pair in (merges or [])]
        self.tokens = [bytes([i]) for i in range(self.BASE_VOCAB_SIZE)]

        for left, right in self.merges:
            if left >= len(self.tokens) or right >= len(self.tokens):
                raise ValueError(f"Invalid merge ({left}, {right})")
            self.tokens.append(self.tokens[left] + self.tokens[right])

    @classmethod
    def train(cls, text: str, vocab_size: int = 512, min_pair_frequency: int = 2):
        if vocab_size < cls.BASE_VOCAB_SIZE:
            raise ValueError(f"vocab_size must be at least {cls.BASE_VOCAB_SIZE}")
        if min_pair_frequency < 1:
            raise ValueError("min_pair_frequency must be at least 1")

        sequence = list(text.encode("utf-8"))
        merges = []
        max_merges = vocab_size - cls.BASE_VOCAB_SIZE

        for _ in range(max_merges):
            pair_counts = {}
            for a, b in zip(sequence, sequence[1:]):
                pair = (a, b)
                pair_counts[pair] = pair_counts.get(pair, 0) + 1

            if not pair_counts:
                break

            best_pair, frequency = max(
                pair_counts.items(),
                key=lambda item: (item[1], item[0]),
            )
            if frequency < min_pair_frequency:
                break

            new_id = cls.BASE_VOCAB_SIZE + len(merges)
            merges.append(best_pair)
            sequence = cls._merge_pair(sequence, best_pair, new_id)

        return cls(merges)

    @staticmethod
    def _merge_pair(sequence, pair, new_id):
        merged = []
        i = 0

        while i < len(sequence):
            if i + 1 < len(sequence) and (sequence[i], sequence[i + 1]) == pair:
                merged.append(new_id)
                i += 2
            else:
                merged.append(sequence[i])
                i += 1

        return merged

    def encode(self, text: str) -> list[int]:
        ids = list(text.encode("utf-8"))

        for merge_index, pair in enumerate(self.merges):
            if len(ids) < 2:
                break
            new_id = self.BASE_VOCAB_SIZE + merge_index
            ids = self._merge_pair(ids, pair, new_id)

        return ids

    def decode(self, ids) -> str:
        raw = b"".join(self.tokens[int(token_id)] for token_id in ids)
        return raw.decode("utf-8", errors="replace")

    @property
    def vocab_size(self) -> int:
        return len(self.tokens)

    @property
    def merge_count(self) -> int:
        return len(self.merges)

    def to_dict(self) -> dict:
        return {
            "type": "byte_bpe",
            "base_vocab_size": self.BASE_VOCAB_SIZE,
            "merges": [list(pair) for pair in self.merges],
        }

    @classmethod
    def from_dict(cls, state: dict):
        if state.get("type") != "byte_bpe":
            raise ValueError("Checkpoint does not contain a byte-BPE tokenizer.")
        return cls(state.get("merges", []))
