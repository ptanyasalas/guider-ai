from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn import functional as F


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        variance = x.float().pow(2).mean(dim=-1, keepdim=True)
        return self.weight * (x * torch.rsqrt(variance + self.eps)).to(x.dtype)


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, max_seq_len: int, theta: float = 10000.0):
        super().__init__()
        if head_dim % 2:
            raise ValueError("RoPE requires an even head dimension.")
        inv_freq = 1.0 / (
            theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
        )
        positions = torch.arange(max_seq_len, dtype=torch.float32)
        freqs = torch.outer(positions, inv_freq)
        self.register_buffer(
            "cos",
            freqs.cos().repeat(1, 2),
            persistent=False,
        )
        self.register_buffer(
            "sin",
            freqs.sin().repeat(1, 2),
            persistent=False,
        )

    def forward(self, q, k):
        seq_len = q.size(-2)
        cos = self.cos[:seq_len].to(dtype=q.dtype).unsqueeze(0).unsqueeze(0)
        sin = self.sin[:seq_len].to(dtype=q.dtype).unsqueeze(0).unsqueeze(0)
        return q * cos + rotate_half(q) * sin, k * cos + rotate_half(k) * sin


class CausalSelfAttention(nn.Module):
    def __init__(
        self,
        n_embd: int,
        n_head: int,
        block_size: int,
        dropout: float,
        rope_theta: float,
    ):
        super().__init__()
        if n_embd % n_head:
            raise ValueError("n_embd must be divisible by n_head.")
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.qkv = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.proj = nn.Linear(n_embd, n_embd, bias=False)
        self.dropout = dropout
        self.rope = RotaryEmbedding(self.head_dim, block_size, rope_theta)

    def forward(self, x):
        b, t, c = x.shape
        q, k, v = self.qkv(x).split(c, dim=2)
        q = q.view(b, t, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(b, t, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(b, t, self.n_head, self.head_dim).transpose(1, 2)
        q, k = self.rope(q, k)
        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            is_causal=True,
            dropout_p=self.dropout if self.training else 0.0,
        )
        y = y.transpose(1, 2).contiguous().view(b, t, c)
        return self.proj(y)


class SwiGLU(nn.Module):
    def __init__(self, n_embd: int, n_ffn: int, dropout: float):
        super().__init__()
        self.w1 = nn.Linear(n_embd, n_ffn, bias=False)
        self.w3 = nn.Linear(n_embd, n_ffn, bias=False)
        self.w2 = nn.Linear(n_ffn, n_embd, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


class Block(nn.Module):
    def __init__(
        self,
        n_embd: int,
        n_head: int,
        n_ffn: int,
        block_size: int,
        dropout: float,
        rope_theta: float,
        rms_norm_eps: float,
    ):
        super().__init__()
        self.ln1 = RMSNorm(n_embd, rms_norm_eps)
        self.attn = CausalSelfAttention(
            n_embd,
            n_head,
            block_size,
            dropout,
            rope_theta,
        )
        self.ln2 = RMSNorm(n_embd, rms_norm_eps)
        self.mlp = SwiGLU(n_embd, n_ffn, dropout)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        return x + self.mlp(self.ln2(x))


class GuiderLM(nn.Module):
    ARCHITECTURE = "guider-transformer-v0.4"

    def __init__(
        self,
        vocab_size: int,
        block_size: int,
        n_layer: int,
        n_head: int,
        n_embd: int,
        n_ffn: int,
        dropout: float,
        rope_theta: float = 10000.0,
        rms_norm_eps: float = 1e-5,
    ):
        super().__init__()
        self.block_size = block_size
        self.vocab_size = vocab_size

        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        self.blocks = nn.ModuleList(
            [
                Block(
                    n_embd,
                    n_head,
                    n_ffn,
                    block_size,
                    dropout,
                    rope_theta,
                    rms_norm_eps,
                )
                for _ in range(n_layer)
            ]
        )
        self.ln_f = RMSNorm(n_embd, rms_norm_eps)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)

        self.apply(self._init_weights)
        self.lm_head.weight = self.token_embedding.weight

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        _, t = idx.shape
        if t > self.block_size:
            raise ValueError(
                f"Sequence length {t} exceeds block size {self.block_size}."
            )

        x = self.token_embedding(idx)
        for block in self.blocks:
            x = block(x)

        logits = self.lm_head(self.ln_f(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
            )
        return logits, loss

    @staticmethod
    def _apply_repetition_penalty(logits, idx, penalty):
        if penalty == 1.0:
            return logits
        for batch_idx in range(logits.size(0)):
            token_ids = torch.unique(idx[batch_idx])
            values = logits[batch_idx, token_ids]
            adjusted = torch.where(values < 0, values * penalty, values / penalty)
            logits[batch_idx].scatter_(0, token_ids, adjusted)
        return logits

    @torch.no_grad()
    def generate(
        self,
        idx,
        max_new_tokens: int,
        temperature: float = 0.8,
        top_k: int | None = 50,
        top_p: float = 1.0,
        repetition_penalty: float = 1.0,
        eos_token_id: int | None = None,
    ):
        if max_new_tokens < 1:
            return idx
        if repetition_penalty < 1.0:
            raise ValueError("repetition_penalty must be >= 1.0")

        finished = torch.zeros(idx.size(0), dtype=torch.bool, device=idx.device)

        for _ in range(max_new_tokens):
            context = idx[:, -self.block_size :]
            logits, _ = self(context)
            logits = logits[:, -1, :]
            logits = self._apply_repetition_penalty(
                logits, context, repetition_penalty
            )

            if temperature <= 0:
                next_token = torch.argmax(logits, dim=-1, keepdim=True)
            else:
                logits = logits / max(temperature, 1e-6)

                if top_k is not None:
                    k = min(int(top_k), logits.size(-1))
                    values, _ = torch.topk(logits, k)
                    logits = logits.masked_fill(
                        logits < values[:, [-1]],
                        float("-inf"),
                    )

                if 0.0 < top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(
                        logits,
                        descending=True,
                        dim=-1,
                    )
                    sorted_probs = F.softmax(sorted_logits, dim=-1)
                    cumulative = torch.cumsum(sorted_probs, dim=-1)
                    remove = cumulative > top_p
                    remove[:, 1:] = remove[:, :-1].clone()
                    remove[:, 0] = False
                    sorted_logits = sorted_logits.masked_fill(
                        remove,
                        float("-inf"),
                    )
                    logits = torch.full_like(logits, float("-inf"))
                    logits.scatter_(1, sorted_indices, sorted_logits)

                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

            if eos_token_id is not None:
                next_token = torch.where(
                    finished.unsqueeze(1),
                    torch.full_like(next_token, eos_token_id),
                    next_token,
                )

            idx = torch.cat((idx, next_token), dim=1)

            if eos_token_id is not None:
                finished |= next_token.squeeze(1).eq(eos_token_id)
                if finished.all():
                    break

        return idx

    @classmethod
    def from_config(cls, vocab_size: int, config: dict):
        return cls(
            vocab_size=vocab_size,
            block_size=config["block_size"],
            n_layer=config["n_layer"],
            n_head=config["n_head"],
            n_embd=config["n_embd"],
            n_ffn=config["n_ffn"],
            dropout=config["dropout"],
            rope_theta=config.get("rope_theta", 10000.0),
            rms_norm_eps=config.get("rms_norm_eps", 1e-5),
        )
