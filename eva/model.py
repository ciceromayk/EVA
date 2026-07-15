"""Modelo de linguagem GPT em miniatura (a "EVA").

Junta as peças de `nn.py` em um Transformer decoder completo, com
embeddings de token e posição, uma pilha de blocos, normalização final
e uma cabeça de projeção para o vocabulário. Inclui `generate()` para
amostragem autoregressiva de texto.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .autograd import Tensor, cross_entropy, no_grad
from .nn import Block, Embedding, LayerNorm, Linear, Module


@dataclass
class GPTConfig:
    """Hiperparâmetros da arquitetura."""

    vocab_size: int
    block_size: int = 64      # comprimento máximo de contexto
    n_layer: int = 3          # número de blocos Transformer
    n_head: int = 4           # cabeças de atenção por bloco
    n_embd: int = 96          # dimensão dos embeddings


class GPT(Module):
    """Transformer decoder autoregressivo em nível de caractere."""

    def __init__(self, config: GPTConfig):
        self.config = config
        self.token_emb = Embedding(config.vocab_size, config.n_embd)
        self.pos_emb = Embedding(config.block_size, config.n_embd)
        self.blocks = [Block(config.n_embd, config.n_head) for _ in range(config.n_layer)]
        self.ln_f = LayerNorm(config.n_embd)
        self.head = Linear(config.n_embd, config.vocab_size, bias=False)

    def forward(self, idx: np.ndarray, targets: np.ndarray | None = None):
        """idx: (B, T) inteiros. Retorna (logits, loss)."""
        idx = np.asarray(idx)
        B, T = idx.shape
        assert T <= self.config.block_size, "sequência maior que o block_size"

        positions = np.arange(T)
        x = self.token_emb(idx) + self.pos_emb(positions)  # (B, T, C) via broadcast
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.head(x)  # (B, T, vocab_size)

        loss = None
        if targets is not None:
            flat = logits.reshape(B * T, self.config.vocab_size)
            loss = cross_entropy(flat, np.asarray(targets).reshape(-1))
        return logits, loss

    @no_grad()
    def generate(self, idx: np.ndarray, max_new_tokens: int, temperature: float = 1.0,
                 top_k: int | None = None, rng: np.random.Generator | None = None):
        """Gera `max_new_tokens` continuando a partir de `idx` (1, T)."""
        rng = rng or np.random.default_rng()
        idx = np.asarray(idx)
        for _ in range(max_new_tokens):
            context = idx[:, -self.config.block_size:]
            logits, _ = self.forward(context)
            logits = logits.data[:, -1, :] / max(temperature, 1e-8)  # (1, vocab)

            if top_k is not None:
                kth = np.sort(logits, axis=-1)[:, -top_k][:, None]
                logits = np.where(logits < kth, -np.inf, logits)

            logits -= logits.max(axis=-1, keepdims=True)
            probs = np.exp(logits)
            probs /= probs.sum(axis=-1, keepdims=True)
            next_id = rng.choice(self.config.vocab_size, p=probs[0])
            idx = np.concatenate([idx, [[next_id]]], axis=1)
        return idx

    def num_params(self) -> int:
        return sum(int(p.data.size) for p in self.parameters())
