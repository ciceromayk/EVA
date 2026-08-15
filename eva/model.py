"""Modelo de linguagem estilo Llama em miniatura (a "EVA").

Junta as peças de `nn.py` em um Transformer decoder completo, com a
arquitetura do Llama: embeddings de token (posições entram via RoPE na
atenção), uma pilha de blocos pré-RMSNorm com MLP SwiGLU, RMSNorm final
e uma cabeça de projeção para o vocabulário. Inclui `generate()` para
amostragem autoregressiva de texto.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as _np

from .autograd import Tensor, cross_entropy, dropout, no_grad
from .backend import asnumpy
from .backend import xp as np
from .nn import Block, Embedding, Linear, Module, RMSNorm, rope_tables


@dataclass
class GPTConfig:
    """Hiperparâmetros da arquitetura (estilo Llama)."""

    vocab_size: int
    block_size: int = 64      # comprimento máximo de contexto
    n_layer: int = 3          # número de blocos Transformer
    n_head: int = 4           # cabeças de atenção por bloco
    n_embd: int = 96          # dimensão dos embeddings
    dropout: float = 0.0      # regularização (0 = desligado)
    rope_base: float = 10000.0  # base das frequências do RoPE (Llama usa 10000)


class GPT(Module):
    """Transformer decoder autoregressivo, arquitetura estilo Llama."""

    def __init__(self, config: GPTConfig):
        self.config = config
        self.token_emb = Embedding(config.vocab_size, config.n_embd)
        self.blocks = [Block(config.n_embd, config.n_head, config.dropout)
                       for _ in range(config.n_layer)]
        self.ln_f = RMSNorm(config.n_embd)
        self.head = Linear(config.n_embd, config.vocab_size, bias=False)

        # Tabelas do RoPE para todas as posições possíveis, calculadas uma
        # vez só. Não são parâmetros: são fixas, como no Llama.
        head_dim = config.n_embd // config.n_head
        rope_base = getattr(config, "rope_base", 10000.0)  # configs antigas não têm o campo
        self._rope_cos, self._rope_sin = rope_tables(config.block_size, head_dim, rope_base)

        # Init de fluxo residual (GPT-2/Llama): encolhe as projeções de saída
        # de cada bloco por 1/sqrt(2*n_layer). Sem isso, a variância cresce
        # ao longo do fluxo residual e o treino fica instável / lento.
        residual_scale = 1.0 / math.sqrt(2 * config.n_layer)
        for block in self.blocks:
            block.attn.proj.weight.data *= residual_scale
            block.mlp.down.weight.data *= residual_scale

    def forward(self, idx: np.ndarray, targets: np.ndarray | None = None):
        """idx: (B, T) inteiros. Retorna (logits, loss)."""
        idx = np.asarray(idx)
        B, T = idx.shape
        assert T <= self.config.block_size, "sequência maior que o block_size"

        # No Llama não há embedding de posição somado: a posição entra na
        # atenção, girando Q e K com as tabelas do RoPE (fatiadas até T).
        cos, sin = self._rope_cos[:T], self._rope_sin[:T]
        x = self.token_emb(idx)  # (B, T, C)
        x = dropout(x, self.config.dropout)
        for block in self.blocks:
            x = block(x, cos, sin)
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
        rng = rng or _np.random.default_rng()
        idx = _np.asarray(idx)  # o histórico de tokens fica na CPU (é leve)
        for _ in range(max_new_tokens):
            context = idx[:, -self.config.block_size:]
            logits, _ = self.forward(context)
            # a amostragem é feita na CPU (numpy): traz só a última linha
            logits = asnumpy(logits.data[:, -1, :]) / max(temperature, 1e-8)

            if top_k is not None:
                kth = _np.sort(logits, axis=-1)[:, -top_k][:, None]
                logits = _np.where(logits < kth, -_np.inf, logits)

            logits -= logits.max(axis=-1, keepdims=True)
            probs = _np.exp(logits)
            probs /= probs.sum(axis=-1, keepdims=True)
            next_id = rng.choice(self.config.vocab_size, p=probs[0])
            idx = _np.concatenate([idx, [[next_id]]], axis=1)
        return idx

    def num_params(self) -> int:
        return sum(int(p.data.size) for p in self.parameters())
