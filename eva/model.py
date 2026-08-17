"""Modelo de linguagem estilo Llama em miniatura (a "EVA").

Junta as peças de `nn.py` em um Transformer decoder completo, com a
arquitetura do Llama: embeddings de token (posições entram via RoPE na
atenção, com QK-Norm estabilizando Q e K), uma pilha de blocos pré-RMSNorm
com MLP SwiGLU, RMSNorm final e uma cabeça de projeção para o vocabulário.
Inclui `generate()`/`stream()` para amostragem autoregressiva de texto,
com temperatura, top-k, nucleus (top-p) e penalidade de repetição.
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


def sample_probs(logits: _np.ndarray, generated_ids: _np.ndarray, temperature: float = 1.0,
                  top_k: int | None = None, top_p: float | None = None,
                  repetition_penalty: float = 1.0) -> _np.ndarray:
    """Transforma logits crus (1 posição, vocab_size) em probabilidades de
    amostragem, na ordem estilo Llama/HF: penalidade de repetição ->
    temperatura -> top-k -> top-p (nucleus). Função pura em NumPy (sem
    Tensor/autograd) — só decide COMO amostrar, não participa do treino.

    `logits` é mutado in-place (quem chama já deve passar uma cópia).

    - `repetition_penalty` (1.0 = desligado): reduz a chance de tokens já
      presentes em `generated_ids` — combate o "eeeee..." de modelos
      pequenos entrando em loop.
    - `top_p` (None = desligado): mantém só o menor conjunto de tokens cuja
      probabilidade acumulada cobre `top_p` (ex.: 0.9), descartando a cauda
      improvável — o "nucleus sampling" do GPT-3/Llama.
    """
    if repetition_penalty != 1.0:
        for tid in set(_np.asarray(generated_ids).tolist()):
            logits[tid] = (logits[tid] / repetition_penalty if logits[tid] > 0
                           else logits[tid] * repetition_penalty)

    logits /= max(temperature, 1e-8)

    if top_k is not None:
        kth = _np.sort(logits)[-top_k]
        logits[logits < kth] = -_np.inf

    logits -= logits.max()
    probs = _np.exp(logits)
    probs /= probs.sum()

    if top_p is not None:
        order = _np.argsort(-probs)
        cumulative = _np.cumsum(probs[order])
        # menor prefixo cuja soma cobre top_p; sempre mantém >= 1 token
        cutoff = max(int(_np.searchsorted(cumulative, top_p)) + 1, 1)
        mask = _np.zeros_like(probs, dtype=bool)
        mask[order[:cutoff]] = True
        probs = _np.where(mask, probs, 0.0)
        probs /= probs.sum()

    return probs


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
    def stream(self, idx: np.ndarray, max_new_tokens: int, temperature: float = 1.0,
               top_k: int | None = None, top_p: float | None = None,
               repetition_penalty: float = 1.0, rng: np.random.Generator | None = None):
        """Gera e ENTREGA um token por vez (generator) — base do chat ao vivo.

        Amostragem estilo Llama/HF (ver `sample_probs`). Cada token novo é
        `yield`ado assim que sai do modelo, permitindo mostrar o texto
        surgindo. Assume uma única sequência por vez (idx tem shape (1, T)).
        """
        rng = rng or _np.random.default_rng()
        idx = _np.asarray(idx)  # o histórico de tokens fica na CPU (é leve)
        for _ in range(max_new_tokens):
            context = idx[:, -self.config.block_size:]
            logits, _ = self.forward(context)
            # a amostragem é feita na CPU (numpy): traz só a última linha.
            # .copy() é necessário: sample_probs muta `logits` in-place
            # (penalidade de repetição) e a fatia pode ser só uma VIEW dos
            # dados internos do tensor — mutar sem copiar corromperia o
            # grafo/reuso de memória do forward.
            logits = asnumpy(logits.data[:, -1, :])[0].copy()  # (vocab_size,)
            probs = sample_probs(logits, idx[0], temperature, top_k, top_p,
                                 repetition_penalty)

            next_id = int(rng.choice(self.config.vocab_size, p=probs))
            idx = _np.concatenate([idx, [[next_id]]], axis=1)
            yield next_id

    def generate(self, idx: np.ndarray, max_new_tokens: int, temperature: float = 1.0,
                 top_k: int | None = None, top_p: float | None = None,
                 repetition_penalty: float = 1.0, rng: np.random.Generator | None = None):
        """Gera `max_new_tokens` continuando a partir de `idx` (1, T)."""
        idx = _np.asarray(idx)
        new = list(self.stream(idx, max_new_tokens, temperature, top_k, top_p,
                               repetition_penalty, rng))
        return _np.concatenate([idx, [new]], axis=1)

    def num_params(self) -> int:
        return sum(int(p.data.size) for p in self.parameters())
