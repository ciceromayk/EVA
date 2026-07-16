"""Camadas e módulos de rede neural construídos sobre o autograd da EVA.

A API imita, de forma enxuta, o `torch.nn`: todo componente herda de
`Module`, expõe seus parâmetros via `parameters()` e é chamável. Aqui
montamos as peças de um Transformer decoder (estilo GPT): embeddings,
LayerNorm, atenção multi-cabeça causal e MLP.
"""

from __future__ import annotations

import math

import numpy as np

from .autograd import Tensor, embedding, softmax


class Module:
    """Base de todos os componentes com parâmetros treináveis."""

    def parameters(self) -> list[Tensor]:
        params: list[Tensor] = []
        for value in vars(self).values():
            if isinstance(value, Tensor) and value.requires_grad:
                params.append(value)
            elif isinstance(value, Module):
                params.extend(value.parameters())
            elif isinstance(value, (list, tuple)):
                for item in value:
                    if isinstance(item, Module):
                        params.extend(item.parameters())
                    elif isinstance(item, Tensor) and item.requires_grad:
                        params.append(item)
        return params

    def zero_grad(self) -> None:
        for p in self.parameters():
            p.grad = None

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, *args, **kwargs):  # pragma: no cover - abstrato
        raise NotImplementedError


class Linear(Module):
    """Camada densa: y = x @ W + b (com inicialização estilo Kaiming)."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        scale = 1.0 / math.sqrt(in_features)
        w = np.random.randn(in_features, out_features).astype(np.float32) * scale
        self.weight = Tensor(w, requires_grad=True)
        self.bias = (
            Tensor(np.zeros(out_features, dtype=np.float32), requires_grad=True)
            if bias
            else None
        )

    def forward(self, x: Tensor) -> Tensor:
        out = x @ self.weight
        if self.bias is not None:
            out = out + self.bias
        return out


class Embedding(Module):
    """Tabela de vetores indexada por inteiros (tokens ou posições)."""

    def __init__(self, num_embeddings: int, dim: int):
        w = np.random.randn(num_embeddings, dim).astype(np.float32) * 0.02
        self.weight = Tensor(w, requires_grad=True)

    def forward(self, idx: np.ndarray) -> Tensor:
        return embedding(self.weight, idx)


class LayerNorm(Module):
    """Normalização por camada com escala (gamma) e deslocamento (beta)."""

    def __init__(self, dim: int, eps: float = 1e-5):
        self.eps = eps
        self.gamma = Tensor(np.ones(dim, dtype=np.float32), requires_grad=True)
        self.beta = Tensor(np.zeros(dim, dtype=np.float32), requires_grad=True)

    def forward(self, x: Tensor) -> Tensor:
        mean = x.mean(axis=-1, keepdims=True)
        centered = x - mean
        var = (centered * centered).mean(axis=-1, keepdims=True)
        normalized = centered * (var + self.eps) ** -0.5
        return normalized * self.gamma + self.beta


class CausalSelfAttention(Module):
    """Atenção multi-cabeça com máscara causal (não olha o futuro)."""

    def __init__(self, dim: int, n_heads: int):
        assert dim % n_heads == 0, "dim precisa ser divisível por n_heads"
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.qkv = Linear(dim, 3 * dim)
        self.proj = Linear(dim, dim)

    def forward(self, x: Tensor) -> Tensor:
        B, T, C = x.shape
        qkv = self.qkv(x)  # (B, T, 3C)
        # separa Q, K, V e reorganiza para (B, n_heads, T, head_dim)
        q = qkv[:, :, :C].reshape(B, T, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = qkv[:, :, C:2 * C].reshape(B, T, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = qkv[:, :, 2 * C:].reshape(B, T, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)

        scale = 1.0 / math.sqrt(self.head_dim)
        scores = (q @ k.transpose(0, 1, 3, 2)) * scale  # (B, nh, T, T)

        # máscara causal: posições futuras recebem -inf antes do softmax
        mask = np.triu(np.ones((T, T), dtype=np.float32), k=1) * -1e9
        scores = scores + Tensor(mask)

        attn = softmax(scores, axis=-1)
        out = attn @ v  # (B, nh, T, head_dim)
        out = out.transpose(0, 2, 1, 3).reshape(B, T, C)
        return self.proj(out)


class MLP(Module):
    """Rede feed-forward com expansão 4x e ativação GELU."""

    def __init__(self, dim: int):
        self.fc = Linear(dim, 4 * dim)
        self.proj = Linear(4 * dim, dim)

    def forward(self, x: Tensor) -> Tensor:
        return self.proj(self.fc(x).gelu())


class Block(Module):
    """Bloco Transformer: atenção e MLP, cada um com conexão residual."""

    def __init__(self, dim: int, n_heads: int):
        self.ln1 = LayerNorm(dim)
        self.attn = CausalSelfAttention(dim, n_heads)
        self.ln2 = LayerNorm(dim)
        self.mlp = MLP(dim)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x
