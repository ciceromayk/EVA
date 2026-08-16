"""Camadas e módulos de rede neural construídos sobre o autograd da EVA.

A API imita, de forma enxuta, o `torch.nn`: todo componente herda de
`Module`, expõe seus parâmetros via `parameters()` e é chamável. Aqui
montamos as peças de um Transformer decoder estilo **Llama**: embeddings,
RMSNorm, atenção multi-cabeça causal com RoPE e MLP SwiGLU — todas as
camadas lineares sem bias, como no Llama.
"""

from __future__ import annotations

import math

from .autograd import Tensor, dropout, embedding, layer_norm, rms_norm, rope, softmax
from .backend import xp as np


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
        return layer_norm(x, self.gamma, self.beta, self.eps)


class RMSNorm(Module):
    """Normalização RMS (Llama): só reescala, sem centrar e sem beta."""

    def __init__(self, dim: int, eps: float = 1e-5):
        self.eps = eps
        self.gamma = Tensor(np.ones(dim, dtype=np.float32), requires_grad=True)

    def forward(self, x: Tensor) -> Tensor:
        return rms_norm(x, self.gamma, self.eps)


def rope_tables(block_size: int, head_dim: int, base: float = 10000.0):
    """Pré-computa as tabelas (cos, sin) do RoPE para todas as posições.

    Cada par de coordenadas (i, i + head_dim/2) gira com frequência
    base^(-2i/head_dim): pares "rápidos" codificam posições próximas,
    pares "lentos" codificam distâncias longas.
    """
    half = head_dim // 2
    freqs = base ** (-np.arange(0, half, dtype=np.float32) / half)
    angles = np.outer(np.arange(block_size, dtype=np.float32), freqs)  # (T, half)
    return np.cos(angles).astype(np.float32), np.sin(angles).astype(np.float32)


class CausalSelfAttention(Module):
    """Atenção multi-cabeça causal com RoPE (não olha o futuro).

    Estilo Llama: as posições entram GIRANDO os vetores Q e K (RoPE),
    não somando um embedding de posição; projeções sem bias.
    """

    def __init__(self, dim: int, n_heads: int):
        assert dim % n_heads == 0, "dim precisa ser divisível por n_heads"
        assert (dim // n_heads) % 2 == 0, "head_dim precisa ser par (RoPE gira pares)"
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.qkv = Linear(dim, 3 * dim, bias=False)
        self.proj = Linear(dim, dim, bias=False)

    def forward(self, x: Tensor, cos: np.ndarray, sin: np.ndarray) -> Tensor:
        B, T, C = x.shape
        qkv = self.qkv(x)  # (B, T, 3C)
        # separa Q, K, V e reorganiza para (B, n_heads, T, head_dim)
        q = qkv[:, :, :C].reshape(B, T, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = qkv[:, :, C:2 * C].reshape(B, T, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = qkv[:, :, 2 * C:].reshape(B, T, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)

        # RoPE: rotaciona Q e K pela posição (V fica intacto)
        q = rope(q, cos, sin)
        k = rope(k, cos, sin)

        scale = 1.0 / math.sqrt(self.head_dim)
        scores = (q @ k.transpose(0, 1, 3, 2)) * scale  # (B, nh, T, T)

        # máscara causal: posições futuras recebem -inf antes do softmax
        mask = np.triu(np.ones((T, T), dtype=np.float32), k=1) * -1e9
        scores = scores + Tensor(mask)

        attn = softmax(scores, axis=-1)
        out = attn @ v  # (B, nh, T, head_dim)
        out = out.transpose(0, 2, 1, 3).reshape(B, T, C)
        return self.proj(out)


def swiglu_hidden(dim: int, multiple_of: int = 8) -> int:
    """Dimensão oculta do SwiGLU: ~8/3 * dim, arredondada para cima.

    Com três matrizes (gate, up, down) de dim x hidden, escolher
    hidden = 8/3 * dim mantém o total de parâmetros igual ao MLP GELU
    clássico de expansão 4x (2 matrizes de dim x 4dim) — mesma conta
    que o Llama faz.
    """
    hidden = (8 * dim + 2) // 3
    return multiple_of * ((hidden + multiple_of - 1) // multiple_of)


class MLP(Module):
    """Rede feed-forward SwiGLU (Llama): down(silu(gate(x)) * up(x)).

    Em vez de uma ativação simples, metade da rede (gate) decide, via
    SiLU, quanto da outra metade (up) passa adiante — um "portão"
    aprendido, que na prática supera o MLP GELU clássico.
    """

    def __init__(self, dim: int):
        hidden = swiglu_hidden(dim)
        self.gate = Linear(dim, hidden, bias=False)
        self.up = Linear(dim, hidden, bias=False)
        self.down = Linear(hidden, dim, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.down(self.gate(x).silu() * self.up(x))


class Block(Module):
    """Bloco Transformer estilo Llama: atenção e MLP com pré-RMSNorm.

    Cada sub-camada tem conexão residual, e o dropout residual continua
    disponível como regularização (o Llama original treina sem, mas em
    corpus pequeno como o da EVA ele ajuda contra overfitting).
    """

    def __init__(self, dim: int, n_heads: int, dropout: float = 0.0):
        self.dropout = dropout
        self.ln1 = RMSNorm(dim)
        self.attn = CausalSelfAttention(dim, n_heads)
        self.ln2 = RMSNorm(dim)
        self.mlp = MLP(dim)

    def forward(self, x: Tensor, cos: np.ndarray, sin: np.ndarray) -> Tensor:
        x = x + dropout(self.attn(self.ln1(x), cos, sin), self.dropout)
        x = x + dropout(self.mlp(self.ln2(x)), self.dropout)
        return x
