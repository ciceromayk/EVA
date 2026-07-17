"""Otimizadores para treinar a EVA.

Implementa o AdamW, o otimizador padrão para treinar Transformers.
Ele combina momentos de primeira e segunda ordem (como o Adam) com
decaimento de peso desacoplado (o "W" de AdamW).
"""

from __future__ import annotations

from .autograd import Tensor
from .backend import xp as np


class AdamW:
    def __init__(self, params: list[Tensor], lr: float = 1e-3, betas=(0.9, 0.999),
                 eps: float = 1e-8, weight_decay: float = 0.01):
        self.params = list(params)
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0
        # momentos m (1ª ordem) e v (2ª ordem) por parâmetro
        self.m = [np.zeros_like(p.data) for p in self.params]
        self.v = [np.zeros_like(p.data) for p in self.params]

    def step(self) -> None:
        self.t += 1
        bias1 = 1.0 - self.beta1 ** self.t
        bias2 = 1.0 - self.beta2 ** self.t
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            g = p.grad
            self.m[i] = self.beta1 * self.m[i] + (1.0 - self.beta1) * g
            self.v[i] = self.beta2 * self.v[i] + (1.0 - self.beta2) * (g * g)
            m_hat = self.m[i] / bias1
            v_hat = self.v[i] / bias2
            # decaimento de peso desacoplado, aplicado direto no parâmetro
            if self.weight_decay:
                p.data -= self.lr * self.weight_decay * p.data
            p.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

    def zero_grad(self) -> None:
        for p in self.params:
            p.grad = None


class SGDMomentum:
    """SGD com momentum e decaimento de peso: 1 buffer por parâmetro (12
    bytes/param no total: peso+gradiente+velocidade) contra os 16 bytes/param
    do AdamW (peso+gradiente+dois momentos). Converge um pouco mais devagar,
    mas cabe ~25% mais parâmetros na mesma VRAM — a troca certa quando o
    limite é memória, não velocidade de convergência.
    """

    def __init__(self, params: list[Tensor], lr: float = 1e-2, momentum: float = 0.9,
                 weight_decay: float = 0.01):
        self.params = list(params)
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.velocity = [np.zeros_like(p.data) for p in self.params]

    def step(self) -> None:
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            g = p.grad
            if self.weight_decay:
                p.data -= self.lr * self.weight_decay * p.data
            self.velocity[i] = self.momentum * self.velocity[i] + g
            p.data -= self.lr * self.velocity[i]

    def zero_grad(self) -> None:
        for p in self.params:
            p.grad = None


def clip_grad_norm(params: list[Tensor], max_norm: float) -> float:
    """Recorta o gradiente global para estabilizar o treino. Retorna a norma."""
    total = 0.0
    for p in params:
        if p.grad is not None:
            total += float(np.sum(p.grad * p.grad))
    total = total ** 0.5
    if total > max_norm:
        scale = max_norm / (total + 1e-6)
        for p in params:
            if p.grad is not None:
                p.grad *= scale
    return total
