"""Motor de autograd minimalista sobre NumPy.

Este módulo implementa diferenciação automática em modo reverso
(backpropagation) — o mesmo mecanismo que PyTorch e TensorFlow usam,
em miniatura. Cada `Tensor` guarda seus dados (um array NumPy), o
gradiente acumulado e uma referência aos tensores que o geraram,
formando um grafo computacional. Chamar `.backward()` no resultado
final percorre o grafo em ordem topológica reversa aplicando a regra
da cadeia.
"""

from __future__ import annotations

import contextlib

import numpy as np

# Flag global: quando False, as operações não constroem o grafo
# (equivalente ao torch.no_grad()). Útil na geração de texto.
_grad_enabled = True


@contextlib.contextmanager
def no_grad():
    """Contexto em que nenhum grafo de gradientes é construído."""
    global _grad_enabled
    previous = _grad_enabled
    _grad_enabled = False
    try:
        yield
    finally:
        _grad_enabled = previous


def _unbroadcast(grad: np.ndarray, shape: tuple) -> np.ndarray:
    """Reduz `grad` de volta a `shape`, desfazendo o broadcasting do NumPy.

    Se um operando de shape (1, C) foi expandido para (B, C) durante a
    operação, o gradiente que chega tem shape (B, C) e precisa ser somado
    ao longo das dimensões expandidas para voltar a (1, C).
    """
    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)
    for axis, size in enumerate(shape):
        if size == 1 and grad.shape[axis] != 1:
            grad = grad.sum(axis=axis, keepdims=True)
    return grad


class Tensor:
    """Array NumPy com suporte a diferenciação automática."""

    __slots__ = ("data", "grad", "requires_grad", "_backward", "_prev")

    def __init__(self, data, requires_grad: bool = False):
        if isinstance(data, Tensor):
            data = data.data
        arr = np.asarray(data)
        # Preserva ponto flutuante de 64 bits quando fornecido (útil para
        # checagem numérica de gradiente); caso contrário usa float32.
        if arr.dtype != np.float64:
            arr = arr.astype(np.float32)
        self.data = arr
        self.grad: np.ndarray | None = None
        self.requires_grad = requires_grad
        self._backward = None  # closure que propaga o gradiente aos pais
        self._prev: tuple = ()

    # ------------------------------------------------------------------
    # Infraestrutura do grafo
    # ------------------------------------------------------------------
    @staticmethod
    def _result(data, parents, backward) -> "Tensor":
        """Cria o tensor resultado de uma operação, ligando-o ao grafo."""
        out = Tensor(data)
        if _grad_enabled and any(p.requires_grad for p in parents):
            out.requires_grad = True
            out._prev = tuple(parents)
            out._backward = backward
        return out

    def _accumulate(self, grad: np.ndarray) -> None:
        if self.grad is None:
            self.grad = np.zeros_like(self.data)
        self.grad += grad

    def backward(self) -> None:
        """Backpropagation a partir deste tensor (tipicamente a loss)."""
        if not self.requires_grad:
            raise RuntimeError("backward() em tensor que não requer gradiente")

        topo: list[Tensor] = []
        visited: set[int] = set()

        def build(t: Tensor) -> None:
            if id(t) in visited:
                return
            visited.add(id(t))
            for parent in t._prev:
                build(parent)
            topo.append(t)

        build(self)
        self.grad = np.ones_like(self.data)
        for node in reversed(topo):
            if node._backward is not None and node.grad is not None:
                node._backward(node.grad)

    def detach(self) -> "Tensor":
        return Tensor(self.data)

    # ------------------------------------------------------------------
    # Propriedades utilitárias
    # ------------------------------------------------------------------
    @property
    def shape(self):
        return self.data.shape

    @property
    def ndim(self):
        return self.data.ndim

    def __repr__(self):
        return f"Tensor(shape={self.data.shape}, requires_grad={self.requires_grad})"

    # ------------------------------------------------------------------
    # Operações aritméticas
    # ------------------------------------------------------------------
    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        data = self.data + other.data

        def backward(grad):
            if self.requires_grad:
                self._accumulate(_unbroadcast(grad, self.data.shape))
            if other.requires_grad:
                other._accumulate(_unbroadcast(grad, other.data.shape))

        return Tensor._result(data, (self, other), backward)

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        data = self.data * other.data

        def backward(grad):
            if self.requires_grad:
                self._accumulate(_unbroadcast(grad * other.data, self.data.shape))
            if other.requires_grad:
                other._accumulate(_unbroadcast(grad * self.data, other.data.shape))

        return Tensor._result(data, (self, other), backward)

    def __pow__(self, exponent):
        if not isinstance(exponent, (int, float)):
            raise TypeError("apenas expoentes escalares são suportados")
        data = self.data ** exponent

        def backward(grad):
            if self.requires_grad:
                self._accumulate(grad * exponent * self.data ** (exponent - 1))

        return Tensor._result(data, (self,), backward)

    def __matmul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        data = self.data @ other.data

        def backward(grad):
            if self.requires_grad:
                ga = grad @ other.data.swapaxes(-1, -2)
                self._accumulate(_unbroadcast(ga, self.data.shape))
            if other.requires_grad:
                gb = self.data.swapaxes(-1, -2) @ grad
                other._accumulate(_unbroadcast(gb, other.data.shape))

        return Tensor._result(data, (self, other), backward)

    def __neg__(self):
        return self * -1.0

    def __sub__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        return self + (-other)

    def __truediv__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        return self * other ** -1.0

    def __radd__(self, other):
        return self + other

    def __rmul__(self, other):
        return self * other

    def __rsub__(self, other):
        return (-self) + other

    def __rtruediv__(self, other):
        return Tensor(other) * self ** -1.0

    def __getitem__(self, idx):
        data = self.data[idx]

        def backward(grad):
            if self.requires_grad:
                full = np.zeros_like(self.data)
                np.add.at(full, idx, grad)
                self._accumulate(full)

        return Tensor._result(data, (self,), backward)

    # ------------------------------------------------------------------
    # Funções elementares
    # ------------------------------------------------------------------
    def exp(self):
        data = np.exp(self.data)

        def backward(grad):
            if self.requires_grad:
                self._accumulate(grad * data)

        return Tensor._result(data, (self,), backward)

    def log(self):
        data = np.log(self.data)

        def backward(grad):
            if self.requires_grad:
                self._accumulate(grad / self.data)

        return Tensor._result(data, (self,), backward)

    def tanh(self):
        data = np.tanh(self.data)

        def backward(grad):
            if self.requires_grad:
                self._accumulate(grad * (1.0 - data * data))

        return Tensor._result(data, (self,), backward)

    # ------------------------------------------------------------------
    # Reduções e mudanças de forma
    # ------------------------------------------------------------------
    def sum(self, axis=None, keepdims=False):
        data = self.data.sum(axis=axis, keepdims=keepdims)

        def backward(grad):
            if not self.requires_grad:
                return
            g = grad
            if axis is not None and not keepdims:
                g = np.expand_dims(g, axis)
            self._accumulate(np.broadcast_to(g, self.data.shape))

        return Tensor._result(data, (self,), backward)

    def mean(self, axis=None, keepdims=False):
        if axis is None:
            count = self.data.size
        else:
            axes = axis if isinstance(axis, tuple) else (axis,)
            count = 1
            for ax in axes:
                count *= self.data.shape[ax]
        return self.sum(axis=axis, keepdims=keepdims) * (1.0 / count)

    def reshape(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        original = self.data.shape
        data = self.data.reshape(shape)

        def backward(grad):
            if self.requires_grad:
                self._accumulate(grad.reshape(original))

        return Tensor._result(data, (self,), backward)

    def transpose(self, *axes):
        if len(axes) == 1 and isinstance(axes[0], (tuple, list)):
            axes = tuple(axes[0])
        data = self.data.transpose(axes)
        inverse = tuple(np.argsort(axes))

        def backward(grad):
            if self.requires_grad:
                self._accumulate(grad.transpose(inverse))

        return Tensor._result(data, (self,), backward)


# ----------------------------------------------------------------------
# Operações funcionais com gradiente analítico
# ----------------------------------------------------------------------
def embedding(weight: Tensor, idx: np.ndarray) -> Tensor:
    """Seleciona linhas de `weight` pelos índices inteiros `idx`.

    No backward, os gradientes são acumulados de volta nas linhas
    correspondentes (scatter-add), inclusive quando um índice se repete.
    """
    data = weight.data[idx]

    def backward(grad):
        if weight.requires_grad:
            if weight.grad is None:
                weight.grad = np.zeros_like(weight.data)
            np.add.at(weight.grad, idx, grad)

    return Tensor._result(data, (weight,), backward)


def cross_entropy(logits: Tensor, targets: np.ndarray) -> Tensor:
    """Entropia cruzada média entre `logits` (N, C) e classes-alvo (N,).

    Combina log-softmax e negative log-likelihood em uma única operação,
    com o gradiente analítico bem conhecido: (softmax(logits) - onehot) / N.
    """
    targets = np.asarray(targets).reshape(-1)
    n = logits.data.shape[0]
    shifted = logits.data - logits.data.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    probs = exp / exp.sum(axis=1, keepdims=True)
    loss = -np.log(probs[np.arange(n), targets] + 1e-12).mean()

    def backward(grad):
        if logits.requires_grad:
            g = probs.copy()
            g[np.arange(n), targets] -= 1.0
            logits._accumulate(grad * g / n)

    return Tensor._result(loss, (logits,), backward)


def softmax(x: Tensor, axis: int = -1) -> Tensor:
    """Softmax numericamente estável construído com as primitivas do grafo."""
    # Subtrair o máximo (constante, sem gradiente) só melhora a estabilidade
    # numérica; não altera o resultado nem o gradiente do softmax.
    stable = x + Tensor(-x.data.max(axis=axis, keepdims=True))
    e = stable.exp()
    return e * e.sum(axis=axis, keepdims=True) ** -1.0
