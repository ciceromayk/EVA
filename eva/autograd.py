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

from .backend import scatter_add
from .backend import xp as np

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
        # No primeiro gradiente, guarda uma cópia direto — evita alocar um
        # array de zeros e uma passada de soma a cada acúmulo.
        if self.grad is None:
            self.grad = np.array(grad, dtype=np.float32, copy=True)
        else:
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
                scatter_add(full, idx, grad)
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

    def relu(self):
        data = np.maximum(self.data, 0.0)

        def backward(grad):
            if self.requires_grad:
                self._accumulate(grad * (self.data > 0.0))

        return Tensor._result(data, (self,), backward)

    def gelu(self):
        """GELU (aproximação tanh) — ativação padrão de Transformers.

        Diferente da tanh pura, não satura para entradas positivas, então
        deixa o gradiente fluir e o modelo aprende muito melhor.
        """
        x = self.data
        k = 0.7978845608028654  # sqrt(2/pi)
        # x*x*x é ~10x mais rápido que np.power(x, 3) para arrays grandes
        x2 = x * x
        inner = k * (x + 0.044715 * x2 * x)
        t = np.tanh(inner)
        data = 0.5 * x * (1.0 + t)

        def backward(grad):
            if self.requires_grad:
                d_inner = k * (1.0 + 0.134145 * x2)  # 3 * 0.044715
                dt = (1.0 - t * t) * d_inner
                dx = 0.5 * (1.0 + t) + 0.5 * x * dt
                self._accumulate(grad * dx)

        return Tensor._result(data, (self,), backward)

    def silu(self):
        """SiLU / Swish: x * sigmoid(x) — a ativação usada no Llama.

        É a metade "portão" do SwiGLU: suave como a GELU, mas mais barata
        de calcular (uma sigmoide em vez de uma tanh de polinômio).
        """
        x = self.data
        # sigmoide numericamente estável: exp só recebe valores <= 0, então
        # nunca estoura (overflow) mesmo para |x| grande.
        e = np.exp(-np.abs(x))
        sig = np.where(x >= 0, 1.0 / (1.0 + e), e / (1.0 + e))
        data = x * sig

        def backward(grad):
            if self.requires_grad:
                # d/dx [x*sig] = sig + x*sig*(1-sig) = sig*(1 + x*(1-sig))
                self._accumulate(grad * (sig * (1.0 + x * (1.0 - sig))))

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
        # Permutação inversa calculada em Python puro (não em `xp`): `axes` é
        # uma tupla de inteiros comuns, e o cupy.argsort exige um array de
        # verdade (diferente do numpy, que aceita qualquer sequência).
        inverse = [0] * len(axes)
        for i, ax in enumerate(axes):
            inverse[ax] = i
        inverse = tuple(inverse)

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
    idx = np.asarray(idx)  # garante que os índices estão no mesmo dispositivo
    data = weight.data[idx]

    def backward(grad):
        if weight.requires_grad:
            if weight.grad is None:
                weight.grad = np.zeros_like(weight.data)
            scatter_add(weight.grad, idx, grad)

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


def dropout(x: Tensor, p: float) -> Tensor:
    """Dropout invertido: zera uma fração `p` das ativações no treino.

    Só age quando o grafo de gradientes está ativo (treino). Durante geração
    e validação — que rodam sob `no_grad()` — retorna a entrada intacta, que
    é exatamente o comportamento esperado em modo de avaliação.
    """
    if not _grad_enabled or p <= 0.0:
        return x
    keep = 1.0 - p
    mask = (np.random.random(x.data.shape) >= p).astype(np.float32) / keep
    data = x.data * mask

    def backward(grad):
        if x.requires_grad:
            x._accumulate(grad * mask)

    return Tensor._result(data, (x,), backward)


def layer_norm(x: Tensor, gamma: Tensor, beta: Tensor, eps: float = 1e-5) -> Tensor:
    """LayerNorm fundido (normaliza a última dimensão) com gradiente analítico.

    Montar o LayerNorm com mean/sub/mul/pow cria vários nós e temporários por
    chamada; ele roda 2x por bloco. Fundir numa operação só acelera o passo.
    """
    xd = x.data
    mu = xd.mean(axis=-1, keepdims=True)
    xc = xd - mu
    var = (xc * xc).mean(axis=-1, keepdims=True)
    rstd = 1.0 / np.sqrt(var + eps)
    xhat = xc * rstd
    data = xhat * gamma.data + beta.data

    def backward(grad):
        axes = tuple(range(grad.ndim - 1))
        if gamma.requires_grad:
            gamma._accumulate((grad * xhat).sum(axis=axes))
        if beta.requires_grad:
            beta._accumulate(grad.sum(axis=axes))
        if x.requires_grad:
            dxhat = grad * gamma.data
            dx = rstd * (dxhat
                         - dxhat.mean(axis=-1, keepdims=True)
                         - xhat * (dxhat * xhat).mean(axis=-1, keepdims=True))
            x._accumulate(dx)

    return Tensor._result(data, (x, gamma, beta), backward)


def rms_norm(x: Tensor, gamma: Tensor, eps: float = 1e-5) -> Tensor:
    """RMSNorm fundido (Llama): normaliza pela raiz da média dos quadrados.

    Diferente do LayerNorm, NÃO subtrai a média nem tem deslocamento (beta):
    só reescala cada vetor para norma RMS 1 e aplica o ganho `gamma`. Menos
    contas, menos parâmetros, e na prática treina igual ou melhor.
    """
    xd = x.data
    ms = (xd * xd).mean(axis=-1, keepdims=True)
    rrms = 1.0 / np.sqrt(ms + eps)
    xhat = xd * rrms
    data = xhat * gamma.data

    def backward(grad):
        if gamma.requires_grad:
            axes = tuple(range(grad.ndim - 1))
            gamma._accumulate((grad * xhat).sum(axis=axes))
        if x.requires_grad:
            dxhat = grad * gamma.data
            # dx = rrms * (dxhat - xhat * mean(dxhat * xhat))
            dx = rrms * (dxhat - xhat * (dxhat * xhat).mean(axis=-1, keepdims=True))
            x._accumulate(dx)

    return Tensor._result(data, (x, gamma), backward)


def rope(x: Tensor, cos: np.ndarray, sin: np.ndarray) -> Tensor:
    """RoPE — Rotary Position Embedding (Llama), como uma única operação.

    Em vez de SOMAR um vetor de posição ao embedding (estilo GPT), o RoPE
    GIRA cada par de coordenadas de Q e K por um ângulo proporcional à
    posição do token. O produto escalar entre Q e K passa a depender só da
    DISTÂNCIA relativa entre os tokens — por isso generaliza melhor.

    `x` tem shape (..., T, head_dim); a rotação emparelha a primeira metade
    do vetor com a segunda (convenção do Llama). `cos`/`sin` têm shape
    (T, head_dim/2) e são pré-computados (não são parâmetros treináveis).
    """
    xd = x.data
    assert xd.shape[-1] % 2 == 0, "rope: a última dimensão (head_dim) precisa ser par"
    half = xd.shape[-1] // 2
    x1, x2 = xd[..., :half], xd[..., half:]
    data = np.concatenate([x1 * cos - x2 * sin, x1 * sin + x2 * cos], axis=-1)

    def backward(grad):
        if x.requires_grad:
            g1, g2 = grad[..., :half], grad[..., half:]
            # a rotação é ortogonal: o backward é girar no sentido contrário
            dx = np.concatenate([g1 * cos + g2 * sin, -g1 * sin + g2 * cos], axis=-1)
            x._accumulate(dx)

    return Tensor._result(data, (x,), backward)


def softmax(x: Tensor, axis: int = -1) -> Tensor:
    """Softmax estável como uma ÚNICA operação, com gradiente analítico.

    Construir o softmax a partir de exp/sum/pow cria vários nós e arrays
    temporários no caminho quente da atenção. Fundir tudo numa operação só,
    com o Jacobiano-vetor analítico, reduz bastante o custo por passo.
    """
    z = x.data - x.data.max(axis=axis, keepdims=True)
    e = np.exp(z)
    s = e / e.sum(axis=axis, keepdims=True)

    def backward(grad):
        if x.requires_grad:
            # regra do softmax: dx = s * (grad - sum(grad*s))
            dot = (grad * s).sum(axis=axis, keepdims=True)
            x._accumulate(s * (grad - dot))

    return Tensor._result(s, (x,), backward)
