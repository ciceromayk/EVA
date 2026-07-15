"""Verifica o autograd comparando com gradientes numéricos (diferenças finitas).

Se a derivada analítica implementada no grafo bate com a aproximação
numérica, temos alta confiança de que o backpropagation está correto.
"""

import numpy as np

from eva.autograd import Tensor, cross_entropy, softmax


def numerical_grad(fn, x, eps=1e-4):
    grad = np.zeros_like(x.data)
    flat = x.data.reshape(-1)
    gflat = grad.reshape(-1)
    for i in range(flat.size):
        original = flat[i]
        flat[i] = original + eps
        plus = float(fn(x).data.sum())
        flat[i] = original - eps
        minus = float(fn(x).data.sum())
        flat[i] = original
        gflat[i] = (plus - minus) / (2 * eps)
    return grad


def check(fn, shape, seed=0):
    rng = np.random.default_rng(seed)
    x = Tensor(rng.standard_normal(shape).astype(np.float64), requires_grad=True)
    out = fn(x)
    out.sum().backward()
    numeric = numerical_grad(fn, x)
    assert np.allclose(x.grad, numeric, atol=1e-3), f"{fn}: {np.abs(x.grad-numeric).max()}"


def test_elementwise():
    check(lambda t: t * t + t, (4, 3))
    check(lambda t: (t * 2.0).tanh(), (4, 3))
    check(lambda t: (t * t + 1.0).log(), (5,))
    check(lambda t: t.exp(), (3, 2))


def test_matmul_and_reduce():
    w = Tensor(np.random.default_rng(1).standard_normal((3, 2)), requires_grad=False)
    check(lambda t: (t @ w.data).sum(axis=1), (4, 3))
    check(lambda t: t.mean(axis=0), (4, 3))


def test_softmax_rows_sum_to_one():
    x = Tensor(np.random.default_rng(2).standard_normal((5, 7)), requires_grad=True)
    p = softmax(x, axis=-1)
    assert np.allclose(p.data.sum(axis=-1), 1.0, atol=1e-5)


def test_cross_entropy_matches_manual():
    rng = np.random.default_rng(3)
    logits = Tensor(rng.standard_normal((6, 4)), requires_grad=True)
    targets = rng.integers(0, 4, size=6)
    loss = cross_entropy(logits, targets)
    loss.backward()
    # gradiente analítico esperado: (softmax - onehot) / N
    exp = np.exp(logits.data - logits.data.max(axis=1, keepdims=True))
    probs = exp / exp.sum(axis=1, keepdims=True)
    expected = probs.copy()
    expected[np.arange(6), targets] -= 1
    expected /= 6
    assert np.allclose(logits.grad, expected, atol=1e-5)


if __name__ == "__main__":
    test_elementwise()
    test_matmul_and_reduce()
    test_softmax_rows_sum_to_one()
    test_cross_entropy_matches_manual()
    print("Todos os testes de autograd passaram.")
