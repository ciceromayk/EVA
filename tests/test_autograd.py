"""Verifica o autograd comparando com gradientes numéricos (diferenças finitas).

Se a derivada analítica implementada no grafo bate com a aproximação
numérica, temos alta confiança de que o backpropagation está correto.
"""

import numpy as np

from eva.autograd import Tensor, cross_entropy, dropout, no_grad, rms_norm, rope, softmax


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
    check(lambda t: t.gelu(), (4, 3))
    check(lambda t: t.silu(), (4, 3))
    check(lambda t: (t * 1.5).relu() + t, (4, 3))


def test_matmul_and_reduce():
    w = Tensor(np.random.default_rng(1).standard_normal((3, 2)), requires_grad=False)
    check(lambda t: (t @ w.data).sum(axis=1), (4, 3))
    check(lambda t: t.mean(axis=0), (4, 3))


def test_softmax_rows_sum_to_one():
    x = Tensor(np.random.default_rng(2).standard_normal((5, 7)), requires_grad=True)
    p = softmax(x, axis=-1)
    assert np.allclose(p.data.sum(axis=-1), 1.0, atol=1e-5)


def test_dropout_off_em_no_grad():
    x = Tensor(np.ones((100, 100)), requires_grad=True)
    # sob no_grad (validação/geração), dropout é identidade
    with no_grad():
        assert np.array_equal(dropout(x, 0.5).data, x.data)
    # no treino, zera ~metade e escala o resto por 1/(1-p)
    out = dropout(x, 0.5)
    zeros = (out.data == 0).mean()
    assert 0.4 < zeros < 0.6
    assert np.allclose(out.data[out.data != 0], 2.0)


def test_rms_norm_grad():
    gamma = Tensor(np.random.default_rng(4).standard_normal(6), requires_grad=False)
    check(lambda t: rms_norm(t, gamma), (5, 6))
    # gradiente do gamma: comparação com diferenças finitas
    rng = np.random.default_rng(5)
    x = Tensor(rng.standard_normal((4, 6)), requires_grad=False)
    g = Tensor(rng.standard_normal(6).astype(np.float64), requires_grad=True)
    rms_norm(x, g).sum().backward()
    numeric = numerical_grad(lambda t: rms_norm(x, t), g)
    assert np.allclose(g.grad, numeric, atol=1e-3)


def test_rope_grad_and_norm():
    # tabelas de um RoPE pequeno: T=5 posições, head_dim=6 (half=3)
    rng = np.random.default_rng(6)
    half = 3
    angles = np.outer(np.arange(5.0), 10000.0 ** (-np.arange(half) / half))
    cos, sin = np.cos(angles), np.sin(angles)
    check(lambda t: rope(t, cos, sin), (2, 5, 6))
    # rotação é ortogonal: preserva a norma de cada vetor
    x = Tensor(rng.standard_normal((2, 5, 6)))
    out = rope(x, cos, sin)
    assert np.allclose((out.data ** 2).sum(-1), (x.data ** 2).sum(-1), atol=1e-5)


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
    test_dropout_off_em_no_grad()
    test_rms_norm_grad()
    test_rope_grad_and_norm()
    test_cross_entropy_matches_manual()
    print("Todos os testes de autograd passaram.")
