"""Backend de array selecionável: NumPy (CPU) ou CuPy (GPU NVIDIA).

Nosso autograd é escrito à mão sobre um "módulo de array". Trocando esse
módulo de NumPy para CuPy, o MESMO código passa a rodar na GPU via CUDA —
sem reescrever a lógica de gradientes, atenção etc.

Escolha o dispositivo pela variável de ambiente antes de importar `eva`:

    EVA_DEVICE=gpu   -> tenta a GPU (CuPy); cai para CPU se indisponível
    EVA_DEVICE=cpu   -> NumPy (padrão)

O `train.py` também aceita `--device gpu`, que apenas define essa variável.
"""

from __future__ import annotations

import os

import numpy as _np

xp = _np          # módulo de array em uso (numpy ou cupy)
_gpu = False      # estamos na GPU?


def using_gpu() -> bool:
    return _gpu


def device_name() -> str:
    return "GPU (CuPy)" if _gpu else "CPU (NumPy)"


def _init() -> None:
    global xp, _gpu
    dev = os.environ.get("EVA_DEVICE", "cpu").lower()
    if dev in ("gpu", "cuda"):
        try:
            import cupy as cp
            # Exercita as libs realmente usadas no treino — random (curand) e
            # matmul (cublas) — para detectar CUDA incompleto AGORA e cair para
            # a CPU com elegância, em vez de quebrar no meio do treino.
            a = cp.random.random(4).astype(cp.float32)
            float((a @ a).sum())
            xp, _gpu = cp, True
            return
        except Exception as exc:  # sem CuPy, CUDA incompleto, driver antigo…
            print(f"[EVA] GPU solicitada, mas indisponível ({type(exc).__name__}). "
                  "Usando CPU.")
            print("[EVA] Faltam bibliotecas CUDA (ex.: curand). "
                  "Veja o README, seção 'Rodar na GPU'.")
    xp, _gpu = _np, False


_init()


def asnumpy(a):
    """Traz um array para a CPU como numpy (identidade quando já é numpy)."""
    if _gpu and hasattr(a, "get"):
        return a.get()
    return _np.asarray(a)


def to_device(a):
    """Leva um array (ex.: numpy vindo de um checkpoint) para o dispositivo atual."""
    return xp.asarray(a)


def scatter_add(target, indices, updates) -> None:
    """Equivale a `np.add.at`: acumula em índices que podem se repetir."""
    if _gpu:
        import cupyx
        cupyx.scatter_add(target, indices, updates)
    else:
        _np.add.at(target, indices, updates)
