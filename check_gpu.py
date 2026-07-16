"""Autoteste de GPU da EVA — confirma que a RTX está sendo usada e compara
a velocidade de um passo de treino entre CPU e GPU.

Uso:
    pip install cupy-cuda12x        # (ver README para a versão de CUDA certa)
    python check_gpu.py
"""

from __future__ import annotations

import os
import time


def bench(device: str, steps: int = 5):
    """Roda alguns passos de treino no dispositivo e devolve ms/passo."""
    os.environ["EVA_DEVICE"] = device
    # importa DEPOIS de setar o dispositivo (o backend é fixado no import)
    import importlib

    import eva.backend as backend
    importlib.reload(backend)
    for mod in ("eva.autograd", "eva.nn", "eva.model", "eva.optim", "eva"):
        importlib.reload(importlib.import_module(mod))

    import numpy as np
    from eva import GPT, GPTConfig, AdamW, clip_grad_norm

    if device == "gpu" and not backend.using_gpu():
        return None  # GPU indisponível

    cfg = GPTConfig(vocab_size=134, block_size=96, n_layer=4, n_head=6,
                    n_embd=192, dropout=0.1)
    model = GPT(cfg)
    opt = AdamW(model.parameters(), lr=3e-3)
    rng = np.random.default_rng(0)
    x = rng.integers(0, 134, size=(16, 96))
    y = rng.integers(0, 134, size=(16, 96))

    for _ in range(2):  # aquecimento (compila kernels na GPU)
        _, loss = model.forward(x, y)
        model.zero_grad(); loss.backward(); opt.step()

    best = float("inf")
    for _ in range(3):
        t0 = time.time()
        for _ in range(steps):
            _, loss = model.forward(x, y)
            model.zero_grad(); loss.backward()
            clip_grad_norm(model.parameters(), 1.0); opt.step()
        best = min(best, (time.time() - t0) / steps)
    return best * 1000


def main():
    print("== Autoteste de GPU da EVA ==\n")
    cpu = bench("cpu")
    print(f"CPU (NumPy):  {cpu:6.0f} ms/passo")

    gpu = bench("gpu")
    if gpu is None:
        print("\nGPU não disponível (CuPy/CUDA não instalado ou driver antigo).")
        print("Veja o README, seção 'Rodar na GPU', para instalar o CuPy certo.")
        return

    print(f"GPU (CuPy):   {gpu:6.0f} ms/passo")
    print(f"\n⚡ A GPU está {cpu / gpu:.1f}x mais rápida que a CPU neste modelo.")
    print("   Use: python train.py --preset medium --device gpu")


if __name__ == "__main__":
    main()
