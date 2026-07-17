"""Regressão: o loop de treino não pode acumular tensores entre passos.

Um modelo gigante (ex.: o preset "xlarge") revelou que manter a variável
`loss` viva até a PRÓXIMA iteração reatribuí-la faz duas iterações terem o
grafo computacional retido ao mesmo tempo — quase dobrando o pico de
memória. train.py agora solta (`del`) o grafo logo após cada passo; este
teste confirma que a contagem de tensores vivos se estabiliza, em vez de
crescer a cada iteração.
"""

import gc

import numpy as np

from eva import AdamW, GPT, GPTConfig, clip_grad_norm
from eva.autograd import Tensor


def count_live_tensors() -> int:
    gc.collect()
    return sum(1 for o in gc.get_objects() if isinstance(o, Tensor))


def test_treino_nao_acumula_tensores_entre_passos():
    cfg = GPTConfig(vocab_size=30, block_size=16, n_layer=2, n_head=2, n_embd=16)
    model = GPT(cfg)
    opt = AdamW(model.parameters(), lr=1e-3)
    rng = np.random.default_rng(0)
    x = rng.integers(0, 30, size=(4, 16))
    y = rng.integers(0, 30, size=(4, 16))

    counts = []
    for _ in range(12):
        # mesmo padrão do loop de treino real em train.py: descarta o
        # grafo (logits/loss) assim que o passo termina.
        logits, loss = model.forward(x, y)
        del logits
        model.zero_grad()
        loss.backward()
        clip_grad_norm(model.parameters(), max_norm=1.0)
        opt.step()
        del loss
        counts.append(count_live_tensors())

    # depois do aquecimento, a contagem de tensores vivos (basicamente só
    # os parâmetros do modelo) deve estabilizar — não crescer a cada passo.
    steady = counts[-4:]
    assert max(steady) - min(steady) <= 2, f"contagem de tensores instável: {counts}"


if __name__ == "__main__":
    test_treino_nao_acumula_tensores_entre_passos()
    print("Todos os testes de vazamento de memória passaram.")
