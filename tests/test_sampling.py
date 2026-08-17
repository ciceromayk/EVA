"""Testa a lógica de amostragem (temperatura, top-k, top-p, anti-repetição)
isoladamente, sem precisar de um modelo treinado — via `sample_probs`.
"""

import numpy as np

from eva.model import sample_probs


def test_probs_sempre_somam_um():
    logits = np.array([2.0, 1.0, 0.1, -3.0, 0.5], dtype=np.float32)
    probs = sample_probs(logits.copy(), generated_ids=[], temperature=0.7,
                         top_k=3, top_p=0.9, repetition_penalty=1.15)
    assert np.isclose(probs.sum(), 1.0, atol=1e-6)


def test_top_k_zera_fora_do_conjunto():
    logits = np.array([5.0, 4.0, 3.0, 2.0, 1.0], dtype=np.float32)
    probs = sample_probs(logits.copy(), generated_ids=[], top_k=2)
    # só os 2 maiores (índices 0 e 1) podem ter probabilidade > 0
    assert (probs[2:] == 0).all()
    assert probs[0] > 0 and probs[1] > 0


def test_top_p_mantem_pelo_menos_um_token():
    # um logit MUITO maior que os outros -> top_p baixo deveria isolar só ele
    logits = np.array([100.0, 1.0, 1.0, 1.0], dtype=np.float32)
    probs = sample_probs(logits.copy(), generated_ids=[], top_p=0.01)
    assert probs[0] == 1.0
    assert probs[1:].sum() == 0.0


def test_top_p_desligado_nao_filtra_nada():
    logits = np.array([3.0, 2.0, 1.0, 0.0], dtype=np.float32)
    probs = sample_probs(logits.copy(), generated_ids=[], top_p=None)
    assert (probs > 0).all()


def test_repetition_penalty_reduz_prob_de_token_repetido():
    logits = np.array([2.0, 2.0, 2.0], dtype=np.float32)
    baseline = sample_probs(logits.copy(), generated_ids=[], repetition_penalty=1.0)
    penalizado = sample_probs(logits.copy(), generated_ids=[0], repetition_penalty=1.5)
    # os 3 tokens começam empatados; penalizar o token 0 (já usado) deve
    # deixar sua probabilidade abaixo da que tinha sem penalidade
    assert penalizado[0] < baseline[0]
    assert penalizado[1] > penalizado[0]


def test_repetition_penalty_desligado_e_neutro():
    logits = np.array([2.0, 1.0, 0.0], dtype=np.float32)
    com_ids_mas_desligado = sample_probs(logits.copy(), generated_ids=[0, 1, 2],
                                          repetition_penalty=1.0)
    sem_ids = sample_probs(logits.copy(), generated_ids=[], repetition_penalty=1.0)
    assert np.allclose(com_ids_mas_desligado, sem_ids)


def test_temperatura_baixa_concentra_probabilidade():
    logits = np.array([2.0, 1.0, 0.0], dtype=np.float32)
    fria = sample_probs(logits.copy(), generated_ids=[], temperature=0.2)
    neutra = sample_probs(logits.copy(), generated_ids=[], temperature=1.0)
    assert fria[0] > neutra[0]  # temperatura baixa favorece ainda mais o líder


if __name__ == "__main__":
    test_probs_sempre_somam_um()
    test_top_k_zera_fora_do_conjunto()
    test_top_p_mantem_pelo_menos_um_token()
    test_top_p_desligado_nao_filtra_nada()
    test_repetition_penalty_reduz_prob_de_token_repetido()
    test_repetition_penalty_desligado_e_neutro()
    test_temperatura_baixa_concentra_probabilidade()
    print("Todos os testes de amostragem passaram.")
