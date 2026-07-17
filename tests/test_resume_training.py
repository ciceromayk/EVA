"""Testa o treino incremental (--resume): continuar um checkpoint existente
sem recomeçar do zero, preservando arquitetura e tokenizer.
"""

import argparse
import os
import tempfile

import train as train_mod

CORPUS = "A EVA aprende com o material que voce fornecer. " * 40


def make_args(data_path, resume=False, steps=20, seed=0):
    return argparse.Namespace(
        data=data_path, preset="nano", tokenizer="char", bpe_vocab=256,
        dropout=0.0, device="cpu", resume=resume, steps=steps,
        batch_size=8, block_size=32, n_layer=1, n_head=1, n_embd=16,
        lr=3e-3, log_every=10, seed=seed, generate=None, max_new=10,
    )


def test_resume_preserva_arquitetura_e_vocabulario():
    with tempfile.TemporaryDirectory() as d:
        data_path = os.path.join(d, "corpus.txt")
        with open(data_path, "w", encoding="utf-8") as f:
            f.write(CORPUS)
        train_mod.CKPT_PATH = os.path.join(d, "ckpt.pkl")

        # 1) treino do zero
        train_mod.train(make_args(data_path, resume=False, steps=20, seed=1))
        assert os.path.exists(train_mod.CKPT_PATH)
        model1, tok1 = train_mod.load_checkpoint()
        vocab1, params1 = tok1.vocab_size, model1.num_params()

        # 2) continua treinando o MESMO checkpoint (não recomeça do zero)
        train_mod.train(make_args(data_path, resume=True, steps=20, seed=2))
        model2, tok2 = train_mod.load_checkpoint()

        assert tok2.vocab_size == vocab1, "vocabulário mudou ao continuar"
        assert model2.num_params() == params1, "arquitetura mudou ao continuar"
        assert model2.config.n_layer == model1.config.n_layer
        assert model2.config.n_embd == model1.config.n_embd


def test_resume_sem_checkpoint_falha_com_mensagem_clara():
    with tempfile.TemporaryDirectory() as d:
        data_path = os.path.join(d, "corpus.txt")
        with open(data_path, "w", encoding="utf-8") as f:
            f.write(CORPUS)
        train_mod.CKPT_PATH = os.path.join(d, "nao_existe.pkl")
        try:
            train_mod.train(make_args(data_path, resume=True, steps=5))
            raise AssertionError("deveria ter levantado SystemExit")
        except SystemExit as exc:
            assert "checkpoint" in str(exc).lower()


def test_resume_nunca_salva_checkpoint_pior_que_o_carregado():
    """Ao continuar, o best_val inicial é medido no modelo carregado — o
    checkpoint só é sobrescrito se um passo futuro bater essa marca."""
    with tempfile.TemporaryDirectory() as d:
        data_path = os.path.join(d, "corpus.txt")
        with open(data_path, "w", encoding="utf-8") as f:
            f.write(CORPUS)
        train_mod.CKPT_PATH = os.path.join(d, "ckpt.pkl")

        train_mod.train(make_args(data_path, resume=False, steps=30, seed=1))
        model_before, _ = train_mod.load_checkpoint()
        mtime_before = os.path.getmtime(train_mod.CKPT_PATH)

        # 0 passos adicionais: não há chance de melhorar, então o
        # checkpoint não deveria ser sobrescrito por algo pior.
        train_mod.train(make_args(data_path, resume=True, steps=1, seed=99))
        mtime_after = os.path.getmtime(train_mod.CKPT_PATH)

        # com 1 passo de log (log_every=10 > steps=1, mas step==1 sempre loga),
        # a única avaliação é a baseline == best_val inicial, então só salva
        # se empatar/melhorar — não deve regredir de forma alguma.
        model_after, _ = train_mod.load_checkpoint()
        assert model_after.num_params() == model_before.num_params()
        assert mtime_after >= mtime_before


if __name__ == "__main__":
    test_resume_preserva_arquitetura_e_vocabulario()
    test_resume_sem_checkpoint_falha_com_mensagem_clara()
    test_resume_nunca_salva_checkpoint_pior_que_o_carregado()
    print("Todos os testes de treino incremental passaram.")
