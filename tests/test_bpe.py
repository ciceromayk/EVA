"""Testes do tokenizador BPE: round-trip, compressão e persistência."""

import os
import tempfile

from eva.bpe import BPETokenizer

SAMPLE = (
    "A arte da guerra é vital para o Estado. "
    "Um algoritmo é uma sequência de passos. "
    "Modelos de linguagem aprendem padrões do texto. " * 20
)


def test_round_trip_preserva_texto():
    tok = BPETokenizer.train(SAMPLE, vocab_size=320)
    assert tok.decode(tok.encode(SAMPLE)) == SAMPLE


def test_round_trip_texto_novo():
    tok = BPETokenizer.train(SAMPLE, vocab_size=320)
    novo = "A lógica de programação é a base do raciocínio computacional."
    assert tok.decode(tok.encode(novo)) == novo


def test_comprime_mais_que_bytes():
    tok = BPETokenizer.train(SAMPLE, vocab_size=400)
    n_tokens = len(tok.encode(SAMPLE))
    n_bytes = len(SAMPLE.encode("utf-8"))
    assert n_tokens < n_bytes  # subpalavras encurtam a sequência


def test_unicode_acentos_e_emoji():
    tok = BPETokenizer.train("ação coração 😀 informação", vocab_size=300)
    s = "informação: ação e coração 😀"
    assert tok.decode(tok.encode(s)) == s


def test_save_load():
    tok = BPETokenizer.train(SAMPLE, vocab_size=320)
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "bpe.json")
        tok.save(path)
        loaded = BPETokenizer.load(path)
    assert loaded.vocab_size == tok.vocab_size
    assert loaded.encode(SAMPLE) == tok.encode(SAMPLE)


if __name__ == "__main__":
    test_round_trip_preserva_texto()
    test_round_trip_texto_novo()
    test_comprime_mais_que_bytes()
    test_unicode_acentos_e_emoji()
    test_save_load()
    print("Todos os testes de BPE passaram.")
