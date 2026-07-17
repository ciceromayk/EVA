"""Tokenizador BPE (Byte-Pair Encoding) construído do zero.

O tokenizador em nível de caractere é simples, mas gasta um token por
letra — o modelo "enxerga" pouquíssimo texto por janela de contexto. O BPE
resolve isso aprendendo *subpalavras*: parte dos bytes brutos e vai
fundindo, iterativamente, o par de símbolos adjacentes mais frequente. Ao
final, sequências comuns ("ção", "modelo", " de ") viram um único token.

É o mesmo algoritmo usado por GPT-2/GPT-3/GPT-4 (aqui em miniatura, em
nível de byte, sem regex de pré-tokenização).

Implementado com NumPy vetorizado: contar pares e aplicar uma fusão em um
array de milhões de elementos com laços Python puro é impraticável (cada
fusão faz uma varredura completa, e são até `vocab_size - 256` fusões). As
mesmas duas operações em NumPy (`np.unique` para contar, comparação
vetorizada + laço só sobre as poucas posições que casam para aplicar a
fusão) chegam a ser centenas de vezes mais rápidas no mesmo corpus.
"""

from __future__ import annotations

import json

import numpy as np


def _count_pairs(ids: np.ndarray) -> dict[tuple[int, int], int]:
    """Conta pares adjacentes com NumPy: codifica cada par (a, b) num único
    inteiro (a*BASE + b) e usa np.unique para contar em lote, em vez de um
    laço Python por elemento."""
    if ids.size < 2:
        return {}
    base = int(ids.max()) + 1
    codes = ids[:-1].astype(np.int64) * base + ids[1:].astype(np.int64)
    unique, counts = np.unique(codes, return_counts=True)
    a, b = np.divmod(unique, base)
    return {(int(x), int(y)): int(c) for x, y, c in zip(a, b, counts)}


def _merge(ids: np.ndarray, pair: tuple[int, int], new_id: int) -> np.ndarray:
    """Substitui toda ocorrência (não sobreposta) de `pair` por `new_id`.

    Acha os candidatos com comparação vetorizada (rápido mesmo em milhões
    de elementos) e só usa um laço Python sobre as POSIÇÕES que casaram —
    tipicamente uma fração pequena do array — para descartar sobreposições
    (ex.: em "aaa", só a primeira ocorrência de "aa" pode ser fundida).
    """
    if ids.size < 2:
        return ids
    match = (ids[:-1] == pair[0]) & (ids[1:] == pair[1])
    positions = np.flatnonzero(match)
    if positions.size == 0:
        return ids

    keep = np.ones(positions.size, dtype=bool)
    last = -2
    for i, pos in enumerate(positions.tolist()):
        if pos == last + 1:
            keep[i] = False  # sobrepõe a fusão anterior; pula
        else:
            last = pos
    positions = positions[keep]

    drop = np.zeros(ids.size, dtype=bool)
    drop[positions + 1] = True
    out = ids.copy()
    out[positions] = new_id
    return out[~drop]


class BPETokenizer:
    """Byte-Pair Encoding em nível de byte."""

    def __init__(self, merges: dict | None = None):
        # merges: {(a, b): new_id} na ordem em que foram criados
        self.merges: dict[tuple[int, int], int] = merges or {}
        self._build_vocab()

    def _build_vocab(self) -> None:
        # vocab: {id: bytes} — os 256 bytes base mais os tokens fundidos
        self.vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
        for (a, b), new_id in self.merges.items():
            self.vocab[new_id] = self.vocab[a] + self.vocab[b]

    @property
    def vocab_size(self) -> int:
        return 256 + len(self.merges)

    # ------------------------------------------------------------------
    @classmethod
    def train(cls, text: str, vocab_size: int, verbose: bool = False) -> "BPETokenizer":
        """Aprende as fusões a partir do texto até atingir `vocab_size`."""
        assert vocab_size >= 256, "vocab_size precisa ser >= 256 (bytes base)"
        ids = np.frombuffer(text.encode("utf-8"), dtype=np.uint8).astype(np.int64)
        merges: dict[tuple[int, int], int] = {}
        num_merges = vocab_size - 256

        for i in range(num_merges):
            pairs = _count_pairs(ids)
            if not pairs:
                break
            # par mais frequente (desempate estável pelo próprio par)
            pair = max(pairs, key=lambda p: (pairs[p], p))
            if pairs[pair] < 2:
                break  # nada mais vale a pena fundir
            new_id = 256 + i
            ids = _merge(ids, pair, new_id)
            merges[pair] = new_id
            if verbose and (i + 1) % 50 == 0:
                print(f"  merge {i + 1}/{num_merges}: {pair} -> {new_id} "
                      f"({pairs[pair]} ocorrências)")

        return cls(merges)

    # ------------------------------------------------------------------
    def encode(self, text: str) -> list[int]:
        ids = np.frombuffer(text.encode("utf-8"), dtype=np.uint8).astype(np.int64)
        # aplica as fusões na ordem de criação (menor new_id primeiro)
        while ids.size >= 2:
            pairs = _count_pairs(ids)
            # escolhe o par cuja fusão foi aprendida mais cedo (menor id)
            candidate = min(pairs, key=lambda p: self.merges.get(p, float("inf")))
            if candidate not in self.merges:
                break
            ids = _merge(ids, candidate, self.merges[candidate])
        return ids.tolist()

    def decode(self, ids) -> str:
        data = b"".join(self.vocab[int(i)] for i in ids)
        return data.decode("utf-8", errors="replace")

    # ------------------------------------------------------------------
    def save(self, path: str) -> None:
        # JSON não tem tuplas nem chaves não-string: serializa como lista
        payload = [[a, b, new_id] for (a, b), new_id in self.merges.items()]
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f)

    @classmethod
    def load(cls, path: str) -> "BPETokenizer":
        with open(path, encoding="utf-8") as f:
            payload = json.load(f)
        merges = {(a, b): new_id for a, b, new_id in payload}
        return cls(merges)
