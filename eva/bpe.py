"""Tokenizador BPE (Byte-Pair Encoding) construído do zero.

O tokenizador em nível de caractere é simples, mas gasta um token por
letra — o modelo "enxerga" pouquíssimo texto por janela de contexto. O BPE
resolve isso aprendendo *subpalavras*: parte dos bytes brutos e vai
fundindo, iterativamente, o par de símbolos adjacentes mais frequente. Ao
final, sequências comuns ("ção", "modelo", " de ") viram um único token.

É o mesmo algoritmo usado por GPT-2/GPT-3/GPT-4 (aqui em miniatura, em
nível de byte, sem regex de pré-tokenização).
"""

from __future__ import annotations

import json
from collections import Counter


def _get_pairs(ids: list[int]) -> Counter:
    """Conta cada par de símbolos adjacentes na sequência."""
    counts: Counter = Counter()
    for a, b in zip(ids, ids[1:]):
        counts[(a, b)] += 1
    return counts


def _merge(ids: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
    """Substitui toda ocorrência de `pair` pelo token `new_id`."""
    out: list[int] = []
    i = 0
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out


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
        ids = list(text.encode("utf-8"))
        merges: dict[tuple[int, int], int] = {}
        num_merges = vocab_size - 256

        for i in range(num_merges):
            pairs = _get_pairs(ids)
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
        ids = list(text.encode("utf-8"))
        # aplica as fusões na ordem de criação (menor new_id primeiro)
        while len(ids) >= 2:
            pairs = _get_pairs(ids)
            # escolhe o par cuja fusão foi aprendida mais cedo (menor id)
            candidate = min(pairs, key=lambda p: self.merges.get(p, float("inf")))
            if candidate not in self.merges:
                break
            ids = _merge(ids, candidate, self.merges[candidate])
        return ids

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
