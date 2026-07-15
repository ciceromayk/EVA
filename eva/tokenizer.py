"""Tokenizador em nível de caractere.

O tokenizador mais simples possível: cada caractere único do corpus vira
um inteiro. É suficiente para treinar e ver a EVA aprender a estrutura da
língua sem depender de bibliotecas externas de subword (BPE etc.).
"""

from __future__ import annotations

import json


class CharTokenizer:
    def __init__(self, chars: list[str]):
        self.chars = list(chars)
        self.stoi = {c: i for i, c in enumerate(self.chars)}
        self.itos = {i: c for i, c in enumerate(self.chars)}

    @classmethod
    def from_text(cls, text: str) -> "CharTokenizer":
        return cls(sorted(set(text)))

    @property
    def vocab_size(self) -> int:
        return len(self.chars)

    def encode(self, text: str) -> list[int]:
        return [self.stoi[c] for c in text if c in self.stoi]

    def decode(self, ids) -> str:
        return "".join(self.itos[int(i)] for i in ids)

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.chars, f, ensure_ascii=False)

    @classmethod
    def load(cls, path: str) -> "CharTokenizer":
        with open(path, encoding="utf-8") as f:
            return cls(json.load(f))
