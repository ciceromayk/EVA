"""Extrai texto de PDFs e monta um corpus de treino limpo para a EVA.

Uso:
    python tools/build_corpus.py entrada1.pdf entrada2.pdf -o data/corpus.txt

A limpeza remove cabeçalhos/rodapés repetidos, hifenização de quebra de
linha e espaços em excesso, deixando um texto contínuo adequado para
treino em nível de caractere.
"""

from __future__ import annotations

import argparse
import re
import unicodedata
from collections import Counter

import fitz  # PyMuPDF


def extract_pages(path: str) -> list[str]:
    doc = fitz.open(path)
    pages = [page.get_text("text") for page in doc]
    doc.close()
    return pages


def strip_repeated_lines(pages: list[str]) -> list[str]:
    """Remove linhas que se repetem em muitas páginas (cabeçalho/rodapé)."""
    counts: Counter[str] = Counter()
    for page in pages:
        for line in {ln.strip() for ln in page.splitlines() if ln.strip()}:
            counts[line] += 1
    threshold = max(3, len(pages) // 3)
    boilerplate = {ln for ln, c in counts.items() if c >= threshold and len(ln) < 80}

    cleaned = []
    for page in pages:
        kept = [ln for ln in page.splitlines() if ln.strip() not in boilerplate]
        cleaned.append("\n".join(kept))
    return cleaned


def clean_text(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    # espaços não-quebráveis e afins viram espaço comum
    text = text.replace("\xa0", " ").replace("​", "")
    # líderes pontilhados de sumário ("....") e marcadores de citação [12]
    text = re.sub(r"\.{4,}", " ", text)
    text = re.sub(r"\[\d+\]", "", text)
    # junta palavras hifenizadas quebradas no fim da linha: "progra-\nmação"
    text = re.sub(r"-\n(?=\w)", "", text)
    # quebra de linha isolada dentro de parágrafo vira espaço
    text = re.sub(r"(?<![\n.:;!?])\n(?![\n])", " ", text)
    # remove números de página soltos
    text = re.sub(r"\n\s*\d+\s*\n", "\n", text)
    # normaliza espaços e linhas em branco excessivas
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    # descarta caracteres de controle raros, preservando acentos e pontuação
    text = "".join(ch for ch in text if ch == "\n" or unicodedata.category(ch)[0] != "C")
    return text.strip()


def extract_file(path: str) -> str:
    """Extrai e limpa o texto de um arquivo (.pdf ou .txt/.md)."""
    if path.lower().endswith(".pdf"):
        pages = strip_repeated_lines(extract_pages(path))
        return clean_text("\n".join(pages))
    with open(path, encoding="utf-8", errors="replace") as f:
        return clean_text(f.read())


def _dedup_key(paragraph: str) -> str:
    """Normaliza um parágrafo para comparação (ignora maiúsculas/espaços)."""
    return re.sub(r"\s+", " ", paragraph).strip().lower()


def dedup_paragraphs(text: str, min_len: int = 40) -> tuple[str, int]:
    """Remove parágrafos DUPLICADOS entre as fontes, mantendo a 1ª ocorrência.

    Comum quando o mesmo prefácio/licença/capítulo aparece em mais de um
    arquivo (ex.: dois PDFs do mesmo livro, ou um artigo citado inteiro em
    outro). Só compara parágrafos com `min_len`+ caracteres — parágrafos
    curtos (títulos, diálogos de uma linha) repetem naturalmente e não são
    "boilerplate", então ficam de fora da checagem.
    """
    seen: set[str] = set()
    kept: list[str] = []
    dropped = 0
    for para in text.split("\n\n"):
        key = _dedup_key(para)
        if len(key) >= min_len:
            if key in seen:
                dropped += 1
                continue
            seen.add(key)
        kept.append(para)
    return "\n\n".join(kept), dropped


def build(paths: list[str]) -> str:
    parts = [extract_file(path) for path in paths]
    joined = "\n\n".join(p for p in parts if p) + "\n"
    deduped, _ = dedup_paragraphs(joined)
    return deduped


def main():
    parser = argparse.ArgumentParser(description="Monta corpus a partir de PDFs/TXT")
    parser.add_argument("files", nargs="+", help="arquivos de entrada (.pdf ou .txt)")
    parser.add_argument("-o", "--output", default="data/corpus.txt")
    args = parser.parse_args()

    parts = [extract_file(path) for path in args.files]
    joined = "\n\n".join(p for p in parts if p) + "\n"
    corpus, dropped = dedup_paragraphs(joined)

    with open(args.output, "w", encoding="utf-8") as f:
        f.write(corpus)

    vocab = sorted(set(corpus))
    print(f"Corpus escrito em {args.output}")
    print(f"  {len(corpus):,} caracteres")
    print(f"  {len(corpus.split()):,} palavras (aprox.)")
    print(f"  {len(vocab)} caracteres distintos (vocabulário)")
    if dropped:
        print(f"  {dropped} parágrafo(s) duplicado(s) removido(s) entre as fontes")


if __name__ == "__main__":
    main()
