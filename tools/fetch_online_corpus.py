"""Busca texto de fontes online públicas para alimentar o corpus da EVA.

Duas fontes, sem dependências além da biblioteca padrão do Python:

- **Wikipédia** (qualquer idioma): baixa o texto puro de artigos por título,
  via a API oficial do MediaWiki (`action=query&prop=extracts`).
- **Project Gutenberg**: busca e baixa livros de domínio público em texto
  puro, via a API pública Gutendex (https://gutendex.com), filtrando por
  idioma.

Os arquivos baixados vão para `materials/` como `.txt`, no mesmo formato
que `tools/build_corpus.py` já sabe limpar e juntar ao corpus — não é
preciso nenhuma integração especial, só rodar a reconstrução depois.

Uso:
    python tools/fetch_online_corpus.py --wikipedia "Inteligência artificial,Rede neural"
    python tools/fetch_online_corpus.py --gutenberg "Machado de Assis" --max-books 3
    python tools/build_corpus.py materials/*.pdf materials/*.txt -o data/corpus.txt
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request

USER_AGENT = "EVA-training-bot/1.0 (projeto educacional; github.com/ciceromayk/EVA)"


def _get_json(url: str, timeout: int = 30) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _get_text(url: str, timeout: int = 60) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read()
    for enc in ("utf-8", "latin-1"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def safe_filename(name: str) -> str:
    name = re.sub(r"[^\w\s-]", "", name, flags=re.UNICODE).strip()
    name = re.sub(r"\s+", "_", name)
    return name[:80] or "arquivo"


def fetch_wikipedia(titles: list[str], lang: str = "pt",
                     out_dir: str = "materials") -> list[str]:
    """Baixa o texto puro de artigos da Wikipédia, um arquivo por título."""
    os.makedirs(out_dir, exist_ok=True)
    saved = []
    base = f"https://{lang}.wikipedia.org/w/api.php"
    for title in titles:
        title = title.strip()
        if not title:
            continue
        params = {
            "action": "query", "prop": "extracts", "explaintext": "1",
            "redirects": "1", "format": "json", "titles": title,
        }
        url = base + "?" + urllib.parse.urlencode(params)
        try:
            data = _get_json(url)
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            print(f"  [wikipedia] falhou '{title}': {exc}")
            continue

        for page in data.get("query", {}).get("pages", {}).values():
            if "missing" in page:
                print(f"  [wikipedia] artigo não encontrado: '{title}'")
                continue
            extract = (page.get("extract") or "").strip()
            real_title = page.get("title", title)
            if not extract:
                print(f"  [wikipedia] artigo vazio: '{real_title}'")
                continue
            fname = os.path.join(out_dir, f"wiki_{safe_filename(real_title)}.txt")
            with open(fname, "w", encoding="utf-8") as f:
                f.write(f"{real_title}\n\n{extract}\n")
            saved.append(fname)
            print(f"  [wikipedia] salvo: {real_title} ({len(extract):,} chars)")
        time.sleep(0.3)  # gentil com a API pública
    return saved


def fetch_gutenberg(query: str, lang: str = "pt", max_books: int = 5,
                     out_dir: str = "materials") -> list[str]:
    """Busca e baixa livros de domínio público via a API pública Gutendex."""
    os.makedirs(out_dir, exist_ok=True)
    saved = []
    url = "https://gutendex.com/books?" + urllib.parse.urlencode(
        {"search": query, "languages": lang})
    try:
        data = _get_json(url)
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        print(f"  [gutenberg] busca falhou: {exc}")
        return saved

    for book in data.get("results", [])[:max_books]:
        title = book.get("title", "livro")
        text_url = next((link for key, link in book.get("formats", {}).items()
                         if key.startswith("text/plain")), None)
        if not text_url:
            print(f"  [gutenberg] sem versão em texto puro: '{title}'")
            continue
        try:
            text = _get_text(text_url)
        except (urllib.error.URLError, TimeoutError) as exc:
            print(f"  [gutenberg] download falhou '{title}': {exc}")
            continue
        fname = os.path.join(out_dir, f"gutenberg_{safe_filename(title)}.txt")
        with open(fname, "w", encoding="utf-8") as f:
            f.write(text)
        saved.append(fname)
        print(f"  [gutenberg] salvo: {title} ({len(text):,} chars)")
        time.sleep(0.3)
    return saved


def main():
    parser = argparse.ArgumentParser(
        description="Busca texto online (Wikipédia / Project Gutenberg) para o corpus da EVA")
    parser.add_argument("--wikipedia",
                        help="títulos separados por vírgula, ex: 'Inteligência artificial,Redes neurais'")
    parser.add_argument("--gutenberg",
                        help="termo de busca no Project Gutenberg, ex: 'Machado de Assis'")
    parser.add_argument("--lang", default="pt", help="idioma (padrão: pt)")
    parser.add_argument("--max-books", type=int, default=5)
    parser.add_argument("-o", "--output", default="materials")
    args = parser.parse_args()

    saved = []
    if args.wikipedia:
        titles = [t.strip() for t in args.wikipedia.split(",")]
        saved += fetch_wikipedia(titles, lang=args.lang, out_dir=args.output)
    if args.gutenberg:
        saved += fetch_gutenberg(args.gutenberg, lang=args.lang,
                                 max_books=args.max_books, out_dir=args.output)

    if not saved:
        print("Nada foi baixado. Use --wikipedia e/ou --gutenberg.")
    else:
        print(f"\n{len(saved)} arquivo(s) salvo(s) em {args.output}/")
        print("Rode 'python tools/build_corpus.py' (ou use o painel web) para "
              "reconstruir o corpus com o novo material.")


if __name__ == "__main__":
    main()
