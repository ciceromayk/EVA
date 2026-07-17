"""Testa fetch_online_corpus.py com respostas HTTP simuladas (sem rede real).

O sandbox de CI não tem acesso irrestrito à internet, então validamos a
lógica de parsing/erro com mocks — o comportamento das APIs (Wikipédia,
Gutendex) é bem documentado e estável, então isso dá confiança real sem
depender de rede externa disponível no momento do teste.
"""

import io
import json
import os
import tempfile
from unittest.mock import patch

from tools import fetch_online_corpus as f


class FakeResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *a):
        pass


def fake_urlopen_factory(mapping):
    """mapping: {url_substring: bytes_ou_callable}"""
    def _urlopen(req, timeout=30):
        url = req.full_url if hasattr(req, "full_url") else req
        for key, value in mapping.items():
            if key in url:
                data = value() if callable(value) else value
                return FakeResponse(data)
        raise AssertionError(f"URL inesperada no mock: {url}")
    return _urlopen


def test_fetch_wikipedia_salva_artigo():
    payload = json.dumps({
        "query": {"pages": {"123": {"title": "Inteligência artificial",
                                     "extract": "A IA é um campo da computação. " * 5}}}}
    ).encode("utf-8")
    with tempfile.TemporaryDirectory() as d:
        with patch("urllib.request.urlopen", fake_urlopen_factory({"wikipedia.org": payload})):
            saved = f.fetch_wikipedia(["Inteligência artificial"], out_dir=d)
        assert len(saved) == 1
        assert os.path.exists(saved[0])
        content = open(saved[0], encoding="utf-8").read()
        assert "Inteligência artificial" in content
        assert "campo da computação" in content


def test_fetch_wikipedia_artigo_inexistente_nao_quebra():
    payload = json.dumps({"query": {"pages": {"-1": {"title": "Xyzabc123", "missing": ""}}}}).encode()
    with tempfile.TemporaryDirectory() as d:
        with patch("urllib.request.urlopen", fake_urlopen_factory({"wikipedia.org": payload})):
            saved = f.fetch_wikipedia(["Xyzabc123"], out_dir=d)
        assert saved == []


def test_fetch_wikipedia_erro_de_rede_nao_quebra():
    import urllib.error

    def raise_error(req, timeout=30):
        raise urllib.error.URLError("sem conexão")

    with tempfile.TemporaryDirectory() as d:
        with patch("urllib.request.urlopen", raise_error):
            saved = f.fetch_wikipedia(["Qualquer coisa"], out_dir=d)
        assert saved == []


def test_fetch_gutenberg_salva_livro():
    search_payload = json.dumps({"results": [
        {"title": "Dom Casmurro",
         "formats": {"text/plain; charset=utf-8": "https://gutendex.example/dc.txt",
                     "text/html": "https://gutendex.example/dc.html"}},
    ]}).encode()
    book_text = "Dom Casmurro\n\nCapítulo primeiro...".encode("utf-8")

    with tempfile.TemporaryDirectory() as d:
        mapping = {"gutendex.com": search_payload, "gutendex.example/dc.txt": book_text}
        with patch("urllib.request.urlopen", fake_urlopen_factory(mapping)):
            saved = f.fetch_gutenberg("Machado de Assis", out_dir=d)
        assert len(saved) == 1
        content = open(saved[0], encoding="utf-8").read()
        assert "Dom Casmurro" in content


def test_fetch_gutenberg_sem_texto_puro_eh_ignorado():
    search_payload = json.dumps({"results": [
        {"title": "Somente HTML", "formats": {"text/html": "https://x/y.html"}},
    ]}).encode()
    with tempfile.TemporaryDirectory() as d:
        with patch("urllib.request.urlopen", fake_urlopen_factory({"gutendex.com": search_payload})):
            saved = f.fetch_gutenberg("teste", out_dir=d)
        assert saved == []


def test_safe_filename_remove_caracteres_invalidos():
    assert f.safe_filename('a/b:c*d"e') == "abcde" or "_" in f.safe_filename('a/b:c*d"e')
    assert f.safe_filename("Dom Casmurro") == "Dom_Casmurro"


if __name__ == "__main__":
    test_fetch_wikipedia_salva_artigo()
    test_fetch_wikipedia_artigo_inexistente_nao_quebra()
    test_fetch_wikipedia_erro_de_rede_nao_quebra()
    test_fetch_gutenberg_salva_livro()
    test_fetch_gutenberg_sem_texto_puro_eh_ignorado()
    test_safe_filename_remove_caracteres_invalidos()
    print("Todos os testes de fetch_online_corpus passaram.")
