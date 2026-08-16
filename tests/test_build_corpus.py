"""Testa a curadoria do corpus: deduplicação de parágrafos entre fontes."""

from tools.build_corpus import clean_text, dedup_paragraphs


def test_remove_paragrafo_duplicado_entre_fontes():
    a = "Este é um parágrafo bem específico que aparece duas vezes no corpus."
    b = "Este segundo parágrafo é diferente e deve ser mantido sempre."
    text = f"{a}\n\n{b}\n\n{a}"
    deduped, dropped = dedup_paragraphs(text)
    assert dropped == 1
    assert deduped.count(a) == 1
    assert b in deduped


def test_ignora_espacos_e_maiusculas_na_comparacao():
    a = "Texto Repetido   com espaçamento  diferente e maiúsculas variadas aqui."
    b = "texto repetido com espaçamento diferente e maiúsculas variadas aqui."
    deduped, dropped = dedup_paragraphs(f"{a}\n\n{b}")
    assert dropped == 1
    assert deduped == a


def test_paragrafos_curtos_nao_sao_deduplicados():
    # títulos/diálogos curtos repetem naturalmente (ex.: "Capítulo 1", "— Sim.")
    text = "Sim.\n\nSim.\n\nSim."
    deduped, dropped = dedup_paragraphs(text)
    assert dropped == 0
    assert deduped == text


def test_preserva_ordem_e_mantem_primeira_ocorrencia():
    a = "Primeiro parágrafo, longo o suficiente para entrar na checagem de duplicata."
    b = "Segundo parágrafo, também longo o bastante para a mesma checagem aqui."
    deduped, _ = dedup_paragraphs(f"{a}\n\n{b}\n\n{a}\n\n{b}")
    assert deduped == f"{a}\n\n{b}"


def test_clean_text_ainda_funciona_com_dedup_no_pipeline():
    para = "Isto é um exemplo de programa-\nção quebrada no fim da linha do PDF."
    text = f"{para}\n\n{para}"
    cleaned = clean_text(text)
    deduped, dropped = dedup_paragraphs(cleaned)
    assert dropped == 1
    assert "programação" in deduped


if __name__ == "__main__":
    test_remove_paragrafo_duplicado_entre_fontes()
    test_ignora_espacos_e_maiusculas_na_comparacao()
    test_paragrafos_curtos_nao_sao_deduplicados()
    test_preserva_ordem_e_mantem_primeira_ocorrencia()
    test_clean_text_ainda_funciona_com_dedup_no_pipeline()
    print("Todos os testes de curadoria do corpus passaram.")
