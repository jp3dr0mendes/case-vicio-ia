from vicio_classifier.audit import audit
from vicio_classifier.data import clean_html, load_folha, load_questions
from vicio_classifier.legal import LegalRules

from conftest import ARTIFACTS, DATA


def test_clean_html_preserva_texto():
    assert clean_html("Art. 1º<br><br>I - <b>dolo</b> &amp; culpa<p>fim</p>") == "Art. 1º I - dolo & culpa fim"


def test_auditoria_confere_numeros_dos_anexos():
    a = audit(DATA)
    assert a["questoes"]["total"] == a["questoes"]["ids_unicos"] == 659
    assert a["temas"]["total"] == 377 and a["temas"]["selecionaveis"] == 371
    assert {t["tema_id"] for t in a["temas"]["inativos"]} == {5078, 5234}
    assert a["folha_verdade"]["total"] == 100 and a["folha_verdade"]["fora_da_taxonomia"] == 8
    assert a["folha_verdade"]["ids_ou_temas_invalidos"] == []
    assert a["questoes"]["html_residual_apos_limpeza"] == 0


def test_folha_e_questoes_coerentes():
    ids = {q.id for q in load_questions(DATA / "questoes.jsonl")}
    assert {g.id for g in load_folha(DATA / "folha_verdade.csv")} <= ids


def test_extracao_de_leis_e_artigos():
    r = LegalRules(ARTIFACTS / "regras")
    h = r.extract("Nos termos da Lei nº 11.340/2006, ...")
    assert {4704, 6724} <= h.temas_lei
    assert 5149 in r.extract("conforme o art. 121 do Código Penal").temas_artigo
    h = r.extract("nos termos do art. 312 do CPP, a preventiva")
    assert 5517 in h.temas_artigo and 5087 not in h.temas_artigo  # CPP, não CP
    assert not r.extract("o art. 5º da Constituição").temas_artigo
