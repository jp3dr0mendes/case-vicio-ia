import pytest

from vicio_classifier.data import DataError, Tema
from vicio_classifier.taxonomy import Taxonomy


def test_same_branch_aceita_pai_e_filho_e_recusa_irmao(small_tax):
    assert small_tax.same_branch(10, 11) and small_tax.same_branch(11, 10)
    assert small_tax.same_branch(11, 11)
    assert not small_tax.same_branch(11, 12)  # irmãos
    assert not small_tax.same_branch(11, 20)  # primos


def test_is_correct_usa_extras_e_fora_nunca_acerta(small_tax):
    assert small_tax.is_correct(10, 11, [])        # publicar no pai do gabarito
    assert small_tax.is_correct(20, 11, [20])      # publicar num extra
    assert not small_tax.is_correct(12, 11, [])    # irmão
    assert not small_tax.is_correct(10, None, [])  # questão fora da taxonomia


def test_selecionaveis_excluem_raiz_inativo_e_descendente_de_inativo(small_tax):
    assert small_tax.selectable_ids == {10, 11, 12, 20}
    assert small_tax.level1_selectable() == [10, 20]
    assert small_tax.level1_of(12) == 10


def test_arvore_invalida_e_rejeitada():
    with pytest.raises(DataError):
        Taxonomy([Tema(1, 99, "P", 1, "a", "a", "", "active")])
    with pytest.raises(DataError):
        Taxonomy([Tema(1, None, "P", 0, "r", "r", "", "active"), Tema(2, 1, "P", 2, "x", "x", "", "active")])
