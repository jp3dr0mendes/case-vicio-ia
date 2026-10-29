from pathlib import Path

import pytest

from vicio_classifier.data import Tema
from vicio_classifier.taxonomy import Taxonomy

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "anexos"
ARTIFACTS = ROOT / "artifacts"


@pytest.fixture
def small_tax() -> Taxonomy:
    def t(i, p, nivel, titulo, status="active", descricao="x"):
        return Tema(i, p, "Penal", nivel, titulo, titulo, descricao, status)
    return Taxonomy([
        t(1, None, 0, "Penal"),
        t(10, 1, 1, "Teoria da Pena"),
        t(11, 10, 2, "Dosimetria"),
        t(12, 10, 2, "Regimes"),
        t(20, 1, 1, "Prescrição"),
        t(30, 1, 1, "Inativo", status="inactive"),
        t(31, 30, 2, "Filho de inativo"),
    ])
