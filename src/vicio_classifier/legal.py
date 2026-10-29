"""Extração determinística (regex) de leis e artigos citados, e mapeamento para temas.

Não usa modelo de linguagem: as regras vêm de artifacts/regras/*.json (curadoria manual).
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from vicio_classifier.data import normalize

_LEI = re.compile(r"\b(?:lei|decreto-lei|dl|lc|lei complementar)\s*(?:federal\s*)?(?:n[ºo°.]*\s*)?(\d{1,2}\.?\d{3})\b")
_ART = re.compile(r"\bart(?:igo)?s?\.?\s*(\d{1,3})(?:\s*[-º°o]*\s*[a-z]\b)?")
_CP = re.compile(r"\b(codigo penal|cp)\b")
_CPP = re.compile(r"\b(codigo de processo penal|cpp)\b")


@dataclass
class LegalHits:
    leis: set[str]
    temas_lei: set[int]
    temas_artigo: set[int]


class LegalRules:
    def __init__(self, rules_dir: Path):
        leis = json.loads((rules_dir / "leis.json").read_text(encoding="utf-8"))["leis"]
        self.numero_para_temas: dict[str, set[int]] = {}
        self.expressoes: list[tuple[re.Pattern, set[int], str]] = []
        for lei in leis:
            temas = set(lei["temas"])
            for n in lei["numeros"]:
                self.numero_para_temas.setdefault(n, set()).update(temas)
            for e in lei["expressoes"]:
                self.expressoes.append((re.compile(rf"\b{re.escape(e)}\b"), temas, lei["nome"]))
        arts = json.loads((rules_dir / "artigos.json").read_text(encoding="utf-8"))
        self.faixas = {code: [tuple(x) for x in arts[code]] for code in ("CP", "CPP")}

    def all_temas(self) -> set[int]:
        out = set().union(*self.numero_para_temas.values())
        for _, temas, _ in self.expressoes:
            out |= temas
        for faixas in self.faixas.values():
            out |= {t for _, _, t in faixas}
        return out

    def _artigo_temas(self, code: str, n: int) -> set[int]:
        return {t for a, b, t in self.faixas[code] if a <= n <= b}

    def extract(self, text: str) -> LegalHits:
        norm = normalize(text)
        leis: set[str] = set()
        temas_lei: set[int] = set()
        for m in _LEI.finditer(norm):
            num = m.group(1).replace(".", "")
            if num in self.numero_para_temas:
                leis.add(num)
                temas_lei |= self.numero_para_temas[num]
        for pat, temas, nome in self.expressoes:
            if pat.search(norm):
                leis.add(nome)
                temas_lei |= temas
        temas_art: set[int] = set()
        for m in _ART.finditer(norm):
            janela = norm[m.end(): m.end() + 60]
            n = int(m.group(1))
            # o código precisa ser nomeado logo depois do artigo; CPP antes de CP (CP é prefixo)
            if _CPP.search(janela):
                temas_art |= self._artigo_temas("CPP", n)
            elif _CP.search(janela):
                temas_art |= self._artigo_temas("CP", n)
        return LegalHits(leis, temas_lei, temas_art)
