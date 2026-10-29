"""Árvore de temas e a regra de acerto do case.

Regra (LEIA-ME): publicar no tema central, num extra, ou num nó do mesmo ramo de um deles
(pai ou filho) conta como certo; outro subtema do mesmo tema (irmão) conta como erro.
"""
from __future__ import annotations

from collections import defaultdict

from vicio_classifier.data import DataError, Tema


class Taxonomy:
    def __init__(self, temas: list[Tema]):
        self.nodes: dict[int, Tema] = {}
        for t in temas:
            if t.tema_id in self.nodes:
                raise DataError(f"tema_id duplicado: {t.tema_id}")
            self.nodes[t.tema_id] = t
        self.children: dict[int | None, list[int]] = defaultdict(list)
        for t in temas:
            if t.parent_id is not None and t.parent_id not in self.nodes:
                raise DataError(f"tema {t.tema_id} aponta para pai inexistente {t.parent_id}")
            self.children[t.parent_id].append(t.tema_id)
        for tid in self.nodes:  # sem ciclos
            seen = set()
            cur: int | None = tid
            while cur is not None:
                if cur in seen:
                    raise DataError(f"ciclo na árvore envolvendo {tid}")
                seen.add(cur)
                cur = self.nodes[cur].parent_id
        for t in temas:
            parent = self.nodes.get(t.parent_id) if t.parent_id else None
            expected = 0 if parent is None else parent.nivel + 1
            if t.nivel != expected:
                raise DataError(f"tema {t.tema_id}: nivel {t.nivel} incoerente com a árvore")
        self.roots = [tid for tid in self.children[None]]
        # Selecionável: ativo, não é disciplina (nível 0) e sem ancestral inativo.
        self.selectable_ids = {
            tid for tid, t in self.nodes.items()
            if t.status == "active" and t.nivel >= 1
            and all(self.nodes[a].status == "active" for a in self.ancestors(tid))
        }

    def ancestors(self, tid: int) -> list[int]:
        out = []
        cur = self.nodes[tid].parent_id
        while cur is not None:
            out.append(cur)
            cur = self.nodes[cur].parent_id
        return out

    def descendants(self, tid: int) -> list[int]:
        out, stack = [], list(self.children.get(tid, []))
        while stack:
            c = stack.pop()
            out.append(c)
            stack.extend(self.children.get(c, []))
        return out

    def root_of(self, tid: int) -> int:
        anc = self.ancestors(tid)
        return anc[-1] if anc else tid

    def level1_of(self, tid: int) -> int | None:
        """Tema de nível 1 que contém tid (ele mesmo se já for nível 1)."""
        t = self.nodes[tid]
        if t.nivel == 0:
            return None
        while self.nodes[tid].nivel > 1:
            tid = self.nodes[tid].parent_id  # type: ignore[assignment]
        return tid

    def same_branch(self, a: int, b: int) -> bool:
        """a e b iguais, ou um é ancestral do outro. Irmãos/primos não contam."""
        return a == b or a in self.ancestors(b) or b in self.ancestors(a)

    def is_correct(self, predicted: int, central: int | None, extras: list[int]) -> bool:
        if central is None:  # questão fora da taxonomia: qualquer publicação é erro
            return False
        return any(self.same_branch(predicted, g) for g in [central, *extras])

    def level1_selectable(self) -> list[int]:
        return sorted(t for t in self.selectable_ids if self.nodes[t].nivel == 1)
