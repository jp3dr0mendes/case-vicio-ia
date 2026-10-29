# Case Vício (Método VDE): classificação de tópicos, etapa 2

Sistema que decide, para cada uma das **659 questões** de Penal, Processo Penal, Legislação Penal Especial e
Criminologia, se ela é **publicada** num tema da árvore, vai para **revisão** do professor ou fica **fora** da
taxonomia, com uma probabilidade calibrada de acerto. **A decisão é do sistema e roda sem chamar modelo de
linguagem**: o modelo só entrou antes, para enriquecer a descrição dos temas e gerar embeddings, e o resultado
está versionado como dado em `artifacts/`.

| Entrega | Arquivo |
|---|---|
| Classificação das 659 questões | [`publicacao.csv`](publicacao.csv) |
| Previsão de precisão e cobertura | [`previsao.txt`](previsao.txt) |
| Relatório (até 4 páginas) | [`relatorio.md`](relatorio.md) |
| Memo para o professor (1 página) | [`memo_professor.md`](memo_professor.md) |
| Enunciado e anexos originais | [`case-ia-classificacao-de-topicos.pdf`](case-ia-classificacao-de-topicos.pdf), [`anexos-case-ia.zip`](anexos-case-ia.zip) (cópia extraída em `data/anexos/`) |

## Como rodar

Requisitos: Python 3.12+ (testado em 3.12 no Docker e 3.13 no macOS). Nenhum serviço externo é necessário.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
PYTHONPATH=src python -m vicio_classifier classify      # regenera publicacao.csv e previsao.txt
```

Leva **cerca de 5 segundos** num notebook (Apple Silicon; cerca de 15 s com o Docker): a maior parte é a
validação cruzada repetida (10 x 10) que escolhe o limiar e alimenta a previsão.

Outros comandos:

```bash
PYTHONPATH=src python -m vicio_classifier audit                        # auditoria dos anexos -> artifacts/data_audit.json
PYTHONPATH=src python -m vicio_classifier validate publicacao.csv      # confere o contrato do LEIA-ME
PYTHONPATH=src pytest -q                                               # testes (inclui classificação com a rede bloqueada)
PYTHONPATH=src python scripts/experimentos.py 3 > artifacts/experimentos.json   # ablações do relatório
PYTHONPATH=src python scripts/estresse_origem.py                       # estresse: origem indisponível
```

Com Docker:

```bash
docker compose build app
docker compose run --rm app            # roda com network_mode: none; saída em ./saida/
cp .env.example .env                   # defina POSTGRES_PASSWORD
docker compose --profile db up -d db
docker compose --profile db run --rm app-db   # mesma classificação + registro da execução no PostgreSQL
```

O serviço `app` roda **sem rede** (`network_mode: none`), o que prova que a decisão não depende de nenhum
modelo. O PostgreSQL é opcional: guarda cada execução (`runs`) e as 659 decisões com a auditoria (`decisions`);
reprocessar cria um novo `run_id` em vez de sobrescrever.

## O que cada comando gera

| Arquivo | Conteúdo |
|---|---|
| `publicacao.csv` (raiz, cópia idêntica em `artifacts/`) | Contrato do LEIA-ME: `id,decisao,tema_central_id,extras_ids,confianca,motivo` |
| `previsao.txt` (raiz, cópia idêntica em `artifacts/`) | Previsão e metodologia |
| `artifacts/run_metadata.json` | Hashes SHA-256 dos anexos e artefatos, configuração, pesos do ranqueador, calibrador, limiar, contagens, validação cruzada, tempo |
| `artifacts/decision_audit.jsonl` | Por questão: decisão, 5 candidatos com probabilidade, concordância entre sinais, origem |
| `artifacts/validacao_cruzada_oof.jsonl` | Predições fora da amostra na folha-verdade (base da calibração e da previsão) |
| `artifacts/data_audit.json` | Auditoria dos anexos |
| `artifacts/experimentos.json` | Ablações (o que foi testado e descartado) |

## Arquitetura

```
anexos ──► data.py (limpa HTML, valida) ──► taxonomy.py (árvore, regra pai/filho/irmão)
                                    │
artifacts/ (versionados)            ▼
  regras/leis.json  ───────► features.py: 12 sinais estáticos + 4 aprendidos por (questão, tema de nível 1)
  regras/artigos.json        embedding · TF-IDF · lei citada · artigo do CP/CPP · assunto e disciplina de origem
  temas_enriquecidos.jsonl   · clone · volume do tema na plataforma · kNN e convenção da folha-verdade
  embeddings/*.npz                  │
                                    ▼
                            model.py: Ranker (logit condicional, softmax sobre 109 temas)
                                    │       Calibrator (logística: p do topo, margem, concordância)
                                    ▼
                            pipeline.py: validação cruzada na folha → limiar → decisão → contract.py
```

| Módulo | Papel |
|---|---|
| `data.py` | Lê e valida `questoes.jsonl`, `temas.csv`, `folha_verdade.csv`; remove HTML preservando o texto |
| `taxonomy.py` | Árvore, temas selecionáveis (ativos, não raiz, sem ancestral inativo) e a regra de acerto |
| `legal.py` | Regex de leis e artigos citados + dicionários curados em `artifacts/regras/` |
| `profiles.py` | Texto que representa cada tema (caminho, subtemas, descrição e termos enriquecidos) |
| `features.py` | Sinais por par (questão, tema de nível 1) |
| `model.py` | Ranqueador, calibrador e entradas de confiança |
| `pipeline.py` | Validação cruzada, escolha do limiar, decisões, metadados |
| `forecast.py` | `previsao.txt` |
| `contract.py` | Validação automática do contrato do `publicacao.csv` |
| `db.py` | Registro opcional no PostgreSQL |
| `offline/` | **Única parte que usa modelo de linguagem** (Ollama local), nunca importada pelo `classify` |

### Por que o tema publicado é sempre de nível 1

Pela régua, publicar no pai do tema da equipe conta como certo e publicar num subtema irmão conta como erro.
Escolher o tema de nível 1 acerta nos dois casos em que a equipe usou o próprio tema ou um subtema dele, e não
corre o risco do irmão. A sugestão de subtema mais provável fica em `decision_audit.jsonl` para o professor.

## De onde veio cada artefato

| Artefato | Origem | Como regenerar |
|---|---|---|
| `artifacts/temas_enriquecidos.jsonl` | `qwen2.5:14b` local (Ollama), temperatura 0, prompt `prompts/enriquecer_tema_v1.txt`. O modelo vê só o caminho do tema, os vizinhos e os subtemas, **nunca uma questão**. Gera descrição e termos típicos para os 371 temas selecionáveis | `PYTHONPATH=src python -m vicio_classifier.offline.enrich_temas --model qwen2.5:14b` (≈ 1h10) |
| `artifacts/embeddings/questoes.npz`, `temas.npz` | `bge-m3` local (Ollama), vetores normalizados em float16 com o SHA-256 do texto de origem | `python -m vicio_classifier.offline.embed --what questoes` e `--what temas` (≈ 2 min) |
| `artifacts/regras/leis.json` | **Curadoria manual**: 26 leis (número e expressões que as nomeiam) mapeadas para temas de nível 1 | Editar o JSON |
| `artifacts/regras/artigos.json` | **Curadoria manual**: faixas de artigos do CP e do CPP mapeadas para temas, a partir da estrutura dos códigos | Editar o JSON |

Se o texto de uma questão mudar e o embedding em cache não corresponder mais ao hash, a questão vai para
`revisar` com o motivo, em vez de o `classify` chamar o modelo. Para questões novas, roda-se primeiro o passo
offline de embeddings (que é "gerar embeddings", permitido antes da decisão).

## Decisões tomadas à mão

1. **Dicionários de leis e artigos** (`artifacts/regras/`): escritos por mim, sem olhar a folha-verdade, a partir
   dos números das leis e da divisão dos códigos. Quando a árvore tem o mesmo tema em duas disciplinas
   (por exemplo Maria da Penha em Penal e em Legislação Especial), os dois entram e o modelo aprende qual a equipe usa.
2. **Revisão do enriquecimento**: conferi por amostragem; termos fracos ou genéricos ficam, porque o peso de cada
   sinal é aprendido na folha-verdade. Os números de artigo que o modelo propôs num teste estavam errados, por isso
   o prompt proíbe citar artigos e os artigos vêm só do dicionário manual.
3. **As 100 questões da folha-verdade** saem com o rótulo da equipe (`publicar` no tema da equipe, ou `fora`),
   confiança 1.0 e motivo "folha-verdade". O LEIA-ME diz que elas não entram na avaliação.
4. **Nenhuma questão fora da folha recebe `fora`**: com 8 exemplos não há como reconhecer "fora da taxonomia"
   com segurança, e dúvida vai para `revisar`, não para `fora`.
5. **`extras_ids` vazio** nas decisões do sistema: não há sinal confiável para afirmar que um segundo tema é
   coberto, e a lista não deve ser uma coleção de alternativas.

## Histórico

A etapa 1 do case (359 questões, LLM classificando e verificando) está em
[jp3dr0mendes/case-vicio](https://github.com/jp3dr0mendes/case-vicio). Esta etapa reconstruiu o classificador
para a regra nova (decisão sem modelo de linguagem). Do código anterior ficaram as ideias de validação de
dados e do contrato; o restante foi reescrito.

## Verificações executadas (29/10/2026, macOS Apple Silicon, Docker Desktop 28.4)

| Verificação | Resultado |
|---|---|
| `python -m compileall -q src tests scripts` | OK |
| `PYTHONPATH=src pytest -q` (local e dentro do container sem rede) | 29 testes passando |
| `python -m vicio_classifier audit` / `validate publicacao.csv` | OK: 659 questões, 377 temas (371 selecionáveis), contrato cumprido |
| `docker compose config --quiet`, `docker compose build app` | OK |
| `docker compose run --rm app` (sem rede) | Mesmas 659 decisões e temas do macOS; confiança difere ≤ 0,0012 (BLAS) |
| `docker compose --profile db run --rm app-db` | 2 execuções gravadas (`run_id` 1 e 2), 659 decisões cada, histórico preservado |
| Ollama no host: `qwen2.5:14b` (enriquecimento, 371 temas, 0 falhas) e `bge-m3` (embeddings) | OK |
| Ollama em container (`--profile offline-llm`, CPU): `ollama pull bge-m3` + serviço `embed` | 659 embeddings em 2 min 16 s; cosseno ≥ 0,9998 com o cache versionado |

Não verificado: a precisão real nas questões avaliadas (depende do gabarito da equipe).
