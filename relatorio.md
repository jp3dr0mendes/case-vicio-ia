# Relatório: sistema de classificação de tópicos (case Vício, etapa 2)

**Resultado:** das 559 questões fora da folha-verdade, o sistema publica **135 (24,2%)** e manda 424 para revisão
com sugestão. Precisão prevista nas publicadas: **97%** (intervalo de 80%: 94% a 100%). A decisão roda em ~4 s,
sem rede e sem modelo de linguagem; o modelo só foi usado antes, para enriquecer os temas e gerar embeddings.

## a) O sistema, peça por peça

**1. Dados organizados (versionados em `artifacts/`).**

| Peça | O que é | Por que está ali |
|---|---|---|
| Árvore validada | 377 nós, 371 selecionáveis, sem ciclos; 2 inativos (5078, 5234) excluídos | Só tema ativo e não raiz pode ser publicado |
| Perfil de cada tema | Caminho + subtemas + descrição e 10-15 termos escritos pelo `qwen2.5:14b` (o modelo vê só o tema e os vizinhos, nunca uma questão) | A coluna `descricao` original é quase sempre "N questões no source"; sem texto, o tema não tem como ser comparado com a questão |
| Embeddings `bge-m3` | Das 659 questões e dos 371 perfis, em cache com hash do texto | Similaridade semântica multilíngue, roda num notebook |
| Dicionário de leis | 26 leis por número e por nome (ex.: 11.340, "Maria da Penha") → temas | Lei citada é o sinal mais explícito para Legislação Especial |
| Faixas de artigos | CP e CPP por capítulo → tema (ex.: CP 121-128 → Crimes contra a Vida) | Usado só quando o artigo vem junto do nome do código |
| Metadados da árvore | `clone` (nó sem descrição, ex.: "Teoria da Pena (Parte 1)", "Penal > Tráfico de Drogas") e volume de questões do tema na plataforma | A folha mostrou que a equipe quase nunca usa os clones |

**2. Granularidade da decisão: tema de nível 1.** Pela régua, publicar no pai acerta e publicar no irmão erra.
O sistema escolhe entre os **109 temas de nível 1**, e cada um agrega os sinais dos seus subtemas (máximo). Isso
transforma 371 classes com 100 exemplos num problema de 109 classes, sem perder acerto pela régua.

**3. Sinais por par (questão, tema)**: cosseno de embedding e TF-IDF (absolutos e distância para o melhor tema),
lei citada, artigo citado, similaridade do assunto de origem com os títulos, disciplina de origem compatível com
a raiz, clone, volume, kNN sobre os exemplos rotulados, convenção da equipe por assunto de origem e frequência do
tema na folha.

**4. Ranqueador**: logit condicional (softmax sobre os 109 temas, 16 pesos, regularização L2), treinado para
maximizar a probabilidade do conjunto de temas corretos (central e extras). Escolhi um modelo linear de 16 pesos
porque só há 92 exemplos dentro da taxonomia: qualquer coisa mais flexível decoraria a folha.

**5. Calibrador**: regressão logística que transforma três entradas em P(acerto): probabilidade do topo, margem
para o 2º tema e **concordância** (quantos dos sinais embedding, TF-IDF, kNN e assunto de origem, sozinhos,
apontam para o mesmo tema). A concordância é a entrada de maior peso (coeficiente 1,02). Ajustado em predições
fora da amostra.

**6. Regra de decisão**: publica se a confiança calibrada ≥ **0,94**; senão, revisa com o tema sugerido e a 2ª
opção no motivo. O limiar é o menor valor em que a precisão fora da amostra fica ≥ 97% para todo limiar acima.
Mirei 97%, não 95%, para ter margem para a amostra pequena e para a importação nova (ver c). Sem embedding em
cache para o texto atual, a questão vai para revisão. O sistema nunca declara `fora` sozinho (ver c).

**Testado e descartado** (`artifacts/experimentos.json`, validação cruzada 10x5):

| Variante | Acerto top-1 | Log-loss calibrado | Cobertura com ≥95% |
|---|---|---|---|
| **Completo (adotado)** | **0,69** | **0,447** | **0,25** |
| Só TF-IDF | 0,49 | 0,500 | 0,09 |
| Só embedding | 0,58 | 0,540 | 0,11 |
| Só assunto/disciplina de origem | 0,40 | 0,601 | 0 |
| Sem sinais de origem | 0,68 | 0,504 | 0,21 |
| Sem leis e artigos | 0,70 | 0,525 | 0,14 |
| Sem embedding | 0,64 | 0,530 | 0,06 |
| Com só 26 dos 371 temas enriquecidos (10x5, L2=3) | 0,66 | 0,525 | 0,10 |

O enriquecimento dos temas foi a peça de maior ganho (cobertura de 10% para 25%). Também descartei: escolher
subtema (risco de irmão sem ganho pela régua), calibrar só com a probabilidade do topo (precisão de 83% no topo
da confiança, porque o softmax fica confiante quando sinais fracos concordam por acaso), usar o LLM para propor
artigos (inventou faixas erradas num teste) e usar o modelo `qwen2.5:7b` no enriquecimento (termos inventados,
ex.: "concurso de motivos").

## b) Os 100 rótulos

**Como usei.** Treino do ranqueador e do calibrador, escolha do limiar e previsão, sempre em validação cruzada
repetida (10 partes x 10 repetições): cada questão é avaliada por um modelo que não a viu. O modelo final é
treinado com os 100. As 100 questões saem no `publicacao.csv` com o rótulo da equipe (não entram na avaliação).

**O que mostraram sobre o sistema.** Acerto do tema de nível 1: 70%. No limiar 0,94 ele publicaria 21,8 das 100
por rodada, com 213 acertos em 218 publicações (97,7%). Os 5 erros vêm de 2 questões: Q0043 ("Sobre a fixação
da pena", que a equipe marcou como fora) e Q0039 (cartel de GLP, também fora). Ou seja, com confiança alta o
sistema erra principalmente onde a equipe disse que **nenhum** tema serve.

**O que mostraram sobre a convenção da equipe.**
- **Clones nunca**: nenhum dos 13 clones de Penal (Parte 1/2, duplicatas de leis) aparece; Lei de Execução Penal
  aparece em Legislação Especial e em Processo Penal (2 vezes no clone 6734).
- **Nível 1 é a norma**: 53 dos 92 rótulos são temas de nível 1; publicar no nível 1 não perde nada.
- **Fronteiras por convenção**: livramento condicional e regimes vão às vezes para *Lei de Execução Penal*, às vezes
  para *Teoria da Pena*; Juizado Especial vai para Legislação Especial, não para Procedimento Sumaríssimo.
- **Fora não é "difícil"**: as 8 questões fora têm assunto plausível (fixação da pena, aplicação da lei penal).
  Pela leitura, parecem temas ausentes da árvore (ordem econômica, rito de crimes de responsabilidade, procedimento da Lei de Drogas)
  ou questões que misturam vários temas.
- **A folha só cobre o banco antigo**: todos os 100 ids estão entre Q0002 e Q0356. Nenhum exemplo da importação nova.

## c) Diagnóstico

**O que atrapalha nos dados** (auditoria em `artifacts/data_audit.json`):
1. *Temas duplicados entre disciplinas* (7 leis em Penal e em Legislação Especial; LEP em Processo Penal) e
   *"Partes"* vazias: a probabilidade se divide entre nós quase iguais, e escolher o "errado" conta como erro.
2. *Descrição dos temas vazia*: 333 de 375 descrições são só a contagem de questões. Resolvido parcialmente com o
   enriquecimento; o professor revisar essas descrições é o ganho mais barato.
3. *Assunto de origem*: 234 valores distintos, com lixeiras ("Concurso de Crimes" em 43 questões de assuntos
   variados). 19 questões sem assunto, 1 sem disciplina, 18 com disciplina fora das 4 áreas (Penal Militar,
   Constitucional, Tributário). É pista, não rótulo.
4. *Importação nova diferente*: Q0360-Q0659 não têm comando (401 sem comando no total), o assunto vem em formato de
   edital e a disciplina "Legislação Especial" some (as questões de lei especial vêm como "Direito Penal").

**Tipos de questão que o sistema ainda erra** e o que resolveria:

| Tipo | Exemplo | O que resolve |
|---|---|---|
| Fora da taxonomia com cara de tema | Q0043, Q0039 | Criar os temas ausentes ou rotular ~50 "fora" para treinar um detector |
| Fronteira por convenção | LEP x Teoria da Pena; Provas em Espécie x Teoria Geral das Provas (Q0338) | Regra escrita do professor, vira regra no dicionário |
| Questão que mistura temas | Q0252 (estelionato, idoso, regime, sursis) | Permitir extras e publicar só quando o central domina |
| Pouco texto específico | itens "julgue o item" curtos | Mais exemplos rotulados (kNN e convenção passam a pesar) |

**Teste de estresse.** Zerando os sinais de origem nas questões testadas (simulando o formato novo), a cobertura
cai de 21,8% para 16,4% e a precisão fica em 99% (`scripts/estresse_origem.py`): o sistema fica mais cauteloso,
não mais errado. Mesmo assim, 94 das 135 publicações são da importação nova, que nenhum rótulo cobre: é o maior
risco da previsão. Auditei 22 publicações sorteadas: 20 me parecem claramente no tema certo e 2 dependem de
convenção (Q0392, livramento condicional em Teoria da Pena; Q0631, advocacia administrativa fazendária em
Crimes contra a Administração Pública, que pela Lei 8.137 poderia ser Crimes Tributários). Não alterei nenhuma.

## d) Produção

**Custo e tempo para 50 mil questões por semana** (medido neste notebook, Apple Silicon, sem GPU dedicada):
- Embeddings `bge-m3` local: as 659 levaram ~2 min (≈ 5 questões/s) → 50 mil ≈ 3 h/semana numa máquina comum, ou
  minutos numa GPU. Custo de API equivalente (embeddings comerciais): poucos dólares por semana.
- Classificação: linear em 109 temas; as 659 levam < 1 s fora a validação cruzada → 50 mil em ~1 min.
- Enriquecimento dos temas: só quando a árvore muda (~1h10 para 371 temas com o `qwen2.5:14b`).
- Revisão: com a cobertura de hoje, ~76% iriam ao professor (38 mil/semana), inviável. Por isso o centro do plano é
  o ciclo abaixo, que aumenta a cobertura.

**Correção do professor vira dado.** Cada revisão gera uma linha (questão, tema escolhido, extras, fora) no mesmo
formato da folha-verdade, gravada com o `run_id` no PostgreSQL. Essas linhas entram no treino (kNN, convenção por
origem, prior) e, separadas por lote, na calibração. Prioridade da fila: questões com confiança entre 0,6 e 0,94
(as que mais rapidamente viram publicáveis) e uma amostra aleatória de publicadas (5%) para medir a precisão real.

**Retreino.** Semanal, junto com a importação: retreinar leva segundos. O limiar só muda se a validação cruzada
com os rótulos novos sustentar ≥ 97%. A amostra de publicadas auditadas é o alarme: se a precisão medida cair
abaixo de 95%, o limiar sobe automaticamente até a próxima revisão.

## e) Memo para o professor

Ver [`memo_professor.md`](memo_professor.md).

## Limitações e o que eu faria com mais tempo

- **A precisão é prevista, não medida.** O único gabarito são os 100 rótulos; ~22 publicações por rodada sustentam
  a estimativa (intervalo de Wilson de 95% por rodada: 78% a 99%).
- **A importação nova não tem rótulo** e recebe 70% das publicações. Com mais tempo: pedir 50-100 rótulos dela e
  calibrar por lote.
- **`fora` automático e `extras_ids`** não implementados por falta de exemplos; dúvida vai para revisão.
- **Descrições geradas por modelo** têm erros pontuais (ex.: "Exercício Arbitrário das Próprias Razões"); entram
  como sinal e o peso é aprendido, mas revisá-las aumentaria a cobertura.
- Reprodutibilidade: no Docker (Linux) as decisões são idênticas às do macOS; as confianças diferem na 4ª casa
  decimal por diferença de BLAS.
