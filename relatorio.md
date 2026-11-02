# Relatório: sistema de classificação de tópicos (case Vício, etapa 2)

**Resultado:** das 559 questões fora da folha-verdade, o sistema publica **147 (26,3%)** e manda 412 para revisão,
cada uma com uma sugestão de tema. A precisão prevista nas publicadas é de **96%** (intervalo de 80%: 90% a 100%).
O comando que gera a entrega leva cerca de 5 segundos e roda sem rede e sem modelo de linguagem. O modelo só
entrou antes da decisão, para escrever a descrição dos temas e gerar embeddings.

## a) O sistema, peça por peça

**1. Dados organizados (versionados em `artifacts/`).**

| Peça | O que é | Por que está ali |
|---|---|---|
| Árvore validada | 377 nós, 371 selecionáveis, sem ciclos; 2 inativos (5078, 5234) excluídos | Só tema ativo e que não seja disciplina pode ser publicado |
| Perfil de cada tema | Caminho, subtemas, descrição e 10 a 17 termos típicos escritos pelo `qwen2.5:14b`, que vê só o tema e os vizinhos, nunca uma questão | Em `temas.csv`, 351 das 371 descrições ativas são só a contagem de questões ou estão vazias |
| Embeddings `bge-m3` | Das 659 questões e dos 371 perfis, em cache com o hash do texto | Similaridade semântica; roda num notebook |
| Dicionário de leis | 26 leis por número e por nome (ex.: 11.340, "Maria da Penha") apontando para temas | Lei citada é o sinal mais explícito de Legislação Especial |
| Faixas de artigos | Capítulos do CP e do CPP apontando para temas (ex.: CP 121-128 → Crimes contra a Vida) | Usado só quando o artigo vem junto do nome do código |
| Metadados da árvore | `clone` (nó sem descrição, como "Teoria da Pena (Parte 1)") e volume do tema na plataforma | A equipe quase nunca usa os clones |

**2. A decisão é sempre no tema de nível 1.** Pela régua, publicar no pai acerta e publicar num irmão erra. O
sistema escolhe entre os **109 temas de nível 1**, e cada um soma os sinais dos seus subtemas. Isso troca 371 classes
com 100 exemplos por 109 classes, sem perder nenhum acerto pela régua.

**3. Sinais por par (questão, tema):** similaridade de embedding e de TF-IDF (o valor e a distância para o melhor
tema), lei citada, artigo citado, semelhança do assunto de origem com os títulos, disciplina de origem compatível,
clone, volume, vizinhos rotulados mais próximos (kNN), convenção da equipe por assunto de origem e frequência do
tema na folha.

**4. Ranqueador:** um modelo linear (logit condicional, softmax sobre os 109 temas, 16 pesos e regularização L2),
treinado para dar probabilidade alta ao tema certo (central ou extra). Só há 92 exemplos dentro da taxonomia; um
modelo mais flexível decoraria a folha.

**5. Calibrador:** regressão logística que transforma três números em chance de acerto: a probabilidade do topo, a
margem para o 2º tema e a **concordância** (quantos dos sinais embedding, TF-IDF, kNN e assunto de origem apontam
sozinhos para o mesmo tema). Uma primeira versão usava o logit da probabilidade e extrapolava quando o softmax ficava
muito confiante; a avaliação aninhada mostrou precisão de 91%, e a escala de 0 a 1 levou a 96%.

**6. Regra de decisão:** publica se a confiança calibrada for pelo menos **0,945**; senão, vai para revisão com a
sugestão e a 2ª opção no motivo. O limiar é o mais baixo em que, dali para cima, a precisão na validação cruzada nunca
cai abaixo de 97%. Mirei 97% para ter folga sobre os 95%. Na prática, só passa quem tem pelo menos 3 dos 4 sinais
concordando: com 2, a confiança máxima possível é 0,92. Se o embedding em cache não corresponder ao texto atual, a
questão vai para revisão. O sistema nunca marca `fora` sozinho (ver c).

**Testado e descartado** (`artifacts/experimentos.json`; mesma validação cruzada da produção, 10 x 10):

| Variante | Acerto do tema | Log-loss calibrado | Precisão aninhada | Cobertura aninhada |
|---|---|---|---|---|
| **Completo (adotado, L2 = 1)** | **0,70** | **0,430** | **0,96** | **0,25** |
| Completo, L2 = 3 | 0,72 | 0,450 | 0,96 | 0,27 |
| Sem o enriquecimento dos temas | 0,65 | 0,520 | 0,86 | 0,08 |
| Sem embedding | 0,64 | 0,522 | 0,94 | 0,17 |
| Sem leis e artigos | 0,69 | 0,527 | 0,98 | 0,12 |
| Sem sinais de origem | 0,68 | 0,509 | 0,94 | 0,15 |
| Sem kNN e frequência da folha | 0,72 | 0,461 | 0,95 | 0,21 |
| Sem a convenção por assunto de origem | 0,71 | 0,424 | 0,96 | 0,24 |
| Sem clone e volume | 0,68 | 0,438 | 0,96 | 0,24 |
| Só TF-IDF / só embedding / só origem | 0,49 / 0,58 / 0,40 | 0,50 / 0,56 / 0,62 | publica quase nada | ≤ 0,02 |

O enriquecimento dos temas e o embedding são as peças de maior ganho: sem enriquecimento, a cobertura cai de 25% para
8% e a precisão para 86%. O critério de escolha foi o log-loss (qualidade da probabilidade), não a cobertura, que com
100 rótulos oscila muito; diferenças de 0,01 ou menos tratei como empate e mantive o modelo completo. Também
descartei: escolher subtema (risco de irmão sem ganho pela régua), usar o LLM para propor artigos (inventou faixas
erradas) e enriquecer com o `qwen2.5:7b` (inventou termos como "concurso de motivos").

## b) Os 100 rótulos

**Como usei.** Para treinar o ranqueador e o calibrador, escolher o limiar e fazer a previsão, sempre fora da amostra:
validação cruzada repetida (cada questão é avaliada por um modelo que não a viu) e, por cima, uma **avaliação aninhada**
em que calibrador e limiar também são escolhidos sem as questões testadas. O modelo final é treinado com os 100. As
100 questões saem no `publicacao.csv` com o rótulo da equipe, já que não entram na avaliação.

**O que mostraram sobre o sistema.** Ele acerta o tema de nível 1 em 70% das questões. Na avaliação aninhada, publica
25 de cada 100 com precisão de 96%, e o único erro que se repete é Q0338 (Provas em Espécie no lugar de Teoria Geral
das Provas). Medida sem o aninhamento, a precisão daria 97%; a diferença é o otimismo de escolher o limiar olhando os
próprios erros. As duas questões "fora" que pareciam ter tema (Q0043, "Sobre a fixação da pena", e Q0039, cartel de GLP)
ficam em 0,92, abaixo do limiar, porque só 2 sinais concordam nelas.

**O que mostraram sobre a convenção da equipe.**

- **Clones quase nunca:** nenhum dos 13 clones de Penal aparece. A exceção é a Lei de Execução Penal, usada 2 vezes na
  cópia de Processo Penal (6734) e nas demais na de Legislação Especial.
- **Nível 1 é a norma:** 53 dos 92 rótulos são temas de nível 1, então publicar no nível 1 não perde nada.
- **Fronteiras de convenção:** livramento condicional e regimes vão ora para Lei de Execução Penal, ora para Teoria da
  Pena; Juizado Especial vai para Legislação Especial, não para Procedimento Sumaríssimo.
- **"Fora" não quer dizer difícil:** as 8 questões fora têm assunto plausível. Pela leitura, ou tratam de temas que não
  existem na árvore (ordem econômica, rito dos crimes de responsabilidade, procedimento da Lei de Drogas), ou misturam
  vários temas.
- **A folha só cobre o banco antigo:** os 100 ids vão de Q0002 a Q0356. Não há exemplo da importação nova.

## c) Diagnóstico

**O que atrapalha nos dados** (`artifacts/data_audit.json`):

1. *Temas duplicados entre disciplinas* (7 leis em Penal e em Legislação Especial; LEP também em Processo Penal) e
   *"Partes" vazias*. A probabilidade se divide entre nós quase iguais, e escolher a cópia "errada" conta como erro.
2. *Temas sem descrição* (331 só com a contagem, 20 vazias). O enriquecimento resolveu em parte; uma revisão do
   professor nessas descrições é o ganho mais barato que existe.
3. *Assunto de origem ruidoso*: 234 valores distintos, com lixeiras ("Concurso de Crimes" aparece em 43 questões de
   assuntos variados). 19 questões não têm assunto, 1 não tem disciplina e 17 vêm de disciplinas fora das 4 áreas
   (Penal Militar, Constitucional, Tributário e outras). Uso como pista, nunca como rótulo.
4. *Importação nova diferente*: as 300 questões Q0360-Q0659 vêm sem comando, com o assunto no formato de edital e sem a
   disciplina "Legislação Especial" (as de lei especial vêm como "Direito Penal").

**O que o sistema ainda erra** e o que resolveria:

| Tipo | Exemplo | O que resolve |
|---|---|---|
| Fronteira por convenção | Q0338 (Provas em Espécie x Teoria Geral das Provas); LEP x Teoria da Pena | Uma regra escrita do professor, que vira regra no dicionário |
| Fora da taxonomia com cara de tema | Q0043, Q0039 (barradas hoje por pouco) | Criar os temas que faltam ou rotular uns 50 casos "fora" para treinar um detector |
| Questão que mistura temas | Q0252 (estelionato, idoso, regime, sursis) | Aceitar extras e publicar só quando o tema central domina |
| Pouco texto específico | Itens curtos de "julgue o item" | Mais exemplos rotulados (o kNN e a convenção passam a pesar) |

**Teste de estresse.** Zerando os sinais de origem nas questões testadas, para imitar a importação nova, a cobertura
cai de 25% para 16% e a precisão de 97% para 95% (`scripts/estresse_origem.py`). Como 103 das 147 publicações vêm desse
lote, é o maior risco da previsão. Li também 22 publicações sorteadas: 20 me pareceram certas e 2 dependem de
convenção (Q0392, livramento condicional em Teoria da Pena; Q0631, advocacia administrativa fazendária, que pela Lei
8.137 poderia ir para Crimes Tributários). Não alterei nenhuma decisão à mão.

## d) Produção

**Custo e tempo para 50 mil questões por semana.**

- *Embeddings*: o `bge-m3` em CPU, no Docker, fez as 659 em 2 min 16 s (cerca de 5 por segundo). Para 50 mil, são
  umas 3 horas de CPU por semana numa máquina comum, ou minutos com GPU.
- *Classificação*: modelo linear sobre 109 temas; quase todo o tempo do comando atual é validação cruzada. Para 50 mil
  questões, estimo poucos minutos.
- *Enriquecimento dos temas*: só quando a árvore muda (cerca de 1h10 para 371 temas no `qwen2.5:14b` local).
- *O custo real é o professor*: com a cobertura de hoje, 74% iriam para revisão (37 mil por semana). Por isso o centro
  do plano é fazer cada revisão virar dado.

**Como entraria numa stack AWS** (proposta). A importação dispara um job de embeddings (Lambda ou container em batch),
a classificação vira um passo no Step Functions ou no Airflow, e as decisões vão para o warehouse com o `run_id`, como
já fazem as tabelas `runs` e `decisions` no PostgreSQL deste repositório. A precisão medida na amostra auditada vira
uma métrica com alarme no CloudWatch.

**Como a correção do professor vira dado.** Cada revisão gera uma linha (questão, tema escolhido, extras, fora) no
mesmo formato da folha-verdade, ligada ao `run_id`. Essas linhas entram no treino (kNN, convenção por origem,
frequência) e, separadas por lote, na calibração. A fila tem duas prioridades: questões com confiança entre 0,6 e 0,945,
que são as que mais rápido viram publicáveis, e uma amostra aleatória de 5% das publicadas, para medir a precisão real.

**Retreino.** Semanal, junto com a importação; retreinar leva segundos. O limiar só baixa se a avaliação aninhada,
com os rótulos novos, sustentar a meta. Em produção, a amostra auditada serviria de alarme: se a precisão medida caísse
abaixo de 95%, o limiar subiria até a revisão seguinte (ainda não implementado).

## e) Memo para o professor

Está em [`memo_professor.md`](memo_professor.md), com uma página.

## Limitações e o que eu faria com mais tempo

- **A precisão é prevista, não medida.** Cerca de 25 publicações por rodada sustentam a estimativa; numa rodada
  isolada, o intervalo de Wilson de 95% vai de 87% a 100%.
- **A importação nova não tem rótulo** e recebe 70% das publicações. Com mais tempo, pediria de 50 a 100 rótulos desse
  lote e calibraria cada lote separado.
- **`fora` automático e `extras_ids`** ficaram de fora por falta de exemplos; dúvida vai para revisão.
- **Descrições escritas por modelo** têm erros pontuais (a de "Exercício Arbitrário das Próprias Razões", por exemplo).
  O peso é aprendido, mas revisá-las aumentaria a cobertura.
- **Reprodutibilidade:** no Docker (Linux) as decisões são idênticas às do macOS; as confianças diferem em até 0,0001
  por diferença de biblioteca numérica.
