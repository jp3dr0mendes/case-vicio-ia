# Memo para o professor

**Assunto:** classificação automática das 659 questões de Penal, Processo Penal, Legislação Especial e Criminologia
**Em uma frase:** o sistema publica sozinho só as questões em que tem segurança (cerca de um quarto das questões que a equipe
não rotulou) e manda as demais para a sua fila já com uma sugestão de tema; para publicar mais, ele precisa de mais
exemplos rotulados e de uma árvore de temas mais limpa.

## O que eu fiz

Montei um classificador que compara cada questão com a descrição dos temas, com as leis e artigos citados, com o
assunto de origem e com as 100 questões que a equipe classificou. Ele não usa inteligência artificial generativa
na hora de decidir; usei um modelo só antes, para escrever uma descrição de cada tema. Cada questão recebe uma
chance de acerto, e só publico quando essa chance passa de 94%. Testei a regra nas 100 questões da equipe,
escondendo de cada vez as que estava testando: nas que ele publicaria, acertou 98%.

## O que encontrei nos temas (e o que recomendo mudar)

1. **Temas duplicados em duas disciplinas.** Lavagem de Dinheiro, Lei Maria da Penha, Organizações Criminosas,
   Tráfico de Drogas, Estatuto do Idoso, Abuso de Autoridade e Crimes Ambientais existem em *Penal* e em
   *Legislação Especial*; Lei de Execução Penal existe em *Legislação Especial* e em *Processo Penal*. Para o aluno
   são o mesmo assunto, mas para a régua publicar no outro é erro. A equipe quase sempre usou a versão de
   Legislação Especial. **Recomendo fundir as cópias** (manter uma e redirecionar a outra).
2. **"Partes" sem conteúdo.** *Teoria da Pena (Parte 1/2)*, *Competência (Parte 1/2)*, *Sentença Penal (Parte 1/2)*,
   *Crimes contra o Patrimônio (Parte 1/2)* e outras são nós vazios que concorrem com o tema principal.
   **Recomendo removê-los** ou transformá-los em subtemas com nome de conteúdo.
3. **Fronteiras que confundem até a regra da equipe.** Livramento condicional, regimes e progressão aparecem tanto
   em *Teoria da Pena* quanto em *Lei de Execução Penal*; Juizado Especial aparece em *Procedimento Sumaríssimo* e
   em *Legislação Especial*. **Preciso de uma regra escrita**: "questão sobre X vai para o tema Y".
4. **Questões "fora".** Oito das 100 questões rotuladas não têm tema (por exemplo, crimes contra a ordem econômica,
   rito dos crimes de responsabilidade de funcionário público, procedimento da Lei de Drogas). Hoje "Direito penal
   econômico" está inativo. **Recomendo criar** esses temas ou dizer explicitamente que não serão cobertos.
5. **O assunto de origem não é confiável.** "Concurso de Crimes" e "Ação Penal (Parte 01)" aparecem em questões de
   quase qualquer assunto, que é o problema das lixeiras. Uso o assunto de origem só como pista.

## Do que preciso do senhor

- **Revisar a fila** (as questões marcadas "revisar"): cada uma já vem com a sugestão e a segunda opção. Cada
  correção vira exemplo novo e aumenta o que o sistema consegue publicar.
- **Rotular cerca de 300 questões da importação nova** (Q0360 em diante). Os 100 exemplos atuais são todos do banco
  antigo, e o formato das questões novas é diferente; sem esses exemplos, não consigo garantir os 95% nelas.
- **Decidir as fusões de temas** dos itens 1 e 2 e a regra de fronteira do item 3.
- **Conferir as descrições dos temas** em `artifacts/temas_enriquecidos.jsonl`: foram escritas por um modelo e
  algumas estão imprecisas. Uma descrição correta de cada tema ajuda diretamente o classificador.
