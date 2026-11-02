# Memo para o professor

**Assunto:** classificação automática das 659 questões de Penal, Processo Penal, Legislação Especial e Criminologia

**Em uma frase:** o sistema publica sozinho só as questões em que tem segurança (cerca de um quarto) e manda as
demais para a sua fila com uma sugestão de tema; para publicar mais, precisa de exemplos e de uma árvore mais limpa.

## O que eu fiz

Montei um sistema que compara cada questão com a descrição dos temas, com as leis e os artigos citados, com o assunto
de origem e com as 100 questões que a equipe classificou. Na hora de decidir, ele não usa IA generativa; usei um
modelo de IA só antes, para escrever a descrição de cada tema. Cada questão recebe uma chance de estar no tema certo,
e o sistema só publica quando ela chega a 94,5%. Testando nas 100 questões da equipe, sempre escondendo as testadas,
ele acertou cerca de 96% das que publicaria. É uma estimativa para as questões novas, não uma medição.

## O que encontrei nos temas (e o que recomendo mudar)

1. **Temas repetidos em duas disciplinas.** Lavagem de Dinheiro, Lei Maria da Penha, Organizações Criminosas, Tráfico
   de Drogas, Estatuto do Idoso, Abuso de Autoridade e Crimes Ambientais existem em *Penal* e em *Legislação Especial*;
   Lei de Execução Penal existe em *Legislação Especial* e em *Processo Penal*. Para o aluno é o mesmo assunto, mas
   quem estuda uma cópia não vê as questões da outra. A equipe quase sempre usou a versão de Legislação Especial.
   **Recomendo juntar as cópias** num tema só.
2. **"Partes" sem conteúdo.** *Teoria da Pena (Parte 1/2)*, *Competência (Parte 1/2)*, *Sentença Penal (Parte 1/2)*,
   *Crimes contra o Patrimônio (Parte 1/2)* e outros são temas sem descrição, que só repetem o nome do tema principal
   e disputam questões com ele. **Recomendo removê-los** ou dar a eles um nome de conteúdo.
3. **Fronteiras que nem a equipe resolve sempre igual.** Livramento condicional, regimes e progressão aparecem tanto em
   *Teoria da Pena* quanto em *Lei de Execução Penal*; Juizado Especial aparece em *Procedimento Sumaríssimo* e em
   *Legislação Especial*; provas oscilam entre *Provas em Espécie* e *Teoria Geral das Provas*. **Preciso de uma regra
   escrita**, do tipo "questão sobre X vai para o tema Y".
4. **Questões sem tema.** A equipe marcou 8 das 100 questões como sem tema. Algumas tratam de assuntos que não existem
   na árvore (crimes contra a ordem econômica, rito dos crimes de responsabilidade de funcionário público, procedimento
   da Lei de Drogas). Outras, como uma sobre fixação da pena, parecem misturar temas. **Recomendo criar** os temas
   que faltam e me dizer o critério da equipe para "sem tema".
5. **O assunto de origem engana.** "Concurso de Crimes" aparece em questões de quase qualquer assunto, e é daí que
   vêm os temas-lixeira. O sistema usa a origem só como pista.

## Do que preciso do senhor

- **Revisar a fila.** Cada questão em "revisar" já vem com uma sugestão e uma segunda opção, e cada correção vira um
  exemplo novo que aumenta o que o sistema publica com segurança.
- **Classificar 50 a 100 questões da importação nova** (Q0360 em diante). Os 100 exemplos de hoje são do banco antigo,
  e as questões novas chegam em outro formato; sem essa amostra, não consigo confirmar que ele acerta tanto nelas.
- **Decidir as fusões** dos itens 1 e 2 e escrever as regras de fronteira do item 3.
- **Conferir as descrições dos temas** (mando numa planilha). Foram escritas por IA e algumas estão imprecisas.
