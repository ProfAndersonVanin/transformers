# 01. Por que precisamos de Transformers em PLN?

## 1. O que é PLN?
Processamento de Linguagem Natural (PLN) é a área da Inteligência Artificial que desenvolve métodos para trabalhar com textos e linguagem humana.

Exemplos de tarefas:
- análise de sentimentos;
- tradução automática;
- reconhecimento de entidades nomeadas (NER);
- resumo de textos;
- classificação de mensagens;
- perguntas e respostas;
- geração de texto.

## 2. O desafio do contexto
Considere:

- "Sentei no banco da praça."
- "Fui ao banco resolver um problema."

A palavra "banco" aparece nas duas frases, mas o sentido muda conforme as outras palavras.

Outro exemplo:

- "O celular é pequeno, mas excelente."
- "O celular é excelente, mas pequeno demais para mim."

As palavras podem ser parecidas, mas a avaliação depende da relação entre elas e do contexto.

## 3. A ideia central
Transformers são uma arquitetura de redes neurais que utiliza mecanismos de atenção para construir representações de elementos de uma sequência considerando outros elementos da sequência.

Em linguagem simples: o modelo aprende a relacionar partes do texto para produzir uma representação contextualizada.

## 4. Uma analogia
Imagine uma conversa em sala. Para entender a palavra "ele" em "João encontrou Pedro depois que ele saiu", precisamos considerar o restante da frase. O contexto ajuda a interpretar as relações.

A analogia é apenas introdutória. O modelo não compreende como uma pessoa; ele aprende padrões estatísticos a partir de grandes conjuntos de dados.

## 5. Linha histórica simplificada
Antes dos Transformers, muitas abordagens de PLN utilizavam regras, características construídas manualmente ou arquiteturas recorrentes, como RNNs e LSTMs. O artigo "Attention Is All You Need", publicado em 2017, apresentou a arquitetura Transformer, baseada em atenção e processamento paralelo da sequência.

## 6. Conceitos que aparecerão
1. Token: unidade de texto processada pelo modelo.
2. Tokenização: divisão do texto em tokens.
3. ID: número que identifica cada token no vocabulário.
4. Embedding: vetor numérico associado a um token.
5. Attention: mecanismo que calcula relações e combina informações.
6. Transformer: arquitetura que organiza esses mecanismos em camadas.
7. Classificador: componente que transforma representações em uma categoria, quando a tarefa é classificação.

## Para discutir
Por que uma lista de palavras positivas e negativas pode falhar ao analisar "Não gostei do celular" ou "O celular é bom, mas a bateria é péssima"?
