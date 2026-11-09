# 03. Attention: entendendo a ideia antes da fórmula

## 1. O problema
Leia:

"Marina colocou o livro na mochila porque ela estava pesada."

A palavra "ela" precisa ser interpretada considerando o contexto. Em uma frase real, o contexto ajuda a descobrir a que elemento o pronome se refere.

## 2. O que a atenção faz?
O mecanismo de atenção calcula quanto cada posição da sequência deve considerar informações de outras posições ao construir uma nova representação.

Não significa que o modelo escolhe conscientemente uma palavra. É uma operação matemática aprendida.

## 3. Query, Key e Value
Uma analogia introdutória:

- Query (Q): o que esta posição está procurando?
- Key (K): que características cada posição oferece para comparação?
- Value (V): que informação será combinada?

As três matrizes são produzidas por transformações aprendidas a partir das representações de entrada.

Em forma simplificada:

```text
Q = X WQ
K = X WK
V = X WV
```

`X` representa as entradas vetoriais. `WQ`, `WK` e `WV` são matrizes de parâmetros aprendidos.

## 4. Como a atenção é calculada?
A versão clássica da atenção escalada é:

```text
Attention(Q, K, V) = softmax(Q K^T / sqrt(dk)) V
```

Passo a passo:
1. `Q K^T`: calcula pontuações de compatibilidade entre posições.
2. Divisão por `sqrt(dk)`: controla a escala das pontuações.
3. `softmax`: transforma pontuações em pesos normalizados.
4. Multiplicação por `V`: combina os vetores de informação usando esses pesos.

## 5. Self-Attention
É chamada de Self-Attention porque Query, Key e Value são derivados da mesma sequência de entrada. Cada posição pode considerar outras posições da própria sequência.

## 6. Cuidado com interpretações
Uma pontuação alta de atenção não deve ser tratada automaticamente como prova de que uma palavra é a explicação causal da previsão. Os pesos são parte do cálculo interno, não uma explicação completa do modelo.

## Perguntas para a turma
- Em "O banco da praça estava molhado", que palavras ajudam a interpretar "banco"?
- Em "Não gostei do celular", por que a palavra "gostei" isolada não basta?
- O que muda entre olhar uma palavra isoladamente e representá-la dentro de uma frase?
