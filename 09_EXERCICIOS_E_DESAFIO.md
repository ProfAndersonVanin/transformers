# 09. Exercícios e desafio final

## Exercício 1: conceitos
Responda com suas palavras:
1. O que é PLN?
2. O que é um token?
3. Qual é a diferença entre token e ID?
4. O que é embedding?
5. Qual é a função geral de Attention?
6. O que representam Query, Key e Value?
7. Qual é a diferença entre treinamento e inferência?

## Exercício 2: tokenização
Use o tokenizer DistilBERT em inglês para processar:

`"I love learning Python!"`

Mostre:
- tokens;
- IDs;
- correspondência entre token e ID.

## Exercício 3: Self-Attention
No arquivo 04:
1. Altere os vetores de entrada.
2. Execute novamente.
3. Compare a matriz de pesos.
4. Explique por que os resultados mudaram.

## Exercício 4: Backpropagation
No arquivo 05:
1. Altere a taxa de aprendizagem para `0.01`.
2. Execute 10 épocas.
3. Compare com a taxa `0.1`.
4. O que acontece se a taxa for muito alta?

## Exercício 5: classificação
No arquivo 08, teste:
- "I love this phone."
- "I do not like this phone."
- "The phone is expensive, but it is worth it."
- "The camera is good, but the battery is terrible."

Observe se o modelo identifica corretamente o sentimento geral. Registre casos em que a saída não parece representar toda a nuance da frase.

## Desafio: analisador de avaliações
Crie uma lista com 10 avaliações curtas de celulares em inglês. Use o modelo de sentimento para classificar cada uma.

Apresente:
- texto da avaliação;
- rótulo previsto;
- pontuação associada;
- uma observação sua sobre possíveis erros.

## Conclusão
Complete a frase:

"Um Transformer é uma arquitetura que utiliza __________ para relacionar informações da sequência. Durante o treinamento, __________ calcula os gradientes e o otimizador atualiza os parâmetros. Quando usamos um modelo já treinado para classificar uma frase nova, estamos fazendo __________."
