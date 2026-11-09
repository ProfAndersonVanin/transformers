# Transformers em PLN: roteiro da sequência

## Objetivo geral
Construir gradualmente uma compreensão prática de Transformers, partindo de conceitos de PLN e chegando à inspeção de um modelo pré-treinado em Python.

## Público
Alunos iniciantes em programação. Os exemplos priorizam Python básico, NumPy e, ao final, a biblioteca Hugging Face Transformers com PyTorch.

## Sequência sugerida

| Arquivo | Tema | Resultado esperado |
|---|---|---|
| 01 | Por que precisamos de PLN e contexto | Entender o problema que os Transformers ajudam a resolver |
| 02 | Texto, tokens, IDs e embeddings | Ver como texto vira números e vetores |
| 03 | Atenção: a ideia intuitiva | Entender por que palavras dependem do contexto |
| 04 | Self-Attention com NumPy | Calcular pontuações, Softmax e combinação de informações |
| 05 | Treinamento, erro e Backpropagation | Entender como os parâmetros são ajustados |
| 06 | Arquitetura Transformer | Juntar embeddings, posição, atenção e camadas |
| 07 | Abrindo um Transformer pré-treinado | Inspecionar tokens, IDs, embeddings, Q, K, V e atenção |
| 08 | Classificação e inferência | Observar logits, probabilidades e resultado |
| 09 | Exercícios e desafio | Consolidar os conceitos |

## Sugestão de distribuição para uma aula de 2h30
- 15 min: motivação e contexto (arquivo 01)
- 20 min: tokenização e embeddings (arquivo 02)
- 20 min: ideia de Attention (arquivo 03)
- 25 min: implementação com NumPy (arquivo 04)
- 20 min: treinamento e Backpropagation (arquivo 05)
- 15 min: arquitetura completa (arquivo 06)
- 25 min: inspeção do modelo pré-treinado (arquivos 07 e 08)
- 10 min: exercícios e fechamento (arquivo 09)

A duração pode ser dividida em duas aulas, caso os alunos precisem de mais tempo para executar e discutir o código.

## Orientação didática
Não apresente todos os termos técnicos de uma vez. Primeiro mostre o problema, depois a representação numérica, então a atenção e, por fim, o treinamento. O código com biblioteca deve aparecer depois que os alunos já tiverem uma imagem mental do processo.

## Observações importantes
- Attention não é, por si só, uma explicação completa de como o modelo aprende.
- Backpropagation é usado durante o treinamento para calcular gradientes e ajustar parâmetros.
- Na inferência, normalmente não ocorre Backpropagation nem atualização dos pesos.
- Os pesos de atenção não devem ser interpretados automaticamente como explicações perfeitas do raciocínio do modelo.
- Um modelo pré-treinado para classificação precisa ter sido treinado para a tarefa. Carregar apenas `distilbert-base-uncased` com uma cabeça classificadora recém-criada não fornece um classificador de sentimento pronto.
