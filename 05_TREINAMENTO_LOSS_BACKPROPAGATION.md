# 05. Treinamento, Loss, Backpropagation e Gradient Descent

## 1. Inferência não é treinamento
Quando usamos um modelo já treinado para analisar uma frase nova, estamos fazendo inferência. O modelo calcula uma saída usando os parâmetros que já aprendeu.

Durante o treinamento, o modelo recebe exemplos, produz previsões, calcula uma medida de erro e ajusta seus parâmetros.

## 2. Fluxo do treinamento
```text
Entrada
  ↓
Forward pass
  ↓
Previsão
  ↓
Loss: comparação com o alvo
  ↓
Backpropagation: cálculo dos gradientes
  ↓
Otimizador: atualização dos parâmetros
  ↓
Próximo exemplo ou lote
```

## 3. Analogia
Um estudante responde uma questão, compara sua resposta com o gabarito, identifica o erro e tenta melhorar. A analogia ajuda a visualizar o ciclo, mas redes neurais fazem os ajustes por cálculos matemáticos.

## 4. Exemplo numérico muito simples
Vamos ajustar um único peso em um modelo `previsao = x * peso`. Queremos que, para `x = 2`, a previsão seja 4.

A função de erro escolhida é metade do erro ao quadrado:

`loss = 0.5 * (previsao - real) ** 2`

O gradiente em relação ao peso é:

`gradiente = (previsao - real) * x`

```python
peso = 0.5
x = 2.0
real = 4.0
taxa_aprendizado = 0.1

for epoca in range(10):
    # Forward pass
    previsao = x * peso

    # Loss
    loss = 0.5 * (previsao - real) ** 2

    # Backpropagation: derivada da loss em relação ao peso
    gradiente = (previsao - real) * x

    # Gradient Descent: atualização do peso
    peso = peso - taxa_aprendizado * gradiente

    print(
        "Época:", epoca + 1,
        "| previsão:", round(previsao, 4),
        "| loss:", round(loss, 4),
        "| peso atualizado:", round(peso, 4)
    )
```

Observe que o peso se aproxima de 2. Quando `peso = 2`, `x * peso = 4`.

## 5. O que é Backpropagation?
Backpropagation é o procedimento usado para calcular como a função de erro varia em relação aos parâmetros da rede. Ele aplica a regra da cadeia, propagando gradientes da saída em direção às camadas anteriores.

Backpropagation calcula gradientes. O otimizador, como Gradient Descent ou Adam, usa esses gradientes para atualizar os parâmetros.

## 6. Ligação com Transformers
Durante o treinamento de um Transformer, os parâmetros das projeções Q, K e V, das camadas feed-forward e de outros componentes são ajustados com gradientes.

Atenção calcula combinações de informação no forward pass. Backpropagation calcula gradientes através dessas operações durante o treinamento.

## 7. Distinção que os alunos devem guardar
- Attention: mecanismo para combinar informações entre posições.
- Loss: medida de quão distante a previsão está do alvo.
- Backpropagation: cálculo dos gradientes.
- Otimizador: atualização dos parâmetros.
- Inferência: uso do modelo treinado, normalmente sem atualizar pesos.

## Verifique se entendeu
Por que não executamos Backpropagation toda vez que fazemos uma pergunta a um chatbot que já está treinado?
