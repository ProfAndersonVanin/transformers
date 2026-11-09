# 04. Implementando uma Self-Attention simplificada com NumPy

## Objetivo
Calcular, com vetores pequenos, os principais passos da atenção: pontuações, Softmax e combinação dos Values.

Este exemplo é didático. Ele usa Query, Key e Value fornecidos manualmente. Ainda não aprende as matrizes de projeção `WQ`, `WK` e `WV` e não é um Transformer completo.

## 1. Instalação e dados
```python
import numpy as np

# Três vetores fictícios, cada um com duas dimensões.
# Em um modelo real, esses vetores seriam aprendidos/produzidos pelo modelo.
X = np.array([
    [1.0, 0.0],  # token 1
    [0.0, 1.0],  # token 2
    [1.0, 1.0]   # token 3
])

# Para simplificar, usamos X diretamente como Q, K e V.
Q = X
K = X
V = X

print("Q:\n", Q)
print("K:\n", K)
print("V:\n", V)
```

## 2. Calcular as pontuações
```python
pontuacoes = Q @ K.T
print(pontuacoes)
```

Cada linha corresponde a uma Query. Cada coluna corresponde a uma Key. Os valores indicam compatibilidade entre os vetores neste exemplo.

## 3. Aplicar Softmax
Softmax é aplicada em cada linha, para que os pesos de cada Query somem 1.

```python
def softmax(x):
    exp = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp / np.sum(exp, axis=-1, keepdims=True)

pesos = softmax(pontuacoes)
print(pesos)
print("Soma de cada linha:", pesos.sum(axis=1))
```

## 4. Combinar os Values
```python
resultado = pesos @ V

print("Representações de saída:")
print(resultado)
```

Cada linha de saída é uma combinação ponderada dos vetores `V`.

## 5. Usar a escala da atenção
Na atenção escalada, dividimos as pontuações pela raiz quadrada da dimensão das Keys.

```python
dk = K.shape[1]
pontuacoes_escaladas = (Q @ K.T) / np.sqrt(dk)
pesos_escalados = softmax(pontuacoes_escaladas)
resultado_escalado = pesos_escalados @ V

print("Pesos escalados:")
print(pesos_escalados)
print("Resultado:")
print(resultado_escalado)
```

## 6. Função completa
```python
def self_attention(Q, K, V):
    dk = K.shape[-1]
    pontuacoes = (Q @ K.T) / np.sqrt(dk)
    pesos = softmax(pontuacoes)
    saida = pesos @ V
    return saida, pesos

saida, pesos = self_attention(Q, K, V)

print("Pesos de atenção:\n", pesos)
print("Saída:\n", saida)
```

## 7. O que ainda falta para ser um Transformer?
- Projeções aprendidas para produzir Q, K e V.
- Múltiplas cabeças de atenção (Multi-Head Attention).
- Informação posicional.
- Conexões residuais e normalização.
- Rede feed-forward por posição.
- Empilhamento de camadas.
- Treinamento dos parâmetros.

## Atividade
Altere os vetores em `X`. Observe como as pontuações e os pesos mudam. Explique por que o resultado também muda.
