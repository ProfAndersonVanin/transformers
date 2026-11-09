# 02. Do texto aos números: tokens, IDs e embeddings

## 1. Por que converter texto?
Modelos de redes neurais realizam operações matemáticas. Por isso, o texto precisa ser convertido em unidades e representações numéricas.

## 2. Tokenização
Tokenização é a divisão do texto em unidades chamadas tokens. Um token pode ser uma palavra, parte de uma palavra ou pontuação.

Exemplo ilustrativo:

```text
"Eu gostei do celular!"
        |
        v
["Eu", "gostei", "do", "celular", "!"]
```

A divisão real depende do tokenizer e do vocabulário do modelo.

## 3. IDs dos tokens
Cada token do vocabulário possui um identificador numérico.

Exemplo fictício:

| Token | ID ilustrativo |
|---|---:|
| Eu | 104 |
| gostei | 825 |
| do | 21 |
| celular | 938 |
| ! | 5 |

Os números acima são inventados para facilitar a explicação. Os IDs reais dependem do tokenizer.

## 4. Embeddings
O ID é usado para consultar uma matriz de embeddings. Cada token recebe um vetor de números.

Exemplo fictício, com apenas quatro dimensões:

```text
"gostei" -> [0.20, -0.10, 0.75, 0.31]
```

Em modelos reais, os vetores podem ter centenas ou milhares de dimensões.

O embedding não é uma definição de dicionário. É uma representação numérica aprendida durante o treinamento.

## 5. Demonstração com Python
Instale a biblioteca:

```python
!pip -q install transformers
```

Código:

```python
from transformers import AutoTokenizer

nome_modelo = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(nome_modelo)

texto = "I liked the cell phone."

tokens = tokenizer.tokenize(texto)
ids = tokenizer.convert_tokens_to_ids(tokens)

print("Texto:", texto)
print("Tokens:", tokens)
print("IDs:", ids)

for token, identificador in zip(tokens, ids):
    print(token, "->", identificador)
```

A saída exata depende da versão do tokenizer. O modelo escolhido é em inglês, por isso usamos uma frase em inglês.

## 6. Inspecionando os embeddings
O trecho abaixo carrega o modelo-base para observar a matriz de embeddings. Ele não é um classificador de sentimento pronto.

```python
import torch
from transformers import AutoModel

modelo_base = AutoModel.from_pretrained(nome_modelo)

ids_tensor = torch.tensor([ids])
camada_embeddings = modelo_base.embeddings.word_embeddings

vetores = camada_embeddings(ids_tensor)

print("Formato:", vetores.shape)
```

O formato esperado é `[1, quantidade_de_tokens, dimensão_do_embedding]`. O primeiro número representa a quantidade de exemplos no lote.

## 7. Uma distinção importante
Em arquiteturas como BERT e DistilBERT, o vetor de entrada combina embeddings de tokens com informação posicional. Assim, o modelo também recebe informação sobre a posição dos tokens na sequência.

## Verifique se entendeu
1. Token e ID são a mesma coisa?
2. O que é um embedding?
3. Por que os IDs reais não são números universais para todas as bibliotecas e modelos?
