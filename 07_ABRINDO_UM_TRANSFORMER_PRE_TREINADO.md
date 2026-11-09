# 07. Abrindo um Transformer pré-treinado com Python

## Objetivo
Inspecionar um modelo real e observar tokenização, IDs, embeddings, dimensões, camadas e pesos de atenção.

## Atenção à escolha do modelo
Nesta demonstração usaremos `distilbert-base-uncased` para inspecionar a arquitetura-base. Ele é um modelo pré-treinado de linguagem, mas não é, por si só, um classificador de sentimento pronto.

Para a classificação de sentimento, usaremos no arquivo seguinte um modelo que foi ajustado especificamente para essa tarefa.

## 1. Instalação
No Google Colab:

```python
!pip -q install transformers torch
```

## 2. Carregar tokenizer e modelo-base
```python
import torch
from transformers import AutoTokenizer, AutoModel

nome_modelo = "distilbert-base-uncased"

tokenizer = AutoTokenizer.from_pretrained(nome_modelo)
modelo = AutoModel.from_pretrained(nome_modelo)
```

## 3. Entrada
```python
texto = "I liked the cell phone."
print(texto)
```

## 4. Tokenização
```python
tokens = tokenizer.tokenize(texto)
print(tokens)
```

O tokenizer pode incluir tokens especiais quando a entrada é preparada para o modelo. A tokenização exibida aqui é uma visualização dos tokens do texto.

## 5. IDs
```python
ids = tokenizer.convert_tokens_to_ids(tokens)
print(ids)

for token, identificador in zip(tokens, ids):
    print(token, "->", identificador)
```

## 6. Preparar a entrada completa
É preferível deixar o tokenizer preparar também os tokens especiais e a máscara:

```python
entrada = tokenizer(texto, return_tensors="pt")
print(entrada)
print("IDs:", entrada["input_ids"])
print("Máscara:", entrada["attention_mask"])
```

## 7. Embeddings de entrada
```python
with torch.no_grad():
    embeddings_entrada = modelo.embeddings(entrada["input_ids"])

print("Formato dos embeddings:", embeddings_entrada.shape)
```

O formato esperado é `[lote, sequência, dimensão_oculta]`. DistilBERT-base usa dimensão oculta 768.

Observação: `modelo.embeddings(...)` inclui a composição de embeddings de palavras e posições, além da normalização/dropout definidos na arquitetura. Não é apenas a matriz de embeddings de palavras.

## 8. Inspecionar Q, K e V
```python
primeira_atencao = modelo.transformer.layer[0].attention

print("Projeção Query:")
print(primeira_atencao.q_lin)

print("Projeção Key:")
print(primeira_atencao.k_lin)

print("Projeção Value:")
print(primeira_atencao.v_lin)
```

Essas camadas lineares projetam as representações de entrada em Q, K e V. Os parâmetros dessas projeções foram aprendidos no treinamento do modelo.

## 9. Obter os pesos de atenção
```python
with torch.no_grad():
    saida = modelo(
        **entrada,
        output_attentions=True,
        output_hidden_states=True
    )

print("Quantidade de camadas de atenção:", len(saida.attentions))
print("Formato da atenção na primeira camada:", saida.attentions[0].shape)
```

Para DistilBERT-base, a primeira dimensão é o lote, a segunda é o número de cabeças, e as duas últimas dimensões representam posições de consulta e posições observadas.

## 10. Visualizar uma matriz de atenção
```python
import matplotlib.pyplot as plt

tokens_com_especiais = tokenizer.convert_ids_to_tokens(
    entrada["input_ids"][0]
)

# Primeiro exemplo, primeira cabeça, primeira camada
matriz = saida.attentions[0][0, 0].cpu().numpy()

plt.figure(figsize=(8, 6))
plt.imshow(matriz)
plt.xticks(range(len(tokens_com_especiais)), tokens_com_especiais, rotation=45)
plt.yticks(range(len(tokens_com_especiais)), tokens_com_especiais)
plt.xlabel("Posições observadas")
plt.ylabel("Posição que consulta")
plt.title("Atenção: primeira camada, primeira cabeça")
plt.colorbar()
plt.tight_layout()
plt.show()
```

A visualização mostra os pesos usados naquela cabeça e camada. Não a interprete como explicação completa do resultado do modelo.

## 11. Estados ocultos
```python
print("Quantidade de estados ocultos:", len(saida.hidden_states))
print("Formato do último estado:", saida.hidden_states[-1].shape)
```

O último estado contém uma representação contextual para cada posição da sequência.

## 12. Uma observação técnica
Este arquivo inspeciona um encoder DistilBERT. Outros modelos têm nomes de módulos e detalhes internos diferentes. Por isso, acessar atributos internos como `q_lin` depende da arquitetura escolhida.
