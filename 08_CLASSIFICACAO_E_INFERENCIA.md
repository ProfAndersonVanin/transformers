# 08. Do Transformer à classificação: inferência

## Objetivo
Usar um modelo realmente ajustado para análise de sentimentos e observar a diferença entre logits, probabilidades e rótulos.

## 1. Escolher um modelo adequado
Usaremos:

`distilbert-base-uncased-finetuned-sst-2-english`

Esse modelo foi ajustado para classificação de sentimentos em inglês, com as classes POSITIVE e NEGATIVE.

## 2. Instalação
```python
!pip -q install transformers torch
```

## 3. Carregar tokenizer e modelo
```python
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

nome_modelo = "distilbert-base-uncased-finetuned-sst-2-english"

tokenizer = AutoTokenizer.from_pretrained(nome_modelo)
modelo = AutoModelForSequenceClassification.from_pretrained(nome_modelo)
```

## 4. Preparar uma frase
```python
texto = "I really liked this cell phone."

entrada = tokenizer(texto, return_tensors="pt")
print(entrada)
```

## 5. Fazer inferência
```python
modelo.eval()

with torch.no_grad():
    resultado = modelo(**entrada)

print("Logits:", resultado.logits)
```

`eval()` coloca o modelo em modo de avaliação. `torch.no_grad()` evita calcular gradientes, pois não estamos treinando.

## 6. Converter logits em probabilidades
```python
probabilidades = torch.softmax(resultado.logits, dim=1)

print("Probabilidades:", probabilidades)
print("Rótulos:", modelo.config.id2label)
```

## 7. Mostrar a classe prevista
```python
indice = torch.argmax(probabilidades, dim=1).item()
rotulo = modelo.config.id2label[indice]
pontuacao = probabilidades[0, indice].item()

print("Sentimento:", rotulo)
print("Pontuação:", round(pontuacao, 4))
```

A pontuação é a probabilidade atribuída pelo modelo à classe prevista. Não deve ser automaticamente interpretada como uma medida perfeitamente calibrada de certeza.

## 8. Testar várias frases
```python
frases = [
    "I love this phone.",
    "This phone is terrible.",
    "The camera is excellent.",
    "I am disappointed with the battery."
]

for frase in frases:
    entrada = tokenizer(frase, return_tensors="pt")

    with torch.no_grad():
        resultado = modelo(**entrada)

    probabilidades = torch.softmax(resultado.logits, dim=1)
    indice = torch.argmax(probabilidades, dim=1).item()
    rotulo = modelo.config.id2label[indice]

    print(frase, "->", rotulo)
```

## 9. Limitações
- O modelo foi ajustado para inglês. Não espere desempenho confiável em português.
- O modelo prevê sentimento geral, não necessariamente o sentimento de cada aspecto do produto.
- Ironia, contexto cultural, negação complexa e frases ambíguas podem causar erros.
- Um score alto não garante que a previsão esteja correta.

## 10. Conectar com o que foi estudado
O tokenizer prepara IDs. O Transformer produz representações contextuais. A cabeça de classificação transforma uma representação em logits. Softmax pode converter logits em probabilidades para interpretação.
