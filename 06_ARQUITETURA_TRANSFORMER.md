# 06. Como as peças formam um Transformer?

## 1. O fluxo de alto nível
```text
Texto
  ↓
Tokenização e IDs
  ↓
Embeddings + informação posicional
  ↓
Blocos Transformer
  ↓
Representações contextuais
  ↓
Cabeça específica da tarefa
  ↓
Saída
```

## 2. Informação posicional
A atenção, por si só, não informa automaticamente a ordem original das palavras. Por isso, arquiteturas Transformer incorporam informação posicional, de diferentes maneiras conforme o modelo.

## 3. Bloco Transformer
Um bloco típico de encoder contém:
1. Multi-Head Self-Attention.
2. Conexão residual e normalização.
3. Rede feed-forward.
4. Outra conexão residual e normalização.

A implementação exata varia conforme a arquitetura.

## 4. Multi-Head Attention
Em vez de calcular apenas uma atenção, o modelo calcula várias cabeças em paralelo. Cada cabeça possui projeções próprias e pode aprender padrões diferentes de relacionamento.

Não devemos afirmar que cada cabeça tem uma função humana fixa. O comportamento é aprendido durante o treinamento.

## 5. Encoder e Decoder
- Encoder: transforma a sequência de entrada em representações contextualizadas. BERT é baseado em encoder.
- Decoder: gera sequências de forma autoregressiva, usando contexto anterior. GPT é baseado em decoder.
- Encoder-decoder: combina um encoder de entrada e um decoder de geração. O Transformer original foi apresentado nessa organização.

## 6. Classificador
Em tarefas de classificação, uma cabeça de classificação recebe uma representação do Transformer e produz pontuações para as classes. Em modelos BERT, frequentemente é usada uma representação especial associada ao token `[CLS]`, mas detalhes dependem do modelo e da tarefa.

## 7. Um mapa mental
```text
TRANSFORMER
  |
  +-- Entrada: tokens, IDs, posições
  |
  +-- Representações: embeddings
  |
  +-- Blocos:
  |     +-- Self-Attention
  |     +-- Multi-Head
  |     +-- Feed-forward
  |     +-- Residual e normalização
  |
  +-- Saída:
        +-- classificação
        +-- geração
        +-- outras tarefas
```

## Para discutir
Qual é a diferença entre a arquitetura Transformer e um modelo específico como BERT ou GPT?
