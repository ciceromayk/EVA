# EVA — uma IA construída do zero

**EVA** é um modelo de linguagem (um *Transformer* estilo GPT) implementado
inteiramente do zero em **NumPy puro** — sem PyTorch, sem TensorFlow. O
objetivo é educacional: mostrar, peça por peça, como uma IA moderna que
gera texto realmente funciona por dentro.

Tudo o que faz uma rede neural aprender está aqui e é legível:

- **Diferenciação automática** (backpropagation) escrita à mão
- **Atenção multi-cabeça causal** — o coração dos Transformers
- **Otimizador AdamW** e recorte de gradiente
- **Tokenizador** e laço de **treino** completos

## Estrutura

```
eva/
  autograd.py    motor de autograd (Tensor + backpropagation)
  nn.py          camadas: Linear, LayerNorm, atenção, MLP, Block
  model.py       o modelo GPT completo + geração de texto
  optim.py       otimizador AdamW e clip de gradiente
  tokenizer.py   tokenizador em nível de caractere
train.py         script de treino e amostragem
data/corpus.txt  corpus de exemplo (português)
tests/           checagem numérica do autograd
```

## Como usar

Requisito único: `numpy`.

```bash
pip install numpy

# treinar (CPU, alguns minutos)
python train.py --steps 2000

# gerar texto a partir do checkpoint salvo
python train.py --generate "A EVA "

# verificar que o autograd está correto
PYTHONPATH=. python tests/test_autograd.py
```

Ajuste o tamanho do modelo pelos argumentos `--n-layer`, `--n-head`,
`--n-embd`, `--block-size`. Treine no seu próprio texto com `--data meu.txt`.

## Como a EVA funciona (visão geral)

1. **Tokenização** — o texto vira uma sequência de inteiros (um por caractere).
2. **Embeddings** — cada token e sua posição viram vetores.
3. **Blocos Transformer** — cada bloco tem *atenção causal* (cada posição
   olha só para o passado) seguida de uma pequena rede feed-forward, ambas
   com conexões residuais e LayerNorm.
4. **Cabeça de saída** — projeta para o vocabulário; o *softmax* dá a
   probabilidade do próximo caractere.
5. **Treino** — a *entropia cruzada* mede o erro; o **autograd** calcula os
   gradientes por backpropagation; o **AdamW** ajusta os pesos.
6. **Geração** — amostra-se um caractere de cada vez, realimentando o modelo.

O modelo padrão tem ~350 mil parâmetros e roda tranquilamente em CPU.

## Nota

Este é um projeto didático. Modelos de produção usam as mesmas ideias em
escala muito maior (bilhões de parâmetros, tokenização por subpalavras,
kernels em GPU), mas a mecânica fundamental é exatamente esta.
