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
  autograd.py         motor de autograd (Tensor + backpropagation)
  nn.py               camadas: Linear, LayerNorm, atenção, MLP, Block
  model.py            o modelo GPT completo + geração de texto
  optim.py            otimizador AdamW e clip de gradiente
  tokenizer.py        tokenizador em nível de caractere
train.py              script de treino e amostragem (com presets)
tools/build_corpus.py extrai texto de PDFs para montar o corpus
data/corpus.txt       corpus de treino (gerado a partir de PDFs)
tests/                checagem numérica do autograd
```

## Como usar

Requisito único: `numpy`.

```bash
pip install numpy

# treinar (CPU) — usa o preset "medium" por padrão
python train.py --steps 1500

# gerar texto a partir do checkpoint salvo
python train.py --generate "A EVA "

# verificar que o autograd está correto
PYTHONPATH=. python tests/test_autograd.py
```

### Tamanho do modelo (presets)

Escolha o tamanho com `--preset`. Modelos maiores aprendem padrões mais
ricos, mas exigem mais tempo de CPU:

| preset   | parâmetros | camadas | contexto | tempo aprox. (1500 passos) |
|----------|-----------:|:-------:|:--------:|:--------------------------:|
| `small`  |    ~350 mil |    3    |    64    |  ~9 min                    |
| `medium` |    ~1,8 mi  |    4    |    96    |  ~19 min (padrão)          |
| `large`  |    ~4,8 mi  |    6    |   128    |  ~40 min                   |

```bash
python train.py --preset large --steps 2000
```

Também dá para sobrescrever qualquer dimensão individual
(`--n-layer`, `--n-head`, `--n-embd`, `--block-size`, `--batch-size`).

### Treinar no seu próprio material (PDFs)

O corpus incluído foi gerado a partir de PDFs sobre lógica de programação,
Python e modelos de linguagem. Para montar o seu:

```bash
pip install pymupdf
python tools/build_corpus.py livro1.pdf livro2.pdf -o data/corpus.txt
python train.py --preset medium --steps 1500
```

O script extrai o texto, remove cabeçalhos/rodapés repetidos, junta
palavras hifenizadas e normaliza o espaçamento.

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

O preset padrão (`medium`) tem ~1,8 milhão de parâmetros e roda em CPU.

## Nota

Este é um projeto didático. Modelos de produção usam as mesmas ideias em
escala muito maior (bilhões de parâmetros, tokenização por subpalavras,
kernels em GPU), mas a mecânica fundamental é exatamente esta.
