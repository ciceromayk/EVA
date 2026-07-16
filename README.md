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

### Tokenizador e regularização

```bash
# subpalavras (BPE) com dropout — gera palavras inteiras, sem soletrar
python train.py --tokenizer bpe --bpe-vocab 512 --dropout 0.2 --steps 2000
```

- `--tokenizer {char,bpe}`: caractere (padrão) ou subpalavras
- `--bpe-vocab N`: tamanho do vocabulário BPE (≥ 256)
- `--dropout P`: taxa de dropout (regularização; padrão 0,1)

O treino salva sempre o checkpoint de **menor loss de validação** (early
stopping), então o overfitting no fim não estraga o modelo final.

### Treinar no seu próprio material (PDFs)

O corpus incluído foi gerado a partir de PDFs sobre lógica de programação,
Python, modelos de linguagem e *A Arte da Guerra*. Para montar o seu:

```bash
pip install pymupdf
python tools/build_corpus.py livro1.pdf livro2.pdf -o data/corpus.txt
python train.py --preset medium --steps 2000
```

O script extrai o texto, remove cabeçalhos/rodapés repetidos, junta
palavras hifenizadas e normaliza o espaçamento.

## Resultados

Treinando o preset `medium` (~1,8 mi de parâmetros) por 2000 passos no
corpus de ~231 mil caracteres (4 livros), a EVA sai de texto aleatório
para português reconhecível:

```
passo    1/2000 | treino 5.20 | val 4.88
passo 1000/2000 | treino 1.54 | val 1.77
passo 2000/2000 | treino 0.84 | val 1.57
```

Amostra gerada (prompt "Um algoritmo é"):

> Um algoritmo é destador o custo de treinamento de múltiplos modelos com
> um mecanismo de aprovem recebença do inimigo. Esses cinco pode ser camada
> capacidade das comunstraras e trabalhadas para medidade, modelos com os
> de seguir desempenho do uso de treinamento profundo.

O modelo aprendeu vocabulário e gramática local do português e mistura os
temas dos quatro livros (guerra, algoritmos, modelos de linguagem). Não é
um texto perfeito — é um modelo minúsculo em CPU — mas demonstra que toda
a mecânica (autograd, atenção, otimização) funciona de ponta a ponta.

### O que fez a diferença

O primeiro treino estagnava em loss ~2,5 gerando texto incoerente. Três
ajustes destravaram o aprendizado:

- **GELU** no lugar de `tanh` no MLP (a `tanh` saturava e matava o gradiente)
- **Agendamento de learning rate** (warmup + decaimento cosseno)
- **Init estilo GPT-2** nas projeções residuais (escala `1/sqrt(2*n_layer)`)

### Tokenizador BPE e regularização

Trocar o tokenizador de caractere pelo **BPE** (subpalavras) comprime o
corpus ~1,9× e faz o modelo gerar *palavras inteiras* em vez de soletrar.
Mas, com um corpus pequeno, o BPE **sofreu overfitting** — a val loss
começava a subir na metade do treino. A cura foram dois clássicos:

- **Dropout** (`--dropout 0.2`): regularização que zera ativações no treino
- **Early stopping**: o treino guarda o checkpoint de menor val loss, não o
  último

Comparação por caractere (métrica justa entre tokenizadores):

| configuração                | val loss/caractere |
|-----------------------------|:------------------:|
| char                        | 1,57               |
| BPE sem dropout             | ~1,99 (overfit)    |
| **BPE + dropout 0,2**       | **~1,55**          |

Amostra do BPE regularizado (prompt "A arte da guerra ensina"):

> A arte da guerra ensina. Quando os LLMs [...] a melhoria e a informação
> de palavras [...] pode ser usado para inteligência ou treinado [...] o
> inimigo que o está [...] para reforçar sua variável.

Repare como a EVA mistura, num mesmo texto, os temas dos três domínios do
corpus: estratégia militar, modelos de linguagem e programação.

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
