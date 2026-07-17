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
  backend.py          seletor de array: NumPy (CPU) ou CuPy (GPU)
  autograd.py         motor de autograd (Tensor + backpropagation)
  nn.py               camadas: Linear, LayerNorm, atenção, MLP, Block
  model.py            o modelo GPT completo + geração de texto
  optim.py            otimizador AdamW e clip de gradiente
  tokenizer.py        tokenizador em nível de caractere
app.py                painel web (http.server) para uso local / Render / Docker
gradio_ui.py          mesma interface em Gradio (para o Hugging Face grátis)
hf_space/             arquivos prontos para colar num Space Gradio
train.py              script de treino e amostragem (com presets)
check_gpu.py          autoteste de GPU (CPU vs CuPy)
tools/build_corpus.py extrai texto de PDFs/TXT para montar o corpus
data/corpus.txt       corpus de treino (gerado a partir de PDFs)
tests/                checagem numérica do autograd
```

## Manter a EVA atualizada (Windows)

Para receber melhorias na interface e no código, obtenha o projeto com
**git** (assim atualizar é um comando, não rebaixar o .zip):

```powershell
git clone -b claude/custom-ai-from-scratch-dx8d5p https://github.com/ciceromayk/EVA.git
cd EVA
```

Depois é só clicar duas vezes:

- **`iniciar.bat`** — instala o que falta e abre o painel no navegador.
- **`atualizar.bat`** — baixa a versão mais recente do GitHub (`git pull`).

Seus materiais (`materials/`) e modelos treinados (`*.pkl`) ficam fora do
controle de versão, então atualizar **nunca apaga o que você treinou**.

## Painel web (a forma mais fácil)

Uma interface local para você alimentar, treinar e conversar com a EVA
**sem tocar em código** — só a biblioteca padrão do Python, nenhum
framework:

```bash
pip install numpy pymupdf
python app.py            # abre em http://localhost:8000
```

No painel você pode:

- **Adicionar material** — arraste PDFs/TXT ou cole texto; o corpus é
  reconstruído automaticamente (extração e limpeza inclusas)
- **Treinar** — escolha tamanho, tokenizador, passos e dropout, e acompanhe
  o progresso ao vivo
- **Gerar texto** — dê um começo de frase e veja a EVA continuar

O material enviado fica em `materials/` e o corpus é montado a partir dele.
Na primeira execução, o corpus atual é preservado como material inicial.

## Rodar na GPU (NVIDIA / CUDA)

A EVA é escrita sobre um "módulo de array" (`eva/backend.py`). Trocando
NumPy por **CuPy**, o mesmo código roda na GPU — muito mais rápido. Ótimo
para quem tem placa NVIDIA (ex.: RTX 5060).

1. Tenha um **driver NVIDIA recente** instalado (para as placas mais novas,
   série Blackwell/RTX 50, use o driver mais atual disponível).
2. Instale o CuPy compatível com CUDA 12:
   ```bash
   pip install cupy-cuda12x
   ```
3. Confira se a GPU foi reconhecida e veja o ganho de velocidade:
   ```bash
   python check_gpu.py
   ```
4. Treine na GPU:
   ```bash
   python train.py --preset medium --tokenizer bpe --device gpu --steps 3000
   ```
   No painel web, escolha **Processador: GPU · CUDA**.

Se a GPU não estiver disponível, a EVA **avisa e volta para a CPU**
automaticamente — nada quebra.

### Erro "curand*.dll não encontrado" / "CUDA path could not be detected"

Isso significa que o CuPy instalou, mas faltam as **bibliotecas de runtime
da CUDA**. Instale-as via pip (não precisa do CUDA Toolkit completo):

```bash
py -m pip install nvidia-cuda-runtime-cu12 nvidia-cublas-cu12 ^
  nvidia-curand-cu12 nvidia-cusparse-cu12 nvidia-cusolver-cu12 ^
  nvidia-cufft-cu12 nvidia-nvrtc-cu12
py check_gpu.py
```

(No PowerShell, tudo numa linha só, sem o `^`.)

### Se ainda falhar: cuidado com a versão do Python

O ecossistema CUDA (CuPy e os pacotes `nvidia-*`) tem suporte mais estável
no **Python 3.11 ou 3.12**. O Python 3.14 é muito novo e pode não ter os
pacotes certos. Se o passo acima não resolver, instale o **Python 3.12** de
python.org e recrie o ambiente da EVA com ele (`py -3.12 -m pip install …`).

O checkpoint é salvo sempre em formato NumPy (CPU), então um modelo
treinado na GPU também abre numa máquina só-CPU, e vice-versa.

## Rodar no iPad / celular (via Replit)

O iPad não roda Python facilmente, mas o painel é uma página web — então
basta rodar a EVA na nuvem e abrir no Safari. O jeito mais simples é o
**Replit**, e o repositório já vem pronto (`.replit` + `requirements.txt`):

1. Crie uma conta grátis em **https://replit.com** (dá para usar o próprio
   iPad).
2. Clique em **Create Repl → Import from GitHub** e cole a URL do
   repositório (`https://github.com/ciceromayk/EVA`), escolhendo o branch
   `claude/custom-ai-from-scratch-dx8d5p`.
3. Clique em **Run**. O Replit instala tudo e sobe o painel; uma janela
   (webview) abre com a EVA.
4. Toque no ícone de **abrir em nova aba** para usar em tela cheia — e
   adicione à tela inicial do iPad para virar um "app".

Tudo (upload de PDF, treino, geração) roda nos servidores do Replit; o
iPad só mostra a tela. Dica: use o preset **`nano`** ou **`small`** para o
treino terminar rápido.

### Alternativa: Render (link fixo, conectado ao GitHub)

Se o Replit parecer limitado, o **Render** publica o painel com um link
fixo `.onrender.com` conectado direto ao repositório. O projeto já traz o
`render.yaml`:

1. Crie uma conta grátis em **https://render.com** (pode ser pelo iPad).
2. **New + → Web Service** e conecte sua conta do GitHub, escolhendo o
   repositório `ciceromayk/EVA` e o branch
   `claude/custom-ai-from-scratch-dx8d5p`.
3. O Render lê o `render.yaml`, instala tudo e sobe o painel. Ao final ele
   te dá uma URL fixa — abra no Safari do iPad e adicione à tela inicial.

No plano grátis o serviço "hiberna" após alguns minutos parado e leva
~30–50s para acordar no primeiro acesso; depois disso responde normal.
Como o disco é temporário, materiais e checkpoints valem para a sessão
atual (para guardar de vez, use "Baixar cérebro" — veja abaixo).

## Publicar no Hugging Face Spaces (Gradio — grátis)

No Hugging Face, o SDK Docker exige hardware pago; o **SDK Gradio** roda no
tier gratuito. O projeto traz uma interface Gradio (`gradio_ui.py`) e uma
pasta `hf_space/` com tudo pronto para colar num Space — link fixo, grátis
e sempre disponível (acorda sozinho ao ser acessado).

1. Conta grátis em **https://huggingface.co**.
2. **New → Space**, escolha **SDK: Gradio → Blank**.
3. No Space, crie três arquivos (copiando o conteúdo da pasta `hf_space/`
   deste repositório, pelo próprio navegador do iPad):
   - `README.md` — cabeçalho do Space (título, `sdk: gradio`, `app_file`)
   - `requirements.txt` — `gradio`, `numpy`, `pymupdf`
   - `app.py` — baixa o código da EVA do GitHub e sobe a interface
4. O Space monta e publica sozinho. A URL fixa abre em qualquer aparelho.

> O `app.py` do Space baixa este repositório automaticamente (via tarball,
> sem depender de git), então você não precisa copiar o projeto todo — só
> os três arquivos da pasta `hf_space/`.

O `Dockerfile` na raiz continua disponível para hosts com Docker (Render,
Fly, etc.), mas para o Hugging Face grátis use o caminho Gradio acima.

## Salvar seu progresso (qualquer dispositivo)

Hosts gratuitos têm disco temporário, então o painel tem persistência
portátil embutida, no cartão **💾 Salvar / restaurar cérebro**:

- **⬇ Baixar cérebro** — salva o modelo treinado (`eva_cerebro.pkl`) no seu
  dispositivo.
- **⬆ Restaurar cérebro** — reenvia esse arquivo depois (em qualquer
  aparelho) para continuar de onde parou, sem treinar de novo.

Assim você treina quando quiser, de onde quiser, e leva a EVA com você.


## Uso por linha de comando

Requisito mínimo: `numpy` (e `pymupdf` para ler PDFs).

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

| preset   | parâmetros | camadas | contexto | velocidade relativa |
|----------|-----------:|:-------:|:--------:|:-------------------:|
| `nano`   |    ~100 mil |    2    |    48    |  relâmpago          |
| `small`  |    ~350 mil |    3    |    64    |  ~4x mais rápido    |
| `medium` |    ~1,8 mi  |    4    |    96    |  base               |
| `large`  |    ~4,8 mi  |    6    |   128    |  ~4x mais lento     |

Para experimentar rápido, use `nano` ou `small`. O painel web já vem com
`small` selecionado por padrão.

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
