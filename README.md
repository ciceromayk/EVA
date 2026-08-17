# EVA — uma IA construída do zero

**EVA** é um modelo de linguagem (um *Transformer* estilo **Llama**)
implementado inteiramente do zero em **NumPy puro** — sem PyTorch, sem
TensorFlow. O objetivo é educacional: mostrar, peça por peça, como uma IA
moderna que gera texto realmente funciona por dentro.

Tudo o que faz uma rede neural aprender está aqui e é legível:

- **Diferenciação automática** (backpropagation) escrita à mão
- **Atenção multi-cabeça causal com RoPE** — posições rotacionais, como no Llama
- **RMSNorm**, **QK-Norm** e **MLP SwiGLU** — os mesmos blocos das versões
  mais recentes do Llama, sem bias
- **Amostragem estilo Llama/GPT-3**: temperatura, top-k, **nucleus (top-p)**
  e penalidade de repetição
- **Tokenizador BPE (subpalavras) por padrão** — como todo LLM moderno
- **Otimizador AdamW** e recorte de gradiente
- Laço de **treino** completo, com checkpoint, retomada e amostragem

## Estrutura

```
eva/
  backend.py          seletor de array: NumPy (CPU) ou CuPy (GPU)
  autograd.py         motor de autograd (Tensor + backpropagation)
  nn.py               camadas: Linear, RMSNorm, atenção com RoPE, SwiGLU, Block
  model.py            o modelo completo (arquitetura Llama) + geração de texto
  optim.py            otimizador AdamW e clip de gradiente
  tokenizer.py        tokenizador em nível de caractere
app.py                painel web (http.server) para uso local / Render / Docker
chat.py               Sala de Conversa: interface de uso com streaming token a token
conversar.bat         abre a Sala de Conversa no Windows (clique duplo)
criar_atalhos.bat     cria atalhos da EVA na área de trabalho (Windows)
train.py              script de treino e amostragem (com presets)
check_gpu.py          autoteste de GPU (CPU vs CuPy)
servidor.bat          sobe a EVA com senha, pronta para acesso remoto
instalar_inicializacao.bat   liga o servidor sozinho com o Windows
tools/build_corpus.py extrai texto de PDFs/TXT para montar o corpus
tools/fetch_online_corpus.py busca texto da Wikipédia / Project Gutenberg
data/corpus.txt       corpus de treino (gerado a partir de PDFs)
tests/                autograd, BPE, curadoria do corpus, treino incremental e vazamento de memória
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
- **`conversar.bat`** — abre a Sala de Conversa (usar o modelo treinado).
- **`atualizar.bat`** — baixa a versão mais recente do GitHub (`git pull`).
- **`criar_atalhos.bat`** — coloca atalhos "EVA · Conversa" e "EVA · Painel"
  na sua área de trabalho (rode uma vez).

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

## Sala de Conversa (interface de USO do modelo)

Depois de treinar, use a interface dedicada a conversar com a EVA — mais
rápida que o botão "Gerar" do painel, porque o cérebro é carregado **uma
única vez** na memória e o texto surge **token a token**, ao vivo, conforme
sai da rede:

```bash
python chat.py           # abre em http://localhost:8001
```

No Windows, é só clicar duas vezes em **`conversar.bat`**.

- **Ficha do cérebro** — parâmetros, camadas, contexto e tokenizador do
  checkpoint atual (recarrega sozinho se você treinar de novo)
- **Regulagem** — temperatura (ousadia), top-k, **top-p/nucleus** e
  **anti-repetição**, além da quantidade de tokens
- **Streaming de verdade** — cada token aparece assim que é amostrado;
  dá para interromper no meio com "Parar"

Lembre: a EVA é um modelo de **continuação** — ela prolonga o texto que
você começar, no estilo do corpus em que treinou (não segue instruções
como um chat assistente). Para exigir senha ao acessar de outra máquina,
defina `EVA_PASSWORD` (usuário `eva`), como no painel.

## Treino incremental (continuar o cérebro salvo)

Por padrão, cada treino começa um modelo **novo**, do zero. Marcando
**🔄 Continuar do cérebro salvo** no cartão "Treinar a mente" (ou passando
`--resume` no `train.py`), a EVA carrega o checkpoint existente e continua
treinando **a partir dele**, em vez de recomeçar — a arquitetura, o
tokenizer e o dropout usados são os do checkpoint (as opções de
preset/percepção/dropout ficam desativadas nesse modo; só "Ciclos" e
"Processador" continuam valendo).

Isso é o que permite **excluir os documentos depois de treinar** sem medo:
o conhecimento já fica todo nos pesos do modelo. Se um dia você quiser
ensinar mais coisas à mesma EVA, é só adicionar o material novo e marcar
"Continuar" — não precisa manter os documentos antigos por perto.

Por segurança, o treino incremental nunca piora o que já existe: antes de
começar, ele mede a qualidade do checkpoint carregado e só sobrescreve o
arquivo se um passo novo bater essa marca de verdade.

```bash
python train.py --resume --steps 1000 --device gpu
```

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
  nvidia-cufft-cu12 nvidia-cuda-nvrtc-cu12 nvidia-nvjitlink-cu12
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

## Transformar seu PC num servidor (treinar de qualquer lugar)

Com uma GPU no PC, o mais poderoso é usar a própria máquina como servidor:
liga em casa, treina/conversa de qualquer aparelho (iPad, celular, outro
PC), sem depender de nuvem gratuita e seus limites. O caminho recomendado
é o **Tailscale** — cria uma rede privada só entre os SEUS aparelhos, sem
abrir portas no roteador e sem expor o PC para a internet toda.

### 1. Instalar o Tailscale

- No **PC** (o servidor): baixe em **https://tailscale.com/download** e
  faça login (Google/Microsoft/GitHub — o que for mais rápido).
- No **iPad/celular** (quem vai acessar): instale o app **Tailscale** na
  App Store / Play Store e faça login **com a mesma conta**.

Pronto — os dois aparelhos agora enxergam um ao outro com segurança,
mesmo em redes diferentes (casa, 4G, wi-fi de outro lugar).

### 2. Ligar o servidor da EVA

Em vez do `iniciar.bat`, use:

```
servidor.bat
```

Na primeira vez, ele **gera uma senha aleatória** (mostrada na tela e
salva em `senha_servidor.txt`, que não vai para o GitHub) e some o
endereço Tailscale do PC.

### 3. Acessar de qualquer lugar

No navegador do iPad/celular, entre em:

```
http://<endereco-tailscale-do-pc>:8000
```

O endereço aparece ao rodar o `servidor.bat` (ex.: `100.x.y.z`), ou
descubra a qualquer momento com `tailscale ip -4`. Ao abrir, o navegador
vai pedir usuário e senha — use **eva** e a senha mostrada pelo script.

### 4. (Opcional) Ligar sozinho com o Windows

Para o PC virar servidor de verdade — sem precisar clicar em nada toda
vez — rode **uma vez**:

```
instalar_inicializacao.bat
```

Isso faz o `servidor.bat` iniciar automaticamente quando o Windows liga.
Para desfazer, rode `desinstalar_inicializacao.bat`.

Vale também impedir o PC de dormir sozinho: **Configurações → Sistema →
Energia e bateria → Tela e suspensão → "Nunca"** (pelo menos enquanto for
usar como servidor).

### Segurança em resumo

- O painel só responde a login/senha quando `EVA_PASSWORD` está definida
  (é o que o `servidor.bat` faz). Sem senha, ele roda aberto — use
  `iniciar.bat` (sem senha) só em `localhost`, nunca exposto à rede.
- O Tailscale mantém o acesso restrito aos aparelhos logados na sua conta
  — ninguém de fora alcança o painel, mesmo sabendo o endereço.
- Se algum dia quiser trocar a senha, apague `senha_servidor.txt` e rode
  o `servidor.bat` de novo (ele gera outra).

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

O `Dockerfile` na raiz continua disponível para hosts com Docker (Render,
Fly, etc.).

## Salvar seu progresso (qualquer dispositivo)

Hosts gratuitos têm disco temporário, então o painel tem persistência
portátil embutida, no cartão **💾 Salvar / restaurar cérebro**:

- **⬇ Baixar cérebro** — salva o modelo treinado (`eva_cerebro.pkl`) no seu
  dispositivo.
- **⬆ Restaurar cérebro** — reenvia esse arquivo depois (em qualquer
  aparelho) para continuar de onde parou, sem treinar de novo.

Assim você treina quando quiser, de onde quiser, e leva a EVA com você.

## Buscar conhecimento online (Wikipédia / Project Gutenberg)

Além de PDFs e texto colado, o painel busca material direto da internet,
no cartão **🌐 Buscar conhecimento online**:

- **Wikipédia** — baixa o texto puro de artigos por título (aceita vários
  separados por vírgula: `Inteligência artificial,Redes neurais,Sun Tzu`).
- **Project Gutenberg** — busca por autor/título e baixa até 5 livros de
  domínio público em texto puro (ex.: `Machado de Assis`).

O texto baixado vira `.txt` em `materials/` e entra no corpus automaticamente,
como qualquer outro material. Funciona sem nenhuma biblioteca extra —
só chamadas HTTP simples (`urllib`, da biblioteca padrão).

Também dá para usar por linha de comando:

```bash
python tools/fetch_online_corpus.py --wikipedia "Inteligência artificial,Redes neurais"
python tools/fetch_online_corpus.py --gutenberg "Machado de Assis" --max-books 3
```

**Atenção ao tamanho**: esses corpora online podem ser bem maiores que os
PDFs que você já usou. Comece com poucos artigos/livros por vez e
acompanhe o crescimento do corpus no cartão "Memória da EVA" antes de
adicionar muito de uma vez — corpus grande demais exige bem mais passos
de treino para não ficar raso.

### Curadoria: deduplicação automática

Ao juntar várias fontes (PDFs, texto colado, downloads online), é comum
algum parágrafo aparecer repetido — o mesmo prefácio em dois PDFs do
mesmo livro, um trecho citado inteiro em outro artigo, etc. O
`tools/build_corpus.py` remove essas duplicatas automaticamente ao
montar o corpus (comparando parágrafos com 40+ caracteres, ignorando
maiúsculas/espaçamento), e avisa quantas removeu:

```bash
python tools/build_corpus.py materials/*.pdf materials/*.txt -o data/corpus.txt
# ... 3 parágrafo(s) duplicado(s) removido(s) entre as fontes
```

Parágrafos curtos (títulos, diálogos de uma linha) ficam de fora da
checagem — eles repetem naturalmente e não são "lixo" de duplicação.

### Sugestão: expandir mantendo o tema do corpus

O corpus incluído mistura estratégia militar, algoritmos e modelos de
linguagem — misturar assuntos aleatórios dilui isso. Para crescer *nessa
mesma linha* (dá pra rodar o comando abaixo na sua máquina, com
internet):

```bash
python tools/fetch_online_corpus.py --wikipedia "Estratégia militar,Sun Tzu,Teoria dos jogos,Aprendizado de máquina,Rede neural artificial,Processamento de linguagem natural,Estrutura de dados,Complexidade de algoritmos"
python tools/fetch_online_corpus.py --gutenberg "Sun Tzu" --max-books 2
python tools/build_corpus.py materials/*.pdf materials/*.txt -o data/corpus.txt
```

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

| preset    | parâmetros | camadas | contexto | ~tempo p/ 1500 passos (CPU) |
|-----------|-----------:|:-------:|:--------:|:----------------------------:|
| `nano`    |    ~118 mil |    2    |    48    |  ~1 min (teste rápido)       |
| `small`   |    ~826 mil |    4    |    96    |  ~20 min                     |
| `medium`  |    ~3,1 mi  |    5    |    96    |  ~50 min (base)              |
| `large`   |    ~8,7 mi  |    7    |   160    |  ~2,5 h                      |
| `xlarge`  |    ~303 mi  |   24    |   256    |  exige GPU com ~6-8GB+       |

Os tempos são uma referência medida em CPU comum (4 núcleos); variam com
sua máquina. Para experimentar rápido, use `nano` ou `small`. O painel web
já vem com `small` selecionado por padrão.

No painel, o cartão **📐 Resumo do modelo** mostra ao vivo — conforme você
troca o preset/percepção — os parâmetros, arquitetura, contexto,
vocabulário, tamanho em disco e VRAM estimada daquela escolha (badge
"estimativa"). Ao marcar **🔄 Continuar do cérebro salvo**, o cartão troca
para os números **reais** do checkpoint (badge "cérebro salvo"), já que
nesse modo é a arquitetura salva que manda, não o preset selecionado.

```bash
python train.py --preset large --steps 2000
```

Também dá para sobrescrever qualquer dimensão individual
(`--n-layer`, `--n-head`, `--n-embd`, `--block-size`, `--batch-size`).

### Quanto dá para treinar no seu hardware (e por que 1 bilhão não cabe)

Treinar (diferente de só *rodar*) um modelo em fp32 exige guardar, **por
parâmetro**: o peso, o gradiente, e o estado do otimizador.

- **AdamW** (padrão): peso + gradiente + 2 momentos = **16 bytes/parâmetro**
- **SGD+momentum** (`--optimizer sgd`): peso + gradiente + 1 buffer =
  **12 bytes/parâmetro** (~25% mais leve — cabe modelo maior na mesma GPU)

Para **1 bilhão de parâmetros**, isso são **16 GB só de peso+gradiente+
otimizador**, em fp32 — antes de contar as ativações. Numa GPU de 8GB, **1
bi não cabe pra treinar em nenhuma combinação de otimizações de código**:
mesmo em fp16 puro (arriscado, tende a divergir sem os truques que
frameworks como PyTorch usam), ainda são ~8GB só de estado do otimizador,
sem sobrar nada para ativações. Isso é física de memória, não é algo que
se resolve otimizando o autograd.

**Teto realista para treinar 100% local** numa GPU de 8GB: o preset
`xlarge` (~304 milhões de parâmetros) usa cerca de 5,6GB com AdamW ou
4,3GB com SGD+momentum — cabe com folga. Empurrar além disso (rumo a 1 bi)
esbarra na parede de memória acima.

**Se 1 bilhão for mesmo a meta**: o caminho é treinar numa GPU maior
alugada na nuvem (ex.: uma A100/H100, só para o treino), e depois baixar o
checkpoint para rodar localmente. **Inferência** (gerar texto, sem
gradiente/otimizador) de 1 bi de parâmetros precisa de só ~4GB — isso
**cabe tranquilo** numa RTX 5060. O gargalo é treinar, não conversar.

O cartão **📐 Resumo do modelo** no painel já mostra a VRAM estimada antes
de você clicar em treinar — se aparecer um número maior que sua placa,
é sinal para escolher um preset menor ou trocar para SGD+momentum.

### Tokenizador e regularização

Desde a migração para Llama, **BPE (subpalavras) é o tokenizador padrão**
— é assim que o Llama real funciona (e todo LLM moderno): tokens que
representam pedaços de palavra, não letra por letra. Um token de contexto
carrega muito mais texto, e o modelo aprende a gerar *palavras inteiras*
em vez de soletrar.

```bash
# padrão atual: BPE com dropout — gera palavras inteiras, sem soletrar
python train.py --bpe-vocab 512 --dropout 0.2 --steps 2000

# para voltar ao tokenizador de caractere (mais simples de inspecionar)
python train.py --tokenizer char --steps 2000
```

- `--tokenizer {bpe,char}`: subpalavras (padrão, como o Llama) ou caractere
- `--bpe-vocab N`: tamanho do vocabulário BPE (≥ 256)
- `--dropout P`: taxa de dropout (regularização; padrão 0,1)
- `--optimizer {adamw,sgd}`: AdamW (padrão, converge melhor) ou SGD+momentum
  (~25% mais leve em memória — ver seção acima sobre limites de hardware)

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

### Arquitetura atual (Llama: RoPE + RMSNorm + SwiGLU)

Treinando o preset `medium` **recalibrado** (~3,1 mi de parâmetros) por
1500 passos no mesmo corpus de ~231 mil caracteres, em CPU comum:

```
passo    1/1500 | treino 5.49 | val 5.26
passo  700/1500 | treino 1.64 | val 1.80
passo 1500/1500 | treino 1.35 | val 1.54
```

Tempo total: **~38 min**. Val loss final **1,54** — melhor que o treino
histórico abaixo (val 1,57), com um preset menor (3,1 mi vs. o teto de
memória de então) e menos passos, graças à arquitetura Llama.

Amostra gerada (continuação livre, sem prompt específico):

> A suas envocainidos se meia de terreno de enviança em três de 2023.
> Consultado em 23 de junho de 2026. Cópia arquivada em 15 de jenho de
> 2023 (https://web.archive.org/web/...) 15. Le, Johan; Albo, Defford
> (2025). «Art overre and Multimodal Intelligence...»

Chama atenção o modelo já reproduzir a *forma* de uma referência
bibliográfica (data, "Consultado em", "Cópia arquivada em", DOI/URL) —
esse padrão vem do material sobre modelos de linguagem no corpus, que
tem muitas notas de rodapé nesse formato. É esperado num modelo desse
tamanho: aprende a estrutura superficial antes do significado.

### Registro histórico (arquitetura GPT, antes da migração)

> Primeiro treino bem-sucedido do projeto, com a arquitetura **GPT**
> original e o preset `medium` **da época** (~1,8 mi de parâmetros — bem
> menor que o `medium` atual). Mantido aqui como referência de que a
> mecânica sempre funcionou de ponta a ponta, mesmo antes da migração
> para Llama.

Treinando o preset `medium` (~1,8 mi de parâmetros, arquitetura GPT) por
2000 passos no corpus de ~231 mil caracteres (4 livros), a EVA sai de
texto aleatório para português reconhecível:

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

1. **Tokenização** — o texto vira uma sequência de inteiros (por padrão,
   pedaços de palavra via **BPE** — subpalavras, como no Llama real).
2. **Embeddings** — cada token vira um vetor; a *posição* não é somada ao
   embedding: ela entra na atenção via **RoPE** (rotação de Q e K), como no
   Llama.
3. **Blocos Transformer (estilo Llama)** — cada bloco tem *atenção causal*
   com **QK-Norm** (estabiliza Q e K antes do produto escalar), seguida de
   uma rede feed-forward **SwiGLU**, ambas com conexões residuais e
   pré-**RMSNorm**.
4. **Cabeça de saída** — projeta para o vocabulário; o *softmax* dá a
   probabilidade do próximo token.
5. **Treino** — a *entropia cruzada* mede o erro; o **autograd** calcula os
   gradientes por backpropagation; o **AdamW** ajusta os pesos.
6. **Geração** — amostra-se um token de cada vez, realimentando o modelo;
   temperatura, top-k, **top-p (nucleus)** e penalidade de repetição
   controlam a ousadia e evitam loops, como no Llama/GPT-3.

O preset padrão (`medium`) tem ~3,1 milhões de parâmetros e roda em CPU.

## Nota

Este é um projeto didático. Modelos de produção usam as mesmas ideias em
escala muito maior (bilhões de parâmetros, tokenização por subpalavras,
kernels em GPU), mas a mecânica fundamental é exatamente esta.
