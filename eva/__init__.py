"""EVA — uma IA (modelo de linguagem) construída do zero em NumPy puro.

Pacote com todas as peças de um Transformer decoder estilo Llama:

- `autograd`  : motor de diferenciação automática (backpropagation)
- `nn`        : camadas (Linear, RMSNorm, atenção com RoPE, MLP SwiGLU, blocos)
- `model`     : o modelo completo (arquitetura Llama), com geração de texto
- `optim`     : otimizadores (AdamW, SGDMomentum) e clipping de gradiente
- `tokenizer` : tokenizador em nível de caractere
- `bpe`       : tokenizador BPE (subpalavras), estilo GPT
"""

from .autograd import Tensor, cross_entropy, no_grad, softmax
from .bpe import BPETokenizer
from .model import GPT, GPTConfig
from .optim import AdamW, SGDMomentum, clip_grad_norm
from .tokenizer import CharTokenizer

__all__ = [
    "Tensor",
    "cross_entropy",
    "no_grad",
    "softmax",
    "GPT",
    "GPTConfig",
    "AdamW",
    "SGDMomentum",
    "clip_grad_norm",
    "CharTokenizer",
    "BPETokenizer",
]
