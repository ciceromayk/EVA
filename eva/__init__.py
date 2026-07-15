"""EVA — uma IA (modelo de linguagem) construída do zero em NumPy puro.

Pacote com todas as peças de um Transformer decoder estilo GPT:

- `autograd`  : motor de diferenciação automática (backpropagation)
- `nn`        : camadas (Linear, LayerNorm, atenção, MLP, blocos)
- `model`     : o modelo GPT completo, com geração de texto
- `optim`     : otimizador AdamW e clipping de gradiente
- `tokenizer` : tokenizador em nível de caractere
"""

from .autograd import Tensor, cross_entropy, no_grad, softmax
from .model import GPT, GPTConfig
from .optim import AdamW, clip_grad_norm
from .tokenizer import CharTokenizer

__all__ = [
    "Tensor",
    "cross_entropy",
    "no_grad",
    "softmax",
    "GPT",
    "GPTConfig",
    "AdamW",
    "clip_grad_norm",
    "CharTokenizer",
]
