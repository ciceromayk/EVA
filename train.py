"""Treina a EVA num corpus de texto e gera amostras.

Uso:
    python train.py                      # treina no data/corpus.txt
    python train.py --data meu.txt --steps 3000
    python train.py --generate "A EVA"   # só gera, usando checkpoint salvo

O modelo é minúsculo de propósito: roda em CPU, sem GPU, em poucos
minutos, e ainda assim aprende a estrutura básica do português do corpus.
"""

from __future__ import annotations

import argparse
import math
import os
import pickle
import sys
import time

# O dispositivo (CPU/GPU) precisa ser escolhido ANTES de importar `eva`,
# porque o backend de array é fixado no momento do import. Um pré-scan de
# --device (ou a variável EVA_DEVICE) resolve isso.
if "--device" in sys.argv:
    os.environ["EVA_DEVICE"] = sys.argv[sys.argv.index("--device") + 1]

import numpy as np  # noqa: E402  (numpy real, para preparar os dados na CPU)

from eva import AdamW, CharTokenizer, GPT, GPTConfig, SGDMomentum, clip_grad_norm, no_grad  # noqa: E402
from eva.backend import asnumpy, device_name, to_device  # noqa: E402
from eva.bpe import BPETokenizer  # noqa: E402

CKPT_PATH = "eva_checkpoint.pkl"

# Presets de arquitetura. Cada um sobe o tamanho do modelo — mais capacidade
# de aprender, ao custo de mais tempo/memória. "xlarge" (~300M parâmetros)
# é o teto realista para treinar por inteiro numa GPU de 8GB (ver README,
# seção "Quanto dá para treinar no seu hardware").
PRESETS = {
    "nano":   dict(n_layer=2,  n_head=2,  n_embd=64,   block_size=48,  batch_size=16),
    "small":  dict(n_layer=3,  n_head=4,  n_embd=96,   block_size=64,  batch_size=16),
    "medium": dict(n_layer=4,  n_head=6,  n_embd=192,  block_size=96,  batch_size=16),
    "large":  dict(n_layer=6,  n_head=8,  n_embd=256,  block_size=128, batch_size=12),
    "xlarge": dict(n_layer=24, n_head=16, n_embd=1024, block_size=256, batch_size=4),
}


def get_batch(data: np.ndarray, block_size: int, batch_size: int, rng):
    """Amostra `batch_size` janelas (x, y) deslocadas de um caractere."""
    ix = rng.integers(0, len(data) - block_size - 1, size=batch_size)
    x = np.stack([data[i:i + block_size] for i in ix])
    y = np.stack([data[i + 1:i + 1 + block_size] for i in ix])
    return x, y


def serialize_tokenizer(tokenizer) -> dict:
    """Empacota qualquer tokenizador (char ou BPE) para salvar no checkpoint."""
    if isinstance(tokenizer, BPETokenizer):
        merges = [[a, b, nid] for (a, b), nid in tokenizer.merges.items()]
        return {"kind": "bpe", "merges": merges}
    return {"kind": "char", "chars": tokenizer.chars}


def deserialize_tokenizer(blob: dict):
    if blob.get("kind") == "bpe":
        merges = {(a, b): nid for a, b, nid in blob["merges"]}
        return BPETokenizer(merges)
    return CharTokenizer(blob["chars"])


def save_checkpoint(model: GPT, tokenizer) -> None:
    # Salva os pesos como numpy (na CPU), para o arquivo funcionar em qualquer
    # máquina, com ou sem GPU.
    params = [asnumpy(p.data) for p in model.parameters()]
    with open(CKPT_PATH, "wb") as f:
        pickle.dump({"config": model.config, "params": params,
                     "tokenizer": serialize_tokenizer(tokenizer)}, f)


def load_checkpoint():
    with open(CKPT_PATH, "rb") as f:
        blob = pickle.load(f)
    # compatível com checkpoints antigos que salvavam só "chars"
    tok_blob = blob.get("tokenizer") or {"kind": "char", "chars": blob["chars"]}
    tokenizer = deserialize_tokenizer(tok_blob)
    model = GPT(blob["config"])
    for p, saved in zip(model.parameters(), blob["params"]):
        p.data = to_device(saved)  # leva os pesos para o dispositivo atual
    return model, tokenizer


def lr_schedule(step, peak_lr, warmup, total, min_ratio=0.1):
    """Warmup linear seguido de decaimento cosseno até min_ratio*peak_lr."""
    if step < warmup:
        return peak_lr * step / warmup
    progress = (step - warmup) / max(1, total - warmup)
    cosine = 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
    return peak_lr * (min_ratio + (1.0 - min_ratio) * cosine)


def train(args) -> None:
    with open(args.data, encoding="utf-8") as f:
        text = f.read()

    if args.resume:
        if not os.path.exists(CKPT_PATH):
            raise SystemExit("Nenhum checkpoint encontrado para continuar. "
                              "Treine do zero primeiro (sem --resume).")
        model, tokenizer = load_checkpoint()
        config = model.config
        print("Continuando treino do checkpoint existente "
              "(arquitetura, tokenizer e dropout vêm do arquivo salvo; "
              "--preset/--tokenizer/--bpe-vocab/--dropout são ignorados)")
    else:
        if args.tokenizer == "bpe":
            print(f"Treinando tokenizador BPE (vocab {args.bpe_vocab})...")
            tokenizer = BPETokenizer.train(text, vocab_size=args.bpe_vocab, verbose=True)
        else:
            tokenizer = CharTokenizer.from_text(text)
        config = GPTConfig(vocab_size=tokenizer.vocab_size, block_size=args.block_size,
                           n_layer=args.n_layer, n_head=args.n_head, n_embd=args.n_embd,
                           dropout=args.dropout)
        model = GPT(config)

    # A EVA sempre reencoda com o tokenizer DELA (novo ou carregado do
    # checkpoint) — nunca com um tokenizer recém-criado sobre o corpus
    # atual, para não descasar do vocabulário que o modelo já conhece.
    data = np.array(tokenizer.encode(text), dtype=np.int64)
    # Split de validação intercalado: reserva 1 de cada 10 blocos contíguos.
    # Como o corpus concatena livros distintos, um corte no fim isolaria um
    # único domínio; intercalar faz a val cobrir a mesma mistura do treino.
    chunk = config.block_size + 1
    blocks = [data[i:i + chunk] for i in range(0, len(data) - chunk, chunk)]
    train_data = np.concatenate([b for i, b in enumerate(blocks) if i % 10 != 0])
    val_data = np.concatenate([b for i, b in enumerate(blocks) if i % 10 == 0])
    tok_kind = "bpe" if isinstance(tokenizer, BPETokenizer) else "char"
    print(f"Corpus: {len(text):,} caracteres -> {len(data):,} tokens "
          f"({tok_kind}, vocabulário {tokenizer.vocab_size})")

    if args.resume:
        print(f"Modelo EVA (continuado): {model.num_params():,} parâmetros | {device_name()}\n")
    else:
        print(f"Modelo EVA ({args.preset}): {model.num_params():,} parâmetros "
              f"| dropout {args.dropout} | {device_name()}\n")

    if args.optimizer == "sgd":
        optimizer = SGDMomentum(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.01)
    else:
        optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    rng = np.random.default_rng(args.seed)
    warmup = max(1, int(args.steps * 0.05))

    # Ao continuar, nunca deixamos o checkpoint regredir: medimos a
    # qualidade do modelo carregado ANTES de treinar mais, e só
    # sobrescrevemos se um passo futuro bater essa marca de verdade.
    if args.resume:
        best_val = estimate_val_loss(model, val_data, config, args.batch_size, rng)
        print(f"Val loss do checkpoint carregado: {best_val:.4f} (referência a bater)\n")
    else:
        best_val = float("inf")

    start = time.time()
    last_print = start
    HEARTBEAT_SECS = 15  # em modelos lentos, avisa que está vivo mesmo
                          # entre logs "oficiais" (que fazem validação)

    for step in range(1, args.steps + 1):
        # agendamento do learning rate: aquecimento linear e depois decaimento
        # cosseno até 10% do pico — estabiliza o início e refina o final.
        optimizer.lr = lr_schedule(step, args.lr, warmup, args.steps)

        x, y = get_batch(train_data, config.block_size, args.batch_size, rng)
        logits, loss = model.forward(x, y)
        del logits  # não é usado no treino; descarta cedo

        model.zero_grad()
        loss.backward()
        clip_grad_norm(model.parameters(), max_norm=1.0)
        optimizer.step()
        loss_value = float(loss.data)
        # Solta o grafo desta iteração ANTES da próxima forward. Sem isso, a
        # variável `loss` continua viva até a linha de cima ser executada de
        # novo, então por um instante duas iterações têm o grafo retido ao
        # mesmo tempo — em modelos grandes (ex.: xlarge) isso quase dobra o
        # pico de memória e pode estourar a RAM/VRAM.
        del loss

        now = time.time()
        if step % args.log_every == 0 or step == 1:
            vloss = estimate_val_loss(model, val_data, config, args.batch_size, rng)
            elapsed = now - start
            # Early stopping: guarda o checkpoint de MENOR val loss, não o
            # último — assim o overfitting no fim não estraga o resultado.
            best = ""
            if vloss < best_val:
                best_val = vloss
                save_checkpoint(model, tokenizer)
                best = "  <- melhor (salvo)"
            print(f"passo {step:5d}/{args.steps} | treino {loss_value:.4f} "
                  f"| val {vloss:.4f} | {elapsed:6.1f}s{best}")
            last_print = now
        elif now - last_print >= HEARTBEAT_SECS:
            # Passou muito tempo real sem uma linha "oficial" (comum em
            # modelos grandes/lentos, onde log-every passos podem levar
            # dezenas de minutos) — avisa que está vivo, sem gastar tempo
            # com validação nem checkpoint.
            print(f"passo {step:5d}/{args.steps} | treino {loss_value:.4f} "
                  f"| ... (em andamento) | {now - start:6.1f}s")
            last_print = now

    print(f"\nMelhor val loss: {best_val:.4f} | checkpoint em {CKPT_PATH}\n")
    # amostra usando o melhor modelo salvo (não o último, possivelmente overfit)
    best_model, tokenizer = load_checkpoint()
    sample(best_model, tokenizer, prompt="A ", max_new_tokens=300, rng=rng)


def estimate_val_loss(model, val_data, config, batch_size, rng, iters=5):
    """Loss média em janelas de validação, sem construir grafo de gradiente."""
    if len(val_data) <= config.block_size + 1:
        return float("nan")
    with no_grad():
        total = 0.0
        for _ in range(iters):
            x, y = get_batch(val_data, config.block_size, batch_size, rng)
            _, loss = model.forward(x, y)
            total += float(loss.data)
    return total / iters


def sample(model, tokenizer, prompt, max_new_tokens, rng=None):
    rng = rng or np.random.default_rng()
    context = np.array([tokenizer.encode(prompt) or [0]], dtype=np.int64)
    out = model.generate(context, max_new_tokens=max_new_tokens,
                         temperature=0.8, top_k=10, rng=rng)
    print("--- amostra gerada ---")
    print(tokenizer.decode(out[0]))
    print("----------------------")


def main():
    parser = argparse.ArgumentParser(description="Treina/roda a EVA")
    parser.add_argument("--data", default="data/corpus.txt")
    parser.add_argument("--preset", choices=list(PRESETS), default="medium",
                        help="tamanho do modelo (padrão: medium)")
    parser.add_argument("--tokenizer", choices=["char", "bpe"], default="char",
                        help="char (1 token/letra) ou bpe (subpalavras)")
    parser.add_argument("--bpe-vocab", type=int, default=512,
                        help="tamanho do vocabulário BPE (>= 256)")
    parser.add_argument("--dropout", type=float, default=0.1,
                        help="taxa de dropout (regularização; 0 desliga)")
    parser.add_argument("--device", choices=["cpu", "gpu"], default="cpu",
                        help="cpu (NumPy) ou gpu (CuPy/CUDA)")
    parser.add_argument("--resume", action="store_true",
                        help="continua treinando o checkpoint existente em vez "
                             "de começar um modelo novo (mantém arquitetura/tokenizer salvos)")
    parser.add_argument("--optimizer", choices=["adamw", "sgd"], default="adamw",
                        help="adamw (padrão, 16 bytes/param) ou sgd (momentum, "
                             "12 bytes/param — ~25%% mais leve, cabe modelo maior)")
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--block-size", type=int, default=None)
    parser.add_argument("--n-layer", type=int, default=None)
    parser.add_argument("--n-head", type=int, default=None)
    parser.add_argument("--n-embd", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None,
                        help="padrão: 3e-3 (adamw) ou 5e-2 (sgd)")
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--generate", metavar="PROMPT",
                        help="gera texto a partir do checkpoint salvo e sai")
    parser.add_argument("--max-new", type=int, default=300,
                        help="tokens a gerar no modo --generate")
    args = parser.parse_args()

    # Preenche a arquitetura pelo preset; flags explícitas têm prioridade.
    preset = PRESETS[args.preset]
    for key, value in preset.items():
        if getattr(args, key) is None:
            setattr(args, key, value)
    if args.lr is None:
        args.lr = 5e-2 if args.optimizer == "sgd" else 3e-3

    if args.generate is not None:
        if not os.path.exists(CKPT_PATH):
            raise SystemExit("Nenhum checkpoint encontrado. Treine primeiro.")
        model, tokenizer = load_checkpoint()
        sample(model, tokenizer, prompt=args.generate, max_new_tokens=args.max_new)
    else:
        train(args)


if __name__ == "__main__":
    main()
