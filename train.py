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
import time

import numpy as np

from eva import AdamW, CharTokenizer, GPT, GPTConfig, clip_grad_norm, no_grad

CKPT_PATH = "eva_checkpoint.pkl"

# Presets de arquitetura. "medium" e "large" são os "modelos maiores":
# aprendem estruturas mais ricas do corpus, ao custo de mais tempo de CPU.
PRESETS = {
    "small":  dict(n_layer=3, n_head=4, n_embd=96,  block_size=64,  batch_size=16),
    "medium": dict(n_layer=4, n_head=6, n_embd=192, block_size=96,  batch_size=16),
    "large":  dict(n_layer=6, n_head=8, n_embd=256, block_size=128, batch_size=12),
}


def get_batch(data: np.ndarray, block_size: int, batch_size: int, rng):
    """Amostra `batch_size` janelas (x, y) deslocadas de um caractere."""
    ix = rng.integers(0, len(data) - block_size - 1, size=batch_size)
    x = np.stack([data[i:i + block_size] for i in ix])
    y = np.stack([data[i + 1:i + 1 + block_size] for i in ix])
    return x, y


def save_checkpoint(model: GPT, tokenizer: CharTokenizer) -> None:
    params = [p.data for p in model.parameters()]
    with open(CKPT_PATH, "wb") as f:
        pickle.dump({"config": model.config, "params": params,
                     "chars": tokenizer.chars}, f)


def load_checkpoint():
    with open(CKPT_PATH, "rb") as f:
        blob = pickle.load(f)
    tokenizer = CharTokenizer(blob["chars"])
    model = GPT(blob["config"])
    for p, saved in zip(model.parameters(), blob["params"]):
        p.data = saved
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

    tokenizer = CharTokenizer.from_text(text)
    data = np.array(tokenizer.encode(text), dtype=np.int64)
    # Split de validação intercalado: reserva 1 de cada 10 blocos contíguos.
    # Como o corpus concatena livros distintos, um corte no fim isolaria um
    # único domínio; intercalar faz a val cobrir a mesma mistura do treino.
    chunk = args.block_size + 1
    blocks = [data[i:i + chunk] for i in range(0, len(data) - chunk, chunk)]
    train_data = np.concatenate([b for i, b in enumerate(blocks) if i % 10 != 0])
    val_data = np.concatenate([b for i, b in enumerate(blocks) if i % 10 == 0])
    print(f"Corpus: {len(text):,} caracteres, vocabulário: {tokenizer.vocab_size}")

    config = GPTConfig(vocab_size=tokenizer.vocab_size, block_size=args.block_size,
                       n_layer=args.n_layer, n_head=args.n_head, n_embd=args.n_embd)
    model = GPT(config)
    print(f"Modelo EVA ({args.preset}): {model.num_params():,} parâmetros\n")

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    rng = np.random.default_rng(args.seed)
    warmup = max(1, int(args.steps * 0.05))
    start = time.time()

    for step in range(1, args.steps + 1):
        # agendamento do learning rate: aquecimento linear e depois decaimento
        # cosseno até 10% do pico — estabiliza o início e refina o final.
        optimizer.lr = lr_schedule(step, args.lr, warmup, args.steps)

        x, y = get_batch(train_data, config.block_size, args.batch_size, rng)
        _, loss = model.forward(x, y)

        model.zero_grad()
        loss.backward()
        clip_grad_norm(model.parameters(), max_norm=1.0)
        optimizer.step()

        if step % args.log_every == 0 or step == 1:
            vloss = estimate_val_loss(model, val_data, config, args.batch_size, rng)
            elapsed = time.time() - start
            print(f"passo {step:5d}/{args.steps} | treino {float(loss.data):.4f} "
                  f"| val {vloss:.4f} | {elapsed:6.1f}s")

    save_checkpoint(model, tokenizer)
    print(f"\nCheckpoint salvo em {CKPT_PATH}\n")
    sample(model, tokenizer, prompt="A ", max_new_tokens=300, rng=rng)


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
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--block-size", type=int, default=None)
    parser.add_argument("--n-layer", type=int, default=None)
    parser.add_argument("--n-head", type=int, default=None)
    parser.add_argument("--n-embd", type=int, default=None)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--generate", metavar="PROMPT",
                        help="gera texto a partir do checkpoint salvo e sai")
    args = parser.parse_args()

    # Preenche a arquitetura pelo preset; flags explícitas têm prioridade.
    preset = PRESETS[args.preset]
    for key, value in preset.items():
        if getattr(args, key) is None:
            setattr(args, key, value)

    if args.generate is not None:
        if not os.path.exists(CKPT_PATH):
            raise SystemExit("Nenhum checkpoint encontrado. Treine primeiro.")
        model, tokenizer = load_checkpoint()
        sample(model, tokenizer, prompt=args.generate, max_new_tokens=300)
    else:
        train(args)


if __name__ == "__main__":
    main()
