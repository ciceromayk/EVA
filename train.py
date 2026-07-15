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
import os
import pickle

import numpy as np

from eva import AdamW, CharTokenizer, GPT, GPTConfig, clip_grad_norm

CKPT_PATH = "eva_checkpoint.pkl"


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


def train(args) -> None:
    with open(args.data, encoding="utf-8") as f:
        text = f.read()

    tokenizer = CharTokenizer.from_text(text)
    data = np.array(tokenizer.encode(text), dtype=np.int64)
    print(f"Corpus: {len(text)} caracteres, vocabulário: {tokenizer.vocab_size}")

    config = GPTConfig(vocab_size=tokenizer.vocab_size, block_size=args.block_size,
                       n_layer=args.n_layer, n_head=args.n_head, n_embd=args.n_embd)
    model = GPT(config)
    print(f"Modelo EVA: {model.num_params():,} parâmetros")

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    rng = np.random.default_rng(args.seed)

    for step in range(1, args.steps + 1):
        x, y = get_batch(data, config.block_size, args.batch_size, rng)
        _, loss = model.forward(x, y)

        model.zero_grad()
        loss.backward()
        clip_grad_norm(model.parameters(), max_norm=1.0)
        optimizer.step()

        if step % args.log_every == 0 or step == 1:
            print(f"passo {step:5d}/{args.steps} | loss {float(loss.data):.4f}")

    save_checkpoint(model, tokenizer)
    print(f"\nCheckpoint salvo em {CKPT_PATH}\n")
    sample(model, tokenizer, prompt="A ", max_new_tokens=200, rng=rng)


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
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument("--n-layer", type=int, default=3)
    parser.add_argument("--n-head", type=int, default=4)
    parser.add_argument("--n-embd", type=int, default=96)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--generate", metavar="PROMPT",
                        help="gera texto a partir do checkpoint salvo e sai")
    args = parser.parse_args()

    if args.generate is not None:
        if not os.path.exists(CKPT_PATH):
            raise SystemExit("Nenhum checkpoint encontrado. Treine primeiro.")
        model, tokenizer = load_checkpoint()
        sample(model, tokenizer, prompt=args.generate, max_new_tokens=300)
    else:
        train(args)


if __name__ == "__main__":
    main()
