"""Experiment 04: The KV-Cache as an In-Memory Data Structure (Prefill vs Decode).

Mental model for Software Devs:
Autoregressive generation produces one token at a time, appending each new token to the sequence.
- Naive recomputation: at step `t`, re-run the whole sequence through every block. Every previous
  Key and Value vector is recomputed from scratch even though the causal mask guarantees past Keys
  and Values never change.
- `Prefill` + `KV-Cache Decode`:
  1. `Prefill`: process the prompt (`T_prompt` tokens) in parallel and store its `K` and `V`
     tensors in a pre-allocated static buffer `(L, B, H_kv, max_seq_len, D_h)`.
  2. `Decode`: for each new token, compute only `q_new, k_new, v_new`, write `k_new, v_new` at
     index `pos` with `jax.lax.dynamic_update_slice` and attend over the cached keys.

What it costs: the write costs the same at any position, but the new query still attends over
every cached key, so each decode step grows linearly with the context. The cache removes the
recomputation of the prefix; it does not make a token O(1). In this file the buffer has a static
size (jit needs static shapes) and a mask hides the empty slots, so every decode step reads all
`max_seq_len` slots, and the naive baseline recomputes the full padded window at every step.
`decode_step_ms_by_cache_len` times one decode step with larger buffers to show the growth.

The model is a toy: a 4-layer decoder with random weights, GQA with 4 query and 2 K/V heads,
RMSNorm, a plain GELU MLP (no GeGLU gate), no RoPE and a tied output projection. Both loops are
compiled with `@jax.jit` inside `lax.scan`, so the speedup measures the algorithm. Times are
medians after a warmup call.
"""

from __future__ import annotations

import statistics
import time
from collections.abc import Callable
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np


class KVCache(NamedTuple):
    k: jax.Array  # (num_layers, batch, num_kv_heads, max_seq_len, head_dim)
    v: jax.Array  # (num_layers, batch, num_kv_heads, max_seq_len, head_dim)


def init_tiny_gemma_weights(
    key: jax.Array,
    num_layers: int = 4,
    vocab_size: int = 256,
    d_model: int = 128,
    num_q_heads: int = 4,
    num_kv_heads: int = 2,
    head_dim: int = 32,
) -> dict[str, jax.Array]:
    """Initialize a toy multi-layer decoder (random weights) with GQA and tied output projection."""
    k_emb, k_wq, k_wk, k_wv, k_wo, k_w1, k_w2 = jax.random.split(key, 7)
    scale = 1.0 / jnp.sqrt(jnp.float32(d_model))
    return {
        "embedding": jax.random.normal(k_emb, (vocab_size, d_model)) * scale,
        "w_q": jax.random.normal(k_wq, (num_layers, d_model, num_q_heads, head_dim)) * scale,
        "w_k": jax.random.normal(k_wk, (num_layers, d_model, num_kv_heads, head_dim)) * scale,
        "w_v": jax.random.normal(k_wv, (num_layers, d_model, num_kv_heads, head_dim)) * scale,
        "w_o": jax.random.normal(k_wo, (num_layers, num_q_heads, head_dim, d_model)) * scale,
        "w_gate": jax.random.normal(k_w1, (num_layers, d_model, d_model * 2)) * scale,
        "w_down": jax.random.normal(k_w2, (num_layers, d_model * 2, d_model)) * scale,
    }


def forward_full_sequence(
    weights: dict[str, jax.Array],
    tokens: jax.Array,
    valid_len: jax.Array,
) -> jax.Array:
    """Naive full-sequence forward pass over a padded buffer `tokens` (B, max_seq_len).
    Recomputes K, V, and full (max_seq_len x max_seq_len) attention for all positions.
    """
    _, max_seq_len = tokens.shape
    num_layers = weights["w_q"].shape[0]
    num_q_heads = weights["w_q"].shape[2]
    num_kv_heads = weights["w_k"].shape[2]
    head_dim = weights["w_q"].shape[3]
    group_size = num_q_heads // num_kv_heads
    scale = 1.0 / jnp.sqrt(jnp.float32(head_dim))

    x = weights["embedding"][tokens]
    causal_mask = jnp.tril(jnp.ones((max_seq_len, max_seq_len), dtype=bool))[None, None, :, :]

    for layer in range(num_layers):
        variance = jnp.mean(jnp.square(x), axis=-1, keepdims=True)
        h = x * jax.lax.rsqrt(variance + 1e-6)

        q = jnp.einsum("btd,dnh->bnth", h, weights["w_q"][layer])
        k = jnp.einsum("btd,dkh->bkth", h, weights["w_k"][layer])
        v = jnp.einsum("btd,dkh->bkth", h, weights["w_v"][layer])

        k_rep = jnp.repeat(k, repeats=group_size, axis=1)
        v_rep = jnp.repeat(v, repeats=group_size, axis=1)

        scores = jnp.matmul(q, jnp.swapaxes(k_rep, -1, -2)) * scale
        scores = jnp.where(causal_mask, scores, -1e9)
        probs = jax.nn.softmax(scores, axis=-1)
        ctx = jnp.matmul(probs, v_rep)
        x = x + jnp.einsum("bnth,nhd->btd", ctx, weights["w_o"][layer])

        var_ff = jnp.mean(jnp.square(x), axis=-1, keepdims=True)
        h_ff = x * jax.lax.rsqrt(var_ff + 1e-6)
        ff = jnp.matmul(jax.nn.gelu(jnp.matmul(h_ff, weights["w_gate"][layer])), weights["w_down"][layer])
        x = x + ff

    last_hidden = x[:, valid_len - 1, :]  # (B, D)
    return jnp.matmul(last_hidden, weights["embedding"].T)


@jax.jit
def naive_generate_loop(
    weights: dict[str, jax.Array],
    prompt_tokens: jax.Array,
    max_seq_len: int = 128,
    gen_steps: int = 64,
) -> tuple[jax.Array, jax.Array]:
    """Baseline: Recompute the full transformer over the whole sequence at every decode step."""
    batch_size, prompt_len = prompt_tokens.shape
    buffer = jnp.zeros((batch_size, max_seq_len), dtype=jnp.int32)
    buffer = jax.lax.dynamic_update_slice(buffer, prompt_tokens, (0, 0))

    def step_fn(carry: jax.Array, step_idx: jax.Array) -> tuple[jax.Array, jax.Array]:
        cur_len = prompt_len + step_idx
        logits = forward_full_sequence(weights, carry, cur_len)
        next_tok = jnp.argmax(logits, axis=-1).astype(jnp.int32)
        carry = jax.lax.dynamic_update_slice(carry, next_tok[:, None], (0, cur_len))
        return carry, logits

    final_buffer, all_logits = jax.lax.scan(step_fn, buffer, jnp.arange(gen_steps))
    return final_buffer, all_logits


def prefill_kv_cache(
    weights: dict[str, jax.Array],
    prompt_tokens: jax.Array,
    max_seq_len: int = 128,
) -> tuple[KVCache, jax.Array]:
    """Phase 1 (Prefill): Process prompt tokens in parallel and populate the static KV-Cache."""
    batch_size, prompt_len = prompt_tokens.shape
    num_layers = weights["w_q"].shape[0]
    num_q_heads = weights["w_q"].shape[2]
    num_kv_heads = weights["w_k"].shape[2]
    head_dim = weights["w_q"].shape[3]
    group_size = num_q_heads // num_kv_heads
    scale = 1.0 / jnp.sqrt(jnp.float32(head_dim))

    k_cache = jnp.zeros(
        (num_layers, batch_size, num_kv_heads, max_seq_len, head_dim), dtype=jnp.float32
    )
    v_cache = jnp.zeros(
        (num_layers, batch_size, num_kv_heads, max_seq_len, head_dim), dtype=jnp.float32
    )

    x = weights["embedding"][prompt_tokens]
    causal_mask = jnp.tril(jnp.ones((prompt_len, prompt_len), dtype=bool))[None, None, :, :]

    for layer in range(num_layers):
        variance = jnp.mean(jnp.square(x), axis=-1, keepdims=True)
        h = x * jax.lax.rsqrt(variance + 1e-6)

        q = jnp.einsum("btd,dnh->bnth", h, weights["w_q"][layer])
        k = jnp.einsum("btd,dkh->bkth", h, weights["w_k"][layer])
        v = jnp.einsum("btd,dkh->bkth", h, weights["w_v"][layer])

        k_cache = k_cache.at[layer, :, :, :prompt_len, :].set(k)
        v_cache = v_cache.at[layer, :, :, :prompt_len, :].set(v)

        k_rep = jnp.repeat(k, repeats=group_size, axis=1)
        v_rep = jnp.repeat(v, repeats=group_size, axis=1)

        scores = jnp.matmul(q, jnp.swapaxes(k_rep, -1, -2)) * scale
        scores = jnp.where(causal_mask, scores, -1e9)
        probs = jax.nn.softmax(scores, axis=-1)
        ctx = jnp.matmul(probs, v_rep)
        x = x + jnp.einsum("bnth,nhd->btd", ctx, weights["w_o"][layer])

        var_ff = jnp.mean(jnp.square(x), axis=-1, keepdims=True)
        h_ff = x * jax.lax.rsqrt(var_ff + 1e-6)
        ff = jnp.matmul(jax.nn.gelu(jnp.matmul(h_ff, weights["w_gate"][layer])), weights["w_down"][layer])
        x = x + ff

    first_logits = jnp.matmul(x[:, -1, :], weights["embedding"].T)
    return KVCache(k=k_cache, v=v_cache), first_logits


def decode_one_token_with_kv_cache(
    weights: dict[str, jax.Array],
    cache: KVCache,
    token_t: jax.Array,
    pos: jax.Array,
) -> tuple[KVCache, jax.Array]:
    """Phase 2 (Decode): Process 1 token `(B, 1)` and update KV-Cache at `pos` via dynamic_update_slice."""
    num_layers = weights["w_q"].shape[0]
    num_q_heads = weights["w_q"].shape[2]
    num_kv_heads = weights["w_k"].shape[2]
    head_dim = weights["w_q"].shape[3]
    max_seq_len = cache.k.shape[3]
    group_size = num_q_heads // num_kv_heads
    scale = 1.0 / jnp.sqrt(jnp.float32(head_dim))

    x = weights["embedding"][token_t]  # (B, 1, D)
    k_cache, v_cache = cache.k, cache.v
    valid_mask = (jnp.arange(max_seq_len) <= pos)[None, None, None, :]

    for layer in range(num_layers):
        variance = jnp.mean(jnp.square(x), axis=-1, keepdims=True)
        h = x * jax.lax.rsqrt(variance + 1e-6)

        q_step = jnp.einsum("btd,dnh->bnth", h, weights["w_q"][layer])  # (B, H_q, 1, D_h)
        k_step = jnp.einsum("btd,dkh->bkth", h, weights["w_k"][layer])  # (B, H_kv, 1, D_h)
        v_step = jnp.einsum("btd,dkh->bkth", h, weights["w_v"][layer])  # (B, H_kv, 1, D_h)

        # The write costs the same at any pos; the attention below still reads every cached slot
        k_layer = jax.lax.dynamic_update_slice(k_cache[layer], k_step, (0, 0, pos, 0))
        v_layer = jax.lax.dynamic_update_slice(v_cache[layer], v_step, (0, 0, pos, 0))
        k_cache = k_cache.at[layer].set(k_layer)
        v_cache = v_cache.at[layer].set(v_layer)

        k_rep = jnp.repeat(k_layer, repeats=group_size, axis=1)
        v_rep = jnp.repeat(v_layer, repeats=group_size, axis=1)

        scores = jnp.matmul(q_step, jnp.swapaxes(k_rep, -1, -2)) * scale  # (B, H_q, 1, max_seq_len)
        scores = jnp.where(valid_mask, scores, -1e9)
        probs = jax.nn.softmax(scores, axis=-1)
        ctx = jnp.matmul(probs, v_rep)  # (B, H_q, 1, D_h)
        x = x + jnp.einsum("bnth,nhd->btd", ctx, weights["w_o"][layer])

        var_ff = jnp.mean(jnp.square(x), axis=-1, keepdims=True)
        h_ff = x * jax.lax.rsqrt(var_ff + 1e-6)
        ff = jnp.matmul(jax.nn.gelu(jnp.matmul(h_ff, weights["w_gate"][layer])), weights["w_down"][layer])
        x = x + ff

    logits = jnp.matmul(x[:, 0, :], weights["embedding"].T)
    return KVCache(k=k_cache, v=v_cache), logits


@jax.jit
def kv_cache_generate_loop(
    weights: dict[str, jax.Array],
    prompt_tokens: jax.Array,
    max_seq_len: int = 128,
    gen_steps: int = 64,
) -> tuple[jax.Array, jax.Array]:
    """Optimized Prefill + KV-Cache Decode loop."""
    batch_size, prompt_len = prompt_tokens.shape
    buffer = jnp.zeros((batch_size, max_seq_len), dtype=jnp.int32)
    buffer = jax.lax.dynamic_update_slice(buffer, prompt_tokens, (0, 0))

    cache, first_logits = prefill_kv_cache(weights, prompt_tokens, max_seq_len=max_seq_len)
    first_tok = jnp.argmax(first_logits, axis=-1).astype(jnp.int32)
    buffer = jax.lax.dynamic_update_slice(buffer, first_tok[:, None], (0, prompt_len))

    def decode_scan_fn(
        carry: tuple[KVCache, jax.Array, jax.Array], step_idx: jax.Array
    ) -> tuple[tuple[KVCache, jax.Array, jax.Array], jax.Array]:
        cur_cache, cur_buf, prev_tok = carry
        pos = prompt_len + step_idx
        cur_cache, logits = decode_one_token_with_kv_cache(
            weights, cur_cache, prev_tok[:, None], pos
        )
        next_tok = jnp.argmax(logits, axis=-1).astype(jnp.int32)
        cur_buf = jax.lax.dynamic_update_slice(cur_buf, next_tok[:, None], (0, pos + 1))
        return (cur_cache, cur_buf, next_tok), logits

    (_, final_buffer, _), rest_logits = jax.lax.scan(
        decode_scan_fn,
        (cache, buffer, first_tok),
        jnp.arange(gen_steps - 1),
    )
    all_logits = jnp.concatenate([first_logits[None, ...], rest_logits], axis=0)
    return final_buffer, all_logits


def median_ms(fn: Callable[[], Any], repeats: int) -> float:
    """Median wall time in ms after one warmup call; JAX results are block_until_ready()."""
    jax.block_until_ready(fn())
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        jax.block_until_ready(fn())
        times.append((time.perf_counter() - t0) * 1000.0)
    return statistics.median(times)


jax_jit_decode_step = jax.jit(decode_one_token_with_kv_cache)


def decode_step_ms_by_cache_len(
    weights: dict[str, jax.Array],
    batch_size: int,
    cache_lens: tuple[int, ...],
    repeats: int,
) -> dict[str, float]:
    """Median time of one compiled decode step for KV-cache buffers of different lengths."""
    num_layers, _, num_kv_heads, head_dim = weights["w_k"].shape
    token = jnp.zeros((batch_size, 1), dtype=jnp.int32)
    out = {}
    for n in cache_lens:
        shape = (num_layers, batch_size, num_kv_heads, n, head_dim)
        cache = KVCache(k=jnp.zeros(shape, jnp.float32), v=jnp.zeros(shape, jnp.float32))
        pos = jnp.int32(n // 2)
        out[str(n)] = median_ms(lambda c=cache, p=pos: jax_jit_decode_step(weights, c, token, p), repeats)
    return out


def run_experiment(
    batch_size: int = 4,
    prompt_len: int = 64,
    gen_steps: int = 64,
    iters: int = 20,
) -> dict[str, float | int | bool | str]:
    max_seq_len = prompt_len + gen_steps
    key = jax.random.key(404)
    k_w, k_p = jax.random.split(key)
    weights = init_tiny_gemma_weights(k_w)
    prompt = jax.random.randint(k_p, (batch_size, prompt_len), 0, 256, dtype=jnp.int32)

    # Warmup and parity verification
    naive_tokens, naive_logits = naive_generate_loop(weights, prompt)
    naive_tokens.block_until_ready()

    kv_tokens, kv_logits = kv_cache_generate_loop(weights, prompt)
    kv_tokens.block_until_ready()

    exact_token_match = bool(jnp.array_equal(naive_tokens, kv_tokens))
    generated = slice(prompt_len, prompt_len + gen_steps)
    tokens_identical = int(jnp.sum(naive_tokens[:, generated] == kv_tokens[:, generated]))
    max_logit_diff = float(np.max(np.abs(np.asarray(naive_logits) - np.asarray(kv_logits))))

    naive_ms = median_ms(lambda: naive_generate_loop(weights, prompt)[0], max(5, iters // 3))
    kv_ms = median_ms(lambda: kv_cache_generate_loop(weights, prompt)[0], iters)
    step_ms = decode_step_ms_by_cache_len(weights, batch_size, (128, 1024, 8192), iters)
    num_layers, _, num_kv_heads, head_dim = weights["w_k"].shape
    cache_shape = (num_layers, batch_size, num_kv_heads, max_seq_len, head_dim)

    return {
        "backend": jax.default_backend(),
        "prompt_len": prompt_len,
        "gen_steps": gen_steps,
        "max_seq_len": max_seq_len,
        "batch_size": batch_size,
        "num_layers": num_layers,
        "cache_shape": str(cache_shape),
        "cache_kb_each": int(np.prod(cache_shape)) * 4 / 1024.0,
        "exact_token_match": exact_token_match,
        "generated_tokens_identical": tokens_identical,
        "generated_tokens_total": batch_size * gen_steps,
        "max_logit_diff": max_logit_diff,
        "naive_ms": naive_ms,
        "kv_ms": kv_ms,
        "speedup": naive_ms / kv_ms,
        "decode_step_ms_by_cache_len": step_ms,
    }


def main() -> None:
    metrics = run_experiment()
    print("=" * 76)
    print("Experiment 04: KV-Cache Data Structure (Prefill vs Decode Loop)")
    print("=" * 76)
    print(f"Backend Device                : {metrics['backend']}")
    print(
        f"Model                         : toy {metrics['num_layers']}-layer decoder, random "
        "weights, GQA 4:2, no RoPE"
    )
    print(
        f"Workload                      : prefill {metrics['prompt_len']} + decode "
        f"{metrics['gen_steps']} tokens, buffer {metrics['max_seq_len']} slots, "
        f"batch {metrics['batch_size']}"
    )
    print(
        f"KV-Cache Buffer (k and v)     : {metrics['cache_shape']} float32, "
        f"{metrics['cache_kb_each']:.0f} KB each"
    )
    print(
        f"Greedy Token Parity           : {metrics['generated_tokens_identical']}/"
        f"{metrics['generated_tokens_total']} generated tokens identical"
    )
    print(f"Max logit diff vs recompute   : {metrics['max_logit_diff']:.2e}")
    print("-" * 76)
    print("Both loops @jax.jit + lax.scan; median ms after warmup")
    print(f"Recompute full window per step: {metrics['naive_ms']:.4f} ms")
    print(f"Prefill + KV-cache decode     : {metrics['kv_ms']:.4f} ms")
    print(f"Speedup, cache vs recompute   : {metrics['speedup']:.2f}x")
    print("-" * 76)
    print("One decode step vs cache length (each step reads the whole cache)")
    for n, ms in metrics["decode_step_ms_by_cache_len"].items():
        print(f"  {int(n):>5} slots                 : {ms:.4f} ms")
    print("=" * 76)


if __name__ == "__main__":
    main()
