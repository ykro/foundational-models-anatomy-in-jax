"""Experiment 03: Anatomy of a Gemma Block (RMSNorm + Grouped-Query Attention + GeGLU).

Mental model for Software Devs:
A modern Foundational Model (like Gemma 2 / Gemma 3) is a pipeline of `N` identical decoder
blocks. Each block transforms a residual stream tensor `x` of shape `(B, T, D)` using three
data-structure optimizations over the 2017 Transformer:
1. `RMSNorm` replaces `LayerNorm`: normalizes by root-mean-square without computing or subtracting
   the mean, saving one reduction pass over `D` per normalization.
2. `Grouped-Query Attention (GQA)`: `H_q` query heads share a smaller number of `H_kv` key/value
   heads (`H_q = 8, H_kv = 2 -> 4x smaller KV-cache footprint in RAM/HBM`).
3. `GeGLU` MLP: replaces a static 2-layer feed-forward network with a multiplicative gate
   `gelu(x @ W_gate) * (x @ W_up)` before projecting back down with `W_down`.
"""

from __future__ import annotations

import time
from typing import TypedDict

import jax
import jax.numpy as jnp
import numpy as np


class GemmaBlockParams(TypedDict):
    rms_attn_scale: jax.Array
    w_q: jax.Array
    w_k: jax.Array
    w_v: jax.Array
    w_o: jax.Array
    rms_ffn_scale: jax.Array
    w_gate: jax.Array
    w_up: jax.Array
    w_down: jax.Array


def rms_norm(x: jax.Array, scale: jax.Array, eps: float = 1e-6) -> jax.Array:
    """Gemma-style RMSNorm: x * rsqrt(mean(x^2) + eps) * (1.0 + scale)."""
    variance = jnp.mean(jnp.square(x), axis=-1, keepdims=True)
    normed = x * jax.lax.rsqrt(variance + eps)
    return normed * (1.0 + scale)


def init_gemma_block_params(
    key: jax.Array,
    d_model: int = 256,
    num_q_heads: int = 8,
    num_kv_heads: int = 2,
    head_dim: int = 32,
    d_ff: int = 512,
) -> GemmaBlockParams:
    """Initialize a PyTree of Gemma decoder block weights."""
    keys = jax.random.split(key, 7)
    scale_d = 1.0 / jnp.sqrt(jnp.float32(d_model))
    scale_ff = 1.0 / jnp.sqrt(jnp.float32(d_ff))
    return {
        "rms_attn_scale": jnp.zeros((d_model,), dtype=jnp.float32),
        "w_q": jax.random.normal(keys[0], (d_model, num_q_heads, head_dim)) * scale_d,
        "w_k": jax.random.normal(keys[1], (d_model, num_kv_heads, head_dim)) * scale_d,
        "w_v": jax.random.normal(keys[2], (d_model, num_kv_heads, head_dim)) * scale_d,
        "w_o": jax.random.normal(keys[3], (num_q_heads, head_dim, d_model)) * scale_d,
        "rms_ffn_scale": jnp.zeros((d_model,), dtype=jnp.float32),
        "w_gate": jax.random.normal(keys[4], (d_model, d_ff)) * scale_d,
        "w_up": jax.random.normal(keys[5], (d_model, d_ff)) * scale_d,
        "w_down": jax.random.normal(keys[6], (d_ff, d_model)) * scale_ff,
    }


def gemma_decoder_block(
    x: jax.Array,
    params: GemmaBlockParams,
    cos: jax.Array,
    sin: jax.Array,
) -> jax.Array:
    """Execute one Gemma block (Pre-RMSNorm -> GQA with RoPE -> Residual -> Pre-RMSNorm -> GeGLU)."""
    _batch_size, seq_len, _ = x.shape
    num_q_heads = params["w_q"].shape[1]
    num_kv_heads = params["w_k"].shape[1]
    head_dim = params["w_q"].shape[2]
    group_size = num_q_heads // num_kv_heads

    # 1. Pre-Attention RMSNorm
    h = rms_norm(x, params["rms_attn_scale"])

    # 2. Project Q (B, H_q, T, D_h) and K, V (B, H_kv, T, D_h)
    q = jnp.einsum("btd,dnh->bnth", h, params["w_q"])
    k = jnp.einsum("btd,dkh->bkth", h, params["w_k"])
    v = jnp.einsum("btd,dkh->bkth", h, params["w_v"])

    # Apply RoPE to Q and K
    half = head_dim // 2
    c = cos[None, None, :, :]
    s = sin[None, None, :, :]
    q = jnp.concatenate([q[..., :half] * c - q[..., half:] * s,
                         q[..., :half] * s + q[..., half:] * c], axis=-1)
    k = jnp.concatenate([k[..., :half] * c - k[..., half:] * s,
                         k[..., :half] * s + k[..., half:] * c], axis=-1)

    # Expand K and V across query groups: (B, H_kv, T, D_h) -> (B, H_q, T, D_h)
    k_grouped = jnp.repeat(k, repeats=group_size, axis=1)
    v_grouped = jnp.repeat(v, repeats=group_size, axis=1)

    scale = 1.0 / jnp.sqrt(jnp.float32(head_dim))
    scores = jnp.matmul(q, jnp.swapaxes(k_grouped, -1, -2)) * scale
    causal_mask = jnp.tril(jnp.ones((seq_len, seq_len), dtype=bool))[None, None, :, :]
    scores = jnp.where(causal_mask, scores, -1e9)
    probs = jax.nn.softmax(scores, axis=-1)
    context = jnp.matmul(probs, v_grouped)  # (B, H_q, T, D_h)

    attn_out = jnp.einsum("bnth,nhd->btd", context, params["w_o"])
    x = x + attn_out

    # 3. Pre-FFN RMSNorm + GeGLU Multiplicative Gate
    h_ffn = rms_norm(x, params["rms_ffn_scale"])
    gate = jax.nn.gelu(jnp.matmul(h_ffn, params["w_gate"]), approximate=True)
    up = jnp.matmul(h_ffn, params["w_up"])
    geglu_out = jnp.matmul(gate * up, params["w_down"])
    return x + geglu_out


jax_jit_gemma_block = jax.jit(gemma_decoder_block)


def run_experiment(
    batch_size: int = 8,
    seq_len: int = 128,
    d_model: int = 256,
    num_q_heads: int = 8,
    num_kv_heads: int = 2,
    head_dim: int = 32,
    d_ff: int = 512,
    iters: int = 40,
) -> dict[str, float | int | str]:
    key = jax.random.key(303)
    k_in, k_params = jax.random.split(key)
    x = jax.random.normal(k_in, (batch_size, seq_len, d_model), dtype=jnp.float32)
    params = init_gemma_block_params(
        k_params,
        d_model=d_model,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        d_ff=d_ff,
    )

    half = head_dim // 2
    inv_freq = 1.0 / (10000.0 ** (jnp.arange(0, half, dtype=jnp.float32) / half))
    angles = jnp.arange(seq_len, dtype=jnp.float32)[:, None] * inv_freq[None, :]
    cos, sin = jnp.cos(angles), jnp.sin(angles)

    # Warmup and parity check between Eager and @jax.jit
    eager_out = gemma_decoder_block(x, params, cos, sin).block_until_ready()
    jit_out = jax_jit_gemma_block(x, params, cos, sin).block_until_ready()
    max_diff = float(np.max(np.abs(np.asarray(eager_out) - np.asarray(jit_out))))

    # Verify RMSNorm unit RMS property on normalized activations
    normed_sample = rms_norm(x, params["rms_attn_scale"])
    rms_val = float(jnp.mean(jnp.sqrt(jnp.mean(jnp.square(normed_sample), axis=-1))))

    # Memory traffic comparison: Standard MHA (H_kv = H_q) vs GQA (H_kv = 2)
    mha_kv_bytes = 2 * batch_size * num_q_heads * seq_len * head_dim * 4
    gqa_kv_bytes = 2 * batch_size * num_kv_heads * seq_len * head_dim * 4
    kv_memory_saving_ratio = mha_kv_bytes / gqa_kv_bytes

    eager_iters = max(6, iters // 4)
    t0 = time.perf_counter()
    for _ in range(eager_iters):
        gemma_decoder_block(x, params, cos, sin).block_until_ready()
    eager_ms = ((time.perf_counter() - t0) / eager_iters) * 1000.0

    t0 = time.perf_counter()
    for _ in range(iters):
        jax_jit_gemma_block(x, params, cos, sin).block_until_ready()
    jit_ms = ((time.perf_counter() - t0) / iters) * 1000.0

    return {
        "backend": jax.default_backend(),
        "shape": f"({batch_size}, {seq_len}, {d_model})",
        "num_q_heads": num_q_heads,
        "num_kv_heads": num_kv_heads,
        "mha_kv_kb": mha_kv_bytes / 1024.0,
        "gqa_kv_kb": gqa_kv_bytes / 1024.0,
        "kv_memory_saving_ratio": kv_memory_saving_ratio,
        "rms_mean": rms_val,
        "eager_ms": eager_ms,
        "jit_ms": jit_ms,
        "speedup": eager_ms / jit_ms,
        "max_diff": max_diff,
    }


def main() -> None:
    metrics = run_experiment()
    print("=" * 72)
    print("Experiment 03: Gemma Decoder Block (RMSNorm + GQA + GeGLU)")
    print("=" * 72)
    print(f"Backend Device                : {metrics['backend']}")
    print(f"Residual Stream Shape         : {metrics['shape']} float32")
    print(
        f"GQA Head Topology             : {metrics['num_q_heads']} Query heads sharing "
        f"{metrics['num_kv_heads']} KV heads (4:1 group)"
    )
    print(
        f"KV Tensor Footprint per Layer : {metrics['mha_kv_kb']:.1f} KB (MHA) -> "
        f"{metrics['gqa_kv_kb']:.1f} KB (GQA) ({metrics['kv_memory_saving_ratio']:.2f}x reduction)"
    )
    print(f"RMSNorm Activation RMS        : {metrics['rms_mean']:.6f} (target ~1.000000)")
    print("-" * 72)
    print(f"JAX Eager Gemma Block Time    : {metrics['eager_ms']:.4f} ms")
    print(f"JAX @jit Fused Gemma Block    : {metrics['jit_ms']:.4f} ms")
    print(f"Speedup (@jit vs Eager)       : {metrics['speedup']:.2f}x")
    print(f"Max |Eager - JAX @jit|        : {metrics['max_diff']:.2e}")
    print("=" * 72)


if __name__ == "__main__":
    main()
