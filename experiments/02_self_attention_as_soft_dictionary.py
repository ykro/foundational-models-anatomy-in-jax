"""Experiment 02: Self-Attention as a Differentiable Dictionary (Q, K, V) with Causal Mask & RoPE.

Mental model for Software Devs:
In Python, a dictionary `d[query]` requires an exact hash match (`key == query`) and returns one
`value`. Self-Attention is a continuous, vectorized dictionary lookup over sequence positions:
1. Every token emits a `Query` (what it is looking for), a `Key` (what it offers), and a `Value`
   (the payload it transfers if matched).
2. Rotary Position Embeddings (`RoPE`) rotate pairs of `(Q, K)` coordinates by angle `pos * theta`
   so the dot product `Q[m] · K[n]` depends only on relative distance `m - n` while preserving
   the L2 norm of every vector.
3. The lower-triangular causal mask (`jnp.tril`) sets future scores `n > m` to `-1e9` before the
   softmax, ensuring token `m` can never read keys from the future (`t > m`).

Timing: the naive baseline loops in Python over the 64 (batch, head) pairs; the fair reference
is the same NumPy math batched over all heads with no Python loop. Times are medians after a
warmup call. The compiled HLO is inspected and reported as it is (several fusions, not one).
"""

from __future__ import annotations

import re
import statistics
import time
from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np


def precompute_rope_cos_sin(
    seq_len: int, head_dim: int, base: float = 10000.0
) -> tuple[jax.Array, jax.Array]:
    """Precompute Rotary Position Embedding (RoPE) cosine and sine tables."""
    half_dim = head_dim // 2
    freq_exponents = jnp.arange(0, half_dim, dtype=jnp.float32) / jnp.float32(half_dim)
    inv_freq = 1.0 / (base**freq_exponents)
    positions = jnp.arange(seq_len, dtype=jnp.float32)[:, None]
    angles = positions * inv_freq[None, :]  # (seq_len, half_dim)
    return jnp.cos(angles), jnp.sin(angles)


def apply_rope(x: jax.Array, cos: jax.Array, sin: jax.Array) -> jax.Array:
    """Apply 2D coordinate rotation to the last dimension of `x` (..., seq_len, head_dim)."""
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    rotated_first = x1 * cos - x2 * sin
    rotated_second = x1 * sin + x2 * cos
    return jnp.concatenate([rotated_first, rotated_second], axis=-1)


def numpy_naive_causal_rope_attention(
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
    cos: np.ndarray,
    sin: np.ndarray,
) -> np.ndarray:
    """Baseline: Explicit Python loops over batch and attention heads in NumPy."""
    batch_size, num_heads, seq_len, head_dim = q.shape
    half = head_dim // 2
    scale = np.float32(1.0 / np.sqrt(head_dim))
    mask = np.tril(np.ones((seq_len, seq_len), dtype=bool))
    out = np.zeros_like(q, dtype=np.float32)

    for b in range(batch_size):
        for h in range(num_heads):
            q_bh = q[b, h]
            k_bh = k[b, h]
            v_bh = v[b, h]

            q_rot = np.concatenate(
                [q_bh[:, :half] * cos - q_bh[:, half:] * sin,
                 q_bh[:, :half] * sin + q_bh[:, half:] * cos],
                axis=-1,
            )
            k_rot = np.concatenate(
                [k_bh[:, :half] * cos - k_bh[:, half:] * sin,
                 k_bh[:, :half] * sin + k_bh[:, half:] * cos],
                axis=-1,
            )
            scores = (q_rot @ k_rot.T) * scale
            scores = np.where(mask, scores, -1e9)
            scores_shifted = scores - np.max(scores, axis=-1, keepdims=True)
            weights = np.exp(scores_shifted)
            weights = weights / np.sum(weights, axis=-1, keepdims=True)
            out[b, h] = weights @ v_bh
    return out


def numpy_batched_causal_rope_attention(
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
    cos: np.ndarray,
    sin: np.ndarray,
) -> np.ndarray:
    """Fair reference: the same NumPy math batched over (batch, heads), no Python loop."""
    seq_len, head_dim = q.shape[-2], q.shape[-1]
    half = head_dim // 2
    scale = np.float32(1.0 / np.sqrt(head_dim))
    q_rot = np.concatenate(
        [q[..., :half] * cos - q[..., half:] * sin, q[..., :half] * sin + q[..., half:] * cos],
        axis=-1,
    )
    k_rot = np.concatenate(
        [k[..., :half] * cos - k[..., half:] * sin, k[..., :half] * sin + k[..., half:] * cos],
        axis=-1,
    )
    scores = (q_rot @ np.swapaxes(k_rot, -1, -2)) * scale
    mask = np.tril(np.ones((seq_len, seq_len), dtype=bool))
    scores = np.where(mask, scores, -1e9)
    scores = scores - np.max(scores, axis=-1, keepdims=True)
    weights = np.exp(scores)
    weights = weights / np.sum(weights, axis=-1, keepdims=True)
    return weights @ v


def median_ms(fn: Callable[[], Any], repeats: int) -> float:
    """Median wall time in ms after one warmup call; JAX results are block_until_ready()."""
    jax.block_until_ready(fn())
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        jax.block_until_ready(fn())
        times.append((time.perf_counter() - t0) * 1000.0)
    return statistics.median(times)


def compiled_hlo_op_counts(jitted: Any, *args: Any) -> dict[str, int]:
    """Count fusion and dot instructions in the optimized HLO that XLA actually compiled."""
    text = jitted.lower(*args).compile().as_text()
    ops = re.findall(r"=\s*\S+\s+([a-z][\w-]*)\(", text)
    return {"fusion": ops.count("fusion"), "dot": ops.count("dot")}


@jax.jit
def jax_jit_causal_rope_attention(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    cos: jax.Array,
    sin: jax.Array,
) -> jax.Array:
    """Causal self-attention with RoPE, vectorized over (batch, heads) and compiled by XLA."""
    seq_len = q.shape[-2]
    head_dim = q.shape[-1]
    q_rot = apply_rope(q, cos[None, None, :, :], sin[None, None, :, :])
    k_rot = apply_rope(k, cos[None, None, :, :], sin[None, None, :, :])

    scale = 1.0 / jnp.sqrt(jnp.float32(head_dim))
    scores = jnp.matmul(q_rot, jnp.swapaxes(k_rot, -1, -2)) * scale
    causal_mask = jnp.tril(jnp.ones((seq_len, seq_len), dtype=bool))[None, None, :, :]
    masked_scores = jnp.where(causal_mask, scores, -1e9)
    weights = jax.nn.softmax(masked_scores, axis=-1)
    return jnp.matmul(weights, v)


def verify_rope_and_causality(
    q: jax.Array, k: jax.Array, v: jax.Array, cos: jax.Array, sin: jax.Array
) -> tuple[float, float, float]:
    """Verify 3 mathematical invariants:
    1. RoPE preserves vector L2 norm: ||RoPE(q)||_2 == ||q||_2.
    2. RoPE dot product depends only on relative distance (m - n).
    3. Causal mask blocks any future token mutation at t > pivot from affecting t <= pivot.
    """
    q_rot = apply_rope(q, cos[None, None, :, :], sin[None, None, :, :])
    norm_before = jnp.linalg.norm(q, axis=-1)
    norm_after = jnp.linalg.norm(q_rot, axis=-1)
    rope_norm_diff = float(jnp.max(jnp.abs(norm_before - norm_after)))

    # Relative distance check: dot(RoPE(q_vec, pos=5), RoPE(k_vec, pos=2)) ==
    #                          dot(RoPE(q_vec, pos=17), RoPE(k_vec, pos=14))
    q_vec = q[0, 0, 0:1, :]
    k_vec = k[0, 0, 0:1, :]
    q_p5 = apply_rope(q_vec, cos[5:6], sin[5:6])
    k_p2 = apply_rope(k_vec, cos[2:3], sin[2:3])
    q_p17 = apply_rope(q_vec, cos[17:18], sin[17:18])
    k_p14 = apply_rope(k_vec, cos[14:15], sin[14:15])
    dot_a = jnp.sum(q_p5 * k_p2)
    dot_b = jnp.sum(q_p17 * k_p14)
    rope_rel_diff = float(jnp.abs(dot_a - dot_b))

    # Causal invariance check: mutate K, V after the pivot (T // 2) and verify the prefix is untouched
    pivot = q.shape[-2] // 2
    out_clean = jax_jit_causal_rope_attention(q, k, v, cos, sin)
    k_mutated = k.at[:, :, pivot + 1 :, :].add(50.0)
    v_mutated = v.at[:, :, pivot + 1 :, :].add(-50.0)
    out_mutated = jax_jit_causal_rope_attention(q, k_mutated, v_mutated, cos, sin)
    causal_prefix_leak = float(
        jnp.max(jnp.abs(out_clean[:, :, : pivot + 1, :] - out_mutated[:, :, : pivot + 1, :]))
    )
    return rope_norm_diff, rope_rel_diff, causal_prefix_leak


def run_experiment(
    batch_size: int = 8,
    num_heads: int = 8,
    seq_len: int = 128,
    head_dim: int = 64,
    iters: int = 40,
) -> dict[str, float | str]:
    key = jax.random.key(102)
    k1, k2, k3 = jax.random.split(key, 3)
    q = jax.random.normal(k1, (batch_size, num_heads, seq_len, head_dim), dtype=jnp.float32)
    k = jax.random.normal(k2, (batch_size, num_heads, seq_len, head_dim), dtype=jnp.float32)
    v = jax.random.normal(k3, (batch_size, num_heads, seq_len, head_dim), dtype=jnp.float32)
    cos, sin = precompute_rope_cos_sin(seq_len, head_dim)

    q_np, k_np, v_np = np.asarray(q), np.asarray(k), np.asarray(v)
    cos_np, sin_np = np.asarray(cos), np.asarray(sin)

    ref_out = numpy_naive_causal_rope_attention(q_np, k_np, v_np, cos_np, sin_np)
    batched_out = numpy_batched_causal_rope_attention(q_np, k_np, v_np, cos_np, sin_np)
    jit_out = jax_jit_causal_rope_attention(q, k, v, cos, sin).block_until_ready()
    max_diff = float(np.max(np.abs(ref_out - np.asarray(jit_out))))
    max_diff_batched = float(np.max(np.abs(batched_out - np.asarray(jit_out))))

    rope_norm_diff, rope_rel_diff, causal_prefix_leak = verify_rope_and_causality(
        q, k, v, cos, sin
    )
    hlo = compiled_hlo_op_counts(jax_jit_causal_rope_attention, q, k, v, cos, sin)

    naive_ms = median_ms(
        lambda: numpy_naive_causal_rope_attention(q_np, k_np, v_np, cos_np, sin_np),
        max(7, iters // 4),
    )
    batched_ms = median_ms(
        lambda: numpy_batched_causal_rope_attention(q_np, k_np, v_np, cos_np, sin_np),
        max(7, iters // 4),
    )
    jit_ms = median_ms(lambda: jax_jit_causal_rope_attention(q, k, v, cos, sin), iters)

    return {
        "backend": jax.default_backend(),
        "shape": f"({batch_size}, {num_heads}, {seq_len}, {head_dim})",
        "rope_norm_diff": rope_norm_diff,
        "rope_rel_diff": rope_rel_diff,
        "causal_prefix_leak": causal_prefix_leak,
        "causal_pivot": seq_len // 2,
        "batch_head_pairs": batch_size * num_heads,
        "hlo_fusions": hlo["fusion"],
        "hlo_dots": hlo["dot"],
        "naive_ms": naive_ms,
        "numpy_batched_ms": batched_ms,
        "jit_ms": jit_ms,
        "speedup": naive_ms / jit_ms,
        "speedup_vs_numpy_batched": batched_ms / jit_ms,
        "max_diff": max_diff,
        "max_diff_numpy_batched": max_diff_batched,
    }


def main() -> None:
    metrics = run_experiment()
    print("=" * 76)
    print("Experiment 02: Self-Attention as Soft Dictionary (Causal Mask + RoPE)")
    print("=" * 76)
    print(f"Backend Device                : {metrics['backend']}")
    print(f"Attention Tensor Shape        : {metrics['shape']} float32 [B, H, T, D_h]")
    print(f"RoPE Norm Preservation Error  : {metrics['rope_norm_diff']:.2e} (||RoPE(q)|| vs ||q||)")
    print(f"RoPE Relative Shift Error     : {metrics['rope_rel_diff']:.2e} (q@5.k@2 vs q@17.k@14)")
    print(
        f"Causal Prefix Leakage         : {metrics['causal_prefix_leak']:.2e} "
        f"(K, V changed after pos {metrics['causal_pivot']})"
    )
    print(
        f"Compiled HLO (XLA CPU)        : {metrics['hlo_fusions']} fusions, "
        f"{metrics['hlo_dots']} dot ops (several kernels)"
    )
    print("-" * 76)
    print("Median ms after warmup")
    print(f"NumPy, Python loop of {metrics['batch_head_pairs']} (b,h): {metrics['naive_ms']:.4f} ms")
    print(f"NumPy batched, no Python loop : {metrics['numpy_batched_ms']:.4f} ms")
    print(f"JAX @jit RoPE attention       : {metrics['jit_ms']:.4f} ms")
    print(f"@jit vs NumPy loop (naive)    : {metrics['speedup']:.2f}x")
    print(f"@jit vs NumPy batched (fair)  : {metrics['speedup_vs_numpy_batched']:.2f}x")
    print(f"Max |NumPy - JAX @jit|        : {metrics['max_diff']:.2e}")
    print("=" * 76)


if __name__ == "__main__":
    main()
