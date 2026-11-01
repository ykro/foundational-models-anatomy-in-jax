"""Experiment 02: Self-Attention as a Differentiable Dictionary (Q, K, V) with Causal Mask & RoPE.

Mental model for Software Devs:
In Python, a dictionary `d[query]` requires an exact hash match (`key == query`) and returns one
`value`. Self-Attention is a continuous, vectorized dictionary lookup over sequence positions:
1. Every token emits a `Query` (what it is looking for), a `Key` (what it offers), and a `Value`
   (the payload it transfers if matched).
2. Rotary Position Embeddings (`RoPE`) rotate pairs of `(Q, K)` coordinates by angle `pos * theta`
   so the dot product `Q[m] · K[n]` depends only on relative distance `m - n` while preserving
   the L2 norm of every vector.
3. The lower-triangular causal mask (`jnp.tril`) sets future scores `n > m` to `-inf`, ensuring
   token `m` can never read keys from the future (`t > m`).
"""

from __future__ import annotations

import time

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


@jax.jit
def jax_jit_causal_rope_attention(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    cos: jax.Array,
    sin: jax.Array,
) -> jax.Array:
    """Vectorized and fused @jax.jit Causal Self-Attention with RoPE."""
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

    # Causal invariance check: mutate tokens after pivot=32 and verify prefix [0..32] is untouched
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
    jit_out = jax_jit_causal_rope_attention(q, k, v, cos, sin).block_until_ready()
    max_diff = float(np.max(np.abs(ref_out - np.asarray(jit_out))))

    rope_norm_diff, rope_rel_diff, causal_prefix_leak = verify_rope_and_causality(
        q, k, v, cos, sin
    )

    py_iters = max(4, iters // 8)
    t0 = time.perf_counter()
    for _ in range(py_iters):
        _ = numpy_naive_causal_rope_attention(q_np, k_np, v_np, cos_np, sin_np)
    naive_ms = ((time.perf_counter() - t0) / py_iters) * 1000.0

    t0 = time.perf_counter()
    for _ in range(iters):
        jax_jit_causal_rope_attention(q, k, v, cos, sin).block_until_ready()
    jit_ms = ((time.perf_counter() - t0) / iters) * 1000.0

    return {
        "backend": jax.default_backend(),
        "shape": f"({batch_size}, {num_heads}, {seq_len}, {head_dim})",
        "rope_norm_diff": rope_norm_diff,
        "rope_rel_diff": rope_rel_diff,
        "causal_prefix_leak": causal_prefix_leak,
        "naive_ms": naive_ms,
        "jit_ms": jit_ms,
        "speedup": naive_ms / jit_ms,
        "max_diff": max_diff,
    }


def main() -> None:
    metrics = run_experiment()
    print("=" * 72)
    print("Experiment 02: Self-Attention as Soft Dictionary (Causal Mask + RoPE)")
    print("=" * 72)
    print(f"Backend Device                : {metrics['backend']}")
    print(f"Attention Tensor Shape        : {metrics['shape']} float32")
    print(f"RoPE Norm Preservation Error  : {metrics['rope_norm_diff']:.2e} (||RoPE(q)|| == ||q||)")
    print(f"RoPE Relative Shift Error     : {metrics['rope_rel_diff']:.2e} (shift-invariant dot)")
    print(f"Causal Prefix Future Leakage  : {metrics['causal_prefix_leak']:.2e} (zero future leak)")
    print("-" * 72)
    print(f"Naive Python/NumPy Head Loop  : {metrics['naive_ms']:.4f} ms")
    print(f"JAX @jit Fused RoPE Attention : {metrics['jit_ms']:.4f} ms")
    print(f"Speedup vs Naive Loop         : {metrics['speedup']:.2f}x")
    print(f"Max |Naive - JAX @jit|        : {metrics['max_diff']:.2e}")
    print("=" * 72)


if __name__ == "__main__":
    main()
