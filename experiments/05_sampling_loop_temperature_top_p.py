"""Experiment 05: Autoregressive Sampling Loop (Greedy, Temperature, Top-K, Top-P & JSON Grammar).

Mental model for Software Devs:
A Foundational Model does not output text; at every step it outputs a flat `float32[vocab_size]`
array of unnormalized scores called `logits`. The `Sampling Loop` is a pure post-processing
pipeline over that vector before calling a categorical random number generator:
1. `Temperature` (`logits / T`): Rescales entropy. As `T -> 0`, softmax collapses to `argmax`
   (`Greedy` decoding). As `T > 1`, the distribution flattens.
2. `Top-K` & `Top-P` (Nucleus) Filtering: Sorts logits descending, computes cumulative probability
   mass `cumsum(softmax(sorted_logits))`, and masks out the low-probability tail with `-inf` so
   hundreds of tiny noise probabilities cannot hijack the next token.
3. `Grammar / Schema Masking`: constrains the next token by applying a boolean bitmask
   `jnp.where(allowed_mask, logits, -1e9)` before sampling.

What this file proves and what it does not: the logits are random (there is no model) and the
grammar is a fixed schedule of allowed token sets per step over a toy vocabulary, so the 100%
compliance is guaranteed by construction. It proves the mask is applied before every draw.
Real constrained decoding needs a state machine that advances with each chosen token and a real
tokenizer whose tokens carry several characters; the masking mechanism is the same `jnp.where`.

Timing: the naive baseline is a Python loop over every step and every row that argsorts the full
vocabulary on the host; the fair reference is a Python loop that calls the compiled step once
per token. The scan keeps the whole loop on the device. Times are medians after a warmup call.
"""

from __future__ import annotations

import statistics
import time
from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np


def sample_logits_step(
    key: jax.Array,
    logits: jax.Array,
    temperature: float = 0.8,
    top_k: int = 16,
    top_p: float = 0.90,
    allowed_mask: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array]:
    """Pure JAX sampling pipeline: Grammar Mask -> Temperature -> Top-K -> Top-P -> Categorical."""
    # 1. Apply deterministic grammar/schema mask if provided
    masked_logits = (
        jnp.where(allowed_mask, logits, -1e9) if allowed_mask is not None else logits
    )

    # 2. Scale by temperature
    scaled_logits = masked_logits / jnp.maximum(jnp.float32(temperature), 1e-5)

    # 3. Keep the Top-K candidates with lax.top_k instead of sorting the whole vocabulary
    topk_logits, topk_indices = jax.lax.top_k(scaled_logits, top_k)

    # Top-P (nucleus) filter over the Top-K candidates
    sorted_probs = jax.nn.softmax(topk_logits, axis=-1)
    cum_probs = jnp.cumsum(sorted_probs, axis=-1)
    # Always keep the top-1 token, mask out tokens whose prior cumulative mass already exceeded top_p
    nucleus_keep = (cum_probs - sorted_probs) < top_p
    nucleus_keep = nucleus_keep.at[:, 0].set(True)
    filtered_topk_logits = jnp.where(nucleus_keep, topk_logits, -1e9)

    # Sample directly in Top-K nucleus space
    probs = jax.nn.softmax(filtered_topk_logits, axis=-1)
    sampled_rank = jax.random.categorical(key, filtered_topk_logits, axis=-1)[:, None]
    sampled_token = jnp.take_along_axis(topk_indices, sampled_rank, axis=-1)[:, 0].astype(
        jnp.int32
    )
    return sampled_token, probs


jax_jit_sample_step = jax.jit(sample_logits_step, static_argnames=("top_k",))


def python_host_sampling_loop(
    logits_seq: np.ndarray,
    grammar_masks: np.ndarray,
    temperature: float = 0.8,
    top_k: int = 16,
    top_p: float = 0.90,
) -> np.ndarray:
    """Baseline: Host-side Python/NumPy autoregressive sampling loop across steps and batch."""
    steps, batch_size, vocab_size = logits_seq.shape
    rng = np.random.default_rng(77)
    out_tokens = np.zeros((steps, batch_size), dtype=np.int32)

    for t in range(steps):
        for b in range(batch_size):
            row = np.where(grammar_masks[t], logits_seq[t, b], -1e9) / max(temperature, 1e-5)
            order = np.argsort(-row)
            s_logits = row[order]
            s_logits[top_k:] = -1e9
            s_logits_shift = s_logits - np.max(s_logits)
            s_probs = np.exp(s_logits_shift)
            s_probs = s_probs / np.sum(s_probs)
            cum = np.cumsum(s_probs)
            keep = (cum - s_probs) < top_p
            keep[0] = True
            s_logits = np.where(keep, s_logits, -1e9)
            s_logits_shift = s_logits - np.max(s_logits)
            p = np.exp(s_logits_shift)
            p = p / np.sum(p)
            choice_in_sorted = rng.choice(vocab_size, p=p)
            out_tokens[t, b] = order[choice_in_sorted]
    return out_tokens


@jax.jit
def jax_jit_scan_sampling_loop(
    key: jax.Array,
    logits_seq: jax.Array,
    grammar_masks: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Compiled on-device sampling loop using jax.lax.scan + Threefry PRNG splitting."""
    steps = logits_seq.shape[0]
    step_keys = jax.random.split(key, steps)

    def scan_body(
        carry: None, inputs: tuple[jax.Array, jax.Array, jax.Array]
    ) -> tuple[None, tuple[jax.Array, jax.Array]]:
        k_t, logits_t, mask_t = inputs
        tok_t, probs_t = sample_logits_step(
            k_t,
            logits_t,
            temperature=0.8,
            top_k=16,
            top_p=0.90,
            allowed_mask=mask_t[None, :],
        )
        return None, (tok_t, probs_t)

    _, (sampled_tokens, all_probs) = jax.lax.scan(
        scan_body, None, (step_keys, logits_seq, grammar_masks)
    )
    return sampled_tokens, all_probs


def python_loop_of_jit_step(
    key: jax.Array, logits_seq: jax.Array, grammar_masks: jax.Array
) -> jax.Array:
    """Fair reference: a Python loop that calls the compiled sampling step once per token."""
    steps = logits_seq.shape[0]
    step_keys = jax.random.split(key, steps)
    out = []
    for t in range(steps):
        tok_t, _ = jax_jit_sample_step(
            step_keys[t],
            logits_seq[t],
            temperature=0.8,
            top_k=16,
            top_p=0.90,
            allowed_mask=grammar_masks[t][None, :],
        )
        out.append(tok_t)
    return jnp.stack(out)


def median_ms(fn: Callable[[], Any], repeats: int) -> float:
    """Median wall time in ms after one warmup call; JAX results are block_until_ready()."""
    jax.block_until_ready(fn())
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        jax.block_until_ready(fn())
        times.append((time.perf_counter() - t0) * 1000.0)
    return statistics.median(times)


# Toy JSON vocabulary for the grammar schedule (token id -> text), used to print a sampled row.
TOY_JSON_VOCAB = {
    10: "{", 20: '"status"', 21: '"code"', 30: ":", 40: '"ok"', 41: '"ready"', 42: '"cached"',
    50: ",", 60: '"latency_ms"', 61: '"tokens"', 70: "12", 71: "24", 72: "64",
}


def run_experiment(
    steps: int = 32,
    batch_size: int = 64,
    vocab_size: int = 512,
    iters: int = 40,
) -> dict[str, float | int | bool | str]:
    key = jax.random.key(505)
    k_logits, k_sample = jax.random.split(key)
    logits_seq = jax.random.normal(k_logits, (steps, batch_size, vocab_size), dtype=jnp.float32)

    # Build a deterministic 8-state JSON grammar schedule:
    # State 0: '{' (id 10), State 1: '"status"' (id 20), State 2: ':' (id 30),
    # State 3: '"ok"|"retry"' (ids 40, 41), ...
    allowed_sets = [
        [10],            # {
        [20, 21],        # "status" | "code"
        [30],            # :
        [40, 41, 42],    # "ok" | "ready" | "cached"
        [50],            # ,
        [60, 61],        # "latency_ms" | "tokens"
        [30],            # :
        [70, 71, 72],    # 12 | 24 | 64
    ]
    mask_np = np.zeros((steps, vocab_size), dtype=bool)
    for t in range(steps):
        valid_ids = allowed_sets[t % len(allowed_sets)]
        mask_np[t, valid_ids] = True
    grammar_masks = jnp.asarray(mask_np)

    # 1. Low temperature with the same Top-K=16 / Top-P=0.90 filter collapses to greedy argmax
    greedy_ids = jnp.argmax(logits_seq[0], axis=-1)
    low_temp_ids, _ = jax_jit_sample_step(
        k_sample, logits_seq[0], temperature=1e-4, top_k=16, top_p=0.90
    )
    greedy_rows = int(jnp.sum(greedy_ids == low_temp_ids))
    greedy_parity = greedy_rows == batch_size

    # 2. Verify Top-P nucleus probability mass and 100% JSON grammar compliance
    sampled_tokens, _all_probs = jax_jit_scan_sampling_loop(
        k_sample, logits_seq, grammar_masks
    )
    sampled_tokens.block_until_ready()

    # Check whether every sampled token at step t is allowed by grammar_masks[t]
    sampled_np = np.asarray(sampled_tokens)
    valid_draws = int(sum(mask_np[t, sampled_np[t]].sum() for t in range(steps)))
    grammar_compliance_rate = 100.0 * valid_draws / (steps * batch_size)
    row0 = "".join(TOY_JSON_VOCAB.get(int(t), "?") for t in sampled_np[: len(allowed_sets), 0])

    # Check active support size after Top-K=16 / Top-P=0.90 filtering on unconstrained logits
    _, unconstrained_probs = jax_jit_sample_step(
        k_sample, logits_seq[0], temperature=0.8, top_k=16, top_p=0.90
    )
    active_support_mean = float(jnp.mean(jnp.sum(unconstrained_probs > 1e-6, axis=-1)))

    # 3. Benchmark: Python host loop (naive), Python loop of @jit step (fair), lax.scan
    loop_tokens = python_loop_of_jit_step(k_sample, logits_seq, grammar_masks)
    loop_matches_scan = bool(jnp.array_equal(loop_tokens, sampled_tokens))
    logits_np = np.asarray(logits_seq)
    py_ms = median_ms(lambda: python_host_sampling_loop(logits_np, mask_np), max(5, iters // 8))
    loop_jit_ms = median_ms(
        lambda: python_loop_of_jit_step(k_sample, logits_seq, grammar_masks), max(9, iters // 4)
    )
    jit_ms = median_ms(lambda: jax_jit_scan_sampling_loop(k_sample, logits_seq, grammar_masks), iters)

    return {
        "backend": jax.default_backend(),
        "shape": f"({steps}, {batch_size}, {vocab_size})",
        "steps": steps,
        "batch_size": batch_size,
        "vocab_size": vocab_size,
        "greedy_parity": greedy_parity,
        "greedy_rows_matching": greedy_rows,
        "active_support_mean": active_support_mean,
        "grammar_compliance_rate": grammar_compliance_rate,
        "grammar_valid_draws": valid_draws,
        "sampled_row0_first_steps": row0,
        "loop_of_jit_matches_scan": loop_matches_scan,
        "py_ms": py_ms,
        "loop_of_jit_ms": loop_jit_ms,
        "jit_ms": jit_ms,
        "speedup": py_ms / jit_ms,
        "speedup_vs_loop_of_jit": loop_jit_ms / jit_ms,
    }


def main() -> None:
    metrics = run_experiment()
    draws = metrics["steps"] * metrics["batch_size"]
    print("=" * 76)
    print("Experiment 05: Autoregressive Sampling Loop (Top-K, Top-P & Grammar Mask)")
    print("=" * 76)
    print(f"Backend Device                : {metrics['backend']}")
    print(f"Logits (random, no model)     : {metrics['shape']} [steps, batch, vocab]")
    print(
        f"Low-Temp Check (T=1e-4, K=16) : {metrics['greedy_rows_matching']}/"
        f"{metrics['batch_size']} rows equal argmax (greedy)"
    )
    print(
        f"Nucleus Support (T=0.8)       : {metrics['active_support_mean']:.2f} of "
        f"{metrics['vocab_size']} tokens kept (Top-K=16, Top-P=0.90)"
    )
    print(f"Row 0, Steps 0-7 (toy JSON)   : {metrics['sampled_row0_first_steps']}")
    print(
        f"Draws Inside Allowed Set      : {metrics['grammar_compliance_rate']:.1f}% "
        f"({metrics['grammar_valid_draws']}/{draws}, by construction)"
    )
    print("-" * 76)
    print("Median ms after warmup")
    print(f"Python host loop (argsort)    : {metrics['py_ms']:.4f} ms")
    print(f"Python loop of @jit step      : {metrics['loop_of_jit_ms']:.4f} ms")
    print(f"@jit + lax.scan (1 call)      : {metrics['jit_ms']:.4f} ms")
    print(f"Scan vs host loop (naive)     : {metrics['speedup']:.2f}x")
    print(f"Scan vs loop of @jit (fair)   : {metrics['speedup_vs_loop_of_jit']:.2f}x")
    print(f"Loop of @jit == scan tokens   : {metrics['loop_of_jit_matches_scan']} (same Threefry keys)")
    print("=" * 76)


if __name__ == "__main__":
    main()
