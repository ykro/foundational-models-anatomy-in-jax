"""Experiment 01: Byte-Pair Encoding (BPE) Tokenizer as Lossless Data Compression.

Mental model for Software Devs:
Before a Foundational Model touches a single floating-point matrix, it runs a deterministic
lossless data compression algorithm over UTF-8 bytes: Byte-Pair Encoding (BPE).
1. Start with raw UTF-8 byte values (0..255) as the base vocabulary.
2. Repeatedly find the most frequent adjacent pair `(a, b)` in the training corpus and mint a
   new token ID `256 + step` representing the merged sub-word `a + b`.
3. At inference time, a string is compressed into an array of integer Token IDs (`int32[T]`),
   which indexes an embedding lookup table (`float32[V, D]`) in a single `@jax.jit` gather op.

The tokenizer is a toy: it trains on a short paragraph that contains the Spanish sample, so the
in-sample compression is optimistic. A held-out sentence shows the honest number. The gather is
timed against a naive Python loop and against the fair reference, NumPy vectorized indexing.
Times are medians after a warmup call (`block_until_ready()` on every JAX call).
"""

from __future__ import annotations

import itertools
import statistics
import time
from collections import Counter
from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np


def merge_pair(tokens: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
    """Replace all non-overlapping occurrences of `pair` in `tokens` with `new_id`."""
    out: list[int] = []
    i = 0
    n = len(tokens)
    p0, p1 = pair
    while i < n:
        if i + 1 < n and tokens[i] == p0 and tokens[i + 1] == p1:
            out.append(new_id)
            i += 2
        else:
            out.append(tokens[i])
            i += 1
    return out


def train_bpe(
    corpus: str, num_merges: int = 128
) -> tuple[dict[tuple[int, int], int], dict[int, bytes]]:
    """Train a deterministic byte-level BPE merge table and byte vocabulary."""
    tokens = list(corpus.encode("utf-8"))
    merges: dict[tuple[int, int], int] = {}
    vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}

    for step in range(num_merges):
        if len(tokens) < 2:
            break
        pair_counts = Counter(itertools.pairwise(tokens))
        # Tie-break deterministically by pair tuple order
        best_pair, best_freq = max(pair_counts.items(), key=lambda kv: (kv[1], -kv[0][0], -kv[0][1]))
        if best_freq < 2:
            break
        new_id = 256 + step
        merges[best_pair] = new_id
        vocab[new_id] = vocab[best_pair[0]] + vocab[best_pair[1]]
        tokens = merge_pair(tokens, best_pair, new_id)

    return merges, vocab


def encode_bpe(text: str, merges: dict[tuple[int, int], int]) -> list[int]:
    """Encode a UTF-8 string into compressed BPE token IDs."""
    tokens = list(text.encode("utf-8"))
    for pair, new_id in merges.items():
        if len(tokens) < 2:
            break
        tokens = merge_pair(tokens, pair, new_id)
    return tokens


def decode_bpe(token_ids: list[int], vocab: dict[int, bytes]) -> str:
    """Decode BPE token IDs back into the exact original UTF-8 string."""
    raw_bytes = b"".join(vocab[t] for t in token_ids)
    return raw_bytes.decode("utf-8", errors="replace")


def python_embedding_lookup_loop(
    embedding_table: np.ndarray, token_batch: np.ndarray
) -> np.ndarray:
    """Baseline: Python nested loop gathering token embeddings and scaling by sqrt(d_model)."""
    batch_size, seq_len = token_batch.shape
    d_model = embedding_table.shape[1]
    scale = np.float32(np.sqrt(d_model))
    out = np.empty((batch_size, seq_len, d_model), dtype=np.float32)
    for b in range(batch_size):
        for t in range(seq_len):
            out[b, t] = embedding_table[token_batch[b, t]] * scale
    return out


def numpy_vectorized_embedding_lookup(
    embedding_table: np.ndarray, token_batch: np.ndarray
) -> np.ndarray:
    """Fair reference: one vectorized NumPy fancy-indexing gather, no Python loop."""
    scale = np.float32(np.sqrt(embedding_table.shape[1]))
    return embedding_table[token_batch] * scale


def median_ms(fn: Callable[[], Any], repeats: int) -> float:
    """Median wall time in ms after one warmup call; JAX results are block_until_ready()."""
    jax.block_until_ready(fn())
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        jax.block_until_ready(fn())
        times.append((time.perf_counter() - t0) * 1000.0)
    return statistics.median(times)


@jax.jit
def jax_jit_embedding_lookup(
    embedding_table: jax.Array, token_batch: jax.Array
) -> jax.Array:
    """Compiled XLA gather + Gemma-style sqrt(d_model) embedding scaling."""
    d_model = embedding_table.shape[-1]
    return embedding_table[token_batch] * jnp.sqrt(jnp.float32(d_model))


def run_experiment(
    num_merges: int = 128,
    batch_size: int = 128,
    seq_len: int = 128,
    d_model: int = 256,
    iters: int = 40,
) -> dict[str, float | int | bool]:
    corpus = (
        "Los foundational models procesan secuencias de tokens mediante matrices de embeddings "
        "y capas de self-attention compiladas con OpenXLA. Cuando un software engineer entiende "
        "la tokenización Byte-Pair Encoding como compresión determinista de sub-palabras, "
        "descubre por qué el ratio chars/token define la ventana de contexto y la latencia del "
        "KV-cache tanto en inglés como en español. "
        "Foundational models process token sequences through embedding matrices and self-attention "
        "layers compiled with OpenXLA. Understanding Byte-Pair Encoding as lossless data "
        "compression explains how sub-word merges shrink sequence length T and quadratic O(T^2) "
        "attention cost."
    ) * 6

    merges, vocab = train_bpe(corpus, num_merges=num_merges)

    sample_es = (
        "Cuando un software engineer entiende la tokenización Byte-Pair Encoding como compresión "
        "determinista, descubre por qué el ratio chars/token define la latencia del KV-cache."
    )
    sample_en = (
        "Understanding Byte-Pair Encoding as lossless data compression explains how sub-word "
        "merges shrink sequence length T and quadratic O(T^2) attention cost."
    )
    # Not in the training corpus: the honest compression number for this toy tokenizer.
    sample_heldout = (
        "El primer token tarda más porque el modelo procesa todo el prompt antes de responder."
    )

    raw_bytes_es = list(sample_es.encode("utf-8"))
    bpe_ids_es = encode_bpe(sample_es, merges)
    decoded_es = decode_bpe(bpe_ids_es, vocab)

    bpe_ids_en = encode_bpe(sample_en, merges)
    decoded_en = decode_bpe(bpe_ids_en, vocab)

    bpe_ids_heldout = encode_bpe(sample_heldout, merges)
    decoded_heldout = decode_bpe(bpe_ids_heldout, vocab)

    roundtrip_exact = (
        decoded_es == sample_es and decoded_en == sample_en and decoded_heldout == sample_heldout
    )
    heldout_bytes = len(sample_heldout.encode("utf-8"))
    compression_ratio_es = len(raw_bytes_es) / len(bpe_ids_es)
    chars_per_token_es = len(sample_es) / len(bpe_ids_es)
    chars_per_token_en = len(sample_en) / len(bpe_ids_en)
    quadratic_attention_saving = (len(raw_bytes_es) / len(bpe_ids_es)) ** 2

    # Benchmark token ID batch -> Embedding Table gather + Gemma sqrt(d_model) scaling
    vocab_size = 512
    rng = np.random.default_rng(42)
    emb_np = rng.standard_normal((vocab_size, d_model)).astype(np.float32) * 0.02
    tokens_np = rng.integers(0, len(vocab), size=(batch_size, seq_len), dtype=np.int32)

    emb_jax = jnp.asarray(emb_np)
    tokens_jax = jnp.asarray(tokens_np)

    ref_out = python_embedding_lookup_loop(emb_np, tokens_np)
    np_out = numpy_vectorized_embedding_lookup(emb_np, tokens_np)
    jit_out = jax_jit_embedding_lookup(emb_jax, tokens_jax).block_until_ready()
    max_diff = float(np.max(np.abs(ref_out - np.asarray(jit_out))))
    max_diff_np = float(np.max(np.abs(np_out - np.asarray(jit_out))))

    py_ms = median_ms(lambda: python_embedding_lookup_loop(emb_np, tokens_np), max(7, iters // 4))
    np_ms = median_ms(lambda: numpy_vectorized_embedding_lookup(emb_np, tokens_np), iters)
    jit_ms = median_ms(lambda: jax_jit_embedding_lookup(emb_jax, tokens_jax), iters)

    return {
        "backend": jax.default_backend(),
        "vocab_size": len(vocab),
        "raw_byte_len_es": len(raw_bytes_es),
        "bpe_token_len_es": len(bpe_ids_es),
        "compression_ratio_es": compression_ratio_es,
        "chars_per_token_es": chars_per_token_es,
        "chars_per_token_en": chars_per_token_en,
        "quadratic_attention_saving": quadratic_attention_saving,
        "heldout_byte_len": heldout_bytes,
        "heldout_token_len": len(bpe_ids_heldout),
        "heldout_compression_ratio": heldout_bytes / len(bpe_ids_heldout),
        "first_merges": [vocab[256 + i].decode("utf-8", errors="replace") for i in range(4)],
        "roundtrip_exact": roundtrip_exact,
        "gather_rows": batch_size * seq_len,
        "py_lookup_ms": py_ms,
        "numpy_vectorized_ms": np_ms,
        "jit_lookup_ms": jit_ms,
        "speedup_vs_py": py_ms / jit_ms,
        "speedup_vs_numpy_vectorized": np_ms / jit_ms,
        "max_diff": max_diff,
        "max_diff_numpy_vectorized": max_diff_np,
    }


def main() -> None:
    metrics = run_experiment()
    merges = " | ".join(repr(m) for m in metrics["first_merges"])
    print("=" * 76)
    print("Experiment 01: BPE Tokenizer as Lossless Data Compression & Embedding Gather")
    print("=" * 76)
    print(f"Backend Device                : {metrics['backend']}")
    print(f"Trained Vocabulary Size       : {metrics['vocab_size']} tokens (256 UTF-8 bytes + 128 merges)")
    print(f"First Merges Learned          : {merges}")
    print(f"Lossless UTF-8 Round-Trip     : {metrics['roundtrip_exact']} (decode(encode(s)) == s)")
    print(
        f"Spanish Sample (in corpus)    : {metrics['raw_byte_len_es']} bytes -> "
        f"{metrics['bpe_token_len_es']} tokens ({metrics['compression_ratio_es']:.2f}x shorter)"
    )
    print(
        f"Held-out Sentence (unseen)    : {metrics['heldout_byte_len']} bytes -> "
        f"{metrics['heldout_token_len']} tokens ({metrics['heldout_compression_ratio']:.2f}x shorter)"
    )
    print(
        f"Attention Token Pairs         : {metrics['quadratic_attention_saving']:.2f}x fewer "
        f"({metrics['compression_ratio_es']:.2f}^2, arithmetic)"
    )
    print("-" * 76)
    print(f"Gather {metrics['gather_rows']:,} token rows, median ms after warmup")
    print(f"Python for-loop, row by row   : {metrics['py_lookup_ms']:.4f} ms")
    print(f"NumPy vectorized emb[ids]     : {metrics['numpy_vectorized_ms']:.4f} ms")
    print(f"JAX @jit gather               : {metrics['jit_lookup_ms']:.4f} ms")
    print(f"@jit vs Python loop (naive)   : {metrics['speedup_vs_py']:.2f}x")
    print(f"@jit vs NumPy emb[ids] (fair) : {metrics['speedup_vs_numpy_vectorized']:.2f}x")
    print(f"Max |Python loop - JAX @jit|  : {metrics['max_diff']:.2e}")
    print("=" * 76)


if __name__ == "__main__":
    main()
