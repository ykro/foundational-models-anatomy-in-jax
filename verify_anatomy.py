"""The Outside Judge: Deterministic Verification & Benchmark Runner for Foundational Models Anatomy.

Executes all 5 Foundational Model anatomy experiments in JAX:
01. Byte-Pair Encoding (BPE) Tokenizer as Lossless Data Compression & @jit Gather
02. Self-Attention as Soft Dictionary with Causal Mask & Rotary Position Embeddings (RoPE)
03. Gemma-Style Block Anatomy (RMSNorm + Grouped-Query Attention + GeGLU)
04. KV-Cache In-Memory Data Structure (Prefill vs Decode Loop with dynamic_update_slice)
05. Autoregressive Sampling Loop (Temperature, Top-K, Top-P & Grammar Mask)

Writes `benchmarks/latest_results.json` and `benchmarks/terminal_output.txt`.

The checks (round-trip, causal leak, parity, token match, mask compliance) are deterministic
with fixed seeds. The timings are medians after a warmup and vary run to run. Every row prints a
naive baseline next to a fairer reference (see "Cómo leer estas cifras" in the README). The
models are toy-sized with random weights; nothing here loads a real Gemma checkpoint.
"""

from __future__ import annotations

import importlib
import json
import os
import platform
import subprocess
from pathlib import Path
from typing import Any

import jax
import numpy as np

exp01 = importlib.import_module("experiments.01_bpe_tokenizer_as_data_compression")
exp02 = importlib.import_module("experiments.02_self_attention_as_soft_dictionary")
exp03 = importlib.import_module("experiments.03_gemma_block_rms_geglu")
exp04 = importlib.import_module("experiments.04_kv_cache_prefill_vs_decode")
exp05 = importlib.import_module("experiments.05_sampling_loop_temperature_top_p")


def _round_metrics(d: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in d.items():
        if isinstance(v, float):
            out[k] = round(v, 6) if abs(v) < 1e-2 else round(v, 4)
        elif isinstance(v, dict):
            out[k] = _round_metrics(v)
        else:
            out[k] = v
    return out


def _cpu_model() -> str:
    """Human-readable CPU model (platform.processor() only says 'arm' on macOS)."""
    try:
        if platform.system() == "Darwin":
            out = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True,
                text=True,
                check=True,
            )
            return out.stdout.strip()
        for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
            if line.lower().startswith("model name"):
                return line.split(":", 1)[1].strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return platform.processor() or "unknown"


def _load_average() -> list[float]:
    try:
        return [round(x, 2) for x in os.getloadavg()]
    except OSError:
        return []


def run_all_verifications() -> dict[str, Any]:
    """Run all 5 experiments and enforce deterministic verification assertions."""
    load_before = _load_average()
    r01 = _round_metrics(exp01.run_experiment())
    r02 = _round_metrics(exp02.run_experiment())
    r03 = _round_metrics(exp03.run_experiment())
    r04 = _round_metrics(exp04.run_experiment())
    r05 = _round_metrics(exp05.run_experiment())

    # Experiment 01: BPE Tokenizer & Embedding Lookup
    assert r01["roundtrip_exact"] is True, "Exp01 BPE round-trip must be 100% lossless"
    assert r01["compression_ratio_es"] >= 2.0, (
        f"Exp01 in-sample BPE compression ratio too low: {r01['compression_ratio_es']}"
    )
    assert r01["heldout_compression_ratio"] < r01["compression_ratio_es"], (
        "Exp01 a held-out sentence should compress less than the in-sample one"
    )
    assert r01["max_diff"] < 1e-5, f"Exp01 lookup parity failed: {r01['max_diff']}"
    assert r01["max_diff_numpy_vectorized"] < 1e-5, "Exp01 NumPy vectorized parity failed"
    assert r01["speedup_vs_py"] >= 10.0, f"Exp01 speedup too low: {r01['speedup_vs_py']}x"

    # Experiment 02: Self-Attention, Causal Mask & RoPE
    assert r02["rope_norm_diff"] < 1e-5, (
        f"Exp02 RoPE norm preservation failed: {r02['rope_norm_diff']}"
    )
    assert r02["rope_rel_diff"] < 1e-5, (
        f"Exp02 RoPE relative shift invariance failed: {r02['rope_rel_diff']}"
    )
    assert r02["causal_prefix_leak"] == 0.0, (
        f"Exp02 causal mask leaked future tokens: {r02['causal_prefix_leak']}"
    )
    assert r02["max_diff"] < 1e-4, f"Exp02 attention parity failed: {r02['max_diff']}"
    assert r02["max_diff_numpy_batched"] < 1e-4, "Exp02 batched NumPy parity failed"
    assert r02["hlo_fusions"] > 1, "Exp02 expected several fusions in the compiled HLO"
    assert r02["speedup"] >= 2.0, f"Exp02 speedup too low: {r02['speedup']}x"

    # Experiment 03: Gemma-style block (RMSNorm + GQA + GeGLU)
    assert r03["kv_memory_saving_ratio"] == 4.0, "Exp03 GQA (8 Q / 2 KV) must store 4.0x less K/V"
    assert abs(r03["rms_mean"] - 1.0) < 1e-2, f"Exp03 RMSNorm RMS drift: {r03['rms_mean']}"
    assert r03["max_diff"] < 1e-4, f"Exp03 Eager vs JIT parity failed: {r03['max_diff']}"
    assert r03["speedup"] >= 1.2, f"Exp03 Gemma block JIT speedup too low: {r03['speedup']}x"

    # Experiment 04: KV-Cache Prefill vs Decode
    assert r04["exact_token_match"] is True, (
        "Exp04 KV-Cache greedy tokens must match full recompute"
    )
    assert r04["max_logit_diff"] < 1e-4, (
        f"Exp04 KV-Cache logit parity failed: {r04['max_logit_diff']}"
    )
    assert r04["speedup"] >= 2.0, f"Exp04 KV-Cache speedup too low: {r04['speedup']}x"
    steps = r04["decode_step_ms_by_cache_len"]
    assert steps["8192"] > 2.0 * steps["128"], (
        f"Exp04 a decode step should grow with the cache length: {steps}"
    )

    # Experiment 05: Sampling Loop (Top-K, Top-P & Grammar Mask)
    assert r05["greedy_parity"] is True, (
        "Exp05 T=1e-4 with Top-K=16 / Top-P=0.90 must match argmax on every row"
    )
    assert r05["grammar_compliance_rate"] == 100.0, (
        "Exp05 the grammar mask must be applied before every draw"
    )
    assert r05["loop_of_jit_matches_scan"] is True, (
        "Exp05 Python loop of @jit step and lax.scan must draw the same tokens"
    )
    assert r05["speedup"] >= 5.0, (
        f"Exp05 on-device scan sampling speedup too low: {r05['speedup']}x"
    )

    summary = {
        "environment": {
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "processor": platform.processor(),
            "cpu_model": _cpu_model(),
            "cpu_count": os.cpu_count(),
            "load_average_before": load_before,
            "load_average_after": _load_average(),
            "timing": "median ms after a warmup; varies run to run",
            "jax_version": jax.__version__,
            "numpy_version": np.__version__,
            "jax_backend": jax.default_backend(),
            "jax_device": str(jax.devices()[0]),
        },
        "experiments": {
            "01_bpe_tokenizer_as_data_compression": r01,
            "02_self_attention_as_soft_dictionary": r02,
            "03_gemma_block_rms_geglu": r03,
            "04_kv_cache_prefill_vs_decode": r04,
            "05_sampling_loop_temperature_top_p": r05,
        },
        "all_verifications_passed": True,
    }
    return summary


def main() -> None:
    summary = run_all_verifications()
    bench_dir = Path(__file__).resolve().parent / "benchmarks"
    bench_dir.mkdir(parents=True, exist_ok=True)
    out_json = bench_dir / "latest_results.json"
    out_txt = bench_dir / "terminal_output.txt"
    out_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    env = summary["environment"]
    e01 = summary["experiments"]["01_bpe_tokenizer_as_data_compression"]
    e02 = summary["experiments"]["02_self_attention_as_soft_dictionary"]
    e03 = summary["experiments"]["03_gemma_block_rms_geglu"]
    e04 = summary["experiments"]["04_kv_cache_prefill_vs_decode"]
    e05 = summary["experiments"]["05_sampling_loop_temperature_top_p"]

    def load1(values: list[float]) -> str:
        return f"{values[0]:.2f}" if values else "n/a"

    steps04 = e04["decode_step_ms_by_cache_len"]
    rows = [
        (
            "01. Embedding gather (BPE ids)",
            f"loop {e01['py_lookup_ms']:.2f} ms",
            f"@jit {e01['jit_lookup_ms']:.3f} ms",
            f"{e01['speedup_vs_py']:.2f}x",
            (
                f"fair: NumPy emb[ids] {e01['numpy_vectorized_ms']:.3f} ms -> @jit "
                f"{e01['speedup_vs_numpy_vectorized']:.2f}x; BPE {e01['raw_byte_len_es']} B -> "
                f"{e01['bpe_token_len_es']} tok in-sample, held-out "
                f"{e01['heldout_compression_ratio']:.2f}x"
            ),
        ),
        (
            "02. Causal attention + RoPE",
            f"NumPy loop {e02['naive_ms']:.2f} ms",
            f"@jit {e02['jit_ms']:.3f} ms",
            f"{e02['speedup']:.2f}x",
            (
                f"fair: NumPy batched {e02['numpy_batched_ms']:.2f} ms -> @jit "
                f"{e02['speedup_vs_numpy_batched']:.2f}x; leak {e02['causal_prefix_leak']:.1f}; "
                f"HLO {e02['hlo_fusions']} fusions"
            ),
        ),
        (
            "03. Gemma-style block",
            f"eager {e03['eager_ms']:.2f} ms",
            f"@jit {e03['jit_ms']:.3f} ms",
            f"{e03['speedup']:.2f}x",
            (
                f"K/V stored {e03['mha_kv_kb']:.0f} -> {e03['gqa_kv_kb']:.0f} KB "
                f"({e03['num_q_heads']}/{e03['num_kv_heads']} heads, arithmetic); "
                f"{e03['eager_top_level_ops']} eager ops -> {e03['hlo_fusions']} fusions"
            ),
        ),
        (
            "04. KV-cache, 64 + 64 tokens",
            f"recompute {e04['naive_ms']:.1f} ms",
            f"cache {e04['kv_ms']:.2f} ms",
            f"{e04['speedup']:.2f}x",
            (
                f"fair: both @jit+scan; {e04['generated_tokens_identical']}/"
                f"{e04['generated_tokens_total']} tokens equal; one decode step "
                f"{steps04['128']:.2f} -> {steps04['8192']:.2f} ms, 128 -> 8192 slots"
            ),
        ),
        (
            "05. Sampling loop + mask",
            f"host loop {e05['py_ms']:.1f} ms",
            f"scan {e05['jit_ms']:.3f} ms",
            f"{e05['speedup']:.2f}x",
            (
                f"fair: Python loop of @jit step {e05['loop_of_jit_ms']:.2f} ms -> scan "
                f"{e05['speedup_vs_loop_of_jit']:.2f}x; mask applied "
                f"{e05['grammar_compliance_rate']:.0f}% (by construction)"
            ),
        ),
    ]
    width = 96
    lines = [
        "=" * width,
        "THE OUTSIDE JUDGE — Foundational Models Anatomy in JAX Verification Report",
        "=" * width,
        (
            f"Runtime : Python {env['python_version']} | JAX {env['jax_version']} | "
            f"NumPy {env['numpy_version']} | Backend: {env['jax_backend']}"
        ),
        (
            f"CPU     : {env['cpu_model']} ({env['cpu_count']} cores) | 1-min load average "
            f"{load1(env['load_average_before'])} before, {load1(env['load_average_after'])} after"
        ),
        "Model   : toy Gemma-style blocks, random weights (no real checkpoint)",
        "-" * width,
        f"{'Experiment':<31} | {'Baseline':<21} | {'JAX/OpenXLA':<16} | Speedup",
        "-" * width,
    ]
    for name, base, jax_col, result, fair in rows:
        lines.append(f"{name:<31} | {base:<21} | {jax_col:<16} | {result}")
        lines.append(f"    {fair}")
    lines += [
        "-" * width,
        "Checks are deterministic (fixed seeds). Times are medians after a warmup and vary",
        "run to run.",
        f"Saved JSON report to: {out_json.relative_to(Path(__file__).resolve().parent)}",
        "ALL DETERMINISTIC CHECKS PASSED.",
    ]
    report = "\n".join(lines)
    out_txt.write_text(report + "\n", encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
