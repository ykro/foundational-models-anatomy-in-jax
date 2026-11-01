"""The Outside Judge: Deterministic Verification & Benchmark Runner for Foundational Models Anatomy.

Executes all 5 Foundational Model anatomy experiments in JAX:
01. Byte-Pair Encoding (BPE) Tokenizer as Lossless Data Compression & @jit Gather
02. Self-Attention as Soft Dictionary with Causal Mask & Rotary Position Embeddings (RoPE)
03. Gemma Block Anatomy (RMSNorm + Grouped-Query Attention + GeGLU)
04. KV-Cache In-Memory Data Structure (Prefill vs Decode Loop with dynamic_update_slice)
05. Autoregressive Sampling Loop (Greedy, Temperature, Top-K, Top-P & JSON Grammar Mask)

Writes `benchmarks/latest_results.json` and `benchmarks/terminal_output.txt`.
"""

from __future__ import annotations

import importlib
import json
import platform
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
        else:
            out[k] = v
    return out


def run_all_verifications() -> dict[str, Any]:
    """Run all 5 experiments and enforce deterministic verification assertions."""
    r01 = _round_metrics(exp01.run_experiment())
    r02 = _round_metrics(exp02.run_experiment())
    r03 = _round_metrics(exp03.run_experiment())
    r04 = _round_metrics(exp04.run_experiment())
    r05 = _round_metrics(exp05.run_experiment())

    # Assertions for Experiment 01: BPE Tokenizer & Embedding Lookup
    assert r01["roundtrip_exact"] is True, "Exp01 BPE round-trip must be 100% lossless"
    assert r01["compression_ratio_es"] >= 2.0, (
        f"Exp01 BPE compression ratio too low: {r01['compression_ratio_es']}"
    )
    assert r01["max_diff"] < 1e-5, f"Exp01 lookup parity failed: {r01['max_diff']}"
    assert r01["speedup_vs_py"] >= 10.0, f"Exp01 speedup too low: {r01['speedup_vs_py']}x"

    # Assertions for Experiment 02: Self-Attention, Causal Mask & RoPE
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
    assert r02["speedup"] >= 2.0, f"Exp02 speedup too low: {r02['speedup']}x"

    # Assertions for Experiment 03: Gemma Block (RMSNorm + GQA + GeGLU)
    assert r03["kv_memory_saving_ratio"] == 4.0, (
        "Exp03 GQA (8 Q / 2 KV) must save 4.0x KV memory"
    )
    assert abs(r03["rms_mean"] - 1.0) < 1e-2, f"Exp03 RMSNorm RMS drift: {r03['rms_mean']}"
    assert r03["max_diff"] < 1e-4, f"Exp03 Eager vs JIT parity failed: {r03['max_diff']}"
    assert r03["speedup"] >= 1.5, f"Exp03 Gemma block JIT speedup too low: {r03['speedup']}x"

    # Assertions for Experiment 04: KV-Cache Prefill vs Decode
    assert r04["exact_token_match"] is True, (
        "Exp04 KV-Cache greedy tokens must match full recompute"
    )
    assert r04["max_logit_diff"] < 1e-4, (
        f"Exp04 KV-Cache logit parity failed: {r04['max_logit_diff']}"
    )
    assert r04["speedup"] >= 2.0, f"Exp04 KV-Cache speedup too low: {r04['speedup']}x"

    # Assertions for Experiment 05: Sampling Loop (Top-K, Top-P & JSON Grammar Mask)
    assert r05["greedy_parity"] is True, "Exp05 low-temp sampling must match argmax"
    assert r05["grammar_compliance_rate"] == 100.0, (
        "Exp05 grammar mask must enforce 100% valid tokens"
    )
    assert r05["speedup"] >= 5.0, (
        f"Exp05 on-device scan sampling speedup too low: {r05['speedup']}x"
    )

    summary = {
        "environment": {
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "processor": platform.processor(),
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

    b01 = f"{e01['py_lookup_ms']:.4f} ms"
    j01 = f"{e01['jit_lookup_ms']:.4f} ms"
    s01 = f"{e01['speedup_vs_py']:.2f}x ({e01['compression_ratio_es']:.2f}x BPE)"

    b02 = f"{e02['naive_ms']:.4f} ms"
    j02 = f"{e02['jit_ms']:.4f} ms"
    s02 = f"{e02['speedup']:.2f}x (leak=0.0)"

    b03 = f"{e03['eager_ms']:.4f} ms"
    j03 = f"{e03['jit_ms']:.4f} ms"
    s03 = f"{e03['speedup']:.2f}x (4x KV RAM)"

    b04 = f"{e04['naive_ms']:.4f} ms"
    j04 = f"{e04['kv_ms']:.4f} ms"
    s04 = f"{e04['speedup']:.2f}x (64/64 tok)"

    b05 = f"{e05['py_ms']:.4f} ms"
    j05 = f"{e05['jit_ms']:.4f} ms"
    s05 = f"{e05['speedup']:.2f}x (100% JSON)"

    lines = [
        "=" * 78,
        "THE OUTSIDE JUDGE — Foundational Models Anatomy in JAX Verification Report",
        "=" * 78,
        (
            f"Runtime : Python {env['python_version']} | JAX {env['jax_version']} | "
            f"NumPy {env['numpy_version']} | Backend: {env['jax_backend']}"
        ),
        "-" * 78,
        f"{'Experiment':<34} | {'Baseline':<14} | {'JAX/OpenXLA':<14} | {'Speedup / Check':<15}",
        "-" * 78,
        f"{'01. BPE Compression & Gather':<34} | {b01:<14} | {j01:<14} | {s01:<15}",
        f"{'02. Causal Self-Attention + RoPE':<34} | {b02:<14} | {j02:<14} | {s02:<15}",
        f"{'03. Gemma Block (RMS+GQA+GeGLU)':<34} | {b03:<14} | {j03:<14} | {s03:<15}",
        f"{'04. KV-Cache Prefill vs Decode':<34} | {b04:<14} | {j04:<14} | {s04:<15}",
        f"{'05. Sampling Loop + Grammar Mask':<34} | {b05:<14} | {j05:<14} | {s05:<15}",
        "-" * 78,
        f"Saved JSON report to: {out_json.relative_to(Path(__file__).resolve().parent)}",
        "ALL DETERMINISTIC CHECKS PASSED.",
    ]
    report = "\n".join(lines)
    out_txt.write_text(report + "\n", encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
