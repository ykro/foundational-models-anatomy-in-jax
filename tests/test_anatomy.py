"""Automated pytest suite verifying all 5 Foundational Models Anatomy in JAX experiments."""

from __future__ import annotations

import importlib

exp01 = importlib.import_module("experiments.01_bpe_tokenizer_as_data_compression")
exp02 = importlib.import_module("experiments.02_self_attention_as_soft_dictionary")
exp03 = importlib.import_module("experiments.03_gemma_block_rms_geglu")
exp04 = importlib.import_module("experiments.04_kv_cache_prefill_vs_decode")
exp05 = importlib.import_module("experiments.05_sampling_loop_temperature_top_p")


def test_01_bpe_tokenizer_as_data_compression() -> None:
    res = exp01.run_experiment(num_merges=128, batch_size=64, seq_len=64, d_model=128, iters=20)
    assert res["roundtrip_exact"] is True
    assert res["vocab_size"] == 256 + 128
    assert res["compression_ratio_es"] >= 2.0
    assert res["quadratic_attention_saving"] >= 4.0
    assert res["heldout_compression_ratio"] < res["compression_ratio_es"]
    assert res["max_diff"] < 1e-5
    assert res["max_diff_numpy_vectorized"] < 1e-5
    assert res["speedup_vs_py"] >= 10.0


def test_02_self_attention_as_soft_dictionary() -> None:
    res = exp02.run_experiment(batch_size=4, num_heads=4, seq_len=64, head_dim=32, iters=20)
    assert res["rope_norm_diff"] < 1e-5
    assert res["rope_rel_diff"] < 1e-5
    assert res["causal_prefix_leak"] == 0.0
    assert res["max_diff"] < 1e-4
    assert res["max_diff_numpy_batched"] < 1e-4
    assert res["hlo_fusions"] > 1
    assert res["speedup"] >= 2.0


def test_03_gemma_block_rms_geglu() -> None:
    res = exp03.run_experiment(
        batch_size=4, seq_len=64, d_model=128, num_q_heads=8, num_kv_heads=2, head_dim=16, iters=20
    )
    assert res["kv_memory_saving_ratio"] == 4.0
    assert abs(res["rms_mean"] - 1.0) < 1e-2
    assert res["max_diff"] < 1e-4
    assert res["eager_top_level_ops"] > res["hlo_fusions"]
    assert res["speedup"] >= 1.2


def test_04_kv_cache_prefill_vs_decode() -> None:
    res = exp04.run_experiment(batch_size=2, prompt_len=64, gen_steps=64, iters=10)
    assert res["exact_token_match"] is True
    assert res["generated_tokens_identical"] == res["generated_tokens_total"]
    assert res["max_logit_diff"] < 1e-4
    assert res["speedup"] >= 2.0
    steps = res["decode_step_ms_by_cache_len"]
    assert steps["8192"] > 2.0 * steps["128"]


def test_05_sampling_loop_temperature_top_p() -> None:
    res = exp05.run_experiment(steps=16, batch_size=32, vocab_size=256, iters=20)
    assert res["greedy_parity"] is True
    assert res["grammar_compliance_rate"] == 100.0
    assert res["loop_of_jit_matches_scan"] is True
    assert 1.0 <= res["active_support_mean"] <= 16.0
    assert res["speedup"] >= 5.0
