PYTHON ?= $(shell if [ -x ../../.venv-shared/bin/python ]; then echo ../../.venv-shared/bin/python; else command -v python3; fi)
PYTEST ?= $(shell if [ -x ../../.venv-shared/bin/pytest ]; then echo ../../.venv-shared/bin/pytest; else command -v pytest; fi)
RUFF ?= $(shell if [ -x ../../.venv-shared/bin/ruff ]; then echo ../../.venv-shared/bin/ruff; else command -v ruff; fi)

.PHONY: all experiments verify test lint clean gcp-gpu gcp-tpu

all: lint test verify

experiments:
	$(PYTHON) experiments/01_bpe_tokenizer_as_data_compression.py
	$(PYTHON) experiments/02_self_attention_as_soft_dictionary.py
	$(PYTHON) experiments/03_gemma_block_rms_geglu.py
	$(PYTHON) experiments/04_kv_cache_prefill_vs_decode.py
	$(PYTHON) experiments/05_sampling_loop_temperature_top_p.py

verify:
	$(PYTHON) verify_anatomy.py

test:
	$(PYTEST) -v

lint:
	$(RUFF) check .

gcp-gpu:
	bash gcp/run_on_demand_anatomy.sh gpu

gcp-tpu:
	bash gcp/run_on_demand_anatomy.sh tpu

clean:
	rm -rf __pycache__ .pytest_cache .ruff_cache experiments/__pycache__ tests/__pycache__
