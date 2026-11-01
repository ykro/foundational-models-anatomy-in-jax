# Anatomía de un Foundational Model en JAX: Tokenización, Atención, KV-Cache y Sampling Loop para Software Devs

- **Por qué es relevante:** Abrir un *Foundational Model* (estilo Gemma) desde la perspectiva de ingeniería de sistemas demuestra que un LLM es una tubería determinista de estructuras de datos en memoria: un compresor sin pérdida (`BPE`), un diccionario diferenciable con rotación relativa (`Self-Attention` + `RoPE`), bloques modulares (`RMSNorm` + `Grouped-Query Attention` + `GeGLU`), un buffer prealocado (`KV-cache`) y un filtro estadístico sobre `logits` (`Sampling Loop`).
- **Para quién es:** Ingenieros de software y desarrolladores backend que ya conocen los fundamentos de JAX, redes neuronales y embeddings, y quieren entender qué ocurre exactamente en memoria durante las fases de `prefill` y `decode` de un modelo fundacional moderno sin depender de cajas negras.
- **Quién debería verlo:** Equipos que diseñan motores de inferencia, agentes con salidas estructuradas (`JSON Grammar Mask`), optimización de latencia o dimensionamiento de memoria de `KV-cache` sobre CPU, GPU o Cloud TPU.
- **Serie completa:** Cuarto repositorio de la Serie B dentro de [JAX & OpenXLA Field Guide](https://github.com/ykro/jax-openxla-field-guide).

---

## Arquitectura y Modelo Mental

### 1. El Stack de Estructuras de Datos de un Foundational Model

Un modelo fundacional transforma un `string` UTF-8 en el siguiente token atravesando seis capas con invariantes de forma explícitos: compresión de bytes a sub-palabras con Byte-Pair Encoding (`int32[B, T]`), lectura en la tabla de embeddings escalada por `sqrt(d_model)` (`float32[B, T, D]`), `N` bloques Gemma que preservan la forma del tensor residual (`RMSNorm`, `Grouped-Query Attention` con razón 8:2 y compuerta `GeGLU`), proyección final a `logits` sobre el vocabulario (`float32[B, V]`) y el `sampling loop` autorregresivo.

![Stack de estructuras de datos de un Foundational Model en JAX](diagrams/01-foundational-model-data-structures.svg)

### 2. Fase `Prefill` vs Fase `Decode` con `KV-Cache` Estático (`O(T^2)` vs `O(T)` por paso)

Durante la generación autorregresiva, recalcular toda la secuencia desde el primer token en cada paso desperdicia cómputo de forma cuadrática porque la máscara causal garantiza que las llaves (`K`) y valores (`V`) de los tokens pasados jamás cambian. Separar la fase paralela de `prefill` y mantener un `KV-cache` prealocado actualizado en `O(1)` mediante `jax.lax.dynamic_update_slice` reduce cada paso de `decode` a proyectar un único token `[B, 1]` manteniendo paridad exacta de `logits`.

![Comparación entre recómputo cuadrático ingenuo y Prefill + KV-Cache Decode](diagrams/02-prefill-vs-kv-cache-decode.svg)

### 3. Pipeline del `Sampling Loop`: De `Logits` a Token ID con Máscara de Gramática

Un modelo fundacional termina cada pasada entregando un vector plano de `logits` (`float32[B, V]`). Antes de invocar al generador pseudoaleatorio Threefry (`jax.random.categorical`), el `sampling loop` aplica una máscara booleana de gramática o esquema JSON (`jnp.where(allowed_mask, logits, -1e9)`), escala la entropía con `temperature` y recorta la cola de ruido con `jax.lax.top_k` (`Top-K`) y probabilidad acumulada (`Top-P`).

![Pipeline de muestreo desde logits hasta Token ID en JAX](diagrams/03-logits-to-token-sampling-pipeline.svg)

---

## Resultados Reales de Verificación (`verify_anatomy.py`)

Benchmarks deterministas ejecutados con Python `3.12.11`, JAX `0.12.0.1` y NumPy `2.5.4`:

| Experimento | Archivo | Baseline (Python / Loop / Eager) | JAX + OpenXLA | Speedup / Verificación |
| :--- | :--- | :--- | :--- | :--- |
| **01. BPE Compression & Gather** | `experiments/01_bpe_tokenizer_as_data_compression.py` | `13.3156 ms` (Python `for-loop` lookup) | `0.3248 ms` (`@jax.jit` gather) | **41.00x** (`2.22x` compresión BPE, `4.91x` ahorro `O(T^2)`) |
| **02. Causal Self-Attention + RoPE** | `experiments/02_self_attention_as_soft_dictionary.py` | `6.9015 ms` (NumPy head `loop`) | `1.1861 ms` (`@jax.jit` RoPE causal) | **5.82x** (`leak = 0.0`, error norma RoPE `1.00e-06`) |
| **03. Gemma Block (RMS+GQA+GeGLU)** | `experiments/03_gemma_block_rms_geglu.py` | `8.1952 ms` (JAX Eager block) | `4.1231 ms` (`@jax.jit` fused block) | **1.99x** (**4.00x** ahorro RAM de `KV` con GQA 8Q:2KV) |
| **04. KV-Cache Prefill vs Decode** | `experiments/04_kv_cache_prefill_vs_decode.py` | `414.8111 ms` (Naive full recompute) | `43.0929 ms` (`Prefill` + `KV-cache`) | **9.63x** (64/64 tokens idénticos, `max_diff = 8.00e-06`) |
| **05. Sampling Loop + Grammar Mask** | `experiments/05_sampling_loop_temperature_top_p.py` | `74.3556 ms` (Python host `loop`) | `3.6776 ms` (`@jax.jit` `lax.scan`) | **20.22x** (**100.0%** cumplimiento de gramática JSON) |

Salida íntegra de terminal generada por `verify_anatomy.py`:

```text
==============================================================================
THE OUTSIDE JUDGE — Foundational Models Anatomy in JAX Verification Report
==============================================================================
Runtime : Python 3.12.11 | JAX 0.12.0.1 | NumPy 2.5.4 | Backend: cpu
------------------------------------------------------------------------------
Experiment                         | Baseline       | JAX/OpenXLA    | Speedup / Check
------------------------------------------------------------------------------
01. BPE Compression & Gather       | 13.3156 ms     | 0.3248 ms      | 41.00x (2.22x BPE)
02. Causal Self-Attention + RoPE   | 6.9015 ms      | 1.1861 ms      | 5.82x (leak=0.0)
03. Gemma Block (RMS+GQA+GeGLU)    | 8.1952 ms      | 4.1231 ms      | 1.99x (4x KV RAM)
04. KV-Cache Prefill vs Decode     | 414.8111 ms    | 43.0929 ms     | 9.63x (64/64 tok)
05. Sampling Loop + Grammar Mask   | 74.3556 ms     | 3.6776 ms      | 20.22x (100% JSON)
------------------------------------------------------------------------------
Saved JSON report to: benchmarks/latest_results.json
ALL DETERMINISTIC CHECKS PASSED.
```

---

## Estructura del Repositorio

```text
.
├── benchmarks/
│   ├── latest_results.json                          # Métricas verificadas por verify_anatomy.py
│   └── terminal_output.txt                          # Salida íntegra del Outside Judge
├── diagrams/
│   ├── 01-foundational-model-data-structures.{html,svg,png}
│   ├── 02-prefill-vs-kv-cache-decode.{html,svg,png}
│   └── 03-logits-to-token-sampling-pipeline.{html,svg,png}
├── experiments/
│   ├── 01_bpe_tokenizer_as_data_compression.py      # Tokenizador BPE reversible y gather compilado
│   ├── 02_self_attention_as_soft_dictionary.py      # Diccionario suave Q/K/V, máscara causal y RoPE
│   ├── 03_gemma_block_rms_geglu.py                  # Bloque Gemma completo: RMSNorm, GQA (8:2) y GeGLU
│   ├── 04_kv_cache_prefill_vs_decode.py             # Fase Prefill vs Decode con KV-cache y dynamic_update_slice
│   └── 05_sampling_loop_temperature_top_p.py        # Sampling loop con Temperature, Top-K, Top-P y JSON mask
├── gcp/
│   └── run_on_demand_anatomy.sh                     # Runner on-demand con auto-teardown en Cloud TPU v5e y L4 GPU
├── tests/
│   └── test_anatomy.py                              # Suite pytest determinista (5 tests)
├── Makefile
├── pyproject.toml
└── verify_anatomy.py                                # El Outside Judge (aserciones de paridad y speedup)
```

---

## Onboarding Paso a Paso

### 1. Instalación desde cero (macOS / Linux)

```bash
git clone https://github.com/ykro/foundational-models-anatomy-in-jax.git
cd foundational-models-anatomy-in-jax
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e ".[dev]"
```

### 2. Recorrido guiado por los 5 experimentos (Orden recomendado)

Cada script se ejecuta en pocos segundos y desarma una pieza clave de un modelo fundacional moderno:

```bash
# Paso 1 — Tokenización Byte-Pair Encoding (BPE) como compresión sin pérdida y lookup en JAX
python experiments/01_bpe_tokenizer_as_data_compression.py

# Paso 2 — Self-Attention como diccionario diferenciable (Q, K, V), máscara causal y RoPE
python experiments/02_self_attention_as_soft_dictionary.py

# Paso 3 — Anatomía de un bloque Gemma: RMSNorm, Grouped-Query Attention (GQA) y compuerta GeGLU
python experiments/03_gemma_block_rms_geglu.py

# Paso 4 — El KV-cache como estructura de datos en memoria: Prefill paralelo vs Decode O(T)
python experiments/04_kv_cache_prefill_vs_decode.py

# Paso 5 — El Sampling Loop autorregresivo: Temperature, Top-K, Top-P y máscara de gramática JSON
python experiments/05_sampling_loop_temperature_top_p.py
```

### 3. Verificación completa (Linter + Tests + Outside Judge)

```bash
# Ejecutar únicamente el Outside Judge y actualizar benchmarks/latest_results.json
python verify_anatomy.py

# Ejecutar ruff, pytest (5 tests) y verify_anatomy.py en un solo comando
make all
```

### 4. Ejecución on-demand con auto-teardown en Google Cloud (Cloud TPU v5e o NVIDIA L4 GPU)

El script `gcp/run_on_demand_anatomy.sh` aprovisiona instancias on-demand con auto-teardown en el proyecto `ai-experiments-487722` y garantiza la limpieza automática de los recursos mediante `trap cleanup EXIT INT TERM`.

```bash
# Ejecutar benchmark en una instancia on-demand de Cloud TPU v5e-1 (us-central1-a)
make gcp-tpu

# Ejecutar benchmark en un Vertex AI Custom Job on-demand con GPU NVIDIA L4
make gcp-gpu
```
