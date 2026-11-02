#!/usr/bin/env bash
# On-demand GPU (Vertex AI L4) and Cloud TPU (v5litepod-1) benchmark runner with auto-teardown
# Project: your own (set GCP_PROJECT_ID)
# Guarantees zero orphaned cloud resources via EXIT trap.

set -euo pipefail

export CLOUDSDK_METRICS_ENVIRONMENT="${CLOUDSDK_METRICS_ENVIRONMENT:+$CLOUDSDK_METRICS_ENVIRONMENT }datacloud.antigravity"
PROJECT_ID="${GCP_PROJECT_ID:?Set GCP_PROJECT_ID to your Google Cloud project id}"
REGION="${GCP_REGION:-us-central1}"
ZONE="${GCP_ZONE:-us-central1-a}"
MODE="${1:-tpu}"
TPU_NAME="jax-fm-anatomy-bench-$(date +%s)"
GPU_JOB_ID=""

cleanup() {
  if [[ "${MODE}" == "tpu" ]]; then
    echo "[auto-teardown] Deleting on-demand TPU VM ${TPU_NAME} (if present)..."
    gcloud compute tpus tpu-vm delete "${TPU_NAME}" \
      --project="${PROJECT_ID}" \
      --zone="${ZONE}" \
      --quiet >/dev/null 2>&1 || true
  elif [[ "${MODE}" == "gpu" && -n "${GPU_JOB_ID}" ]]; then
    echo "[auto-teardown] Ensuring Vertex AI Custom Job ${GPU_JOB_ID} is cancelled if still active..."
    gcloud ai custom-jobs cancel "${GPU_JOB_ID}" \
      --project="${PROJECT_ID}" \
      --region="${REGION}" \
      --quiet >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT INT TERM

echo "============================================================"
echo " Foundational Models Anatomy in JAX — On-Demand GCP Benchmark"
echo " Project : ${PROJECT_ID}"
echo " Mode    : ${MODE}"
echo "============================================================"

if [[ "${MODE}" == "tpu" ]]; then
  echo "[1/3] Provisioning on-demand Cloud TPU v5e-1 (v5litepod-1) in ${ZONE}..."
  gcloud compute tpus tpu-vm create "${TPU_NAME}" \
    --project="${PROJECT_ID}" \
    --zone="${ZONE}" \
    --accelerator-type="v5litepod-1" \
    --version="v2-alpha-tpuv5-lite" \
    --spot

  echo "[2/3] Copying benchmark suite and installing dependencies..."
  gcloud compute tpus tpu-vm ssh "${TPU_NAME}" \
    --project="${PROJECT_ID}" \
    --zone="${ZONE}" \
    --command="mkdir -p ~/jax-bench"
  gcloud compute tpus tpu-vm scp --recurse \
    experiments verify_anatomy.py pyproject.toml \
    "${TPU_NAME}:~/jax-bench/" \
    --project="${PROJECT_ID}" \
    --zone="${ZONE}"

  echo "[3/3] Executing verify_anatomy.py on Cloud TPU v5e..."
  gcloud compute tpus tpu-vm ssh "${TPU_NAME}" \
    --project="${PROJECT_ID}" \
    --zone="${ZONE}" \
    --command="pip install -q 'jax[tpu]' -f https://storage.googleapis.com/jax-releases/libtpu_releases.html numpy && cd ~/jax-bench && python3 verify_anatomy.py"

elif [[ "${MODE}" == "gpu" ]]; then
  echo "[1/2] Submitting on-demand Vertex AI Custom Job on NVIDIA L4 GPU (g2-standard-4)..."
  BUNDLE_B64="$(tar czf - experiments verify_anatomy.py pyproject.toml | base64 | tr -d '\n')"
  GPU_JOB_ID="$(gcloud ai custom-jobs create \
    --project="${PROJECT_ID}" \
    --region="${REGION}" \
    --display-name="foundational-models-anatomy-l4-$(date +%s)" \
    --worker-pool-spec="machine-type=g2-standard-4,replica-count=1,accelerator-type=NVIDIA_L4,accelerator-count=1,container-image-uri=us-docker.pkg.dev/vertex-ai/training/pytorch-gpu.2-3.py310:latest" \
    --command="bash,-c,mkdir -p /tmp/bench && cd /tmp/bench && echo ${BUNDLE_B64} | base64 -d | tar xzf - && pip install -q 'jax[cuda12]' && python3 verify_anatomy.py" \
    --format="value(name)")"
  echo "[2/2] Streaming logs for ${GPU_JOB_ID}..."
  gcloud ai custom-jobs stream-logs "${GPU_JOB_ID}" \
    --project="${PROJECT_ID}" \
    --region="${REGION}"
else
  echo "Usage: $0 [tpu|gpu]"
  exit 1
fi
