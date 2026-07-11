#!/bin/bash
# Serve a trained gemma-4-26B-A4B SFT checkpoint (orchestrator) on 2xL40S (mkt1),
# then run eval_orchestrator over benchmarks. Self-contained: serve -> health -> eval -> teardown.
# Usage: sbatch eval_orch_gemma_matx.sbatch <1k|2k> <benchmarks> <N> <tag>
#SBATCH --job-name=eval_gemma
#SBATCH --account=mkt
#SBATCH --partition=mkt
#SBATCH -w mkt1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-gpu=10
#SBATCH --mem=200G
#SBATCH -t 24:00:00
#SBATCH --output=logs/eval_gemma_matx_%j.log

set -uo pipefail
SIZE=${1:?usage: <1k|2k> <benchmarks> <N> <tag>}
BENCHES=${2:-gaia}
N=${3:-2}
TAG=${4:-smoke}
PORT=8123

# Paths are resolved relative to the repo, or overridden by env:
#   OJ_ROOT       repo checkout           (default: two levels up from this script)
#   OJ_WORK       working-state dir       (default: $HOME/.openjarvis)
#   OJ_CKPT_ROOT  trained SFT checkpoints (default: $OJ_WORK/lambda_ckpts)
OJ_ROOT=${OJ_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
OJ_WORK=${OJ_WORK:-$HOME/.openjarvis}
OJ_CKPT_ROOT=${OJ_CKPT_ROOT:-$OJ_WORK/lambda_ckpts}

cd "$OJ_ROOT"
set -a
source .env
[ -f "$OJ_WORK/eval_key_override.sh" ] && source "$OJ_WORK/eval_key_override.sh"
set +a
export HF_HOME=${HF_HOME:?set HF_HOME to a cache dir with room for the weights}
export PYTHONPATH="$PWD/src"
# judge/expert cloud engine needs a config with an [engine.cloud] section
export OPENJARVIS_CONFIG=${OPENJARVIS_CONFIG:?set OPENJARVIS_CONFIG to a cell config.toml}
source .venv/bin/activate

if [ "$SIZE" = "base" ]; then
  CKPT=google/gemma-4-26B-A4B-it   # un-fine-tuned base orchestrator (from HF cache)
else
  CKPT=$OJ_CKPT_ROOT/sft_gemma_${SIZE}_bf16/epoch3
fi
OUT=results/orch-eval-gemma-${SIZE}-${TAG}
mkdir -p "$OUT" logs
echo "=== [$(date)] eval_gemma SIZE=$SIZE BENCHES=$BENCHES N=$N TAG=$TAG on $(hostname) ==="
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

# ---- serve: TP=2 bf16 + MANDATORY no-NVLink L40S flags (else NCCL hangs) ----
echo "=== [$(date)] launching vLLM (TP=2) serve of $CKPT ==="
NCCL_P2P_DISABLE=1 NCCL_CUMEM_HOST_ENABLE=0 VLLM_WORKER_MULTIPROC_METHOD=spawn \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python -m vllm.entrypoints.openai.api_server \
  --model "$CKPT" --served-model-name gemma-sft \
  --port $PORT --tensor-parallel-size 2 --dtype bfloat16 \
  --gpu-memory-utilization 0.90 --trust-remote-code \
  --max-model-len 32768 --enforce-eager --disable-custom-all-reduce \
  --enable-auto-tool-choice --tool-call-parser gemma4 \
  --max-num-batched-tokens 8192 --safetensors-load-strategy eager \
  --enable-prefix-caching > logs/serve_gemma_${SIZE}_${SLURM_JOB_ID}.log 2>&1 &
SERVE_PID=$!
echo "serve pid=$SERVE_PID log=logs/serve_gemma_${SIZE}_${SLURM_JOB_ID}.log"
cleanup(){ echo "=== [$(date)] teardown kill $SERVE_PID ==="; kill $SERVE_PID 2>/dev/null; sleep 5; kill -9 $SERVE_PID 2>/dev/null; }
trap cleanup EXIT

# ---- health poll (up to 25 min; TP + 52GB load is slow on L40S) ----
echo "=== [$(date)] waiting for /v1/models ==="
READY=0
for i in $(seq 1 150); do
  if ! kill -0 $SERVE_PID 2>/dev/null; then echo "!! serve died during startup"; tail -40 logs/serve_gemma_${SIZE}_${SLURM_JOB_ID}.log; exit 3; fi
  if curl -sf http://localhost:$PORT/v1/models 2>/dev/null | grep -q gemma-sft; then READY=1; echo "=== [$(date)] READY after ${i}0s ==="; break; fi
  sleep 10
done
[ $READY -eq 1 ] || { echo "!! not ready in 25min"; tail -40 logs/serve_gemma_${SIZE}_${SLURM_JOB_ID}.log; exit 4; }

# ---- eval ----
echo "=== [$(date)] running eval_orchestrator benches=$BENCHES n=$N ==="
python scripts/orchestrator/eval_orchestrator.py \
  --benchmarks "$BENCHES" --n "$N" --seed 42 \
  --orchestrator-endpoint http://localhost:$PORT/v1 \
  --orchestrator-model gemma-sft --orchestrator-api-key EMPTY \
  --judge-model gemini-2.5-flash --judge-engine cloud \
  --max-workers 4 --max-turns 8 \
  --output-dir "$OUT"
RC=$?
echo "=== [$(date)] eval exit=$RC summary=$OUT/summary.json ==="
echo "EVAL_GEMMA_DONE size=$SIZE tag=$TAG rc=$RC"
exit $RC
