#!/usr/bin/env bash
# Serve Qwen3.6-35B-A3B out of an expert pool on this laptop's 8 GB GPU, TensorFold's python-0.6 line.
# Everything but the weights comes off the checkpoint's files: only 1 GiB of experts lives in VRAM.
#
#   ./serve-pool.sh                 # 1 GiB pool, 32K window, serial decoding (what was measured)
#   POOL=0.5 ./serve-pool.sh        # smaller pool: ~0.5 GiB of slack on the card, a little slower
#   CTX=8192 ./serve-pool.sh        # short window: more room for the pool, faster
#   DRAFTS=1 ./serve-pool.sh        # MTP drafting on (more tokens a round, fewer hits)
#
# Ctrl-C stops it (the engine hands its memory back and prints the pool's receipts).
set -uo pipefail
POOL=${POOL:-1.0}
CTX=${CTX:-32768}
PORT=${PORT:-8099}
GPU=${GPU:-0}                        # CUDA_VISIBLE_DEVICES: your 8 GB GPU
SRC=${SRC:-$HOME/Development/TensorFold}      # a checkout of the branch in PR #555 (see README)
V=${V:-$HOME/.venvs/tfx/bin}              # the env from README's setup, with torch cu130
DRAFT_FLAG="--no-drafts"          # serial by default: a 1 GiB pool is under a drafting round's working set
[ -n "${DRAFTS:-}" ] && DRAFT_FLAG=""

SNAP=$(ls -d "$HOME"/.cache/huggingface/hub/models--TensorFold--Qwen3.6*/snapshots/* 2>/dev/null | head -1)
[ -n "$SNAP" ] || { echo "no checkpoint in ~/.cache/huggingface/hub — download it first"; exit 1; }
[ -x "$V/python" ] || { echo "no interpreter at $V/python"; exit 1; }

echo "model      $SNAP"
echo "pool       $POOL GiB of routed experts in VRAM, the rest read from the files on demand"
echo "window     $CTX tokens    decoding: ${DRAFTS:+drafts}drafting${DRAFTS:-serial}"
echo "endpoint   http://127.0.0.1:$PORT/v1   (OpenAI-compatible; first boot compiles kernels, ~35 s)"
echo

exec env CUDA_VISIBLE_DEVICES="$GPU" TENSORFOLD_MEMORY_RESERVE_GIB=2 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH="$SRC/src" \
    "$V/python" -m tensorfold serve "$SNAP" --backend cuda --name qwen3.6-35b-a3b \
    --context "$CTX" --expert-pool "$POOL" --parallel 1 --temperature 0 $DRAFT_FLAG \
    --host 127.0.0.1 --port "$PORT"
