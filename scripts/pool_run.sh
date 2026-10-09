#!/usr/bin/env bash
# The expert pool's first served run on maxq's 5090: boot, the house exactness gate, decode speed, pool stats.
set -uo pipefail
GPU=${GPU:-0}                        # CUDA_VISIBLE_DEVICES
PORT=${PORT:-8099}
POOL=${POOL:-2}
CTX=${CTX:-32768}
cd "${SRC:-$HOME/Development/TensorFold}" || exit 1
SNAP=$(ls -d "$HOME"/.cache/huggingface/hub/models--TensorFold--Qwen3.6*/snapshots/*)
LOG="pool-serve-$POOL.log"
rm -f "$LOG"
# a stale server from an earlier attempt on this port would answer the gate and fake a green receipt
for pid in $(pgrep -f "tensorfold serve.*--port $PORT" 2>/dev/null); do
  echo "killing a server already on port $PORT (pid $pid)"; kill -INT "$pid" 2>/dev/null; sleep 3
  kill -TERM "$pid" 2>/dev/null; sleep 2
done
if ss -tln 2>/dev/null | grep -q ":$PORT "; then echo "port $PORT is still taken; stopping"; exit 1; fi

env CUDA_VISIBLE_DEVICES="$GPU" TENSORFOLD_MEMORY_RESERVE_GIB=2 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH=src \
    setsid nohup "${V:-./venv}/bin/python" -m tensorfold serve "$SNAP" --backend cuda --name pool-smoke \
    --context "$CTX" --expert-pool "$POOL" --parallel 1 --no-thinking --temperature 0 \
    ${EXTRA:-} --host 127.0.0.1 --port "$PORT" > "$LOG" 2>&1 < /dev/null &
echo "booting; log $LOG"
for i in $(seq 1 180); do
  if grep -q "serving pool-smoke.*:$PORT" "$LOG" 2>/dev/null; then echo "up after ${i} polls"; break; fi
  if grep -qE "cannot fit|ValueError|RuntimeError|Traceback|No module named|^tensorfold:" "$LOG" 2>/dev/null; then
    echo "FAILED AT BOOT:"; tail -30 "$LOG"; exit 1
  fi
  sleep 5
done
grep -q "serving pool-smoke.*:$PORT" "$LOG" || { echo "no server of ours came up:"; tail -20 "$LOG"; exit 1; }
PID=$(pgrep -f "tensorfold serve.*--port $PORT" | head -1)
echo "server pid $PID"; tail -8 "$LOG"

echo "=== exactness: drafts against serial, concurrent against solo (house gate) ==="
timeout 1200 ./venv/bin/python tools/bench_concurrent.py "http://127.0.0.1:$PORT" pool-smoke --alone --serial \
    --tokens 24 --reps 1 2>&1 | tail -22

echo "=== decode speed (tokens 64, reps 3, greedy) ==="
timeout 900 ./venv/bin/python tools/bench_openai.py "http://127.0.0.1:$PORT" pool-smoke --tokens 64 --reps 3 \
    --temperatures 0 2>&1 | tail -14

if [ -n "${LONG:-}" ]; then
  echo "=== a long prompt: the prefill path's own time (chunk 4096 rows), ${LONG_TOKENS:-4500} tokens ==="
  timeout 1800 ./venv/bin/python long_prompt.py "http://127.0.0.1:$PORT" pool-smoke --tokens "${LONG_TOKENS:-4500}" 2>&1 | tail -3
  grep -E "done req-" "$LOG" | tail -2
fi

echo "=== shutdown, pool stats ===" 
[ -n "$PID" ] && kill -INT "$PID"
sleep 8
if [ -n "$PID" ] && kill -0 "$PID" 2>/dev/null; then kill -TERM "$PID"; sleep 5; fi
kill -0 "$PID" 2>/dev/null && echo "WARNING: pid $PID still alive" || echo "server stopped"
grep -E "expert pool:" "$LOG" | tail -3
tail -6 "$LOG"
echo "POOL-RUN-DONE"
