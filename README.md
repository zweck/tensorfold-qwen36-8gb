# Qwen3.6-35B-A3B on an 8 GB GPU: a TensorFold expert pool, measured

Qwen3.6-35B-A3B has 35B parameters, about 3B active a token, and **16.88 GiB of routed experts**. TensorFold's
CUDA line normally holds every expert resident, so this checkpoint wants ~23 GiB of VRAM and refuses to start
below that. With `--expert-pool` a **1 GiB pool** of experts lives in VRAM (606 of its 10,240 layer-expert pairs)
and the rest are read from the checkpoint's files on demand — which is what lets an 8 GB card serve the model, at a
32,768-token window, with tool calls working.

**Status: prototype, proposed upstream, not merged.**

| | |
| --- | --- |
| Proposal (the gate question, full results log) | https://github.com/ashhart/TensorFold/issues/554 |
| Engine change (`--expert-pool`) | https://github.com/ashhart/TensorFold/pull/555 |
| Recipe section (docs) | https://github.com/ashhart/TensorFold/pull/556 |
| Fork branch this ran from | `zweck/TensorFold`, `feat/expert-pool-cuda` |

## Measured

One RTX 5070 Laptop (8,151 MiB, 7.53 GiB visible to CUDA), `--expert-pool 1 --context 32768 --no-drafts`,
checkpoint `TensorFold/Qwen3.6-35B-A3B-MLX-4bit-MTP`:

| | |
| --- | --- |
| Admission, planned / granted | 5.33 GiB / 5.36 GiB |
| Decode | 3.4 tok/s on real requests; 5.3-5.8 warm greedy |
| First token | 1.1-2.2 s short prompts; 12.9 s on a 6,057-token prompt |
| Pool | 28-34% of reads hit; residency saturated at 566/606 slots |
| Engine peak | 2.68 GiB allocated, 2.81 GiB reserved |

- Same engine, one RTX 5090, `--expert-pool 1.5 --context 32768`: **11.9-12.6 tok/s**, 40.5% of reads hit.
- Exactness: pooled expert rows are bit-identical to resident ones at the kernel level, including after eviction
  churn; at the model level every reply equals its `"draft": false` run and every concurrent reply equals its solo
  run (four cells, none failed).

## What it costs, honestly

- **3.4-5.8 tok/s is below the 8 tok/s bar this work set itself.** The measured cause is not the disk: an expert's
  read is 4 µs warm, while its pageable host-to-device copy and pack cost 502 µs — 320 of those a round is 99% of
  a 146 ms round. Pinned staging is the next lever and is not in the PR.
- **Rounds decode eagerly, without CUDA graphs** (the fill is host-driven and cannot be captured).
- **A pool must hold a layer's window** (~0.49 GiB here) or a prompt routing more distinct experts is refused.
- **The kernels take at most 1,024 experts in a stack**, so a pool caps at 1,024 slots (1.73 GiB); a larger
  `--expert-pool` is clamped at startup with a line saying so.
- **The fit is tight**: 0.03 GiB of slack at 32,768 tokens on this card, so `--expert-pool 0.5` is the safer
  default. This box has 62 GiB of RAM where the laptop class it targets has 32; the pool's reads come off disk
  (host RAM only warms the page cache), and the read figures above are page-cache-warm.

## Reproduce

```bash
# 1. the engine change (until it lands upstream)
git clone git@github.com:zweck/TensorFold.git && cd TensorFold
git checkout feat/expert-pool-cuda

# 2. an environment: python 3.13, torch 2.14.1+cu130, triton 3.8, ninja, and CUDA 13 headers
python3.13 -m venv ~/.venvs/tfx && uv pip install --python ~/.venvs/tfx/bin/python \
    torch --index-url https://download.pytorch.org/whl/cu130
uv pip install --python ~/.venvs/tfx/bin/python numpy ninja huggingface_hub tokenizers jinja2 triton

# 3. the checkpoint (~20 GB)
~/.venvs/tfx/bin/hf download TensorFold/Qwen3.6-35B-A3B-MLX-4bit-MTP

# 4. serve it: first boot compiles the CUDA kernels (~35 s), then it answers on 127.0.0.1:8099
SRC=$PWD V=~/.venvs/tfx/bin scripts/serve-pool.sh
#    POOL=0.5 CTX=8192 SRC=$PWD V=~/.venvs/tfx/bin scripts/serve-pool.sh   # roomier, faster

# 5. talk to it
scripts/tf ask "Write a Python one-liner that returns the nth Fibonacci number." 24
scripts/tf stream "Count 1 to 5." 20
scripts/tf tool "What is the weather in Reykjavik?" 40
scripts/tf stats          # the engine's own receipts: pool hit rate, per-request tok/s
```

`chat_template_kwargs: {"enable_thinking": false}` matters: thinking is on by default and will spend a small
`max_tokens` budget before the answer, leaving `content` empty. `scripts/tf` sets it.

`scripts/pool_run.sh` is the harness the numbers above came from (boot, the house exactness gate, decode speed,
the pool's receipts); `scripts/fill_bench.py` is the attribution of a round's time into read, pack and place.

## Media

`media/demo.gif` (and `demo.mp4`) show the hardware and two live requests — a code answer and a tool call — with
the engine's own startup and pool receipts in frame. `media/NOTES.md` carries the numbers and the privacy checks
run before it was recorded; `media/transcript.txt` is the session as text.

## Licence and attribution

Apache-2.0 (`LICENSE`, `NOTICE`). The recipe text in `docs/` is proposed for TensorFold's own recipe in PR #556 and
is theirs, Apache-2.0; the checkpoints are `TensorFold/Qwen3.6-35B-A3B-MLX-4bit-MTP` (Apache-2.0, quantized from
`Qwen/Qwen3.6-35B-A3B`). The harnesses here were written for this work.
