"""Attribute gdn_geometry's fixed floor for Qwen3.6-35B-A3B, term by term, with the checkpoint's own config."""

import glob
import os
import sys
from pathlib import Path

sys.path.insert(0, os.environ.get("TF_SRC", "src"))

from tensorfold.cuda import capacity, geometry as G                      # noqa: E402

root = Path(glob.glob(str(Path.home() / ".cache/huggingface/hub/models--TensorFold--Qwen3.6*/snapshots/*"))[0])
t = capacity.config(root)
world = 1
linear, attention = G.layer_counts(t)
d, h = int(t["hidden_size"]), int(t["num_attention_heads"]) // world
hk = int(t["num_key_value_heads"]) // world
hd = int(t.get("head_dim") or d // int(t["num_attention_heads"]))
nk, nv = int(t["linear_num_key_heads"]) // world, int(t["linear_num_value_heads"]) // world
dk, dv = int(t["linear_key_head_dim"]), int(t["linear_value_head_dim"])
conv = int(t["linear_conv_kernel_dim"])
streams = int(t.get("hc_count", 1))
indexed, kept = bool(t.get("indexed", False)), 2
width = 2 * nk * dk + 2 * nv * dv + 2 * nv
rows = 128
slots = int(t.get("num_experts_per_tok", 1)) + 1
intermediate = int(t.get("moe_intermediate_size", t.get("intermediate_size", d))) // world
extent = d * streams + int(t["vocab_size"]) // world + slots * (intermediate + d) + width + h * hd
GIB = 2 ** 30

terms = {
    "recurrent state + retained prefixes (4 x nv*dk*dv x4B, per linear layer)":
        linear * (4 * nv * dk * dv * 4),
    "conv rollback (4 x (conv-1) x (2nk*dk + nv*dv) x2B, per linear layer)":
        linear * (4 * (conv - 1) * (2 * nk * dk + nv * dv) * 2),
    f"row replay inputs (rows={rows}, per linear layer)":
        linear * rows * (width * 2 + nk * dk * 4 + nv * dv * 4 + nv * 8),
    f"activation workspace (16 x rows={rows} x extent x4B)":
        max(16 * rows * extent * 4, 0),
    f"MoE/logits/split-K scratch (2 x 32 x rows={rows} x 2560 x4B, mtp)":
        2 * 32 * rows * 2560 * 4,
}
total = sum(terms.values())
print(f"layers: linear={linear} attention={attention}   dims: d={d} h={h} hk={hk} hd={hd} nk={nk} nv={nv} "
      f"dk={dk} dv={dv} conv={conv} streams={streams} slots={slots} inter={intermediate}")
print(f"{'term':<62} {'GiB':>7}  share")
for name, value in sorted(terms.items(), key=lambda kv: -kv[1]):
    print(f"{name:<62} {value / GIB:7.3f}  {value / total:6.1%}")
print(f"{'FIXED TOTAL':<62} {total / GIB:7.3f}")

g = G.gdn_geometry(t, world, 5, mtp=True, rows=rows)          # the engine's call: reserve = depth + 1
print(f"\ngdn_geometry(rows={rows}) needed: " + "  ".join(
    f"{w}: {g.needed(w) / GIB:.3f} GiB" for w in (4096, 8192, 16384, 32768, 65536)))
for r in (4, 6, 8, 16, 32, 64):
    gr = G.gdn_geometry(t, world, 5, mtp=True, rows=r)
    print(f"  rows={r:>3}: fixed-ish need(4096)={gr.needed(4096) / GIB:6.3f}  need(32768)={gr.needed(32768) / GIB:6.3f}")
print("\nkv bytes/token:", G.kv_bytes(hd, 16) * 4 * attention / 1024, "KiB (attention layers only)")
print("cache at 32768 (the geometry's rounded term):",
      round(4 * attention * 32768 * hk * hd * 4 / GIB, 3), "GiB")
