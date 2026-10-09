"""Where a pooled decode round's 180 ms actually goes: reads, pack (H2D), or place.

The disk probe says 18-20 GiB/s at 4-8 readers, so a round's ~566 MiB of misses should read in ~30 ms. The served
engine does 5.3 tok/s (≈180 ms a round for 63 rounds), so the time is somewhere else in the fill. This times each
stage on the real checkpoint, one layer's 8 experts at a time, the way a round does it.
"""

import glob
import os
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, os.environ.get("TF_SRC", "src"))

import torch                                                            # noqa: E402

from tensorfold.families.qwen3_5_moe.cuda import pool as P               # noqa: E402

root = Path(glob.glob(str(Path.home() / ".cache/huggingface/hub/models--TensorFold--Qwen3.6*/snapshots/*"))[0])
LAYERS, EXPERTS, SLOTS = 40, 256, 606
pool = P.ExpertPool(root, LAYERS, EXPERTS, gs=64, slots=SLOTS, device="cuda")
torch.cuda.synchronize()


def timed(fn, n=32):
    times = []
    for _ in range(n):
        t = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t) * 1e3)                    # ms
    return statistics.median(times)


pairs = [(0, e) for e in range(8)]                                       # one layer's routed experts, all misses
slot_of = {pair: i for i, pair in enumerate(pairs)}

read_ms = timed(lambda: pool.readers.map(lambda p: pool._read(*p), pairs), 16)
blocks = list(pool.readers.map(lambda p: pool._read(*p), pairs))
pack_ms = timed(lambda: [pool._pack(b) for b in blocks], 16)
packed = [pool._pack(b) for b in blocks]
place_ms = timed(lambda: [pool._place(slot_of[p], q) for p, q in zip(pairs, packed)], 16)
fill_ms = timed(lambda: pool._fill(pairs, slot_of), 16)

print(f"one layer's 8 experts (a round has 40 layers)")
print(f"  read  (8 readers over the files)   {read_ms:7.2f} ms   {read_ms / 8 * 1e3:6.0f} us/expert")
print(f"  pack  (H2D + grouped.pack a proj)  {pack_ms:7.2f} ms   {pack_ms / 8 * 1e3:6.0f} us/expert")
print(f"  place (3 copies into the slots)    {place_ms:7.2f} ms")
print(f"  fill  (read + pack + place)        {fill_ms:7.2f} ms")
round_ms = fill_ms * LAYERS
print(f"a whole round, all misses: {round_ms:6.1f} ms -> {1000 / round_ms:4.1f} tok/s "
      f"(the served engine measured 5.3-5.8 tok/s)")
print(f"bytes a round moves: {pairs.__len__() * LAYERS * 1.769472:.0f} MiB of expert rows")
pool.close()
