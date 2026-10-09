# Serving Qwen3.6-35B-A3B on an 8 GB card

This is the section proposed for TensorFold's `docs/recipes/qwen3.6-moe.md` in PR #556, reproduced here so it
travels with the measurements. TensorFold's docs are Apache-2.0 (see NOTICE); the heading level is adapted.

## Serving a card smaller than the checkpoint (`--expert-pool`) (`--expert-pool`)

`--expert-pool GIB` keeps a pool of routed experts in VRAM and reads the rest of the 16.88 GiB from the
checkpoint's files on demand: the pool's slots are the grouped kernels' landing buffers, so a miss is a host read
into the slot the call assigned rather than a separate staging copy. The plan charges the pool's bytes instead of
every expert's, which is what lets a card below the checkpoint's size admit at all.

```bash
tensorfold serve TensorFold/Qwen3.6-35B-A3B-MLX-4bit-MTP --expert-pool 1 --context 32768 --no-drafts
```

Measured on one RTX 5070 Laptop (8,151 MiB, 7.53 GiB visible to CUDA), this checkpoint, a 32,768-token window and
a 1.00 GiB pool (606 slots, 566 of them routed, of 10,240 layer-expert pairs). The pool's reads come off disk, so
host RAM above a few GB only warms the page cache:

| | 1.00 GiB pool, `--no-drafts` |
| --- | ---: |
| Admission, planned / granted | 5.33 GiB / 5.36 GiB |
| Decode | 3.4 tok/s on real requests; 5.3-5.8 warm greedy |
| First token | 1.1-2.2 s on short prompts; 12.9 s on a 6,057-token prompt |
| Pool | 28-34% of reads hit, residency saturated at 566/606 |
| Engine peak | 2.68 GiB allocated, 2.81 GiB reserved |

The same engine at `--expert-pool 1.5 --context 32768` on one RTX 5090 decodes at 11.9-12.6 tok/s (40.5% hits).
Every reply equals its `"draft": false` run and every concurrent reply equals its solo run, as on the resident
path, and pooled rows are bit-identical to resident ones at the kernel level.

What a pool changes, and what it costs:

- **Rounds decode eagerly, without CUDA graphs.** The fill is host-driven, so it cannot run inside graph capture:
  a pooled request does not get the per-width graph replay the resident path has.
- **A pool must hold a layer's window.** Below about 0.49 GiB on this checkpoint it cannot hold a layer's 256
  experts, and a prompt routing more distinct experts than the pool has slots is refused.
- **The kernels take at most 1,024 experts in a stack**, so a pool caps at 1,024 slots (1.73 GiB here); a larger
  `--expert-pool` is clamped at startup rather than failing per request.
- **`--no-drafts` reads fewer bytes a round**: a serial round touches about 320 expert pairs against roughly
  1,600 for a four-row MTP chain, and therefore hits more often — 41.4% against 29.8% measured on the 5090.
- **Mind the margin.** At 32,768 tokens on an 8 GB card the fit has 0.03 GiB of slack, so `--expert-pool 0.5` is
  the safer default there, at some cost in hit rate.

Where the time goes, measured on the same box: one expert's read is 4 µs warm and 88 µs cold, while its
host-to-device copy and pack cost 502 µs — 320 of those a round is 99% of a 146 ms round. The pool is still a
pageable transfer per missing expert; pinned staging is the next lever, not the disk.
