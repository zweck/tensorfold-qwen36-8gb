# Qwen3.6-35B-A3B served from an expert pool on an 8 GB laptop GPU

Files: `demo.cast` (raw asciinema v3), `demo.gif` (primary, 41 frames), `demo.mp4` (24 s), `transcript.txt`,
this note. Recorded 2026-10-09 on the development laptop itself.

## What it shows

1. The hardware the model is running on: `NVIDIA GeForce RTX 5070 Laptop GPU, 8151 MiB, 595.71.05`, with the
   served engine holding **2,764 MiB** of it (a second, unrelated process holds 1,368 MiB — the desktop).
2. The engine's own startup receipt: **startup estimate 5.33 GiB within 5.36 GiB**; 262,144 native context with a
   **32,768-token window**; **expert pool 1.00 GiB = 606 slots** serving **16.88 GiB of routed experts read from
   the checkpoint's files**; loaded in 4.1 s.
3. Two live requests, streamed token by token:
   - a code answer: 29 tokens, 10.8 s, **3.4 tok/s**, first token 2.56 s;
   - a **tool call**: `get_weather({"city":"Reykjavik"})`, 29 tokens, 17.2 s, **3.8 tok/s**, first token 9.92 s.
4. The pool's own live receipt: **219,907 experts read, 90,955 hits (29.3%), residency 566/606**.

## What the numbers mean, in plain terms

Qwen3.6-35B-A3B has 35 billion parameters but only ~3 billion are active per token, and 16.88 GiB of them live in
the routed experts. Instead of holding that in VRAM, only a 1 GiB pool of them is resident (606 of 10,240
layer-expert pairs) and the rest are read from the checkpoint's files on demand. That is why a 35B model fits on an
8 GB card: the plan charges 5.33 GiB for weights, pool and working memory against the 5.36 GiB the card grants.

It is **not fast**: 3.4-3.8 tok/s, against a design bar of 8. The measured cause is not the disk (the pool's reads
take 4 µs an expert here) but the per-missing-expert host-to-device transfer: ~500 µs each, ~320 a round
(see the plan doc's Appendix D).

## Method

- `asciinema` 3.1.0 static x86_64 in `~/x-media/tools/` (the box had none); `agg` x86_64 static for the GIF
  (`--theme dracula --font-size 20 --fps-cap 60 --idle-time-limit 3 --last-frame-duration 5`); `ffmpeg` for the
  MP4 (`-movflags +faststart -pix_fmt yuv420p`).
- Commands are shown by an `echo` of the line actually run; the real commands are shell functions sourced from a
  hidden env file, so **no path, username or hostname is ever typed into the frame**.
- **Playback compresses idle gaps to 3 s** (asciinema `-i 3`, agg `--idle-time-limit 3`). The session's own
  timings in the footers and receipts are the real ones (10.8 s / 17.2 s), not the playback's.
- The final paint's interval in the cast was 1 ms (it landed in the same frame as the line before it), so it is
  set to 0.6 s — the last receipt line now gets a frame of its own. No content was changed.

## Privacy checks run

- `grep` of the transcript text and the raw cast for `/home`, the username, `100.`/`192.168` octets, hostnames
  (`maxq`, `spark`, `numenor`), `.hermes`, `Bearer`, `sk-`, `hf_`: **no hits** (the only match anywhere was the
  benign word "cache" in `buff/cache` and `cache slots`).
- The rendered frames were OCR-read: no path, username or hostname appears.
- Nothing here is posted anywhere; no credentials appear in any frame or file.

## Verification (what was and was not checked)

- OCR of rendered frames at t=1, 7, 13 and the final frame (negated, `tessdata_fast` `eng` downloaded to
  `~/x-media/tools/tessdata` — the system `tesseract` had no language data). Every line quoted above was read back
  off the pixels, including the last two receipt lines.
- No vision-model check was possible (the configured vision endpoint returned 401), so **a human should still
  watch the GIF once before posting** — the pixel check above confirms the text is present and legible, not the
  aesthetic.
- Honest caveats for any caption: this box has 8 GB of VRAM but **62 GiB of RAM** where the target machine has 32
  GB; the engine was already running when recording began (the 4.1 s load line is that boot's); the branch
  (`host-tier/laptop-experts`) is **unpushed and unmerged**.

## Suggested caption (≤265 chars, no names or internal ids)

> 35B params, 16.9 GiB of routed experts, running on an 8 GB laptop GPU. Only 1 GiB of experts lives in VRAM; the
> rest is read from the checkpoint's files on demand. 3.4 tok/s, tools work. Not fast — but it fits.
