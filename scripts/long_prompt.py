"""Post one long prompt at the served engine: the prefill path's real TTFT and, in the server log, its peak memory.

The gate's prompts are 14-31 tokens, so nothing has exercised a prompt chunk (this family prefills in 4096-row
chunks). This sends roughly ``--tokens`` tokens of prose and prints what the stream did; the server's own
``done req-`` line carries the authoritative prefill time, and its shutdown receipt the peak.
"""

import argparse
import json
import time
import urllib.request

WORDS = ("the quick brown fox jumps over the lazy dog while the engine streams experts from the checkpoint's "
         "files and every routed expert arrives a page at a time until the answer is finished").split()


def build(target_tokens: int) -> str:
    """Roughly ``target_tokens`` of prose: ~1.3 tokens a word for this kind of text, rounded up."""

    words = []
    while len(words) < target_tokens * 1.3:
        words += WORDS
    return " ".join(words[: int(target_tokens * 1.3)])


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("base")
    p.add_argument("model")
    p.add_argument("--tokens", type=int, default=4500)
    p.add_argument("--reply", type=int, default=32)
    args = p.parse_args()

    prompt = build(args.tokens)
    body = json.dumps({"model": args.model, "temperature": 0.0, "max_tokens": args.reply, "stream": True,
                       "messages": [{"role": "system", "content": "Answer in one short sentence."},
                                    {"role": "user", "content": prompt + "\n\nSummarise the above."}]}).encode()
    request = urllib.request.Request(args.base + "/v1/chat/completions", data=body,
                                     headers={"Content-Type": "application/json"})
    started, first, chunks, text = time.perf_counter(), None, 0, []
    with urllib.request.urlopen(request, timeout=1800) as response:
        for line in response:
            line = line.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            try:
                piece = json.loads(line[6:])["choices"][0].get("delta", {}).get("content") or ""
            except (KeyError, ValueError):
                continue
            if piece:
                chunks += 1
                text.append(piece)
                first = first if first is not None else time.perf_counter() - started
    total = time.perf_counter() - started
    print(json.dumps({"prompt_chars": len(prompt), "chunks": chunks, "ttft_s": round(first or -1, 3),
                      "total_s": round(total, 3), "reply_chars": len("".join(text)),
                      "reply": "".join(text)[:80]}))


if __name__ == "__main__":
    main()
