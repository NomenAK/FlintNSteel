"""Per-kernel GPU time of the fns-bench -p prefill (nsys), over the last measured repetition.

  python prof_prefill.py <name> --model <gguf> [--out DIR] [fns-bench args...]
  e.g.: python prof_prefill.py pp32k --model MODEL.gguf -p 4096 -d 32768 -ub 2048

Writes OUT_DIR/fns-<name>.nsys-rep and .sqlite. It then prints total time per kernel for the kernels launched after
the last k_embed_batch of the fill: the timed prefill starts with one k_embed_batch per batch.
Environment and defaults are those of prof_decode.py (FNS_BUILD_DIR, nsys on PATH, FNS_BUILD_DIR/bench/prof).
"""

import argparse
import collections
from pathlib import Path

from prof_decode import DEFAULT_OUT, capture, kernel_rows


def main() -> None:
    ap = argparse.ArgumentParser(description="Per-kernel GPU time of the fns-bench prefill.")
    ap.add_argument("name", help="run name; the report is written as fns-<name>")
    ap.add_argument("--model", required=True, help="GGUF model path passed to fns-bench")
    ap.add_argument("--out", default=str(DEFAULT_OUT), help="output directory (default: FNS_BUILD_DIR/bench/prof)")
    o, bench_args = ap.parse_known_args()  # everything else goes to fns-bench
    rep = capture(o.name, o.model, Path(o.out), bench_args, ["-t", "cuda"])
    rows = kernel_rows(rep)
    n_ub = int(bench_args[bench_args.index("-ub") + 1]) if "-ub" in bench_args else 512
    n_p = int(bench_args[bench_args.index("-p") + 1])
    chunks = (n_p + n_ub - 1) // n_ub
    starts = [s for s, _, n in rows if n.startswith("k_embed_batch")]
    a = starts[-chunks]
    agg, cnt = collections.Counter(), collections.Counter()
    for s, e, n in rows:
        if s >= a:
            agg[n] += e - s
            cnt[n] += 1
    wall = (rows[-1][1] - a) / 1e6
    busy = sum(agg.values()) / 1e6
    print(f"{wall:.1f} ms ({n_p / wall * 1e3:.0f} tok/s), GPU active {busy:.1f} ms ({busy / wall:.0%})")
    for n, t in agg.most_common(16):
        print(f"  {t / 1e6:8.2f} ms {t / 1e6 / wall:6.1%} x{cnt[n]:5d}  {n[:70]}")


if __name__ == "__main__":
    main()
