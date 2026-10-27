"""Alternating pairs of llama-bench runs: A, B, A, B, ... with a fresh process per measurement.

Usage:
  python ab_bench.py --name q3-tg --pairs 5 \
      --a "-m MODEL.gguf -ngl 99 -fa on -p 0 -n 256" \
      --b "-m MODEL.gguf -ngl 99 -fa on -p 0 -n 256 -ctk q8_0 -ctv q8_0"

A single configuration (--a without --b) gives plain repetitions with the same output format.
Writes <out-dir>/<date>-<name>.jsonl (one line per measurement; default out-dir: bench/results)
and prints medians and paired wins.

Environment: FNS_LLAMA_DIR (llama.cpp checkout, default external/llama.cpp) and
FNS_LLAMA_BUILD (CMake build dir, default <FNS_LLAMA_DIR>/build-sm86).
"""

import argparse
import datetime as dt
import json
import os
import shlex
import statistics
import subprocess
from pathlib import Path

LLAMA_DIR = Path(os.environ.get("FNS_LLAMA_DIR", Path(__file__).resolve().parents[2] / "external" / "llama.cpp"))
LLAMA_BUILD = Path(os.environ.get("FNS_LLAMA_BUILD", LLAMA_DIR / "build-sm86"))
BENCH = LLAMA_BUILD / "bin" / "llama-bench.exe"
RESULTS = Path(__file__).resolve().parents[1] / "results"


def gpu_state() -> dict:
    q = "memory.used,memory.free,temperature.gpu,pcie.link.gen.current,clocks.sm,clocks.mem"
    out = subprocess.run(
        ["nvidia-smi", f"--query-gpu={q}", "--format=csv,noheader,nounits"], capture_output=True, text=True
    ).stdout.strip()
    return dict(zip(q.split(","), (v.strip() for v in out.split(",")), strict=False))


def run(args: str) -> list[dict]:
    # -r 1: one process per measurement; llama-bench's own warm-up is not counted.
    cmd = [str(BENCH), *shlex.split(args, posix=False), "-r", "1", "-o", "jsonl"]
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        raise SystemExit(f"llama-bench failed ({p.returncode}): {' '.join(cmd)}\n{p.stderr[-3000:]}")
    return [json.loads(l) for l in p.stdout.splitlines() if l.startswith("{")]


def key(row: dict) -> str:
    return f"pp{row['n_prompt']}+tg{row['n_gen']}@d{row['n_depth']}"


def main() -> None:
    ap = argparse.ArgumentParser(description="Alternating A/B llama-bench runs.")
    ap.add_argument("--name", required=True, help="output name: <date>-<name>.jsonl")
    ap.add_argument("--pairs", type=int, default=5, help="number of alternating rounds")
    ap.add_argument("--a", required=True, help="llama-bench arguments, arm A")
    ap.add_argument("--b", help="llama-bench arguments, arm B (optional)")
    ap.add_argument("--out-dir", type=Path, default=RESULTS, help="output directory (created if missing)")
    o = ap.parse_args()

    o.out_dir.mkdir(parents=True, exist_ok=True)
    out = o.out_dir / f"{dt.date.today():%Y-%m-%d}-{o.name}.jsonl"
    commit = subprocess.run(["git", "-C", str(LLAMA_DIR), "rev-parse", "HEAD"], capture_output=True,
                            text=True).stdout.strip()
    arms = {"A": o.a} | ({"B": o.b} if o.b else {})
    samples: dict[str, dict[str, list[float]]] = {arm: {} for arm in arms}

    with out.open("a", encoding="utf-8") as f:
        for i in range(o.pairs):
            for arm, args in arms.items():
                before = gpu_state()
                for row in run(args):
                    rec = {"pair": i, "arm": arm, "args": args, "llama_commit": commit, "gpu_before": before,
                           "test": key(row), "avg_ts": row["avg_ts"], "raw": row}
                    f.write(json.dumps(rec) + "\n")
                    samples[arm].setdefault(key(row), []).append(row["avg_ts"])
                print(f"pair {i + 1}/{o.pairs} {arm}: " + ", ".join(
                    f"{t} {v[-1]:.1f} tok/s" for t, v in samples[arm].items()), flush=True)

    print(f"\n{out}")
    print("| test | A median tok/s | B median tok/s | B/A | pairs won by B |")
    print("|---|---:|---:|---:|---:|")
    for t, a in samples["A"].items():
        if "B" in samples:
            b = samples["B"][t]
            wins = sum(y > x for x, y in zip(a, b, strict=True))
            ma, mb = statistics.median(a), statistics.median(b)
            print(f"| {t} | {ma:.1f} | {mb:.1f} | {mb / ma:.3f} | {wins}/{len(a)} |")
        else:
            print(f"| {t} | {statistics.median(a):.1f} (min {min(a):.1f}, max {max(a):.1f}) | — | — | — |")


if __name__ == "__main__":
    main()
