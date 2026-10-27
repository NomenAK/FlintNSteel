"""Alternating runs of FlintNSteel (fns-bench) against llama.cpp (llama-bench): tg decode at depth d, or pp prefill.

  python ab_fns_llama.py --name q3-tg --pairs 5 --depth 0 [--n 256] [--model ...]
  python ab_fns_llama.py --name q3-pp --pairs 5 --depth 32768 --pp 4096 [--ub 4096]

A = llama-bench -ngl 99 -fa on -p 0 -n N -d D -b 4096 -ub 4096 -r 1 (KV F16 reference), or -p P -n 0;
B = fns-bench -n N -d D -r 1, or -p P -ub UB (depth D filled in batches of UB). One fresh process per measurement.
Writes <out-dir>/<date>-<name>.jsonl (default out-dir: bench/results) and prints medians and paired wins.

Environment: FNS_LLAMA_DIR, FNS_LLAMA_BUILD, FNS_BUILD_DIR, FNS_MODELS (see ab_server.py).
"""

import argparse
import datetime as dt
import json
import os
import statistics
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MODELS = Path(os.environ.get("FNS_MODELS", ROOT / "models"))
LLAMA_DIR = Path(os.environ.get("FNS_LLAMA_DIR", ROOT / "external" / "llama.cpp"))
LLAMA_BUILD = Path(os.environ.get("FNS_LLAMA_BUILD", LLAMA_DIR / "build-sm86"))
BUILD_DIR = Path(os.environ.get("FNS_BUILD_DIR", ROOT / "build"))
LLAMA_BENCH = str(LLAMA_BUILD / "bin" / "llama-bench.exe")
FNS_BENCH = str(BUILD_DIR / "fns-bench.exe")


def jsonl(cmd: list[str]) -> dict:
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode:
        raise SystemExit(f"failed ({p.returncode}): {' '.join(cmd)}\n{p.stderr[-2000:]}")
    rows = [json.loads(l) for l in p.stdout.splitlines() if l.startswith("{")]
    return [r for r in rows if r.get("n_gen", 0) > 0 or r.get("n_prompt", 0) > 0][0]


def main() -> None:
    ap = argparse.ArgumentParser(description="Alternating FlintNSteel vs llama.cpp runs (tg or pp).")
    ap.add_argument("--name", required=True, help="output name: <date>-<name>.jsonl")
    ap.add_argument("--pairs", type=int, default=5, help="number of alternating rounds")
    ap.add_argument("--depth", type=int, default=0, help="context depth d before the measurement")
    ap.add_argument("--n", type=int, default=256, help="generated tokens (decode mode)")
    ap.add_argument("--pp", type=int, default=0, help="measured prefill size (0: decode mode)")
    ap.add_argument("--ub", type=int, default=4096, help="fns-bench micro-batch size for prefill")
    ap.add_argument("--model", default=str(MODELS / "flint-UD-Q3_K_XL.gguf"), help="GGUF model for both engines")
    ap.add_argument("--fns-args", default="", help="extra fns-bench options (e.g. --fa-int8)")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "bench" / "results",
                    help="output directory (created if missing)")
    o = ap.parse_args()

    p, n = (o.pp, 0) if o.pp else (0, o.n)
    a_cmd = [LLAMA_BENCH, "-m", o.model, "-ngl", "99", "-fa", "on", "-p", str(p), "-n", str(n), "-d", str(o.depth),
             "-b", "4096", "-ub", "4096", "-r", "1", "-o", "jsonl"]
    b_cmd = [FNS_BENCH, "--model", o.model, "-d", str(o.depth), "-r", "1", "-o", "jsonl"]
    b_cmd += ["-p", str(p), "-ub", str(o.ub)] if p else ["-n", str(n)]
    b_cmd += o.fns_args.split()
    commits = {
        "fns": subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip(),
        "llama": subprocess.run(["git", "-C", str(LLAMA_DIR), "rev-parse", "HEAD"], capture_output=True,
                                text=True).stdout.strip(),
    }
    o.out_dir.mkdir(parents=True, exist_ok=True)
    out = o.out_dir / f"{dt.date.today():%Y-%m-%d}-{o.name}.jsonl"
    a, b = [], []
    with out.open("a", encoding="utf-8") as f:
        for i in range(o.pairs):
            ra = jsonl(a_cmd)
            a.append(ra["avg_ts"])
            f.write(json.dumps({"pair": i, "arm": "A-llama", "cmd": a_cmd, "commits": commits, "avg_ts": ra["avg_ts"]}) + "\n")
            rb = jsonl(b_cmd)
            b.append(rb["avg_ts"])
            f.write(json.dumps({"pair": i, "arm": "B-fns", "cmd": b_cmd, "commits": commits, "avg_ts": rb["avg_ts"]}) + "\n")
            print(f"pair {i + 1}/{o.pairs}: llama {a[-1]:.1f}  fns {b[-1]:.1f} tok/s", flush=True)
    wins = sum(y > x for x, y in zip(a, b, strict=True))
    ma, mb = statistics.median(a), statistics.median(b)
    label = f"pp{p}" if p else f"tg{n}"
    print(f"\n{out}\n| {label} @ d{o.depth} | llama.cpp {ma:.1f} | FlintNSteel {mb:.1f} | x{mb / ma:.3f} | {wins}/{o.pairs} |")


if __name__ == "__main__":
    main()
