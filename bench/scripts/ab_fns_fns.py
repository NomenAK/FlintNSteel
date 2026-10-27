"""Alternating pairs of two fns-bench binaries (A = before, B = after a change) under the same load.

  python ab_fns_fns.py --name q3-mmvq-warps-d4096 --a OLD.exe --b NEW.exe --pairs 5 -- -d 4096 -n 256
  python ab_fns_fns.py --name c5-kvq8-d32768 --a fns-bench.exe --b fns-bench.exe --b-args "--kv-global q8_0" -- -d 32768

Arguments after `--` go to both binaries (with --model and -r 1 -o jsonl). One fresh process per measurement.
Writes <out-dir>/<date>-<name>.jsonl (default out-dir: bench/results) and prints medians and paired wins for B.

Environment: FNS_MODELS (model directory, default models).
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


def avg_ts(cmd: list[str]) -> float:
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode:
        raise SystemExit(f"failed ({p.returncode}): {' '.join(cmd)}\n{p.stderr[-2000:]}")
    return json.loads(next(l for l in p.stdout.splitlines() if l.startswith("{")))["avg_ts"]


def main() -> None:
    ap = argparse.ArgumentParser(description="Alternating A/B runs of two fns-bench binaries.")
    ap.add_argument("--name", required=True, help="output name: <date>-<name>.jsonl")
    ap.add_argument("--a", required=True, help="fns-bench binary, arm A")
    ap.add_argument("--b", required=True, help="fns-bench binary, arm B")
    ap.add_argument("--pairs", type=int, default=5, help="number of alternating rounds")
    ap.add_argument("--model", default=str(MODELS / "flint-UD-Q3_K_XL.gguf"), help="GGUF model passed to both arms")
    ap.add_argument("--a-args", default="", help="options for arm A only")
    ap.add_argument("--b-args", default="", help="options for arm B only")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "bench" / "results",
                    help="output directory (created if missing)")
    ap.add_argument("args", nargs=argparse.REMAINDER, help="options for both arms, after --")
    o = ap.parse_args()

    extra = [a for a in o.args if a != "--"]
    common = ["--model", o.model, *extra, "-r", "1", "-o", "jsonl"]
    a_cmd, b_cmd = [o.a, *common, *o.a_args.split()], [o.b, *common, *o.b_args.split()]
    commit = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    o.out_dir.mkdir(parents=True, exist_ok=True)
    out = o.out_dir / f"{dt.date.today():%Y-%m-%d}-{o.name}.jsonl"
    a, b = [], []
    with out.open("a", encoding="utf-8") as f:
        for i in range(o.pairs):
            a.append(avg_ts(a_cmd))
            b.append(avg_ts(b_cmd))
            for arm, cmd, v in (("A", a_cmd, a[-1]), ("B", b_cmd, b[-1])):
                f.write(json.dumps({"pair": i, "arm": arm, "exe": cmd[0], "args": cmd[1:], "commit": commit,
                                    "avg_ts": v}) + "\n")
            print(f"pair {i + 1}/{o.pairs}: A {a[-1]:.1f}  B {b[-1]:.1f} tok/s", flush=True)
    wins = sum(y > x for x, y in zip(a, b, strict=True))
    ma, mb = statistics.median(a), statistics.median(b)
    print(f"\n{out}\n| {' '.join(extra)} | A {ma:.1f} | B {mb:.1f} | x{mb / ma:.3f} | {wins}/{o.pairs} |")


if __name__ == "__main__":
    main()
