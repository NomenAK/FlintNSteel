"""Alternating pairs of FlintNSteel (fns-bench --prompt) against llama-server, same real prompt, greedy decode.

Load as in bench/scripts/ab_server.py: a prompt of --prompt-tokens corpus tokens, then --n generated tokens.
A = llama-server (fresh server, one warm-up request, then one measured request: predicted_per_second);
B = fns-bench --prompt <same ids> -n N (fresh process; 8 warm-up tokens after the prompt, then N tokens).

  python ab_fns_server.py --name q4-fns-vs-server --pairs 5 \
      --a "-m MODEL.gguf -ngl 99 -fa on --n-cpu-moe 12 --moe-cache-mib 2800" \
      --fns "--model MODEL.gguf --expert-profile PROFILE.route" \
      --corpus CORPUS.txt

Writes <out-dir>/<date>-<name>.jsonl (default out-dir: bench/results) and prints medians and paired wins.
Environment: FNS_BUILD_DIR (fns-bench location), plus the variables documented in ab_server.py.
"""

import argparse
import datetime as dt
import json
import os
import shlex
import statistics
import subprocess
import tempfile
from pathlib import Path

import ab_server

ROOT = Path(__file__).resolve().parents[2]
BUILD_DIR = Path(os.environ.get("FNS_BUILD_DIR", ROOT / "build"))
FNS_BENCH = BUILD_DIR / "fns-bench.exe"


def git_head(path: str) -> str:
    return subprocess.run(["git", "-C", path, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()


def main() -> None:
    ap = argparse.ArgumentParser(description="A/B llama-server against fns-bench, same prompt.")
    ap.add_argument("--name", required=True, help="output name: <date>-<name>.jsonl")
    ap.add_argument("--pairs", type=int, default=5, help="number of alternating rounds")
    ap.add_argument("--prompt-tokens", type=int, default=4096, help="corpus tokens used as prompt")
    ap.add_argument("--n", type=int, default=256, help="generated tokens per request")
    ap.add_argument("--a", required=True, help="llama-server arguments")
    ap.add_argument("--fns", required=True, help="fns-bench arguments (without -n or --prompt)")
    ap.add_argument("--corpus", type=Path, default=ab_server.CORPUS, help="UTF-8 text corpus used for the prompt")
    ap.add_argument("--out-dir", type=Path, default=ab_server.RESULTS, help="output directory (created if missing)")
    o = ap.parse_args()

    ids = ab_server.encode_corpus(o.corpus, o.prompt_tokens)

    o.out_dir.mkdir(parents=True, exist_ok=True)
    prompt = Path(tempfile.gettempdir()) / f"fns-prompt-{len(ids)}.txt"
    prompt.write_text(" ".join(map(str, ids)), encoding="utf-8")

    out = o.out_dir / f"{dt.date.today():%Y-%m-%d}-{o.name}.jsonl"
    commits = {"fns": git_head(str(ROOT)), "llama": git_head(str(ab_server.LLAMA_DIR))}
    b_cmd = [str(FNS_BENCH), *shlex.split(o.fns, posix=False), "-n", str(o.n), "--prompt", str(prompt), "-o", "jsonl"]
    a_ts: list[float] = []
    b_ts: list[float] = []
    with out.open("a", encoding="utf-8") as f:
        for i in range(o.pairs):
            log = o.out_dir / f".server-{o.name}-A.log"
            t = ab_server.measure(o.a, ids, o.n, 1, log)[0]
            a_ts.append(t["predicted_per_second"])
            f.write(json.dumps({"pair": i, "arm": "A-llama-server", "args": o.a, "commits": commits,
                                "prompt_tokens": len(ids), "n": o.n, "timings": t}) + "\n")
            p = subprocess.run(b_cmd, capture_output=True, text=True)
            if p.returncode:
                raise SystemExit(f"failed ({p.returncode}): {' '.join(b_cmd)}\n{p.stderr[-2000:]}")
            row = [json.loads(l) for l in p.stdout.splitlines() if l.startswith("{")][0]
            b_ts.append(row["avg_ts"])
            f.write(json.dumps({"pair": i, "arm": "B-fns", "cmd": b_cmd, "commits": commits, "prompt_tokens": len(ids),
                                "n": o.n, "stderr": p.stderr.strip()[-500:], "avg_ts": row["avg_ts"]}) + "\n")
            print(f"pair {i + 1}/{o.pairs}: llama-server {a_ts[-1]:.2f}, FlintNSteel {b_ts[-1]:.2f} tok/s", flush=True)
    ma, mb = statistics.median(a_ts), statistics.median(b_ts)
    wins = sum(b > a for a, b in zip(a_ts, b_ts, strict=True))
    print(f"\n{out}\n| tg{o.n} after {len(ids)} | llama-server {ma:.1f} | FlintNSteel {mb:.1f} | x{mb / ma:.3f} | "
          f"{wins}/{o.pairs} |")


if __name__ == "__main__":
    main()
