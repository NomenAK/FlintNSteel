"""KL divergence and top-1 agreement of quantized models against a Q8_0 base.

  python kld.py base --model <Q8_0.gguf> --corpus <text> [--base-kld FILE] [--out-dir DIR] [-- extra args]
  python kld.py eval --model <gguf> --name <label> --corpus <text> [--base-kld FILE] [--out-dir DIR] [-- extra args]

base: computes the reference logits (llama-perplexity --kl-divergence-base) and writes them to --base-kld.
      Defaults are --ngl 99 and --n-cpu-moe 49, the placement used for the 24 GB reference card.
eval: measures the candidate against that base and prints the summary: Mean KLD, Median KLD, 99% KLD, Maximum KLD,
      Same top p, and PPL. Arguments after `--` are passed to llama-perplexity unchanged.

Common options: -c 4096, -b and -ub 4096, 32 chunks, -fa on. The corpus is a plain text file supplied by the caller.
Environment:
  FNS_LLAMA_BUILD  llama.cpp build directory holding bin/llama-perplexity.exe (default: FNS_LLAMA_DIR/build-sm86)
  FNS_LLAMA_DIR    llama.cpp checkout (default: ROOT/external/llama.cpp)
  FNS_BUILD_DIR    build directory; the default output directory lives under it (default: ROOT/build)
Output (default OUT_DIR: FNS_BUILD_DIR/bench/kld): <date>-kld-<name>.log and .json (eval); base logs as <date>-kld-base.log.
"""

import argparse
import datetime as dt
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LLAMA_DIR = Path(os.environ.get("FNS_LLAMA_DIR", ROOT / "external" / "llama.cpp"))
LLAMA_BUILD = Path(os.environ.get("FNS_LLAMA_BUILD", LLAMA_DIR / "build-sm86"))
BUILD_DIR = Path(os.environ.get("FNS_BUILD_DIR", ROOT / "build"))
BIN = LLAMA_BUILD / "bin" / "llama-perplexity.exe"
DEFAULT_OUT = BUILD_DIR / "bench" / "kld"

PATTERNS = {
    "ppl": r"Mean PPL\(Q\)\s*:\s*([\d.]+)",
    "ppl_base": r"Mean PPL\(base\)\s*:\s*([\d.]+)",
    "kld_mean": r"Mean\s+KLD:\s*([\d.]+)",
    "kld_median": r"Median\s+KLD:\s*([\d.]+)",
    "kld_p99": r"99\.0%\s+KLD:\s*([\d.]+)",
    "kld_max": r"Maximum KLD:\s*([\d.]+)",
    "same_top_p": r"Same top p:\s*([\d.]+)",
}


def run(cmd: list[str], log: Path) -> tuple[int, float]:
    t = time.time()
    with log.open("w", encoding="utf-8") as f:
        f.write(" ".join(cmd) + "\n")
        f.flush()
        code = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT).returncode
    return code, time.time() - t


def main() -> int:
    # Split the caller's extra llama-perplexity arguments at `--` before argparse sees them.
    argv, extra = sys.argv[1:], []
    if "--" in argv:
        i = argv.index("--")
        argv, extra = argv[:i], argv[i + 1:]
    ap = argparse.ArgumentParser(description="KL divergence and top-1 agreement against a Q8_0 base.")
    ap.add_argument("mode", choices=["base", "eval"])
    ap.add_argument("--model", required=True, help="GGUF model (Q8_0 for base, candidate for eval)")
    ap.add_argument("--name", help="label of the candidate (required for eval)")
    ap.add_argument("--corpus", required=True, help="text corpus file")
    ap.add_argument("--base-kld", help="reference logits file (default: OUT_DIR/base.kld)")
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT), help="output directory, created if missing")
    o = ap.parse_args(argv)
    if not BIN.exists():
        raise SystemExit(f"llama-perplexity not found: {BIN} (set FNS_LLAMA_BUILD)")

    out = Path(o.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    base = Path(o.base_kld) if o.base_kld else out / "base.kld"
    day = f"{dt.date.today():%Y-%m-%d}"
    common = ["-f", o.corpus, "-c", "4096", "-b", "4096", "-ub", "4096", "--chunks", "32", "-fa", "on"]

    if o.mode == "base":
        log = out / f"{day}-kld-base.log"
        cmd = [str(BIN), "-m", o.model, *common, "-ngl", "99", "--n-cpu-moe", "49",
               "--kl-divergence-base", str(base), *extra]
        code, secs = run(cmd, log)
        if code == 0:
            print(f"base: code {code}, {secs / 60:.1f} min, {base.stat().st_size / 1e9:.2f} GB")
        else:
            print(f"failed {code}, see {log}")
        return code

    if not o.name:
        raise SystemExit("eval requires --name <label>")
    if not base.exists():
        raise SystemExit(f"base logits not found: {base} (run `kld.py base` first or pass --base-kld)")
    log = out / f"{day}-kld-{o.name}.log"
    cmd = [str(BIN), "-m", o.model, *common, "--kl-divergence-base", str(base), "--kl-divergence", *extra]
    code, secs = run(cmd, log)
    text = log.read_text(encoding="utf-8", errors="replace")
    summary = {"name": o.name, "model": o.model, "options": extra, "seconds": round(secs, 1), "exit": code}
    for k, p in PATTERNS.items():
        m = re.search(p, text)
        summary[k] = float(m.group(1)) if m else None
    (out / f"{day}-kld-{o.name}.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(json.dumps(summary))
    return code


if __name__ == "__main__":
    sys.exit(main())
