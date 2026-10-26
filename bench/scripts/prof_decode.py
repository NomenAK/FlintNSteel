"""Per-kernel breakdown of one fns-bench decode token (nsys), over the last 40 tokens.

  python prof_decode.py <name> --model <gguf> [--out DIR] [fns-bench args...]
  e.g.: python prof_decode.py d0 --model MODEL.gguf -n 64 -d 0

Writes OUT_DIR/fns-<name>.nsys-rep and .sqlite, then prints ms/token, the active GPU share and the top kernels.
Tokens are delimited by the embedding kernel (k_embed_q8_0), one per token.

Environment:
  FNS_BUILD_DIR  build directory holding fns-bench.exe (default: ROOT/build)
nsys (Nsight Systems) must be on PATH. The default OUT_DIR is FNS_BUILD_DIR/bench/prof.
"""

import argparse
import collections
import os
import shutil
import sqlite3
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BUILD = Path(os.environ.get("FNS_BUILD_DIR", ROOT / "build"))
DEFAULT_OUT = BUILD / "bench" / "prof"


def nsys() -> str:
    exe = shutil.which("nsys")
    if exe is None:
        raise SystemExit("nsys not found on PATH (install Nsight Systems and add its target directory to PATH)")
    return exe


def capture(name: str, model: str, out: Path, bench_args: list[str], trace: list[str]) -> Path:
    """Profile one fns-bench run and export it to SQLite; returns the report path without suffix."""
    out.mkdir(parents=True, exist_ok=True)
    rep = out / f"fns-{name}"
    subprocess.run([nsys(), "profile", *trace, "-o", str(rep), "--force-overwrite", "true",
                    str(BUILD / "fns-bench.exe"), "--model", model, *bench_args], capture_output=True, check=True)
    subprocess.run([nsys(), "export", "-t", "sqlite", "-o", f"{rep}.sqlite", "--force-overwrite", "true",
                    f"{rep}.nsys-rep"], capture_output=True, check=True)
    return rep


def kernel_rows(rep: Path) -> list[tuple[int, int, str]]:
    """(start, end, name) of every CUDA kernel in the export, ordered by start."""
    con = sqlite3.connect(f"{rep}.sqlite")
    try:
        return con.execute("select k.start, k.end, s.value from CUPTI_ACTIVITY_KIND_KERNEL k "
                           "join StringIds s on s.id = k.shortName order by k.start").fetchall()
    finally:
        con.close()


def main() -> None:
    ap = argparse.ArgumentParser(description="Per-kernel breakdown of one fns-bench decode token.")
    ap.add_argument("name", help="run name; the report is written as fns-<name>")
    ap.add_argument("--model", required=True, help="GGUF model path passed to fns-bench")
    ap.add_argument("--out", default=str(DEFAULT_OUT), help="output directory (default: FNS_BUILD_DIR/bench/prof)")
    o, bench_args = ap.parse_known_args()  # everything else goes to fns-bench
    rep = capture(o.name, o.model, Path(o.out), bench_args, ["-t", "cuda", "--cuda-graph-trace=node"])
    rows = kernel_rows(rep)
    starts = [s for s, _, n in rows if n.startswith("k_embed_q8_0")]
    k = min(40, len(starts) - 1)
    a, b = starts[-k - 1], starts[-1]
    agg, cnt = collections.Counter(), collections.Counter()
    for s, e, n in rows:
        if a <= s < b:
            agg[n] += e - s
            cnt[n] += 1
    tok_ms = (b - a) / k / 1e6
    busy = sum(agg.values()) / k / 1e6
    print(f"{tok_ms:.3f} ms/token ({1000 / tok_ms:.1f} tok/s), GPU active {busy:.3f} ms ({busy / tok_ms:.0%}), "
          f"{sum(cnt.values()) / k:.0f} kernels/token")
    for n, t in agg.most_common(14):
        print(f"  {t / k / 1e3:8.1f} us {t / k / 1e6 / tok_ms:6.1%} x{cnt[n] / k:5.0f}  {n[:60]}")


if __name__ == "__main__":
    main()
