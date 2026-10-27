"""Alternating A/B runs through llama-server, for options that llama-bench does not expose (e.g. --moe-cache-mib).

Each measurement starts a fresh server, sends one unmeasured warm-up request, then R greedy /completion
requests (prompt of --prompt-tokens corpus tokens, --n generated tokens, prompt cache disabled between requests).
Reported: prompt_per_second and predicted_per_second from `timings` (tools/server/server-common.cpp:140-152).

Environment (defaults relative to the repository root):
  FNS_LLAMA_DIR    llama.cpp checkout             (default: external/llama.cpp)
  FNS_LLAMA_BUILD  llama.cpp CMake build dir      (default: <FNS_LLAMA_DIR>/build-sm86)
  FNS_MODELS       model directory                (default: models)
  FNS_TOKENIZER    explicit tokenizer.json file   (default: <FNS_MODELS>/prof/flint-tokenizer.json, else Hugging Face)

Usage:
  python ab_server.py --name q4-ncmoe12-cache --pairs 5 \
      --a "-m MODEL.gguf -ngl 99 -fa on --n-cpu-moe 12" \
      --b "-m MODEL.gguf -ngl 99 -fa on --n-cpu-moe 20 --moe-cache-mib 3000" \
      --corpus CORPUS.txt
"""

import argparse
import datetime as dt
import json
import os
import shlex
import statistics
import subprocess
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MODELS = Path(os.environ.get("FNS_MODELS", ROOT / "models"))
LLAMA_DIR = Path(os.environ.get("FNS_LLAMA_DIR", ROOT / "external" / "llama.cpp"))
LLAMA_BUILD = Path(os.environ.get("FNS_LLAMA_BUILD", LLAMA_DIR / "build-sm86"))
SERVER = LLAMA_BUILD / "bin" / "llama-server.exe"
CORPUS = MODELS / "kl" / "code.txt"
RESULTS = Path(__file__).resolve().parents[1] / "results"
TOKENIZER_REPO = "RecursiveSI/flint"
TOKENIZER_REV = "2a247be0c0aec895ccb0ac7ca92229ec6df37288"
PORT = 8399


def load_tokenizer():
    """Tokenizer from FNS_TOKENIZER, else MODELS/prof/flint-tokenizer.json, else Hugging Face (pinned revision)."""
    from tokenizers import Tokenizer

    explicit = os.environ.get("FNS_TOKENIZER")
    local = MODELS / "prof" / "flint-tokenizer.json"
    if explicit:
        path = Path(explicit)
        if not path.is_file():
            raise SystemExit(f"FNS_TOKENIZER not found: {path}")
    elif local.is_file():
        path = local
    else:
        from huggingface_hub import hf_hub_download

        path = Path(hf_hub_download(TOKENIZER_REPO, "tokenizer.json", revision=TOKENIZER_REV))
    return Tokenizer.from_file(str(path))


def read_corpus(path: Path) -> str:
    if not path.is_file():
        raise SystemExit(f"corpus not found: {path} (pass --corpus, or generate it with bench/corpus/build.py)")
    return path.read_text(encoding="utf-8")


def encode_corpus(corpus: Path, n: int) -> list[int]:
    """First n tokens of the corpus, without special tokens."""
    return load_tokenizer().encode(read_corpus(corpus), add_special_tokens=False).ids[:n]


def post(path: str, body: dict) -> dict:
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}{path}", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=3600) as r:
        return json.load(r)


def wait_ready(proc: subprocess.Popen) -> None:
    for _ in range(600):
        if proc.poll() is not None:
            raise SystemExit(f"llama-server exited ({proc.returncode})")
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=2) as r:
                if r.status == 200:
                    return
        except Exception:
            pass
        time.sleep(1)
    raise SystemExit("llama-server not ready after 600 s")


def measure(args: str, prompt_ids: list[int], n: int, reps: int, log: Path) -> list[dict]:
    cmd = [str(SERVER), *shlex.split(args, posix=False), "--port", str(PORT), "-np", "1", "-lm", "none",
           "--cache-ram", "0", "--no-warmup", "-c", str(len(prompt_ids) + n + 256)]
    with log.open("w", encoding="utf-8") as lf:
        proc = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT)
    try:
        wait_ready(proc)
        body = {"prompt": prompt_ids, "n_predict": n, "temperature": 0, "cache_prompt": False,
                "ignore_eos": True}
        post("/completion", body | {"n_predict": 16})  # warm-up, not counted
        return [post("/completion", body)["timings"] for _ in range(reps)]
    finally:
        proc.terminate()
        proc.wait(60)


def main() -> None:
    ap = argparse.ArgumentParser(description="A/B llama-server timings (prompt and decode tok/s).")
    ap.add_argument("--name", required=True, help="output name: <date>-<name>.jsonl")
    ap.add_argument("--pairs", type=int, default=5, help="number of alternating rounds")
    ap.add_argument("--reps", type=int, default=1, help="measured requests per server")
    ap.add_argument("--prompt-tokens", type=int, default=4096, help="corpus tokens used as prompt")
    ap.add_argument("--n", type=int, default=256, help="generated tokens per request")
    ap.add_argument("--a", required=True, help="llama-server arguments, arm A")
    ap.add_argument("--b", help="llama-server arguments, arm B (optional)")
    ap.add_argument("--corpus", type=Path, default=CORPUS, help="UTF-8 text corpus used for the prompt")
    ap.add_argument("--out-dir", type=Path, default=RESULTS, help="output directory (created if missing)")
    o = ap.parse_args()

    ids = encode_corpus(o.corpus, o.prompt_tokens)

    o.out_dir.mkdir(parents=True, exist_ok=True)
    out = o.out_dir / f"{dt.date.today():%Y-%m-%d}-{o.name}.jsonl"
    commit = subprocess.run(["git", "-C", str(LLAMA_DIR), "rev-parse", "HEAD"], capture_output=True,
                            text=True).stdout.strip()
    arms = {"A": o.a} | ({"B": o.b} if o.b else {})
    pp: dict[str, list[float]] = {a: [] for a in arms}
    tg: dict[str, list[float]] = {a: [] for a in arms}
    with out.open("a", encoding="utf-8") as f:
        for i in range(o.pairs):
            for arm, args in arms.items():
                log = o.out_dir / f".server-{o.name}-{arm}.log"
                for t in measure(args, ids, o.n, o.reps, log):
                    f.write(json.dumps({"pair": i, "arm": arm, "args": args, "llama_commit": commit,
                                        "prompt_tokens": len(ids), "n": o.n, "timings": t}) + "\n")
                    pp[arm].append(t["prompt_per_second"])
                    tg[arm].append(t["predicted_per_second"])
                print(f"pair {i + 1}/{o.pairs} {arm}: pp {pp[arm][-1]:.1f}, tg {tg[arm][-1]:.2f} tok/s", flush=True)

    print(f"\n{out}")
    print("| metric | A median | B median | B/A | pairs won by B |")
    print("|---|---:|---:|---:|---:|")
    for name, d in (("pp tok/s", pp), ("tg tok/s", tg)):
        a = d["A"]
        if "B" in d:
            b = d["B"]
            wins = sum(y > x for x, y in zip(a, b, strict=True))
            print(f"| {name} | {statistics.median(a):.2f} | {statistics.median(b):.2f} | "
                  f"{statistics.median(b) / statistics.median(a):.3f} | {wins}/{len(a)} |")
        else:
            print(f"| {name} | {statistics.median(a):.2f} (min {min(a):.2f}, max {max(a):.2f}) | — | — | — |")


if __name__ == "__main__":
    main()
