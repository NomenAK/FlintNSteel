"""Compare two logit files of shape [n_tokens][n_vocab] (float32) over the same token ids.

  python parity.py <reference.f32> <candidate.f32> [--vocab 150144]
  python parity.py --run <corpus_dir> --model <gguf> [--floor] [--fns-args "..."] [--work DIR] [--out summary.json]

Per-position metrics, reference = P, candidate = Q: KL(P||Q) over the full softmax, top-1 agreement, max |Δlogit|.

--run: for each p*.txt token file in corpus_dir, run fns-dump (llama.cpp CUDA) and fns-ref. With --floor, also run
fns-dump --ngl 0 (llama.cpp CPU, the noise floor), then aggregate. Each token file holds one prompt's token ids.
The corpus is supplied by the caller; no default corpus or result files are used.

Environment:
  FNS_BUILD_DIR  build directory holding fns-dump.exe and fns-ref.exe (default: ROOT/build)
Intermediate logits are written under --work (default: FNS_BUILD_DIR/bench/parity) and deleted after each prompt.
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
BUILD = Path(os.environ.get("FNS_BUILD_DIR", ROOT / "build"))
VOCAB = 150144


def load(path: Path, vocab: int) -> np.ndarray:
    a = np.fromfile(path, dtype=np.float32)
    if a.size % vocab:
        raise SystemExit(f"{path}: size {a.size} is not a multiple of {vocab}")
    return a.reshape(-1, vocab)


def compare(p_logits: np.ndarray, q_logits: np.ndarray) -> dict:
    n = min(len(p_logits), len(q_logits))
    p, q = p_logits[:n].astype(np.float64), q_logits[:n].astype(np.float64)
    lp = p - p.max(1, keepdims=True)
    lp -= np.log(np.exp(lp).sum(1, keepdims=True))
    lq = q - q.max(1, keepdims=True)
    lq -= np.log(np.exp(lq).sum(1, keepdims=True))
    kl = (np.exp(lp) * (lp - lq)).sum(1)
    top1 = (p.argmax(1) == q.argmax(1))
    return {
        "positions": int(n),
        "kl_mean": float(kl.mean()),
        "kl_max": float(kl.max()),
        "kl_p99": float(np.percentile(kl, 99)),
        "top1": float(top1.mean()),
        "max_abs_logit_diff": float(np.abs(p - q).max()),
        "first_top1_miss": int(np.argmin(top1)) if not top1.all() else -1,
    }


def run(cmd: list[str]) -> None:
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode:
        raise SystemExit(f"failed ({p.returncode}): {' '.join(cmd)}\n{p.stderr[-2000:]}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Logit parity of fns-ref against llama.cpp.")
    ap.add_argument("ref", nargs="?", help="reference logits (.f32), compare mode")
    ap.add_argument("cand", nargs="?", help="candidate logits (.f32), compare mode")
    ap.add_argument("--vocab", type=int, default=VOCAB, help="vocabulary size of the logits")
    ap.add_argument("--run", metavar="CORPUS_DIR", help="directory of p*.txt token files to run")
    ap.add_argument("--model", help="GGUF model path (required with --run)")
    ap.add_argument("--floor", action="store_true", help="also measure llama.cpp CPU against CUDA (noise floor)")
    ap.add_argument("--fns-args", default="", help="extra arguments passed to fns-ref, e.g. \"--ubatch 512\"")
    ap.add_argument("--work", help="directory for intermediate logits (default: FNS_BUILD_DIR/bench/parity)")
    ap.add_argument("--out", help="write the JSON summary to this file")
    o = ap.parse_args()

    if not o.run:
        if not (o.ref and o.cand):
            ap.error("give <reference.f32> <candidate.f32>, or --run <corpus_dir> --model <gguf>")
        print(json.dumps(compare(load(Path(o.ref), o.vocab), load(Path(o.cand), o.vocab))))
        return 0
    if not o.model:
        ap.error("--run requires --model <gguf>")

    work = Path(o.work) if o.work else BUILD / "bench" / "parity"
    work.mkdir(parents=True, exist_ok=True)
    rows = []
    for tok in sorted(Path(o.run).glob("p*.txt")):
        stem = work / tok.stem
        ref, fns, cpu = (Path(f"{stem}.{s}.f32") for s in ("llama", "fns", "cpu"))
        run([str(BUILD / "fns-dump.exe"), "--model", o.model, "--tokens", str(tok), "--out", str(ref)])
        run([str(BUILD / "fns-ref.exe"), "--model", o.model, "--tokens", str(tok), "--out", str(fns),
             *o.fns_args.split()])
        P = load(ref, o.vocab)
        row = {"prompt": tok.name, "fns_vs_llama_cuda": compare(P, load(fns, o.vocab))}
        if o.floor:
            run([str(BUILD / "fns-dump.exe"), "--model", o.model, "--tokens", str(tok), "--out", str(cpu),
                 "--ngl", "0"])
            row["llama_cpu_vs_cuda"] = compare(P, load(cpu, o.vocab))
        for f in (ref, fns, cpu):
            f.unlink(missing_ok=True)
        rows.append(row)
        print(json.dumps(row), flush=True)

    def agg(key: str) -> dict:
        xs = [r[key] for r in rows if key in r]
        if not xs:
            return {}
        n = sum(x["positions"] for x in xs)
        return {
            "positions": n,
            "kl_mean": sum(x["kl_mean"] * x["positions"] for x in xs) / n,
            "kl_max": max(x["kl_max"] for x in xs),
            "top1": sum(x["top1"] * x["positions"] for x in xs) / n,
        }

    summary = {"model": o.model, "fns_vs_llama_cuda": agg("fns_vs_llama_cuda"), "llama_cpu_vs_cuda": agg("llama_cpu_vs_cuda"),
               "prompts": rows}
    if summary["llama_cpu_vs_cuda"]:
        summary["ratio_to_floor"] = summary["fns_vs_llama_cuda"]["kl_mean"] / summary["llama_cpu_vs_cuda"]["kl_mean"]
    print(json.dumps({k: v for k, v in summary.items() if k != "prompts"}, indent=1))
    if o.out:
        Path(o.out).parent.mkdir(parents=True, exist_ok=True)
        Path(o.out).write_text(json.dumps(summary, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
