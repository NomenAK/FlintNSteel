"""Replay a recorded agent session against llama-server and fns_server at equal work.

Inputs (both required, supplied by the caller):
  --trace      request bodies of one session, JSONL, as written by `fns_server.py --record`.
  --trace-log  the journal of the same session, JSONL, as written by `fns_server.py --log`; it gives the
               number of completion tokens generated for each request.

Each request is replayed with the same prompt and the same number of generated tokens on both servers.
The prompt is the chat template rendered by serve/fns_server.py, so both servers receive identical token ids.
The generated length is fixed with `ignore_eos`. Decoding is greedy, or rec sampling with --sample.
Prompts come from the trace bodies, so the reusable prefix between consecutive requests is the same for both
servers, even though the replayed output differs from the recorded output.

Arm A (llama-server): /completion with the token ids and cache_prompt, one fresh server per measurement.
Arm B (fns_server): /v1/chat/completions with the messages and ignore_eos, one fresh server per measurement.
`ignore_eos` also disables fns_server's tool grammar and its stop tokens, so arm B generates without grammar
constraints. Use this tool only for fixed-length replay, not for grammar-constrained output.

The measurement is the client-side total session time; model loading is excluded. Both servers warm up at
startup, and that warm-up is part of loading. Pairs alternate between arms.

Environment:
  FNS_LLAMA_BUILD  llama.cpp build directory holding bin/llama-server.exe (default: FNS_LLAMA_DIR/build-sm86)
  FNS_LLAMA_DIR    llama.cpp checkout, used for the commit recorded in the output (default: ROOT/external/llama.cpp)
  FNS_BUILD_DIR    build directory; the default output directory lives under it (default: ROOT/build)

Usage:
  python agent_replay.py --name q3-agent --pairs 5 --trace T.jsonl --trace-log T.log.jsonl --model MODEL.gguf \
      --a "-ngl 99 -fa on" --a "-ngl 99 -fa on --swa-full" --out OUT_DIR

  fns_server A/B against itself (same load, two fns_server option sets; --a is optional):
  python agent_replay.py --name q3-agent-spec --trace T.jsonl --trace-log T.log.jsonl --model MODEL.gguf \
      --fns "" --fns-b "--spec 3" --out OUT_DIR

Writes OUT_DIR/<date>-<name>.jsonl and prints medians and paired wins.
"""

import argparse
import datetime as dt
import json
import os
import shlex
import statistics
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LLAMA_DIR = Path(os.environ.get("FNS_LLAMA_DIR", ROOT / "external" / "llama.cpp"))
LLAMA_BUILD = Path(os.environ.get("FNS_LLAMA_BUILD", LLAMA_DIR / "build-sm86"))
BUILD_DIR = Path(os.environ.get("FNS_BUILD_DIR", ROOT / "build"))
sys.path.insert(0, str(ROOT / "serve"))
import ab_server  # noqa: E402  (wait_ready and PORT for llama-server)
from fns_server import Chat  # noqa: E402

PORT_FNS = 8092
# Sampling shared by both servers: greedy by default, or rec's parameters with --sample.
SAMPLING: dict[str, float] = {"temperature": 0}
LLAMA_SAMPLING: dict[str, float] = {}


def post(port: int, path: str, body: dict) -> dict:
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=3600) as r:
        return json.loads(r.read())


def git_head(path: Path) -> str:
    return subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()


def run_llama(args: str, model: str, ctx: int, turns: list, log: Path) -> dict:
    cmd = [str(LLAMA_BUILD / "bin" / "llama-server.exe"), "-m", model, *shlex.split(args, posix=False),
           "--port", str(ab_server.PORT), "-np", "1", "-c", str(ctx)]  # default llama-server warm-up, as fns-serve's
    with log.open("w", encoding="utf-8") as lf:
        proc = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT)
        try:
            ab_server.wait_ready(proc)
            per, t0 = [], time.perf_counter()
            for ids, n in turns:
                t = time.perf_counter()
                r = post(ab_server.PORT, "/completion", {"prompt": ids, "n_predict": n, "ignore_eos": True,
                                                         "cache_prompt": True, **SAMPLING, **LLAMA_SAMPLING})
                tm = r["timings"]
                if r["tokens_predicted"] != n:
                    raise SystemExit(f"llama-server: {r['tokens_predicted']} tokens generated instead of {n}")
                per.append({"wall_ms": (time.perf_counter() - t) * 1e3, "prompt_n": tm["prompt_n"],
                            "cached": r.get("tokens_cached"), "prefill_ms": tm["prompt_ms"],
                            "decode_ms": tm["predicted_ms"]})
            return {"total_s": time.perf_counter() - t0, "turns": per}
        finally:
            proc.terminate()
            proc.wait()


def run_fns(model: str, ctx: int, extra: str, bodies: list, turns: list, log: Path) -> dict:
    cmd = [sys.executable, str(ROOT / "serve" / "fns_server.py"), "--model", model, "--ctx", str(ctx),
           "--port", str(PORT_FNS), *shlex.split(extra, posix=False)]
    with log.open("w", encoding="utf-8") as lf:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=lf, text=True, encoding="utf-8")
        try:
            while True:
                line = proc.stdout.readline() if proc.stdout else ""
                if not line:
                    raise SystemExit(f"fns_server stopped ({proc.wait()}), see {log}")
                if line.startswith("fns_server : http"):
                    break
            per, t0 = [], time.perf_counter()
            for body, (ids, n) in zip(bodies, turns):
                t = time.perf_counter()
                r = post(PORT_FNS, "/v1/chat/completions", {
                    "messages": body["messages"], "tools": body.get("tools"), "max_tokens": n, "ignore_eos": True,
                    **SAMPLING})
                if r["usage"]["prompt_tokens"] != len(ids) or r["usage"]["completion_tokens"] != n:
                    raise SystemExit(f"fns_server: usage {r['usage']} instead of {len(ids)} + {n}")
                f = r["fns"]
                per.append({"wall_ms": (time.perf_counter() - t) * 1e3, "reused": f["reused_prefix"],
                            "prefill_ms": f["prefill_ms"], "decode_ms": f["decode_ms"]})
            return {"total_s": time.perf_counter() - t0, "turns": per}
        finally:
            proc.terminate()
            proc.wait()


def main() -> None:
    ap = argparse.ArgumentParser(description="Replay a recorded agent session against llama-server and fns_server.")
    ap.add_argument("--name", required=True, help="run name; output is <out>/<date>-<name>.jsonl")
    ap.add_argument("--pairs", type=int, default=5, help="number of alternating measurement pairs")
    ap.add_argument("--trace", required=True, help="request bodies of the session (fns_server --record JSONL)")
    ap.add_argument("--trace-log", required=True, help="journal of the same session (fns_server --log JSONL)")
    ap.add_argument("--model", required=True, help="GGUF model path, used by both servers")
    ap.add_argument("--ctx", type=int, default=32768, help="context size of both servers")
    ap.add_argument("--a", action="append", default=[], help="llama-server arguments (repeatable)")
    ap.add_argument("--fns", default="", help="extra fns_server arguments (arm fns)")
    ap.add_argument("--fns-b", help="second fns_server arm (arguments), compared with the first arm")
    ap.add_argument("--sample", action="store_true",
                    help="rec sampling (temperature 1, top-p 0.95, top-k 20, seed 0) instead of greedy")
    ap.add_argument("--presence", type=float, default=0.0,
                    help="presence_penalty (fns arms only; llama-server has no equivalent per-turn penalty)")
    ap.add_argument("--out", default=str(BUILD_DIR / "bench" / "agent"),
                    help="output directory, created if missing (default: FNS_BUILD_DIR/bench/agent)")
    o = ap.parse_args()
    if o.sample:
        SAMPLING.update(temperature=1.0, top_p=0.95, top_k=20, seed=0)
        # llama-server: the same filters only (no min-p, no penalties), in its default order.
        LLAMA_SAMPLING.update(min_p=0.0, typical_p=1.0, repeat_penalty=1.0, xtc_probability=0.0, dry_multiplier=0.0)
    if o.presence:
        if o.a:
            raise SystemExit("--presence: fns arms only (--fns / --fns-b)")
        SAMPLING["presence_penalty"] = o.presence

    bodies = [json.loads(l) for l in Path(o.trace).read_text(encoding="utf-8").splitlines() if l.strip()]
    logged = [json.loads(l) for l in Path(o.trace_log).read_text(encoding="utf-8").splitlines() if l.strip()]
    if len(bodies) != len(logged):
        raise SystemExit(f"{len(bodies)} requests in the trace, {len(logged)} in the journal")
    chat = Chat()
    turns = []
    for b, g in zip(bodies, logged):
        ids = chat.prompt_ids(b["messages"], b.get("tools"))
        if len(ids) != g["prompt_tokens"]:
            raise SystemExit(f"prompt of {len(ids)} tokens, {g['prompt_tokens']} in the journal")
        turns.append((ids, g["completion_tokens"]))
    print(f"{len(turns)} requests, {sum(len(i) for i, _ in turns)} prompt tokens, "
          f"{sum(n for _, n in turns)} generated", flush=True)

    arms = [*o.a, "fns"] + (["fns-b"] if o.fns_b is not None else [])
    if len(arms) < 2:
        raise SystemExit("at least two arms: --a (llama-server) or --fns-b")
    out_dir = Path(o.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{dt.date.today():%Y-%m-%d}-{o.name}.jsonl"
    commits = {"fns": git_head(ROOT), "llama": git_head(LLAMA_DIR)}
    totals = {a: [] for a in arms}
    with out.open("a", encoding="utf-8") as f:
        for i in range(o.pairs):
            for arm in arms:
                log = out_dir / f".server-{o.name}.log"
                if arm == "fns":
                    r = run_fns(o.model, o.ctx, o.fns, bodies, turns, log)
                elif arm == "fns-b":
                    r = run_fns(o.model, o.ctx, o.fns_b or "", bodies, turns, log)
                else:
                    r = run_llama(arm, o.model, o.ctx, turns, log)
                totals[arm].append(r["total_s"])
                f.write(json.dumps({"pair": i, "arm": arm, "args": o.fns_b if arm == "fns-b" else o.fns if arm == "fns"
                                    else arm, "model": o.model, "ctx": o.ctx, "commits": commits,
                                    "trace": o.trace, **r}) + "\n")
                f.flush()
            print(f"pair {i + 1}/{o.pairs}: " + ", ".join(f"{a!r} {v[-1]:.2f} s" for a, v in totals.items()),
                  flush=True)
    fm = statistics.median(totals["fns"])
    print(f"\n{out}")
    for a in o.a:
        am = statistics.median(totals[a])
        wins = sum(b < x for x, b in zip(totals[a], totals["fns"]))
        print(f"| llama-server {a} | {am:.2f} s | FlintNSteel {fm:.2f} s | x{am / fm:.3f} | {wins}/{o.pairs} |")
    if o.fns_b is not None:
        bm = statistics.median(totals["fns-b"])
        wins = sum(b < x for x, b in zip(totals["fns"], totals["fns-b"]))
        print(f"| fns {o.fns!r} | {fm:.2f} s | fns {o.fns_b!r} | {bm:.2f} s | x{fm / bm:.3f} | {wins}/{o.pairs} |")


if __name__ == "__main__":
    main()
