"""Expert routing profile and expert-cache hit rate.

  python route_profile.py profile [--model MODEL.gguf] [--corpus CODE.txt] [--start N] [--count N] [--out PROFILE.route]
  python route_profile.py hits --profile P.route --test T.route [--host-bytes N] [--model MODEL.gguf]

Environment:
  FNS_MODELS     model directory (default: ROOT/models)
  FNS_LLAMA_DIR  llama.cpp checkout; its gguf-py is used when present (default: ROOT/external/llama.cpp)
  FNS_BUILD_DIR  build directory containing fns-ref (default: ROOT/build)
  FNS_TOKENIZER  explicit tokenizer.json (default: FNS_MODELS/prof/flint-tokenizer.json, else the pinned
                 Hugging Face revision)

profile : tokenizes the corpus, selects token ids [start:start+count] (default 16384:24576, a slice disjoint
          from the test slice) and runs fns-ref --profile-out, which writes uint32 counters [n_layer][n_expert].
hits    : replays the placement of Engine::place_experts (per-layer slots aligned to 256 bytes, descending
          profile order, then ascending layer and expert on ties) under a VRAM budget that keeps --host-bytes
          of experts in RAM, and compares it with the real selections of another trace (--test): the share
          of selections served from VRAM.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def env_path(name: str, default: Path) -> Path:
    value = os.environ.get(name)
    return Path(value) if value else default


MODELS = env_path("FNS_MODELS", ROOT / "models")
LLAMA = env_path("FNS_LLAMA_DIR", ROOT / "external" / "llama.cpp")
BUILD = env_path("FNS_BUILD_DIR", ROOT / "build")
PROF = MODELS / "prof"
DEFAULT_CORPUS = MODELS / "kl" / "code.txt"
DEFAULT_MODEL = MODELS / "flint-UD-Q4_K_XL.gguf"
DEFAULT_PROFILE = PROF / "q4-code-train-8k.route"
TOKENIZER_REPO = "RecursiveSI/flint"
TOKENIZER_REVISION = "2a247be0c0aec895ccb0ac7ca92229ec6df37288"
DEFAULT_START = 16384
DEFAULT_COUNT = 8192

if (LLAMA / "gguf-py").is_dir():
    sys.path.insert(0, str(LLAMA / "gguf-py"))


def require_file(path: Path, what: str, hint: str) -> None:
    if not path.is_file():
        raise SystemExit(f"{what} not found: {path}\n{hint}")


def tokenizer_file() -> Path:
    override = os.environ.get("FNS_TOKENIZER")
    if override:
        require_file(Path(override), "FNS_TOKENIZER file", "Point FNS_TOKENIZER at an existing tokenizer.json.")
        return Path(override)
    local = PROF / "flint-tokenizer.json"
    if local.is_file():
        return local
    from huggingface_hub import hf_hub_download
    return Path(hf_hub_download(TOKENIZER_REPO, "tokenizer.json", revision=TOKENIZER_REVISION))


def corpus_ids(corpus: Path) -> list[int]:
    require_file(corpus, "corpus", "Build it with: python bench/corpus/build.py <output>, or pass --corpus.")
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(tokenizer_file()))
    return tok.encode(corpus.read_text(encoding="utf-8"), add_special_tokens=False).ids


def select_slice(ids: list[int], start: int, count: int) -> list[int]:
    if start < 0 or count <= 0:
        raise SystemExit("--start must be >= 0 and --count must be > 0.")
    if start + count > len(ids):
        raise SystemExit(f"corpus has {len(ids)} tokens, but --start {start} + --count {count} needs "
                         f"{start + count}. Use a larger corpus or a smaller --start/--count.")
    return ids[start:start + count]


def run_profile(model: str, ids: list[int], tokens: Path, out: Path) -> None:
    tokens.write_text(" ".join(map(str, ids)), encoding="utf-8")
    exe = BUILD / "fns-ref.exe"
    require_file(exe, "fns-ref", "Build the reference binary or point FNS_BUILD_DIR at its build directory.")
    p = subprocess.run([str(exe), "--model", model, "--tokens", str(tokens), "--profile-out", str(out)],
                       capture_output=True, text=True)
    if p.returncode:
        raise SystemExit(p.stderr[-2000:])
    print(p.stderr.strip())


def slots(model: str) -> np.ndarray:
    import gguf
    r = gguf.GGUFReader(model)
    n_layer = 1 + max(int(t.name.split(".")[1]) for t in r.tensors if t.name.startswith("blk."))
    sz = np.zeros(n_layer)
    for t in r.tensors:
        if "_exps" in t.name:
            sz[int(t.name.split(".")[1])] += t.n_bytes / 256
    return np.ceil(sz / 256) * 256


def hits(model: str, prof: np.ndarray, test: np.ndarray, host_bytes: float, slot: np.ndarray | None = None) -> dict:
    slot = slots(model) if slot is None else slot
    n_layer, n_exp = prof.shape
    moe = slot > 0
    budget = slot[moe].sum() * n_exp - host_bytes
    # place_experts order: descending profile count, then ascending (layer, expert) index; stable sort.
    flat_slot = np.repeat(slot, n_exp)
    cand = np.flatnonzero(np.repeat(moe, n_exp))
    order = cand[np.argsort(-prof.ravel()[cand], kind="stable")]
    res = np.zeros(n_layer * n_exp, dtype=bool)
    used = 0.0
    for i in order:  # greedy: a smaller slot from another layer may still fit
        if used + flat_slot[i] <= budget:
            used += flat_slot[i]
            res[i] = True
    res = res.reshape(n_layer, n_exp)
    t = test * moe[:, None]
    n_tok = t[np.flatnonzero(moe)[0]].sum() / 6
    miss_bytes = float((t * ~res * slot[:, None]).sum())
    return {"hit_rate": float(t[res].sum() / t.sum()), "misses_per_token": float(t[~res].sum() / n_tok),
            "host_mb_per_token": miss_bytes / n_tok / 1e6, "tokens": int(round(n_tok))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["profile", "hits"])
    ap.add_argument("--model", default=str(DEFAULT_MODEL))
    ap.add_argument("--corpus", default=str(DEFAULT_CORPUS), help="profile: UTF-8 code corpus text")
    ap.add_argument("--start", type=int, default=DEFAULT_START, help="profile: first token index of the slice")
    ap.add_argument("--count", type=int, default=DEFAULT_COUNT, help="profile: number of tokens in the slice")
    ap.add_argument("--out", default=str(DEFAULT_PROFILE), help="profile: output .route file")
    ap.add_argument("--profile", help="hits: profile .route file")
    ap.add_argument("--test", help="hits: test .route file")
    ap.add_argument("--host-bytes", type=float, default=2.13e9)
    o = ap.parse_args()
    PROF.mkdir(parents=True, exist_ok=True)
    if o.cmd == "profile":
        require_file(Path(o.model), "model", "Download it with bench/scripts/fetch-models.py or pass --model.")
        ids = select_slice(corpus_ids(Path(o.corpus)), o.start, o.count)
        out = Path(o.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        run_profile(o.model, ids, PROF / "code-train-8k.txt", out)
        return
    if not o.test:
        ap.error("hits requires --test")
    if o.profile:
        require_file(Path(o.profile), "profile", "Create it with: route_profile.py profile --out <file>.")
    require_file(Path(o.test), "test trace", "Pass an existing .route file to --test.")
    load = lambda p: np.fromfile(p, dtype=np.uint32).reshape(-1, 256).astype(np.int64)
    prof = load(o.profile) if o.profile else None
    test = load(o.test)
    for name, p in (("without profile", np.zeros_like(test)), ("profile", prof), ("oracle (profile = test)", test)):
        if p is None:
            continue
        print(name, hits(o.model, p, test, o.host_bytes))


if __name__ == "__main__":
    main()
