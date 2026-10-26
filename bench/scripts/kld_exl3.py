"""KL divergence and top-1 agreement of an EXL3 model (exllamav3) against the llama.cpp Q8_0 base.

Same protocol as llama-perplexity --kl-divergence-base. Experimental: EXL3 is not recommended for serving
and its model parity is not validated; the result is a measurement, not a parity claim.

  python bench/scripts/kld_exl3.py <model_dir> <name> --base <base.kld> [--chunks 32]
         [--exllamav3-dir DIR] [--out-dir DIR]
  -> <out-dir>/<date>-kld-<name>.json (default out-dir: <build dir>/bench)

The base file contains the corpus tokens (n_chunk x n_ctx) and, for each chunk, the Q8_0 log-probabilities of
positions n_ctx/2 .. n_ctx - 2 compressed as uint16 (llama.cpp perplexity.cpp:79-107). Tokens are read from the
base file, so tokenization matches the base exactly. As with llama-perplexity on Flint (add_bos_token = false in
the GGUF, perplexity.cpp:1799), chunks are used as-is, each evaluated alone on a cleared context. KL follows
perplexity.cpp:191-252 (base terms below -16 ignored, top-1 = equal argmax).
"""

import argparse
import datetime as dt
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]


def build_dir():
    """Build directory: FNS_BUILD_DIR, else ROOT/build."""
    return Path(os.environ.get("FNS_BUILD_DIR") or ROOT / "build")


def exllamav3_dir(arg):
    """exllamav3 checkout: --exllamav3-dir, else FNS_EXLLAMAV3_DIR."""
    path = arg or os.environ.get("FNS_EXLLAMAV3_DIR")
    if not path:
        sys.exit("error: set FNS_EXLLAMAV3_DIR or pass --exllamav3-dir")
    return Path(path)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir", help="EXL3 model directory")
    ap.add_argument("name", help="result name, used in the output file name")
    ap.add_argument("--base", required=True, help="llama.cpp Q8_0 KLD base file (llama-perplexity --kl-divergence-base)")
    ap.add_argument("--chunks", type=int, default=32, help="number of chunks to evaluate")
    ap.add_argument("--exllamav3-dir", help="exllamav3 checkout (default: FNS_EXLLAMAV3_DIR)")
    ap.add_argument("--out-dir", help="result directory (default: <build dir>/bench)")
    a = ap.parse_args()
    sys.path.insert(0, str(exllamav3_dir(a.exllamav3_dir)))
    from exllamav3.model.config import Config  # noqa: E402
    from exllamav3.model.model import Model  # noqa: E402

    f = Path(a.base).open("rb")
    assert f.read(8) == b"_logits_"
    n_ctx, n_vocab, n_chunk = np.frombuffer(f.read(12), dtype=np.int32).tolist()
    tokens = np.frombuffer(f.read(4 * n_ctx * n_chunk), dtype=np.int32).reshape(n_chunk, n_ctx)
    nv = 2 * ((n_vocab + 1) // 2) + 4
    first = n_ctx // 2
    n_score = n_ctx - 1 - first
    chunks = min(a.chunks, n_chunk)

    config = Config.from_directory(a.model_dir)
    model = Model.from_config(config)
    model.load(progressbar=True)
    dev = torch.device("cuda:0")

    s = dict(kld=0.0, kld2=0.0, nll=0.0, nll_base=0.0, same=0, n=0)
    t0 = time.time()
    with torch.inference_mode():
        for c in range(chunks):
            raw = np.frombuffer(f.read(2 * n_score * nv), dtype=np.uint16).reshape(n_score, nv)
            hdr = raw[:, :4].copy().view(np.float32)  # [scale, min_log_prob] par position
            ids = torch.from_numpy(tokens[c].astype(np.int64))
            logits = model.forward(ids[None, :], {"attn_mode": "flash_attn_nc"})[0, first:n_ctx - 1, :n_vocab]
            tgt_all = torch.from_numpy(tokens[c, first + 1:].astype(np.int64)).to(dev)
            # Chunked: a whole decompressed F32 base chunk does not fit beside the weights.
            for r0 in range(0, n_score, 256):
                r1 = min(r0 + 256, n_score)
                scale = torch.from_numpy(hdr[r0:r1, 0].copy()).to(dev)[:, None]
                minlp = torch.from_numpy(hdr[r0:r1, 1].copy()).to(dev)[:, None]
                q = torch.from_numpy(raw[r0:r1, 4:4 + n_vocab].astype(np.int32)).to(dev)
                lp_base = q.float() * scale + minlp  # base log-probabilities
                lg = logits[r0:r1].float()
                lp = torch.log_softmax(lg, dim=-1)
                pb = torch.where(lp_base > -16.0, lp_base.exp(), torch.zeros_like(lp_base))
                kl = (pb * (lp_base - lp)).sum(-1)
                tgt = tgt_all[r0:r1, None]
                s["kld"] += kl.sum().item()
                s["kld2"] += (kl * kl).sum().item()
                s["nll"] += -lp.gather(1, tgt).sum().item()
                s["nll_base"] += -lp_base.gather(1, tgt).sum().item()
                s["same"] += (lg.argmax(-1) == q.argmax(-1)).sum().item()
            s["n"] += n_score
            del logits
            print(f"chunk {c + 1}/{chunks} KL {s['kld'] / s['n']:.4f} top1 {100 * s['same'] / s['n']:.2f}%",
                  flush=True)
    n = s["n"]
    mean = s["kld"] / n
    sd = math.sqrt(max(s["kld2"] / n - mean * mean, 0.0) / (n - 1))
    out = dict(name=a.name, model=a.model_dir, chunks=chunks, n_ctx=n_ctx, positions=n,
               kld_mean=round(mean, 6), kld_stderr=round(sd, 6), same_top_pct=round(100 * s["same"] / n, 3),
               ppl=round(math.exp(s["nll"] / n), 4), ppl_base=round(math.exp(s["nll_base"] / n), 4),
               seconds=round(time.time() - t0, 1), torch=torch.__version__)
    out_dir = Path(a.out_dir) if a.out_dir else build_dir() / "bench"
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / f"{dt.date.today():%Y-%m-%d}-kld-{a.name}.json"
    p.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
