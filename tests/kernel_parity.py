"""Parity of quantized GEMVs against real GGUF blocks.

  python tests/kernel_parity.py [--model MODEL.gguf]

Paths (environment; defaults relative to the repository root):
  FNS_MODELS     model directory; default model is FNS_MODELS/flint-UD-Q3_K_XL.gguf (default: models/)
  FNS_LLAMA_DIR  llama.cpp checkout providing gguf-py (default: external/llama.cpp)
  FNS_BUILD_DIR  directory with fns-kernel-parity.exe; kernel_parity.json is written there (default: build/)

Runs fns-kernel-parity on a fixed x, then compares each output with W·x computed in float64 from the dequantization
in gguf-py (same commit as ggml). The only approximation on the ggml side is the Q8_1 quantization of x; the reference
reproduces it (quantize_row_q8_1 from ggml: round to nearest, d = amax/127).
Threshold: relative error ||y - ref|| / ||ref|| < 1e-4 against the reference with quantized x, and < 2e-2 against exact x.
Exit code 1 if any tensor fails.
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
LLAMA_DIR = Path(os.environ.get("FNS_LLAMA_DIR") or ROOT / "external" / "llama.cpp")
MODELS_DIR = Path(os.environ.get("FNS_MODELS") or ROOT / "models")
BUILD_DIR = Path(os.environ.get("FNS_BUILD_DIR") or ROOT / "build")
GGUF_PY = LLAMA_DIR / "gguf-py"
if not GGUF_PY.is_dir():
    sys.exit(f"gguf-py not found under {LLAMA_DIR}: set FNS_LLAMA_DIR to a llama.cpp checkout")
sys.path.insert(0, str(GGUF_PY))
import gguf  # noqa: E402
import gguf.quants as gq  # noqa: E402

IDS = [0, 255, 17, 128, 3, 200]


def q8_1(x: np.ndarray) -> np.ndarray:
    """x after ggml-cuda's quantize_q8_1: blocks of 32, d = amax/127, q = round(x/d), d stored as F16."""
    b = x.reshape(-1, 32).astype(np.float32)
    amax = np.abs(b).max(1, keepdims=True)
    d = amax / 127.0
    q = np.where(amax == 0, 0, np.round(b / np.where(d == 0, 1, d)))
    return (q * d.astype(np.float16).astype(np.float32)).reshape(x.shape)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(MODELS_DIR / "flint-UD-Q3_K_XL.gguf"),
                    help="GGUF model (default: FNS_MODELS/flint-UD-Q3_K_XL.gguf)")
    o = ap.parse_args()
    if not Path(o.model).is_file():
        ap.error(f"model not found: {o.model}")

    rng = np.random.default_rng(20261022)
    x = (rng.standard_normal(6 * 6144) * 0.5).astype(np.float32)
    reader = gguf.GGUFReader(o.model)
    tensors = {t.name: t for t in reader.tensors}

    with tempfile.TemporaryDirectory() as tmp:
        xp = Path(tmp) / "x.f32"
        x.tofile(xp)
        p = subprocess.run([str(BUILD_DIR / "fns-kernel-parity.exe"), "--model", o.model, "--x", str(xp),
                            "--out", tmp], capture_output=True, text=True)
        if p.returncode:
            print(p.stderr[-2000:])
            return 1
        rows, failed = [], 0
        for name in p.stdout.split():
            y = np.fromfile(Path(tmp) / f"{name}.f32", dtype=np.float32)
            base, _, form = name.partition(".weight")
            t = tensors[base + ".weight"]
            w = gq.dequantize(t.data, t.tensor_type).astype(np.float64)  # [n_mats][nrows][ncols] or [nrows][ncols]
            if form:  # experts: .x1 (one shared row) or .x6 (one row per expert)
                ncols = w.shape[-1]
                xs = [x[:ncols]] * 6 if form == ".x1" else [x[j * ncols:(j + 1) * ncols] for j in range(6)]
                ref_q = np.concatenate([w[e] @ q8_1(xi).astype(np.float64) for e, xi in zip(IDS, xs, strict=True)])
                ref_x = np.concatenate([w[e] @ xi.astype(np.float64) for e, xi in zip(IDS, xs, strict=True)])
            else:
                ncols = w.shape[-1]
                ref_q = w @ q8_1(x[:ncols]).astype(np.float64)
                ref_x = w @ x[:ncols].astype(np.float64)
            eq = float(np.linalg.norm(y - ref_q) / np.linalg.norm(ref_q))
            ex = float(np.linalg.norm(y - ref_x) / np.linalg.norm(ref_x))
            ok = eq < 1e-4 and ex < 2e-2
            failed += not ok
            rows.append({"tensor": name, "type": t.tensor_type.name, "rel_err_vs_q8_1_ref": eq,
                         "rel_err_vs_exact": ex, "ok": ok})
            print(f"{'OK ' if ok else 'FAIL'} {name:36s} {t.tensor_type.name:8s} vs Q8_1 {eq:.2e}  vs exact {ex:.2e}")
        (BUILD_DIR / "kernel_parity.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
