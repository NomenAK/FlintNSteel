"""Download the reference GGUF models and verify their SHA-256 checksums.

Usage: python fetch-models.py [--out DIR] [q3] [q4] [q8]   (no model key: all three)

The destination is --out, then the FNS_MODELS environment variable, then ROOT/models.
The script refuses to start when the destination lacks free space for the missing files.
"""

import argparse
import hashlib
import os
import shutil
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download

REPO = "unsloth/flint-GGUF"
REVISION = "20468edfc38ab43d6914cc88b08cbecb05d0fc62"
ROOT = Path(__file__).resolve().parents[2]
MARGIN = 2 * 10**9

FILES = {
    "q3": ("flint-UD-Q3_K_XL.gguf", "91b8d85ab6ff7fb095d9a972f9afa6f3fa2b9acb642867a079f3e013ea3c350c"),
    "q4": ("flint-UD-Q4_K_XL.gguf", "63c2ecc461ca545134b670c660e76fa20f47240f8d39b6510bfd21c4dd6e072f"),
    "q8": ("flint-Q8_0.gguf", "5df42b56859c7b5142228005d7d63c327ec9e34c99c72548016d805309a0d284"),
}
SIZES = {"q3": 18.49e9, "q4": 25.05e9, "q8": 42.30e9}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(64 << 20):
            h.update(chunk)
    return h.hexdigest()


def default_out() -> Path:
    value = os.environ.get("FNS_MODELS")
    return Path(value) if value else ROOT / "models"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="*", choices=list(FILES), metavar="MODEL",
                    help="models to fetch: q3, q4, q8 (default: all)")
    ap.add_argument("--out", type=Path, default=None,
                    help="destination directory (default: FNS_MODELS, else ROOT/models)")
    args = ap.parse_args()
    dest = args.out if args.out is not None else default_out()
    keys = args.models or list(FILES)
    dest.mkdir(parents=True, exist_ok=True)
    missing = [k for k in keys if not (dest / FILES[k][0]).exists()]
    need = sum(SIZES[k] for k in missing) + MARGIN
    free = shutil.disk_usage(dest).free
    if free < need:
        print(f"{dest}: free {free / 1e9:.1f} GB < required {need / 1e9:.1f} GB for {missing}")
        return 2
    for k in keys:
        name, digest = FILES[k]
        path = dest / name
        if not path.exists():
            print(f"downloading {name}", flush=True)
            hf_hub_download(REPO, name, revision=REVISION, local_dir=dest)
        got = sha256(path)
        status = "OK" if got == digest else f"FAILED (got {got})"
        print(f"{name}: SHA-256 {status}", flush=True)
        if got != digest:
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
