"""Rebuild the code corpus used by the quality gates from code.manifest.json.

The text is not versioned (it is third-party code). It is regenerated from the source repositories at the
commits recorded in the manifest, and the result is checked against the manifest's SHA-256.

Repository roots (environment variables, all external to this repository):
  FNS_LLAMA_DIR   llama.cpp checkout (default: ROOT/external/llama.cpp)
  FNS_REC_DIR     rec checkout (required when the manifest references rec)
  FNS_STRATA_DIR  Strata checkout (required when the manifest references Strata)

Usage: python build.py [OUTPUT]   (default: FNS_MODELS/kl/code.txt, FNS_MODELS defaults to ROOT/models)
"""

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REPO_ENV = {"llama.cpp": "FNS_LLAMA_DIR", "rec": "FNS_REC_DIR", "Strata": "FNS_STRATA_DIR"}
REPO_DEFAULT = {"llama.cpp": ROOT / "external" / "llama.cpp"}


def repo_root(repo: str) -> Path:
    value = os.environ.get(REPO_ENV[repo])
    if value:
        return Path(value)
    if repo in REPO_DEFAULT:
        return REPO_DEFAULT[repo]
    raise SystemExit(f"set {REPO_ENV[repo]} to the {repo} checkout that matches the manifest commits")


def default_output() -> Path:
    models = Path(os.environ["FNS_MODELS"]) if os.environ.get("FNS_MODELS") else ROOT / "models"
    return models / "kl" / "code.txt"


def main() -> int:
    manifest = json.loads((Path(__file__).parent / "code.manifest.json").read_text(encoding="utf-8"))
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else default_output()
    roots = {}
    parts = []
    for f in manifest["files"]:
        if f["repo"] not in roots:
            roots[f["repo"]] = repo_root(f["repo"])
        blob = subprocess.run(
            ["git", "-C", str(roots[f["repo"]]), "show", f"{f['commit']}:{f['path']}"],
            capture_output=True, check=True,
        ).stdout.decode("utf-8")
        parts.append(blob.replace("\r\n", "\n"))
    text = "\n\n".join(parts)
    digest = hashlib.sha256(text.encode()).hexdigest()
    if digest != manifest["sha256_utf8"]:
        print(f"digest mismatch: {digest}", file=sys.stderr)
        return 1
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8", newline="\n")
    print(f"{out}: {manifest['tokens']} tokens, sha256 {digest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
