"""Writes the Flint tokenizer copy and its token list (ids in order), used by the grammar and by the xgrammar smoke test.

  py -3.14 tools/xgrammar/export_vocab.py [output_dir]

The tokenizer is resolved the same way as by the server (serve/flint_assets.py): FNS_TOKENIZER if set, else an existing
local copy, else the pinned Hugging Face revision. The output directory defaults to FNS_MODELS/prof (ROOT/models/prof).
Writes flint-tokenizer.json (copy of the resolved file) and flint-vocab.json (token strings indexed by id).
"""

import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "serve"))

from flint_assets import TOKENIZER_NAME, VOCAB_NAME, profile_dir, tokenizer_path  # noqa: E402


def main() -> None:
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else profile_dir()
    out.mkdir(parents=True, exist_ok=True)
    src = tokenizer_path()
    dst = out / TOKENIZER_NAME
    if not (src.exists() and dst.exists() and src.samefile(dst)):
        shutil.copy(src, dst)
    t = json.loads(src.read_text(encoding="utf-8"))
    entries = list(t["model"]["vocab"].items()) + [(a["content"], a["id"]) for a in t.get("added_tokens", [])]
    vocab = [""] * (max(i for _, i in entries) + 1)
    for s, i in entries:
        vocab[i] = s
    (out / VOCAB_NAME).write_text(json.dumps(vocab, ensure_ascii=False), encoding="utf-8")
    print(f"{len(vocab)} tokens -> {out / VOCAB_NAME} (tokenizer: {src})")


if __name__ == "__main__":
    main()
