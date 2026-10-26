"""Model, tokenizer, template, and build paths shared by the server, the CLI, and tools/xgrammar/export_vocab.py.

  models dir  : FNS_MODELS, else ROOT/models
  tokenizer   : FNS_TOKENIZER (must exist if set), else <models>/prof/flint-tokenizer.json if present,
                else the pinned Hugging Face revision of RecursiveSI/flint (tokenizer.json)
  vocabulary  : <models>/prof/flint-vocab.json, derived from the tokenizer by tools/xgrammar/export_vocab.py
  build dir   : FNS_BUILD_DIR, else ROOT/build
"""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CHAT_TEMPLATE = Path(__file__).resolve().parent / "chat_template.jinja"
TOKENIZER_REPO = "RecursiveSI/flint"
TOKENIZER_REVISION = "2a247be0c0aec895ccb0ac7ca92229ec6df37288"
TOKENIZER_NAME = "flint-tokenizer.json"
VOCAB_NAME = "flint-vocab.json"


def models_dir() -> Path:
    return Path(os.environ.get("FNS_MODELS") or ROOT / "models")


def profile_dir() -> Path:
    return models_dir() / "prof"


def vocab_path() -> Path:
    return profile_dir() / VOCAB_NAME


def build_dir() -> Path:
    return Path(os.environ.get("FNS_BUILD_DIR") or ROOT / "build")


def local_tokenizer() -> Path | None:
    """Explicit FNS_TOKENIZER or the local profile copy; None when neither applies. Never downloads."""
    explicit = os.environ.get("FNS_TOKENIZER")
    if explicit:
        path = Path(explicit)
        if not path.is_file():
            raise FileNotFoundError(f"FNS_TOKENIZER points to a missing file: {path}")
        return path
    local = profile_dir() / TOKENIZER_NAME
    return local if local.is_file() else None


def cached_hf_tokenizer() -> str | None:
    """Path of the pinned tokenizer in the Hugging Face cache, if already downloaded."""
    from huggingface_hub import try_to_load_from_cache

    cached = try_to_load_from_cache(TOKENIZER_REPO, "tokenizer.json", revision=TOKENIZER_REVISION)
    return cached if isinstance(cached, str) else None


def tokenizer_path() -> Path:
    """The tokenizer every component uses: explicit, local profile copy, or the pinned Hugging Face file."""
    local = local_tokenizer()
    if local is not None:
        return local
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(TOKENIZER_REPO, "tokenizer.json", revision=TOKENIZER_REVISION))
