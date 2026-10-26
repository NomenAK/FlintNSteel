"""Tests for tools/xgrammar/export_vocab.py run as a real subprocess against a temporary tokenizer file.

Each test isolates its outputs under a temporary FNS_MODELS directory, so the repository's models/ is never touched.
"""

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

EXPORTER = Path(__file__).resolve().parents[1] / "tools" / "xgrammar" / "export_vocab.py"
TOKENIZER_NAME = "flint-tokenizer.json"
VOCAB_NAME = "flint-vocab.json"

TOKENIZER = {
    "version": "1.0",
    "model": {
        "type": "WordLevel",
        "vocab": {"<unk>": 0, "hello": 1, "wörld": 2, "!": 4},
        "unk_token": "<unk>",
    },
    "added_tokens": [{"id": 3, "content": "<|eot|>", "special": True}],
}

# Index = token id: 0 <unk>, 1 hello, 2 wörld, 3 <|eot|> (added token), 4 !
EXPECTED_VOCAB = ["<unk>", "hello", "wörld", "<|eot|>", "!"]


class ExportVocabTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.models = self.root / "models"
        self.profile = self.models / "prof"
        self.source = self.root / "source-tokenizer.json"
        self.source.write_text(json.dumps(TOKENIZER, ensure_ascii=False), encoding="utf-8")

    def tearDown(self):
        self._tmp.cleanup()

    def run_exporter(self, tokenizer=None):
        env = {k: v for k, v in os.environ.items() if k != "FNS_TOKENIZER"}
        env["FNS_MODELS"] = str(self.models)
        env["PYTHONIOENCODING"] = "utf-8"
        if tokenizer is not None:
            env["FNS_TOKENIZER"] = str(tokenizer)
        return subprocess.run(
            [sys.executable, str(EXPORTER)], env=env, capture_output=True, text=True, encoding="utf-8"
        )

    def read_vocab(self):
        return json.loads((self.profile / VOCAB_NAME).read_text(encoding="utf-8"))

    def test_first_setup_copies_tokenizer_and_derives_vocab(self):
        result = self.run_exporter(self.source)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual((self.profile / TOKENIZER_NAME).read_bytes(), self.source.read_bytes())
        self.assertEqual(self.read_vocab(), EXPECTED_VOCAB)

    def test_repeated_setup_from_local_copy_preserves_tokenizer_and_restores_vocab(self):
        first = self.run_exporter(self.source)
        self.assertEqual(first.returncode, 0, first.stderr)

        # Second setup: FNS_TOKENIZER names the exported copy itself (same file as the destination).
        second = self.run_exporter(self.profile / TOKENIZER_NAME)
        self.assertEqual(second.returncode, 0, second.stderr)
        self.assertEqual((self.profile / TOKENIZER_NAME).read_bytes(), self.source.read_bytes())
        self.assertEqual(self.read_vocab(), EXPECTED_VOCAB)

        # Third setup: vocab deleted, no FNS_TOKENIZER, so the resolver picks the local exported copy.
        (self.profile / VOCAB_NAME).unlink()
        third = self.run_exporter()
        self.assertEqual(third.returncode, 0, third.stderr)
        self.assertEqual((self.profile / TOKENIZER_NAME).read_bytes(), self.source.read_bytes())
        self.assertEqual(self.read_vocab(), EXPECTED_VOCAB)

    def test_missing_explicit_tokenizer_fails_without_falling_back(self):
        # A valid local copy exists; an explicit FNS_TOKENIZER that is missing must not fall back to it.
        self.profile.mkdir(parents=True)
        (self.profile / TOKENIZER_NAME).write_bytes(self.source.read_bytes())
        missing = self.root / "does-not-exist.json"

        result = self.run_exporter(missing)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("FNS_TOKENIZER points to a missing file", result.stderr)
        self.assertIn(str(missing), result.stderr)
        self.assertFalse((self.profile / VOCAB_NAME).exists())


if __name__ == "__main__":
    unittest.main()
