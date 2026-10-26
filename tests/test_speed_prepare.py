"""Author-created SPEED preparation fixtures: no downloaded benchmark examples.

These tests need only normal service test dependencies, never datasets, pandas,
pyarrow, tiktoken, network access, model weights, or GPU hardware.
"""
import argparse
import contextlib
import hashlib
import importlib.util
import io
import json
import os
import socket
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bench" / "scripts"))
import speed_prepare as prep
import speed_sources as sources


class FixtureChat:
    def prompt_ids(self, messages, tools):
        if tools is not None:
            raise AssertionError("Fixtures must not add tools")
        # Author-written character tokenizer with explicit template overhead.
        return [101] + [ord(c) for c in messages[0]["content"]] + [102, 103]


def fixture_normalize(messages):
    return [{**message, "content": message["content"].replace("\x00", "")} for message in messages]


def fixture_row(identifier, text="written fixture", category="coding"):
    return {"question_id": identifier, "category": category, "source": "https://example.invalid/author-created",
            "src_id": "original-" + identifier, "turns": [text, "Unused fictional second turn"], "sub_category": "fictional"}


def fixture_protocol():
    protocol = prep.load_protocol(prep.DEFAULT_PROTOCOL)
    protocol["ctx"] = 100
    protocol["max_tokens"] = 10
    return protocol


def fixture_lock(work, protocol, corpus):
    raw = work / "raw.jsonl"
    resolved = work / "resolved.jsonl"
    resolver = work / "resolver" / "speed.py"
    license_path = work / "resolver" / "LICENSE"
    for path, data in ((raw, prep.jsonl(corpus)), (resolved, prep.jsonl(corpus)),
                       (resolver, b"# author-created nonexecuting resolver fixture\n"),
                       (license_path, b"Fictional fixture license text\n")):
        sources.immutable_write(path, data)
    lock = {"schema_version": 1, "reproducible": True, "dataset": protocol["dataset"], "resolver": protocol["resolver"],
            "protocol_sha256": sources.digest(prep.DEFAULT_PROTOCOL),
            "dependency_lock_sha256": sources.digest(prep.DEPENDENCIES), "adaptation_sha256": prep.adaptation_hash(),
            "environment": {"fixture": "author-created"}, "raw": {"qualitative": ["raw.jsonl"]},
            "resolved": {"qualitative": "resolved.jsonl"},
            "sources": [{"path": path.relative_to(work).as_posix(), "sha256": sources.digest(path)}
                        for path in (raw, resolved, resolver, license_path)]}
    sources.immutable_write(work / "source-lock.json", sources.canonical(lock) + b"\n")
    return lock


class SelectionTests(unittest.TestCase):
    def select(self, rows, profile="smoke", protocol=None):
        return prep.select_requests({"qualitative": rows}, protocol or fixture_protocol(), profile, FixtureChat(), fixture_normalize)

    def test_deterministic_when_input_order_changes(self):
        rows = [fixture_row(str(i), category=category) for category in ("coding", "writing") for i in range(20)]
        # Question identities must be unique even across categories.
        for row in rows:
            row["question_id"] = row["category"] + "-" + row["question_id"]
        first = self.select(rows)
        second = self.select(list(reversed(rows)))
        self.assertEqual(prep.jsonl(first[0]), prep.jsonl(second[0]))
        self.assertEqual(prep.jsonl(first[1]), prep.jsonl(second[1]))
        self.assertEqual(first[2], second[2])
        self.assertEqual(len(first[0]), 2)

    def test_selection_key_is_specified_compact_json_not_python_hash(self):
        expected = hashlib.sha256(b'["fns-speed-v1","qualitative","fictional-7"]').hexdigest()
        self.assertEqual(prep.selection_key("qualitative", "fictional-7"), (expected, "fictional-7"))

    def test_exact_context_boundary_and_one_token_over(self):
        protocol = fixture_protocol()
        protocol["ctx"], protocol["max_tokens"] = 20, 7
        requests, selection, coverage = self.select([fixture_row("exact", "a" * 10), fixture_row("over", "b" * 11)], protocol=protocol)
        self.assertEqual([row["question_id"] for row in requests], ["exact"])
        decisions = {row["question_id"]: row for row in selection}
        self.assertTrue(decisions["exact"]["admissible"])
        self.assertFalse(decisions["over"]["admissible"])
        self.assertEqual(decisions["over"]["reason"], "context")
        self.assertEqual(protocol["max_tokens"], 7)
        coding = next(c for c in coverage if c["category"] == "coding")
        self.assertEqual((coding["total"], coding["admissible"], coding["selected"], coding["excluded_context"]), (2, 1, 1, 1))
        self.assertEqual(coding["lengths"], {"min": 13, "max": 14, "p50": 13, "p95": 14})

    def test_all_categories_including_empty_and_all_excluded(self):
        requests, _, coverage = self.select([fixture_row("too-long", "x" * 200)])
        self.assertEqual(requests, [])
        self.assertEqual(len(coverage), 11)
        writing = next(c for c in coverage if c["category"] == "writing")
        self.assertEqual(writing["total"], 0)
        self.assertEqual(writing["lengths"], {"min": None, "max": None, "p50": None, "p95": None})
        self.assertEqual(next(c for c in coverage if c["category"] == "coding")["excluded_context"], 1)

    def test_first_turn_only_normalization_hashes_and_unused_count(self):
        row = fixture_row("normalization", "hello\x00world")
        requests, selection, _ = self.select([row])
        request = requests[0]
        self.assertEqual(request["messages"], [{"role": "user", "content": "helloworld"}])
        self.assertEqual(request["unused_turns"], 1)
        self.assertTrue(request["transformed"])
        self.assertEqual(selection[0]["source_text"], "hello\x00world")
        self.assertEqual(selection[0]["normalized_text"], "helloworld")
        self.assertEqual(request["source_text_sha256"], hashlib.sha256(b"hello\x00world").hexdigest())
        self.assertEqual(request["normalized_text_sha256"], hashlib.sha256(b"helloworld").hexdigest())
        self.assertEqual(request["prompt_sha256"], prep.content_hash(request["tokens"]))
        self.assertEqual(request["request_id"], prep.content_hash(["qualitative", "normalization"]))
        self.assertEqual(row["turns"][0], "hello\x00world")

    def test_duplicate_id_missing_source_empty_turn_and_unknown_category_fail(self):
        cases = [[fixture_row("duplicate"), fixture_row("duplicate")],
                 [{**fixture_row("missing"), "source": ""}],
                 [{**fixture_row("empty"), "turns": [""]}],
                 [fixture_row("unexpected", category="private-path-name")]]
        for rows in cases:
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                self.select(rows)

    def test_placeholder_anywhere_in_any_turn_fails(self):
        row = fixture_row("masked")
        row["turns"][1] = "prefix " + sources.PLACEHOLDER
        with self.assertRaisesRegex(ValueError, "placeholder"):
            self.select([row])

    def test_non_idempotent_normalizer_and_invalid_tokens_fail(self):
        def non_idempotent(messages):
            return [{**m, "content": m["content"] + "!"} for m in messages]
        with self.assertRaisesRegex(ValueError, "idempotent"):
            prep.select_requests({"qualitative": [fixture_row("bad-normalizer")]}, fixture_protocol(), "smoke", FixtureChat(), non_idempotent)
        class InvalidChat:
            def prompt_ids(self, messages, tools):
                return [True]
        with self.assertRaisesRegex(ValueError, "token"):
            prep.select_requests({"qualitative": [fixture_row("bad-tokens")]}, fixture_protocol(), "smoke", InvalidChat(), fixture_normalize)

    def test_release_requires_six_configs_and_applies_qualitative_and_throughput_quotas(self):
        protocol = fixture_protocol()
        with self.assertRaisesRegex(ValueError, "six"):
            self.select([fixture_row("only-one-config")], "release")
        corpus = {config: [fixture_row(f"{config}-{i}", category=protocol["categories"][config][0]) for i in range(40)]
                  for config in protocol["configs"]}
        requests, selection, coverage = prep.select_requests(corpus, protocol, "release", FixtureChat(), fixture_normalize)
        self.assertEqual(sum(r["config"] == "qualitative" for r in requests), 40)
        self.assertEqual(sum(r["config"] == "throughput_1k" for r in requests), 32)
        self.assertEqual(len(selection), 240)
        self.assertEqual(len(coverage), 26)
        full, _, _ = prep.select_requests(corpus, protocol, "full", FixtureChat(), fixture_normalize)
        self.assertEqual(len(full), 240)


class LockAndOfflineTests(unittest.TestCase):
    def test_jsonl_preserves_unicode_separators_inside_strings(self):
        rows = [{"text": "before\u0085middle\u2028next\u2029after"}, {"text": "second record"}]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "authored.jsonl"
            path.write_bytes(prep.jsonl(rows))
            self.assertEqual(sources.read_rows([path]), rows)

    def test_missing_and_modified_source_fail(self):
        for missing in (True, False):
            with self.subTest(missing=missing), tempfile.TemporaryDirectory() as directory:
                work = Path(directory)
                protocol = fixture_protocol()
                fixture_lock(work, protocol, [fixture_row("local")])
                prep.verify_lock(work, protocol)
                if missing:
                    (work / "raw.jsonl").unlink()
                else:
                    (work / "raw.jsonl").write_text("altered", encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "Missing or modified"):
                    prep.verify_lock(work, protocol)

    def test_unstable_revision_and_duplicate_artifact_fail(self):
        for modification in ("revision", "duplicate"):
            with self.subTest(modification=modification), tempfile.TemporaryDirectory() as directory:
                work = Path(directory)
                protocol = fixture_protocol()
                lock = fixture_lock(work, protocol, [fixture_row("local")])
                if modification == "revision":
                    lock["sources"][0]["revision"] = "main"
                else:
                    lock["sources"].append(lock["sources"][0])
                (work / "source-lock.json").write_bytes(sources.canonical(lock))
                with self.assertRaises(ValueError):
                    prep.verify_lock(work, protocol)

    def test_paths_cannot_escape_private_lock(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "escapes"):
                prep.locked_path(Path(directory), "../secret.json")
        with self.assertRaisesRegex(ValueError, "perso"):
            sources.private_work_dir(ROOT / "bench" / "forbidden-data")

    def test_immutable_outputs_never_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fixed.json"
            sources.immutable_write(path, b"original")
            sources.immutable_write(path, b"original")
            with self.assertRaisesRegex(ValueError, "overwrite"):
                sources.immutable_write(path, b"different")
            self.assertEqual(path.read_bytes(), b"original")

    def test_authentication_path_preserved_outside_private_cache(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {"HF_HOME": str(Path(directory) / "normal-hf")}, clear=False):
            os.environ.pop("HF_TOKEN_PATH", None)
            work = Path(directory) / "private-work"
            sources.private_work_dir(work)
            self.assertEqual(os.environ["HF_TOKEN_PATH"], str(Path(directory) / "normal-hf" / "token"))
            self.assertNotEqual(Path(os.environ["HF_TOKEN_PATH"]).parent, Path(os.environ["HF_HOME"]))

    def test_nonofficial_cached_resolver_is_never_executed(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "speed.py"
            path.write_text("raise AssertionError('must not execute')", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "authenticated"):
                sources.official_resolver(path, None)

    def test_persisted_acquisition_records_do_not_alias_mutable_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            acquisition = sources.Acquisition(Path(directory), [])
            record = {"path": "fictional.json", "sha256": "a" * 64}
            acquisition.remember("fixture", record)
            record["terms"] = "separate provenance"
            self.assertNotIn("terms", acquisition.plan["fixture"])

    def test_socket_and_dns_blocked_and_restored(self):
        original = socket.create_connection
        with prep.offline_network():
            self.assertEqual(os.environ["HF_HUB_OFFLINE"], "1")
            with self.assertRaisesRegex(RuntimeError, "forbidden"):
                socket.create_connection(("example.invalid", 443))
            with self.assertRaises(RuntimeError):
                socket.getaddrinfo("example.invalid", 443)
        self.assertIs(socket.create_connection, original)

    def test_prepare_twice_produces_identical_files_without_network(self):
        protocol = prep.load_protocol(prep.DEFAULT_PROTOCOL)
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            fixture_lock(work, protocol, [fixture_row("offline", "self authored local text")])
            tokenizer, template = work / "tokenizer.json", work / "template.jinja"
            tokenizer.write_text("author-tokenizer", encoding="utf-8")
            template.write_text("author-template", encoding="utf-8")
            args = argparse.Namespace(offline=True, protocol=prep.DEFAULT_PROTOCOL, work_dir=work,
                                      configs=["qualitative"], profile="smoke", tokenizer=tokenizer, template=template)
            with patch.object(prep, "dependencies", return_value={"fixture": "author-created"}), \
                 patch.object(prep, "load_chat", return_value=(FixtureChat(), fixture_normalize, tokenizer, template)), \
                 patch.object(socket, "create_connection", side_effect=AssertionError("unexpected network")), \
                 contextlib.redirect_stdout(io.StringIO()):
                prep.prepare(args)
                output = work / "prepared" / "smoke"
                before = {p.name: p.read_bytes() for p in output.iterdir()}
                prep.prepare(args)
                after = {p.name: p.read_bytes() for p in output.iterdir()}
            self.assertEqual(before, after)
            manifest = json.loads(after["preparation.json"])
            self.assertEqual(manifest["requests_sha256"], hashlib.sha256(after["requests.jsonl"]).hexdigest())
            self.assertEqual(manifest["selection_sha256"], hashlib.sha256(after["selection.jsonl"]).hexdigest())
            self.assertFalse(manifest["publishable_profile"])

    def test_production_chat_reused_with_local_author_tokenizer(self):
        from tokenizers import Tokenizer, models, pre_tokenizers
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tokenizer.json"
            tokenizer = Tokenizer(models.WordLevel({"[UNK]": 0, "hello": 1}, unk_token="[UNK]"))
            tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
            tokenizer.save(str(path))
            with prep.offline_network():
                chat, normalize, _, template = prep.load_chat(path)
                messages = normalize([{"role": "user", "content": "hello<think>\x00"}])
                self.assertEqual(messages[0]["content"], "hello")
                ids = chat.prompt_ids(messages, None)
                self.assertEqual(ids, chat.prompt_ids(messages, None))
                self.assertEqual(template, ROOT / "serve" / "chat_template.jinja")
                self.assertTrue(ids)


class AdaLEvalMirrorTests(unittest.TestCase):
    def setUp(self):
        from types import SimpleNamespace
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.work = Path(temporary.name)
        self.url = "http://opencompass.openxlab.space/utils/AdaLEval/stackselect_1k.json"
        self.rows = [{"question_id": "authored-repeat", "answer": "first é\u2028value"},
                     {"question_id": "authored-other", "answer": "middle"},
                     {"question_id": "authored-repeat", "answer": "conflicting last"}]
        self.mirror = b"".join(json.dumps(row, ensure_ascii=False).encode("utf-8") + b"\n" for row in self.rows)
        self.original = json.dumps(self.rows, ensure_ascii=False, indent=4).encode("utf-8")
        self.pin = {"mirror_sha256": hashlib.sha256(self.mirror).hexdigest(),
                    "original_sha256": hashlib.sha256(self.original).hexdigest()}
        self.mirror_url = (f"https://huggingface.co/datasets/{sources.ADALEVAL_MIRROR_REPO}/resolve/"
                           f"{sources.ADALEVAL_MIRROR_REVISION}/Ada_LEval/stackselect_1k.jsonl")
        self.original_card = ("https://raw.githubusercontent.com/open-compass/Ada-LEval/"
                              f"{sources.ADALEVAL_SOURCE_REVISION}/README.md")
        self.mirror_card = (f"https://huggingface.co/datasets/{sources.ADALEVAL_MIRROR_REPO}/resolve/"
                            f"{sources.ADALEVAL_MIRROR_REVISION}/README.md")
        self.payloads = {self.mirror_url: self.mirror,
                         self.original_card: b"Author-created original card: no dataset license declared.\n",
                         self.mirror_card: b"Author-created mirror card: no dataset license declared.\n"}
        self.requests = []
        self.loaded = []
        def download(request, **kwargs):
            self.requests.append(request.full_url)
            # No route for the mutable original host or unpinned resources.
            return io.BytesIO(self.payloads[request.full_url])
        def load_dataset(kind, **kwargs):
            self.assertEqual(kind, "json")
            path = Path(kwargs["data_files"]["test"])
            self.loaded.append(path)
            rows = sources.read_rows([path])
            class AuthoredRows:
                def __getitem__(self, column):
                    return [row[column] for row in rows]
            return AuthoredRows()
        for patcher in (patch.dict(sources.ADALEVAL_MIRROR_PINS, {self.url: self.pin}),
                        patch("urllib.request.urlopen", side_effect=download),
                        patch.dict(sys.modules, {"datasets": SimpleNamespace(load_dataset=load_dataset,
                                                                           Dataset=object, concatenate_datasets=None)})):
            patcher.start()
            self.addCleanup(patcher.stop)
        self.acquisition = sources.Acquisition(self.work, ["third-party-sources"])

    def acquire(self, acquisition=None):
        acquisition = acquisition or self.acquisition
        with contextlib.redirect_stdout(io.StringIO()):
            return acquisition.json_source(self.url)

    def original_record(self):
        return next(record for record in self.acquisition.records.values() if record["url"] == self.url)

    def test_reconstructs_exact_bytes_preserving_order_duplicates_and_url_identity(self):
        with contextlib.redirect_stdout(io.StringIO()):
            result = sources.ResolverSources(self.acquisition)("AdaLEval/stackselect", self.url)
        self.assertEqual(result["question_id"], [row["question_id"] for row in self.rows])
        self.assertEqual(result["answer"], [row["answer"] for row in self.rows])
        record = self.original_record()
        expected_path = "external/" + hashlib.sha256(self.url.encode()).hexdigest() + "/stackselect_1k.json"
        self.assertEqual(record["path"], expected_path)
        self.assertEqual(self.loaded, [self.work / expected_path])
        self.assertEqual(self.loaded[0].read_bytes(), self.original)
        self.assertEqual(record["sha256"], self.pin["original_sha256"])
        self.assertTrue(record["independently_reacquirable"])
        self.assertEqual(record["license_status"], "unknown")
        self.assertEqual(record["duplicate_identity_policy"], "official_first_occurrence_in_locked_file_order")
        self.assertEqual((record["source_rows"], record["distinct_question_ids"]), (3, 2))
        provenance = record["acquisition"]
        self.assertEqual(provenance["source_revision"], sources.ADALEVAL_SOURCE_REVISION)
        self.assertEqual(provenance["mirror"]["url"], self.mirror_url)
        self.assertEqual(provenance["mirror"]["sha256"], self.pin["mirror_sha256"])
        self.assertEqual(provenance["mirror"]["revision"], sources.ADALEVAL_MIRROR_REVISION)
        self.assertEqual(provenance["serialization"]["indent"], 4)
        self.assertFalse(provenance["serialization"]["ensure_ascii"])
        self.assertFalse(provenance["serialization"]["trailing_newline"])
        for card in provenance["cards"]:
            self.assertTrue((self.work / card).is_file())
            self.assertEqual(self.acquisition.records[card]["license_status"], "unknown")
        self.assertEqual(set(self.requests), set(self.payloads))
        self.assertNotIn(self.url, self.requests)

    def test_download_corruption_is_rejected_before_json_parse_or_original_write(self):
        self.payloads[self.mirror_url] = b"invalid authored JSONL"
        with patch.object(sources, "read_rows", side_effect=AssertionError("parsed before hash validation")), \
             self.assertRaisesRegex(ValueError, "mirror SHA256 mismatch"):
            self.acquire()
        self.assertEqual(self.loaded, [])
        self.assertFalse((self.work / "external").exists())
        self.assertFalse(any(record.get("independently_reacquirable") for record in self.acquisition.records.values()))

    def test_cached_mirror_corruption_is_rejected_before_parse(self):
        self.acquire()
        record = self.original_record()
        mirror = self.work / record["acquisition"]["mirror"]["path"]
        mirror.write_bytes(b"corrupt cached authored payload")
        replay = sources.Acquisition(self.work, ["third-party-sources"])
        with patch.object(sources, "read_rows", side_effect=AssertionError("corrupt cache parsed")), \
             patch("urllib.request.urlopen", side_effect=AssertionError("unexpected network")), \
             self.assertRaisesRegex(ValueError, "acquisition hash"):
            self.acquire(replay)

    def test_original_hash_mismatch_cannot_write_or_claim_reacquisition(self):
        self.pin["original_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "reconstructed original SHA256 mismatch"):
            self.acquire()
        self.assertEqual(self.loaded, [])
        self.assertFalse((self.work / "external").exists())
        self.assertFalse(any(record.get("independently_reacquirable") for record in self.acquisition.records.values()))

    def test_immutable_replay_uses_cached_mirror_and_retains_exact_plan(self):
        self.acquire()
        before = {path.relative_to(self.work): path.read_bytes() for path in self.work.rglob("*") if path.is_file()}
        replay = sources.Acquisition(self.work, ["third-party-sources"])
        with patch("urllib.request.urlopen", side_effect=AssertionError("replay contacted network")):
            result = self.acquire(replay)
        self.assertEqual(result["answer"], [row["answer"] for row in self.rows])
        self.assertEqual(before, {path.relative_to(self.work): path.read_bytes() for path in self.work.rglob("*") if path.is_file()})
        self.assertEqual(self.original_record(), next(record for record in replay.records.values() if record["url"] == self.url))

    def test_existing_original_is_never_overwritten(self):
        self.acquire()
        path = self.work / self.original_record()["path"]
        path.write_bytes(b"different private original")
        replay = sources.Acquisition(self.work, ["third-party-sources"])
        with self.assertRaisesRegex(ValueError, "overwrite"):
            self.acquire(replay)
        self.assertEqual(path.read_bytes(), b"different private original")

    def test_original_blob_plan_identity_survives_cutover(self):
        relative = "external/" + hashlib.sha256(self.url.encode()).hexdigest() + "/stackselect_1k.json"
        path = self.work / relative
        sources.immutable_write(path, self.original)
        previous = {"url": self.url, "path": relative, "sha256": self.pin["original_sha256"],
                    "bytes": len(self.original)}
        self.acquisition.remember("blob:" + self.url, previous)
        self.acquire()
        self.assertEqual(self.acquisition.plan["blob:" + self.url], previous)
        self.assertEqual(path.read_bytes(), self.original)
        self.assertNotIn(self.url, self.requests)

    def test_provenance_does_not_alias_plan_mirror_record_or_pin(self):
        self.acquire()
        record = self.original_record()
        snapshot = sources.canonical(self.acquisition.plan)
        persisted = self.acquisition.plan["adaleval-mirror:" + self.url]
        mirror_path = record["acquisition"]["mirror"]["path"]
        record["acquisition"]["serialization"]["indent"] = 99
        record["acquisition"]["mirror"]["sha256"] = "edited metadata"
        self.acquisition.records[mirror_path]["url"] = "edited mirror record"
        record["duplicate_identity_policy"] = "edited downstream policy"
        self.assertEqual(sources.canonical(self.acquisition.plan), snapshot)
        self.assertEqual(persisted["acquisition"]["serialization"]["indent"], 4)
        self.assertEqual(self.pin["mirror_sha256"], hashlib.sha256(self.mirror).hexdigest())
        on_disk = json.loads(self.acquisition.plan_path.read_bytes())
        self.assertEqual(on_disk, self.acquisition.plan)

    def test_consent_exposes_both_pinned_cards_before_any_download(self):
        refused = sources.Acquisition(self.work, [])
        with patch.object(refused, "blob", side_effect=AssertionError("download before consent")), \
             contextlib.redirect_stdout(io.StringIO()) as output, \
             self.assertRaisesRegex(ValueError, "accept-license third-party-sources"):
            refused.json_source(self.url)
        self.assertIn(self.original_card, output.getvalue())
        self.assertIn(self.mirror_card, output.getvalue())
        self.assertEqual(refused.records, {})


class AcquisitionConsentTests(unittest.TestCase):
    def test_licence_refusal_precedes_dataset_or_dependency_access(self):
        with tempfile.TemporaryDirectory() as directory:
            args = argparse.Namespace(protocol=prep.DEFAULT_PROTOCOL, work_dir=Path(directory), accept_license=[], configs=None)
            with patch.object(prep, "dependencies", side_effect=AssertionError("dependencies touched")), \
                 patch.object(sources.Acquisition, "hf", side_effect=AssertionError("network touched")), \
                 contextlib.redirect_stdout(io.StringIO()) as output, self.assertRaisesRegex(ValueError, "accept-license"):
                prep.fetch(args)
            self.assertIn("License.pdf", output.getvalue())

    def test_unstable_url_rejected_without_network(self):
        with tempfile.TemporaryDirectory() as directory:
            acquisition = sources.Acquisition(Path(directory), [])
            with patch.object(acquisition, "_json", side_effect=AssertionError("network touched")):
                with self.assertRaisesRegex(ValueError, "immutable"):
                    acquisition.pin_url("https://example.invalid/mutable.json")

    def test_source_loader_failure_is_not_discarded(self):
        loader = sources.ResolverSources(None)
        with patch.object(loader, "load", side_effect=FileNotFoundError("author fixture missing")):
            with self.assertRaisesRegex(RuntimeError, "Required source unavailable"):
                loader("fictional-source")

    def test_adaleval_preserves_conflicting_first_match_and_other_sources_reject(self):
        from types import SimpleNamespace
        url = "http://opencompass.openxlab.space/utils/AdaLEval/stackselect_1k.json"
        rows = [{"question_id": "authored", "answer": "first"},
                {"question_id": "authored", "answer": "different"}]
        class SourceRows:
            def __getitem__(self, key):
                return [row[key] for row in rows]
        dataset = SourceRows()
        acquisition = SimpleNamespace(records={"fixture": {"url": url}})
        loader = sources.ResolverSources(acquisition)
        with patch.object(loader, "load", return_value=dataset):
            resolved = loader("AdaLEval/stackselect", url)
            self.assertEqual(resolved["answer"], ["first", "different"])
            self.assertEqual(acquisition.records["fixture"]["distinct_question_ids"], 1)
            with self.assertRaisesRegex(ValueError, "Duplicate source identities"):
                loader("livecodebench/code_generation_lite")

    def test_livecodebench_uses_official_chunked_json_loader_on_local_files(self):
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            paths = [work / f"test{suffix}.jsonl" for suffix in ("", "2", "3", "4", "5", "6")]
            calls = []
            sentinel = object()
            def local_loader(*args, **kwargs):
                calls.append((args, kwargs))
                return sentinel
            fake_datasets = SimpleNamespace(Dataset=object, concatenate_datasets=None, load_dataset=local_loader)
            acquisition = SimpleNamespace(work=work, hf=lambda *args: paths)
            with patch.dict(sys.modules, {"datasets": fake_datasets}), patch.object(sources, "read_rows", side_effect=AssertionError("giant materialization")):
                result = sources.ResolverSources(acquisition).load("livecodebench/code_generation_lite", "test")
            self.assertIs(result, sentinel)
            self.assertEqual(calls, [(("json",), {"data_files": {"test": [str(path) for path in paths]}, "split": "test", "cache_dir": str(work / "hf" / "datasets")})])

    def test_hle_adapter_matches_official_tuple_external_getter_contract(self):
        from types import SimpleNamespace
        split_calls = []
        class FakeDataset:
            @classmethod
            def from_list(cls, rows):
                return cls()
            def train_test_split(self, **kwargs):
                split_calls.append(kwargs)
                return {"train": "author-train", "test": "author-test"}
        fake = SimpleNamespace(Dataset=FakeDataset, concatenate_datasets=None)
        acquisition = SimpleNamespace(hf=lambda *args: [])
        with patch.dict(sys.modules, {"datasets": fake}), patch.object(sources, "read_rows", return_value=[]):
            result = sources.ResolverSources(acquisition).load("cais/hle", "train_test_split")
        self.assertEqual(result, ("author-train", "author-test"))
        self.assertEqual(split_calls, [{"test_size": .5, "shuffle": True, "seed": 42}])

    def test_unknown_adaleval_file_fails_without_substitution_or_network(self):
        with tempfile.TemporaryDirectory() as directory:
            acquisition = sources.Acquisition(Path(directory), ["third-party-sources"])
            with patch.object(acquisition, "blob", side_effect=AssertionError("network touched")), \
                 patch.object(acquisition, "pin_url", side_effect=AssertionError("mutable fallback")):
                with self.assertRaisesRegex(ValueError, "Unsupported AdaLEval"):
                    acquisition.json_source("http://opencompass.openxlab.space/utils/AdaLEval/stackselect_64k.json")

    def test_import_does_not_import_acquisition_dependencies_or_contact_network(self):
        # Isolated namespace import of both files is safe even when optional data
        # libraries are absent; no subprocess or package installation is needed.
        import builtins
        original_import = builtins.__import__
        blocked = {"datasets", "pandas", "pyarrow", "tiktoken", "requests", "urllib.request"}
        def guarded(name, *args, **kwargs):
            if name in blocked or name.split(".")[0] in blocked:
                raise AssertionError("Unexpected import-time acquisition dependency: " + name)
            return original_import(name, *args, **kwargs)
        with patch.object(builtins, "__import__", side_effect=guarded), \
             patch.object(socket, "create_connection", side_effect=AssertionError("network touched")):
            for name in ("speed_sources", "speed_prepare"):
                spec = importlib.util.spec_from_file_location("fixture_import_" + name, ROOT / "bench" / "scripts" / (name + ".py"))
                assert spec is not None and spec.loader is not None
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)

    def test_cli_help_needs_no_optional_dependencies(self):
        with contextlib.redirect_stdout(io.StringIO()) as output, self.assertRaises(SystemExit) as exited:
            prep.main(["fetch", "--help"])
        self.assertEqual(exited.exception.code, 0)
        self.assertIn("--accept-license", output.getvalue())
        with contextlib.redirect_stdout(io.StringIO()) as output, self.assertRaises(SystemExit) as exited:
            prep.main(["prepare", "--help"])
        self.assertEqual(exited.exception.code, 0)
        self.assertIn("--offline", output.getvalue())


if __name__ == "__main__":
    unittest.main()
