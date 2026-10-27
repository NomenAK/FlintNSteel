"""Authored synthetic campaigns: no downloads, source rows, engines, or optional deps."""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("fns_speed_report_test_module", ROOT / "bench/scripts/speed_report.py")
assert SPEC is not None and SPEC.loader is not None
reporter = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = reporter
SPEC.loader.exec_module(reporter)


def digest(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def fixture(profile="full", comparison="service", mode="greedy", examples=2, pairs=5):
    """All numeric measurements and identities are fabricated specifically for tests."""
    seeds = [0] if mode == "greedy" else [0, 7]
    configs = list(reporter.CONFIGS) if profile == "release" else ["throughput_1k"]
    coverage = []
    requests = []
    for config in configs:
        for category in reporter.CATEGORIES[config]:
            count = examples if (config, category) == ("throughput_1k", "high_entropy") else 0
            if profile == "smoke":
                count = min(count, 1)
            coverage.append({"config": config, "category": category, "total": count + (3 if count else 0), "admissible": count, "selected": count, "excluded_context": 3 if count else 0,
                             "lengths": {"min": 12, "max": 33000, "p50": 12, "p95": 33000} if count else {"min": None, "max": None, "p50": None, "p95": None}})
            requests.extend({"request_id": f"private-example-{index}", "config": config, "category": category, "prompt_tokens": 12} for index in range(count))
    options = {"ctx": 32768, "max_tokens": 256, "ignore_eos": True, "cache_prompt": False, "grammar": False,
               "temperature": 0 if mode == "greedy" else 1, "top_p": 1 if mode == "greedy" else .95, "top_k": 0 if mode == "greedy" else 20,
               "presence_penalty": 0, "frequency_penalty": 0, "repeat_penalty": 1, "min_p": 0, "typical_p": 1, "tfs_z": 1, "mirostat": 0,
               "dynatemp_range": 0, "xtc_probability": 0, "top_n_sigma": -1, "seeds": seeds, "spec": {"A": 0, "B": 3}, "spec_sample": {"A": 0, "B": 2}}
    def fns(arm):
        return ["{python}", "{fns_server}", "--model", "{model}", "--ctx", "{ctx}", "--spec", "0" if arm == "A" else "3", "--spec-sample", "0" if arm == "A" else "2", "--kv-global", "f16", "--log", "{metrics}", "--log-tokens", "--engine", "{fns_binary}", "--vocab", "{vocab}", "--port", "{port}"]
    llama = ["{llama_binary}", "--model", "{model}", "--ctx-size", "{ctx}", "--parallel", "1", "--port", "{port}", "--host", "127.0.0.1", "--jinja", "--chat-template-file", "{template}", "--metrics", "--slots", "--n-gpu-layers", "99", "--flash-attn", "on", "--cache-type-k", "f16", "--cache-type-v", "f16", "--n-cpu-moe", "12", "--moe-cache-mib", "2800"]
    hashes = {key: digest(key) for key in ("model", "tokenizer", "template", "protocol", "source_lock", "dependencies", "fns_binary", "llama_binary")}
    hashes["execution_protocol"] = hashlib.sha256((ROOT / "bench/speed/execution-v2.json").read_bytes()).hexdigest()
    source = {"dirty": False, "patch_sha256": None, "snapshot_sha256": None, "source_tree_sha256": digest("source"), "build_recipe_sha256": digest("recipe")}
    manifest = {"schema_version": 2, "run_id": "private-run-identity", "protocol_name": "FNS-SPEED v2", "profile": profile, "quant": "q3", "comparison": comparison, "mode": mode,
                "seeds": seeds, "pairs": pairs, "max_tokens": 256, "ctx": 32768, "coverage": coverage,
                "expected_requests": requests, "expected_request_ids": [row["request_id"] for row in requests], "protocol_sha256": hashes["protocol"],
                "execution_protocol_sha256": hashes["execution_protocol"], "execution_protocol": copy.deepcopy(reporter.EXECUTION_PROTOCOL),
                "preparation_sha256": digest("preparation"), "requests_sha256": digest("requests"), "source_lock_sha256": hashes["source_lock"], "reproducible": True,
                "selection_sha256": digest("selection"), "warmup_sha256": digest("warmup"),
                "environment_constraints": {"max_gpu_temperature_c": 80},
                "effective_options": options, "public_provenance": {"commits": {"fns": "a" * 40, "llama": "b" * 40}, "hashes": hashes, "sources": {"fns": copy.deepcopy(source), "llama": copy.deepcopy(source)},
                "machine": {"os": "Microsoft Windows 11 Pro", "gpu": "NVIDIA GeForce RTX 3090", "cpu": "GenuineIntel", "ram_bytes": 64 * 1024**3, "driver": "580.88", "cuda": "13.0", "msvc": "19.44"},
                "options": {"A": fns("A") if comparison == "speculation" else llama, "B": fns("B")}}}
    rows = []
    for index, request in enumerate(requests):
        for seed in seeds:
            for pair in range(pairs):
                # Different example scales distinguish ratio-of-medians from pooled medians.
                for arm in ("A", "B"):
                    fns_arm = arm == "B" or comparison == "speculation"
                    latency = ([100, 110, 120, 130, 1000][pair % 5] if index == 0 else [1000, 1100, 1200, 1300, 10000][pair % 5])
                    if arm == "B":
                        latency /= 2 if index == 0 else 4
                    native = {"prefill_ms": 1, "decode_ms": 10, "spec_windows": 10 if arm == "B" else 0, "spec_drafted": 30 if arm == "B" else 0,
                              "spec_accepted": 20 if arm == "B" else 0, "spec_emitted": 30 if arm == "B" else 0, "decode_iterations": 200 if arm == "B" else 255}
                    if not fns_arm:
                        native = {key: (1 if key == "prefill_ms" else 10 if key == "decode_ms" else None) for key in native}
                    rows.append({"schema_version": 2, "run_id": manifest["run_id"], **request, "quant": "q3", "comparison": comparison, "mode": mode, "seed": seed, "pair": pair, "arm": arm,
                                 "attempt": 0, "replaces": None, "pair_generation": 0, "status": "ok", "error_code": None, "latency_ms": latency, "ttft_visible_ms": 1,
                                 "tpot_client_ms": .2, "delta_intervals_ms": [2, 4], "completion_tokens": 256, "prefilled_tokens": request["prompt_tokens"], "reused_tokens": 0, "finish_reason": "length",
                                 "output_tokens_sha256": digest(f"output-{index}-{seed}") if fns_arm else None, "metrics": native,
                                 "effective_options": copy.deepcopy(options)})
                    rows[-1]["gpu_during_request"] = {"samples": 2, "valid_samples": 2, "active_samples": 1, "interval_ms": 200,
                                                      "pcie_gen_min": 4, "pcie_gen_max": 4, "pcie_width_min": 16, "pcie_width_max": 16,
                                                      "memory_used_mib_max": 22000, "temperature_c_max": 60,
                                                      "power_draw_watts_max": 300, "utilization_percent_max": 100}
    inventory = {request["request_id"]: (request["config"], request["category"], request["prompt_tokens"]) for request in requests}
    ordered, sessions = [], []
    for block_id, ids in enumerate(reporter._blocks(inventory)):
        for seed in seeds:
            for pair in range(pairs):
                for arm in (("A", "B") if (block_id + pair) % 2 == 0 else ("B", "A")):
                    sid = f"private-session-{block_id}-{seed}-{pair}-{arm}-0"
                    for rid in ids:
                        row = next(row for row in rows if (row["request_id"], row["seed"], row["pair"], row["arm"]) == (rid, seed, pair, arm))
                        row.update(block_id=block_id, block_generation=0, session_id=sid, observed_order=len(ordered))
                        ordered.append(row)
                    sessions.append({"schema_version": 2, "session_id": sid, "block_id": block_id, "block_generation": 0,
                                     "seed": seed, "pair": pair, "arm": arm, "request_ids": list(ids), "status": "ok", "error_code": None,
                                     "startup_ms": 2, "warmup_ms": 3, "shutdown_ms": 4, "environment_before": {},
                                     "gpu_process_checks": [{"monotonic": 1, "result": {"gpu_state": {"temperature.gpu": "60", "driver_model.current": "WDDM"},
                                                            "owned_pids": [1234], "gpu_processes": [{"pid": 1234, "path": r"C:\private\owned.exe", "type": "C"}]}}]})
    manifest["_sessions"] = sessions
    rows = ordered
    return manifest, rows

def renumber(rows):
    for index, row in enumerate(rows):
        row["observed_order"] = index


def retry_first_block(manifest, rows):
    """Append a complete new generation for the first block/seed/repetition."""
    old = [row for row in rows if (row["block_id"], row["seed"], row["pair"]) == (0, 0, 0)]
    latest = {(row["request_id"], row["arm"]): row for row in old}
    new = []
    template = [session for session in manifest["_sessions"] if (session["block_id"], session["seed"], session["pair"]) == (0, 0, 0)]
    new_sessions = copy.deepcopy(template[:2])
    for session in new_sessions:
        session.update(block_generation=1, session_id=session["session_id"] + "-retry", status="ok", error_code=None)
        for rid in session["request_ids"]:
            row = copy.deepcopy(next(row for row in old if row["request_id"] == rid and row["arm"] == session["arm"]))
            previous = latest[(rid, session["arm"])]
            row.update(attempt=previous["attempt"] + 1, replaces=previous["attempt"], block_generation=1, pair_generation=1,
                       session_id=session["session_id"], status="ok", error_code=None)
            new.append(row)
    end = max(index for index, row in enumerate(rows) if row in old) + 1
    rows[end:end] = new
    session_end = max(index for index, session in enumerate(manifest["_sessions"]) if session in template) + 1
    manifest["_sessions"][session_end:session_end] = new_sessions
    renumber(rows)
    return new, new_sessions



class SpeedReportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.run_path = self.root / "run"
        self.out = self.root / "public"

    def save(self, manifest, rows, path=None):
        path = self.run_path if path is None else path
        path.mkdir(parents=True, exist_ok=True)
        (path / "run-manifest.json").write_text(json.dumps({key: value for key, value in manifest.items() if key != "_sessions"}), encoding="utf-8")
        (path / "sessions.jsonl").write_text("".join(json.dumps(session) + "\n" for session in manifest["_sessions"]), encoding="utf-8")
        (path / "attempts.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        return path

    def generate(self, manifest, rows):
        return reporter.generate_report(self.save(manifest, rows), self.out)

    def invalid(self, manifest, rows):
        with self.assertRaises(ValueError):
            self.generate(manifest, rows)
        self.assertFalse((self.out / "report.json").exists())
        self.assertFalse((self.out / "report.md").exists())

    def test_schema_two_protocol_hash_and_no_legacy_mixing(self):
        for mutation in ("manifest_v1", "attempt_v1", "session_v1", "name", "hash", "provenance_hash", "protocol", "protocol_extra", "pair_order", "inventory_order"):
            with self.subTest(mutation=mutation):
                manifest, rows = fixture()
                if mutation == "manifest_v1":
                    manifest["schema_version"] = 1
                elif mutation == "attempt_v1":
                    rows[0]["schema_version"] = 1
                elif mutation == "session_v1":
                    manifest["_sessions"][0]["schema_version"] = 1
                elif mutation == "name":
                    manifest["protocol_name"] = "FNS-SPEED v1"
                elif mutation == "hash":
                    manifest["execution_protocol_sha256"] = digest("arbitrary-execution")
                    manifest["public_provenance"]["hashes"]["execution_protocol"] = manifest["execution_protocol_sha256"]
                elif mutation == "provenance_hash":
                    manifest["public_provenance"]["hashes"]["execution_protocol"] = digest("different")
                elif mutation == "protocol":
                    manifest["execution_protocol"]["cache_prompt"] = True
                elif mutation == "protocol_extra":
                    manifest["execution_protocol"]["private"] = "ignored-extra"
                elif mutation == "pair_order":
                    manifest["pair_order"] = ["A", "B"]
                else:
                    manifest["expected_request_ids"].reverse()
                self.invalid(manifest, rows)

    def test_block_boundary_and_alternation_for_every_seed(self):
        manifest, rows = fixture(examples=33, mode="sample")
        result = self.generate(manifest, rows)
        self.assertEqual(result["schema_version"], 2)
        self.assertEqual(result["protocol"], "FNS-SPEED v2")
        self.assertEqual(result["runs"][0]["sessions"]["total"], 40)
        self.assertEqual(result["runs"][0]["sessions"]["lifecycle_summed_ms"], {"startup": 80, "warmup": 120, "shutdown": 160})
        for block_id in (0, 1):
            for seed in (0, 7):
                for pair in range(5):
                    sessions = [session for session in manifest["_sessions"] if (session["block_id"], session["seed"], session["pair"]) == (block_id, seed, pair)]
                    self.assertEqual([session["arm"] for session in sessions], ["A", "B"] if (block_id + pair) % 2 == 0 else ["B", "A"])
                    self.assertEqual(len(sessions[0]["request_ids"]), 32 if block_id == 0 else 1)
        self.assertNotIn("lifecycle_summed_ms", result["runs"][0]["categories"][0]["arms"]["A"])

    def test_session_membership_order_completion_and_costs_fail_closed(self):
        for mutation in ("missing", "duplicate", "membership", "request_order", "partial", "arm_order", "row_order", "row_session", "block", "generation", "cost", "row_cost", "observed", "continue_after_error"):
            with self.subTest(mutation=mutation):
                manifest, rows = fixture()
                sessions = manifest["_sessions"]
                if mutation == "missing":
                    sessions.pop()
                elif mutation == "duplicate":
                    sessions.append(copy.deepcopy(sessions[-1]))
                elif mutation == "membership":
                    sessions[0]["request_ids"] = sessions[0]["request_ids"][:1]
                elif mutation == "request_order":
                    sessions[0]["request_ids"].reverse()
                elif mutation == "partial":
                    rows.pop(1)
                elif mutation == "arm_order":
                    sessions[:2] = sessions[1::-1]
                elif mutation == "row_order":
                    rows[:2] = rows[1::-1]
                    renumber(rows)
                elif mutation == "row_session":
                    rows[0]["session_id"] += "-other"
                elif mutation == "block":
                    rows[0]["block_id"] = 1
                elif mutation == "generation":
                    rows[0]["pair_generation"] = 1
                elif mutation == "cost":
                    sessions[0]["shutdown_ms"] = None
                elif mutation == "row_cost":
                    rows[0]["startup_ms"] = 2
                elif mutation == "observed":
                    rows[1]["observed_order"] = rows[0]["observed_order"]
                else:
                    rows[0].update(status="error", error_code="http")
                    sessions[0].update(status="error", error_code="http")
                    retry_first_block(manifest, rows)
                self.invalid(manifest, rows)

    def test_shutdown_failure_with_successful_rows_justifies_whole_block_retry(self):
        manifest, rows = fixture()
        manifest["_sessions"][1].update(status="error", error_code="process")
        retry_first_block(manifest, rows)
        result = self.generate(manifest, rows)["runs"][0]
        self.assertEqual(result["attempts"]["failed"], 0)
        self.assertEqual(result["sessions"]["failed_by_code"], {"process": 1})

    def test_failed_block_generation_is_excluded_for_both_arms(self):
        manifest, rows = fixture(examples=2)
        expected = self.generate(manifest, rows)["runs"][0]["categories"][0]
        manifest["_sessions"][1].update(status="error", error_code="process")
        retry_first_block(manifest, rows)
        for row in rows:
            if row["pair"] == 0 and row["block_generation"] == 0:
                row["latency_ms"] = 50000
        actual = self.generate(manifest, rows)["runs"][0]["categories"][0]
        self.assertEqual(actual, expected)

    def test_same_generation_parity_invalidation_cannot_fabricate_failure(self):
        manifest, rows = fixture(comparison="speculation")
        invalidations = copy.deepcopy([rows[0], rows[2]])
        for row in invalidations:
            row.update(status="error", error_code="spec_parity", attempt=1, replaces=0)
        rows[4:4] = invalidations
        retry_first_block(manifest, rows)
        self.invalid(manifest, rows)

    def test_blocks_group_interleaved_categories_by_first_occurrence(self):
        inventory = {"first": ("throughput_1k", "mixed", 1), "second": ("throughput_1k", "high_entropy", 1),
                     "third": ("throughput_1k", "mixed", 1), "fourth": ("qualitative", "coding", 1)}
        self.assertEqual(reporter._blocks(inventory), [["first", "third"], ["second"], ["fourth"]])

    def test_crash_before_or_during_parity_invalidations_preserves_retry_evidence(self):
        for recorded in (0, 1):
            with self.subTest(recorded=recorded):
                manifest, rows = fixture(comparison="speculation")
                rows[2]["output_tokens_sha256"] = digest("mismatch")
                if recorded:
                    invalidation = copy.deepcopy(rows[0])
                    invalidation.update(status="error", error_code="spec_parity", attempt=1, replaces=0)
                    rows.insert(4, invalidation)
                retry, _ = retry_first_block(manifest, rows)
                for row in retry:
                    if row["request_id"] == rows[0]["request_id"]:
                        row["output_tokens_sha256"] = digest("matched-retry")
                result = self.generate(manifest, rows)
                self.assertEqual(result["runs"][0]["attempts"]["failed"], recorded)

    def test_live_temperature_uses_declared_campaign_ceiling(self):
        manifest, rows = fixture()
        manifest["environment_constraints"]["max_gpu_temperature_c"] = 70
        check = manifest["_sessions"][0]["gpu_process_checks"][0]["result"]
        check["gpu_state"]["temperature.gpu"] = "71"
        self.invalid(manifest, rows)
        manifest["environment_constraints"]["max_gpu_temperature_c"] = 90
        check["gpu_state"]["temperature.gpu"] = "81"
        self.generate(manifest, rows)

    def test_strict_live_monitor_allows_graphics_but_not_foreign_compute(self):
        manifest, rows = fixture()
        processes = manifest["_sessions"][0]["gpu_process_checks"][0]["result"]["gpu_processes"]
        foreign = {"pid": 5678, "path": r"C:\private\graphics.exe", "type": "G"}
        processes.append(foreign)
        self.generate(manifest, rows)
        for kind in ("C", "C+G"):
            foreign["type"] = kind
            with self.assertRaisesRegex(reporter.ReportError, "Unapproved live GPU process"):
                self.generate(manifest, rows)

    def test_malformed_provenance_returns_report_error(self):
        for field in ("options", "artifacts", "python"):
            with self.subTest(field=field):
                manifest, rows = fixture()
                if field == "options":
                    manifest["public_provenance"]["options"]["A"] = [None]
                else:
                    manifest[field] = None
                with self.assertRaises(reporter.ReportError):
                    self.generate(manifest, rows)

    def test_successful_sessions_require_safe_live_owned_process_monitor(self):
        for mutation in ("missing", "empty", "malformed", "time", "error", "temperature", "owned", "unknown", "unlisted", "path", "duplicate"):
            with self.subTest(mutation=mutation):
                manifest, rows = fixture()
                session = manifest["_sessions"][0]
                check = session["gpu_process_checks"][0]
                result = check["result"]
                if mutation == "missing":
                    session.pop("gpu_process_checks")
                elif mutation == "empty":
                    session["gpu_process_checks"] = []
                elif mutation == "malformed":
                    check["result"] = None
                elif mutation == "time":
                    session["gpu_process_checks"].append(copy.deepcopy(check))
                elif mutation == "error":
                    check["error"] = "private-monitor-failure"
                elif mutation == "temperature":
                    result["gpu_state"]["temperature.gpu"] = "81"
                elif mutation == "owned":
                    result["owned_pids"] = []
                elif mutation == "unknown":
                    result["gpu_processes"][0]["type"] = "N/A"
                elif mutation == "unlisted":
                    result["gpu_processes"][0]["pid"] = 5678
                elif mutation == "path":
                    result["gpu_processes"][0]["path"] = "owned.exe"
                else:
                    result["gpu_processes"].append(copy.deepcopy(result["gpu_processes"][0]))
                self.invalid(manifest, rows)

    def test_loaded_monitor_accepts_only_desktop_allowlist_without_idle_thresholds(self):
        manifest, rows = fixture()
        self.desktop_policy(manifest, rows)
        result = manifest["_sessions"][0]["gpu_process_checks"][0]["result"]
        result["gpu_state"].update({"memory.free": "0", "utilization.gpu": "100"})
        result["gpu_processes"].append({"pid": 5678, "path": manifest["gpu_process_policy"]["allowed_graphics_paths"][0], "type": "G"})
        self.generate(manifest, rows)
        for mutation in ("compute", "model", "unlisted"):
            with self.subTest(mutation=mutation):
                altered = copy.deepcopy(manifest)
                observed = altered["_sessions"][0]["gpu_process_checks"][0]["result"]
                if mutation == "compute":
                    observed["gpu_processes"][-1]["type"] = "C"
                elif mutation == "model":
                    observed["gpu_state"]["driver_model.current"] = "TCC"
                else:
                    observed["gpu_processes"][-1]["path"] = r"C:\unlisted.exe"
                with self.assertRaises(ValueError):
                    self.generate(altered, rows)

    def test_independent_statistics_and_nearest_rank(self):
        manifest, rows = fixture()
        report = self.generate(manifest, rows)
        result = report["runs"][0]["categories"][0]
        self.assertEqual(result["median_example_latency_ratio_a_over_b"], 3)
        self.assertEqual(result["valid_pairs"], 10)
        self.assertEqual(result["paired_win_fraction_b"], 1)
        self.assertEqual(result["qualification"], "stable_gain")
        self.assertEqual(result["arms"]["A"]["latency_ms"], {"p50": 1000, "p95": 10000})
        self.assertEqual(result["arms"]["B"]["latency_ms"], {"p50": 250, "p95": 2500})
        self.assertAlmostEqual(result["arms"]["A"]["throughput_tokens_per_second"], 2560000 / 16060)
        self.assertAlmostEqual(result["arms"]["B"]["throughput_tokens_per_second"], 2560000 / 4380)
        speculation = result["arms"]["B"]["speculation"]
        self.assertAlmostEqual(speculation["acceptance"]["value"], 2 / 3)
        self.assertEqual(speculation["emitted_per_window"]["value"], 3)
        self.assertEqual(speculation["effective_advance"]["value"], 1.275)
        self.assertEqual(result["arms"]["B"]["sse_nonempty_delta_intervals_ms"]["p95"], 4)
        self.assertEqual(len(report["runs"][0]["categories"]), 3)
        self.assertEqual(report["runs"][0]["categories"][1]["qualification"], "no_data")
        self.assertIn("Context excluded", (self.out / "report.md").read_text(encoding="utf-8"))

    def test_exact_80_percent_and_two_percent_threshold(self):
        manifest, rows = fixture(examples=1)
        a = [100, 100, 100, 100, 100]
        b = [98, 98, 98, 98, 101]
        for row in rows:
            row["latency_ms"] = (a if row["arm"] == "A" else b)[row["pair"]]
        category = self.generate(manifest, rows)["runs"][0]["categories"][0]
        self.assertEqual(category["paired_win_fraction_b"], .8)
        self.assertEqual(category["qualification"], "stable_gain")
        for row in rows:
            if row["arm"] == "B":
                row["latency_ms"] = 99 if row["pair"] < 4 else 101
        category = self.generate(manifest, rows)["runs"][0]["categories"][0]
        self.assertEqual(category["qualification"], "not_qualified")
        for row in rows:
            if row["arm"] == "B":
                row["latency_ms"] = 50 if row["pair"] < 3 else 101
        category = self.generate(manifest, rows)["runs"][0]["categories"][0]
        self.assertGreater(category["median_example_latency_ratio_a_over_b"], 1.02)
        self.assertEqual(category["qualification"], "not_qualified")

    def test_missing_duplicate_unexpected_and_incompatible_attempts(self):
        for mutation in ("missing", "duplicate", "request", "quant", "config", "category", "seed", "pair", "option", "null_metric", "output", "kv", "reason", "nan"):
            with self.subTest(mutation=mutation):
                manifest, rows = fixture()
                if mutation == "missing":
                    rows.pop()
                elif mutation == "duplicate":
                    rows.append(copy.deepcopy(rows[0]))
                elif mutation in ("request", "quant", "config", "category", "seed", "pair"):
                    key = "request_id" if mutation == "request" else mutation
                    rows[0][key] = 99 if mutation in ("seed", "pair") else "not-declared"
                elif mutation == "option":
                    rows[0]["effective_options"]["cache_prompt"] = True
                elif mutation == "null_metric":
                    next(row for row in rows if row["arm"] == "B")["metrics"]["spec_accepted"] = None
                elif mutation == "output":
                    rows[1]["completion_tokens"] -= 1
                elif mutation == "kv":
                    rows[0]["reused_tokens"] = 1
                elif mutation == "reason":
                    rows[0]["finish_reason"] = "stop"
                else:
                    rows[0]["latency_ms"] = float("nan")
                self.invalid(manifest, rows)

    def test_manifest_missing_inventory_coverage_reproducibility_or_provenance(self):
        for mutation in ("pairs", "reproducible", "ids", "expected", "coverage", "counts", "hash", "source", "recipe", "seed", "empty"):
            with self.subTest(mutation=mutation):
                manifest, rows = fixture()
                if mutation == "pairs":
                    manifest["pairs"] = 4
                elif mutation == "reproducible":
                    manifest["reproducible"] = False
                elif mutation == "ids":
                    manifest["expected_request_ids"].pop()
                elif mutation == "expected":
                    manifest["expected_requests"].append(copy.deepcopy(manifest["expected_requests"][0]))
                elif mutation == "coverage":
                    manifest["coverage"].pop()
                elif mutation == "counts":
                    manifest["coverage"][0]["selected"] -= 1
                elif mutation == "hash":
                    manifest["public_provenance"]["hashes"]["model"] = None
                elif mutation == "source":
                    manifest["public_provenance"]["sources"]["fns"]["dirty"] = True
                elif mutation == "recipe":
                    del manifest["public_provenance"]["sources"]["fns"]["build_recipe_sha256"]
                elif mutation == "seed":
                    manifest["seeds"] = [7]
                else:
                    manifest, rows = fixture(examples=0)
                self.invalid(manifest, rows)

    def test_release_all_six_configs_and_nonpublishable_smoke(self):
        manifest, rows = fixture(profile="release")
        result = self.generate(manifest, rows)
        self.assertEqual(len(result["runs"][0]["coverage"]), 26)
        self.assertNotEqual(result["publication_status"], "NON-PUBLISHABLE")
        manifest, rows = fixture(profile="smoke", pairs=1)
        manifest["reproducible"] = False
        result = self.generate(manifest, rows)
        self.assertEqual(result["publication_status"], "NON-PUBLISHABLE")
        self.assertIn("**NON-PUBLISHABLE**", (self.out / "report.md").read_text(encoding="utf-8"))

    def test_release_missing_config_or_wrong_budget_rejected(self):
        for mutation in ("config", "budget"):
            manifest, rows = fixture(profile="release")
            if mutation == "config":
                manifest["coverage"] = [row for row in manifest["coverage"] if row["config"] != "throughput_32k"]
            else:
                manifest["max_tokens"] = 128
            self.invalid(manifest, rows)

    def test_linked_retry_failures_visible_and_no_free_error_text(self):
        manifest, rows = fixture(examples=2)
        rows[3].update(status="error", error_code="timeout", error="PRIVATE-FAILURE-TEXT")
        manifest["_sessions"][1].update(status="error", error_code="timeout", error="PRIVATE-FAILURE-TEXT")
        retry_first_block(manifest, rows)
        result = self.generate(manifest, rows)
        self.assertEqual(result["runs"][0]["attempts"], {"total": 24, "successful_final": 20, "failed": 1, "failed_by_code": {"timeout": 1}})
        sessions = result["runs"][0]["sessions"]
        self.assertEqual(sessions["total"], 12)
        self.assertEqual(sessions["failed_by_code"], {"timeout": 1})
        self.assertEqual(sessions["lifecycle_summed_ms"], {"startup": 24, "warmup": 36, "shutdown": 48})
        self.assertNotIn("PRIVATE-FAILURE-TEXT", (self.out / "report.json").read_text(encoding="utf-8"))

    def test_unlinked_unresolved_cherry_picked_and_mixed_generation_retries_fail(self):
        for mutation in ("unlinked", "unresolved", "cherry_pick", "mixed", "skip_generation", "session_only", "rows_only", "same_generation"):
            with self.subTest(mutation=mutation):
                manifest, rows = fixture(examples=2)
                if mutation != "cherry_pick":
                    manifest["_sessions"][1].update(status="error", error_code="process")
                retry, sessions = retry_first_block(manifest, rows)
                if mutation == "unlinked":
                    retry[0]["replaces"] = None
                elif mutation == "unresolved":
                    rows[:] = [row for row in rows if row not in retry]
                    manifest["_sessions"][:] = [session for session in manifest["_sessions"] if session not in sessions]
                elif mutation == "mixed":
                    rows.remove(retry[-1])
                elif mutation == "skip_generation":
                    for row in retry:
                        row.update(pair_generation=2, block_generation=2)
                    for session in sessions:
                        session["block_generation"] = 2
                elif mutation == "session_only":
                    rows[:] = [row for row in rows if row not in retry]
                elif mutation == "rows_only":
                    manifest["_sessions"][:] = [session for session in manifest["_sessions"] if session not in sessions]
                elif mutation == "same_generation":
                    for row in retry:
                        row.update(pair_generation=0, block_generation=0)
                self.invalid(manifest, rows)

    def test_interrupted_unfinalized_session_can_retry_only_as_whole_block(self):
        manifest, rows = fixture(examples=2)
        retry_first_block(manifest, rows)
        rows[:] = [row for row in rows if not (row["pair"] == 0 and row["block_generation"] == 0 and row["arm"] == "B")]
        # No finalized record for the interrupted first session or its unstarted second arm.
        manifest["_sessions"][:] = [session for session in manifest["_sessions"] if not (session["pair"] == 0 and session["block_generation"] == 0)]
        for row in rows:
            if row["pair"] == 0 and row["block_generation"] == 1 and row["arm"] == "B":
                row.update(attempt=0, replaces=None)
        renumber(rows)
        result = self.generate(manifest, rows)
        self.assertEqual(result["runs"][0]["attempts"]["total"], 22)
        self.assertEqual(result["runs"][0]["sessions"]["unfinalized"], 1)
        self.assertEqual(result["runs"][0]["sessions"]["total"], 10)

    def test_ablation_exact_hash_parity_for_every_seed_and_pair(self):
        for mode in ("greedy", "sample"):
            manifest, rows = fixture(comparison="speculation", mode=mode)
            result = self.generate(manifest, rows)
            self.assertEqual(result["runs"][0]["categories"][0]["valid_pairs"], 10 * (2 if mode == "sample" else 1))
            spec = result["runs"][0]["categories"][0]["arms"]["A"]["speculation"]
            self.assertEqual(spec["acceptance"], {"value": None, "reason": "zero_denominator"})
            self.assertEqual(spec["effective_advance"]["value"], 1)
            rows[-1]["output_tokens_sha256"] = digest("different-sequence")
            with self.assertRaisesRegex(ValueError, "parity"):
                self.generate(manifest, rows)

    def test_recorded_parity_invalidation_then_linked_whole_block_retry(self):
        manifest, rows = fixture(comparison="speculation", examples=2)
        rows[2]["output_tokens_sha256"] = digest("mismatch")
        invalidations = copy.deepcopy([rows[0], rows[2]])
        for row in invalidations:
            row.update(status="error", error_code="spec_parity", attempt=1, replaces=0)
        rows[4:4] = invalidations
        retry, _ = retry_first_block(manifest, rows)
        for row in retry:
            if row["request_id"] == rows[0]["request_id"]:
                row["output_tokens_sha256"] = digest("matched-retry")
        result = self.generate(manifest, rows)
        self.assertEqual(result["runs"][0]["attempts"]["failed_by_code"], {"spec_parity": 2})
        self.assertEqual(result["runs"][0]["sessions"]["failed"], 0)

    def test_no_interengine_rng_parity_requirement(self):
        manifest, rows = fixture(mode="sample")
        for row in rows:
            if row["arm"] == "A":
                row["output_tokens_sha256"] = digest("unrelated-llama-sequence")
        self.generate(manifest, rows)

    def test_nullable_visible_timing_and_no_draft_ratios(self):
        manifest, rows = fixture()
        for row in rows:
            row.update(ttft_visible_ms=None, tpot_client_ms=None, delta_intervals_ms=[])
            if row["arm"] == "B":
                row["metrics"].update(spec_windows=0, spec_drafted=0, spec_accepted=0, spec_emitted=0, decode_iterations=255)
        result = self.generate(manifest, rows)["runs"][0]["categories"][0]["arms"]["B"]
        self.assertIsNone(result["ttft_visible_ms"]["p50"])
        self.assertEqual(result["ttft_visible_ms"]["null_reason"], "no_visible_delta")
        self.assertIsNone(result["tpot_client_ms"]["p95"])
        self.assertEqual(result["speculation"]["emitted_per_window"]["reason"], "zero_denominator")

    def test_under_load_gpu_summary_aggregates_only_typed_observations(self):
        manifest, rows = fixture(examples=1)
        for row in rows:
            row["gpu_during_request"] = {"samples": 5, "valid_samples": 5, "active_samples": 4, "interval_ms": 200,
                                         "pcie_gen_min": 3, "pcie_gen_max": 4, "pcie_width_min": 16, "pcie_width_max": 16,
                                         "memory_used_mib_max": 22000 + row["pair"], "temperature_c_max": 60 + row["pair"],
                                         "power_draw_watts_max": 300, "utilization_percent_max": 100,
                                         "raw_csv": "PRIVATE-GPU-SECRET", "source_path": "C:/Users/PRIVATE-GPU-SECRET"}
        report = self.generate(manifest, rows)
        summary = report["runs"][0]["categories"][0]["arms"]["B"]["gpu_during_request"]
        self.assertEqual(summary["samples"], 25)
        self.assertEqual(summary["active_samples"], 20)
        self.assertEqual(summary["pcie_gen_min"], 3)
        self.assertEqual(summary["pcie_gen_max"], 4)
        self.assertEqual(summary["memory_used_mib_max"], 22004)
        self.assertEqual(summary["requests_unobserved"], 0)
        self.assertEqual(summary["observation"], "measured_request_active_samples")
        self.assertNotIn("PRIVATE-GPU-SECRET", (self.out / "report.json").read_text(encoding="utf-8"))
        self.assertNotIn("raw_csv", (self.out / "report.md").read_text(encoding="utf-8"))
        manifest, rows = fixture(profile="smoke", examples=1)
        for row in rows:
            row.pop("gpu_during_request")
        summary = self.generate(manifest, rows)["runs"][0]["categories"][0]["arms"]["B"]["gpu_during_request"]
        self.assertEqual(summary["observation"], "no_active_measured_sample")
        self.assertIsNone(summary["pcie_gen_max"])

    def test_publication_requires_under_load_sampler_evidence(self):
        for mutation in ("missing", "inactive", "invalid_counts", "missing_pcie", "bad_interval"):
            with self.subTest(mutation=mutation):
                manifest, rows = fixture(examples=1)
                for row in rows:
                    if row["arm"] != "B":
                        continue
                    summary = row["gpu_during_request"]
                    if mutation == "missing":
                        row.pop("gpu_during_request")
                    elif mutation == "inactive":
                        summary["active_samples"] = 0
                        for key in tuple(summary):
                            if key.endswith(("_min", "_max")):
                                summary[key] = None
                    elif mutation == "invalid_counts":
                        summary["active_samples"] = 3
                    elif mutation == "missing_pcie":
                        for key in ("pcie_gen_min", "pcie_gen_max", "pcie_width_min", "pcie_width_max"):
                            summary[key] = None
                    else:
                        summary["interval_ms"] = 0
                self.invalid(manifest, rows)

    def test_positive_schema_blocks_nested_metadata_identifiers_and_paths(self):
        secret = "SYNTHETIC-PRIVATE-SECRET-DO-NOT-PUBLISH"
        manifest, rows = fixture()
        manifest.update(owner=secret, source_metadata={"annotation": secret}, argv=[secret], run_id=secret)
        manifest["public_provenance"]["api_key"] = secret
        manifest["public_provenance"]["commits"]["author"] = secret
        manifest["public_provenance"]["hashes"]["source_url"] = secret
        manifest["public_provenance"]["sources"]["fns"]["repository"] = secret
        machine = manifest["public_provenance"]["machine"]
        for key in ("os", "gpu", "cpu", "driver", "cuda", "msvc", "hostname", "username"):
            machine[key] = secret
        for row in manifest["coverage"]:
            row.update(annotation=secret, source=secret, question_id=secret)
            row["lengths"]["source_text"] = secret
        for row in manifest["expected_requests"]:
            row["request_id"] += secret
            row.update(messages=[{"role": "user", "content": secret}], token_ids=[123456789], question_id=secret)
        manifest["expected_request_ids"] = [row["request_id"] for row in manifest["expected_requests"]]
        for session in manifest["_sessions"]:
            session["request_ids"] = [rid + secret for rid in session["request_ids"]]
            session["session_id"] += secret
            session.update(raw_argv=[secret], log_path="C:/Users/PRIVATE-USER/" + secret, error=secret)
        for row in rows:
            row["run_id"] = secret
            row["request_id"] += secret
            row["session_id"] += secret
            row.update(prompt=secret, output=secret, tokens=[123456789], source_metadata={"src_id": secret}, path="C:/Users/PRIVATE-USER/secret", finish_metadata=secret)
            row["metrics"]["stdout"] = secret
        for arm in ("A", "B"):
            argv = manifest["public_provenance"]["options"][arm]
            argv[argv.index("--model") + 1] = "C:/Users/PRIVATE-USER/" + secret
        result = self.generate(manifest, rows)
        for name in ("report.json", "report.md"):
            text = (self.out / name).read_text(encoding="utf-8")
            for forbidden in (secret, "PRIVATE-USER", "123456789", "private-example-"):
                self.assertNotIn(forbidden, text)
        private_keys = {"request_id", "question_id", "src_id", "run_id", "expected_request_ids", "expected_requests", "source_metadata", "messages", "token_ids", "tokens", "prompt", "output", "path", "api_key", "owner"}
        def assert_public_structure(value):
            if isinstance(value, dict):
                self.assertTrue(private_keys.isdisjoint(value), "Private row field escaped the positive schema")
                for child in value.values():
                    assert_public_structure(child)
            elif isinstance(value, list):
                for child in value:
                    assert_public_structure(child)
        assert_public_structure(result)
        assert_public_structure(json.loads((self.out / "report.json").read_text(encoding="utf-8")))
        self.assertIsNone(result["runs"][0]["provenance"]["machine"]["gpu"])
        safe_argv = result["runs"][0]["provenance"]["options"]["A"]
        self.assertEqual(safe_argv[safe_argv.index("--model") + 1], "{model}")

    def desktop_policy(self, manifest, rows):
        path = r"C:\Users\PRIVATE-DESKTOP-PATH\desktop-fixture.exe"
        policy = {"mode": "wddm_desktop", "allowed_graphics_paths": [path]}
        observed = {"gpu_process_policy": copy.deepcopy(policy), "gpu_state": {"driver_model.current": "WDDM"},
                    "permitted_graphics_processes": [{"pid": 765432101, "path": path, "type": "C+G"}]}
        manifest["gpu_process_policy"] = copy.deepcopy(policy)
        manifest["machine"] = copy.deepcopy(observed)
        for row in rows:
            row["gpu_process_policy"] = copy.deepcopy(policy)
            row["environment_before"] = copy.deepcopy(observed)
        for session in manifest["_sessions"]:
            session["environment_before"] = copy.deepcopy(observed)

    def test_desktop_session_policy_exports_only_label_counts_and_limitations(self):
        manifest, rows = fixture(profile="release", examples=1)
        self.desktop_policy(manifest, rows)
        manifest["_sessions"][0]["environment_before"]["permitted_graphics_processes"].append(
            {"pid": 765432102, "path": manifest["gpu_process_policy"]["allowed_graphics_paths"][0], "type": "G"})
        result = self.generate(manifest, rows)
        summary = result["runs"][0]["gpu_process_policy"]
        self.assertEqual(set(summary), {"mode", "measurement_environment", "allowed_graphics_path_count", "permitted_context_count_max", "limitations"})
        self.assertEqual(summary["mode"], "wddm_desktop")
        self.assertEqual(summary["measurement_environment"], "desktop-session")
        self.assertEqual(summary["allowed_graphics_path_count"], 1)
        self.assertEqual(summary["permitted_context_count_max"], 2)
        self.assertIn("not absolute GPU isolation", " ".join(summary["limitations"]))
        for name in ("report.json", "report.md"):
            text = (self.out / name).read_text(encoding="utf-8")
            self.assertIn("desktop-session", text)
            for secret in ("PRIVATE-DESKTOP-PATH", "765432101", "765432102", "allowed_graphics_paths", "permitted_graphics_processes"):
                self.assertNotIn(secret, text)

    def test_desktop_report_rejects_malformed_or_inconsistent_private_policy(self):
        for mutation in ("mode", "empty", "relative", "duplicate", "attempt_policy", "missing_attempt_policy", "missing_observation",
                         "non_wddm", "unlisted", "compute", "unknown", "missing_pid", "duplicate_pid", "missing_contexts", "manifest_observation"):
            with self.subTest(mutation=mutation):
                manifest, rows = fixture(examples=1)
                self.desktop_policy(manifest, rows)
                observed = manifest["_sessions"][0]["environment_before"]
                context = observed["permitted_graphics_processes"][0]
                if mutation == "mode":
                    manifest["gpu_process_policy"]["mode"] = "private-invalid-mode"
                elif mutation == "empty":
                    manifest["gpu_process_policy"]["allowed_graphics_paths"] = []
                elif mutation == "relative":
                    manifest["gpu_process_policy"]["allowed_graphics_paths"] = ["desktop-fixture.exe"]
                elif mutation == "duplicate":
                    manifest["gpu_process_policy"]["allowed_graphics_paths"] *= 2
                elif mutation == "attempt_policy":
                    rows[0]["gpu_process_policy"] = {"mode": "strict"}
                elif mutation == "missing_attempt_policy":
                    rows[0].pop("gpu_process_policy")
                elif mutation == "missing_observation":
                    manifest["_sessions"][0].pop("environment_before")
                elif mutation == "non_wddm":
                    observed["gpu_state"]["driver_model.current"] = "TCC"
                elif mutation == "unlisted":
                    context["path"] = r"C:\Other\desktop-fixture.exe"
                elif mutation in ("compute", "unknown"):
                    context["type"] = "C" if mutation == "compute" else "N/A"
                elif mutation == "missing_pid":
                    context.pop("pid")
                elif mutation == "duplicate_pid":
                    observed["permitted_graphics_processes"].append(copy.deepcopy(context))
                elif mutation == "missing_contexts":
                    observed.pop("permitted_graphics_processes")
                else:
                    manifest.pop("machine")
                self.invalid(manifest, rows)

    def test_old_manifest_retains_strict_policy_without_inventing_isolation(self):
        manifest, rows = fixture()
        summary = self.generate(manifest, rows)["runs"][0]["gpu_process_policy"]
        self.assertEqual(summary["mode"], "strict")
        self.assertEqual(summary["allowed_graphics_path_count"], 0)
        self.assertEqual(summary["permitted_context_count_max"], 0)
        self.assertIn("does not establish absolute GPU isolation", " ".join(summary["limitations"]))

    def test_typed_machine_model_build_and_dll_hashes(self):
        manifest, rows = fixture()
        manifest["public_provenance"]["machine"]["cpu"] = "AMD Ryzen 7 7800X3D 8-Core Processor"
        manifest["public_provenance"]["machine"]["os"] = "Microsoft Windows 11 Pro for Workstations"
        manifest["machine"] = {"version": "10.0.26300", "architecture": "AMD64", "declared": {"build_architecture": "x64"},
                               "power_plan": "Power Scheme GUID: 8c5e7fda-e8bf-4a96-9a85-a6e23a8c635c (PRIVATE LABEL)",
                               "gpu_state": {"driver_model.current": "WDDM", "display_active": "Enabled", "memory.free": "22000", "memory.total": "24576", "pcie.link.gen.current": "4", "pcie.link.width.current": "16"}}
        manifest["artifacts"] = {"dlls": [{"path": "C:/Users/PRIVATE-USER/private-name.dll", "sha256": digest("dll")}]}
        provenance = self.generate(manifest, rows)["runs"][0]["provenance"]
        machine = provenance["machine"]
        self.assertEqual(machine["cpu"], {"vendor": "AMD", "family": "Ryzen", "tier": 7, "model": 7800, "suffix": "X3D"})
        self.assertEqual(machine["os"]["build_version"], [10, 0, 26300])
        self.assertEqual(machine["driver_model"], "WDDM")
        self.assertEqual(machine["power_plan"], "high_performance")
        self.assertEqual(machine["pcie_observed_width"], 16)
        self.assertEqual(provenance["dll_sha256"], [digest("dll")])
        self.assertNotIn("PRIVATE", (self.out / "report.json").read_text(encoding="utf-8"))

    def test_allowlisted_fields_reject_privacy_payloads_and_unknown_options(self):
        for mutation in ("category", "hash", "commit", "numeric", "unknown_option", "argv_secret", "boolean"):
            with self.subTest(mutation=mutation):
                manifest, rows = fixture()
                secret = "C:/Users/PRIVATE-SECRET/api-key"
                if mutation == "category":
                    manifest["coverage"][0]["category"] = secret
                elif mutation == "hash":
                    manifest["public_provenance"]["hashes"]["model"] = secret
                elif mutation == "commit":
                    manifest["public_provenance"]["commits"]["fns"] = secret
                elif mutation == "numeric":
                    manifest["public_provenance"]["machine"]["ram_bytes"] = secret
                elif mutation == "unknown_option":
                    manifest["public_provenance"]["options"]["B"] += ["--api-key", secret]
                elif mutation == "argv_secret":
                    argv = manifest["public_provenance"]["options"]["A"]
                    argv[argv.index("--n-gpu-layers") + 1] = secret
                else:
                    manifest["effective_options"]["cache_prompt"] = 0
                self.invalid(manifest, rows)

    def test_ablation_options_and_disabled_counters_validated(self):
        for mutation in ("args", "counter", "hash"):
            manifest, rows = fixture(comparison="speculation")
            if mutation == "args":
                manifest["public_provenance"]["options"]["A"] += ["--ubatch", "64"]
            elif mutation == "counter":
                rows[0]["metrics"]["spec_windows"] = 1
            else:
                rows[0]["output_tokens_sha256"] = None
            self.invalid(manifest, rows)

    def test_collection_is_deterministic_and_rejects_duplicates_or_incompatibility(self):
        manifest, rows = fixture()
        first = self.save(manifest, rows, self.root / "collection" / "z")
        other, other_rows = fixture(comparison="speculation")
        other["run_id"] = "second-private-run"
        for row in other_rows:
            row["run_id"] = other["run_id"]
        self.save(other, other_rows, first.parent / "a")
        report = reporter.generate_report(first.parent, self.out)
        before = ((self.out / "report.json").read_bytes(), (self.out / "report.md").read_bytes())
        self.assertEqual(reporter.generate_report(first.parent, self.out), report)
        self.assertEqual(before, ((self.out / "report.json").read_bytes(), (self.out / "report.md").read_bytes()))
        other["comparison"] = "service"
        self.save(other, other_rows, first.parent / "a")
        with self.assertRaises(ValueError):
            reporter.generate_report(first.parent, self.out)
        other, other_rows = fixture(comparison="speculation")
        other["run_id"] = "third-private-run"
        other["requests_sha256"] = digest("changed-selection")
        self.save(other, other_rows, first.parent / "a")
        with self.assertRaises(ValueError):
            reporter.generate_report(first.parent, self.out)


if __name__ == "__main__":
    unittest.main()
