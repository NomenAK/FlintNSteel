"""Authored SPEED fixtures: no corpus, dataset package, server binary or network access."""
import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bench" / "scripts"))
import speed_bench as bench
from speed_runtime import BenchError, SSEParser, StreamCapture


class StreamingTests(unittest.TestCase):
    def event(self, delta=None, finish=None, usage=None):
        return json.dumps({"choices": [] if delta is None else [{"index": 0, "delta": delta,
                            "finish_reason": finish}], "usage": usage}, ensure_ascii=False)

    def usage(self):
        return {"prompt_tokens": 3, "completion_tokens": 4, "total_tokens": 7}

    def test_fragmented_unicode_reasoning_role_and_crlf(self):
        frames = [self.event({"role": "assistant"}), self.event({"reasoning_content": "Réfléchir 🦊"}),
                  self.event({"content": "answer"}), self.event({}, "length", self.usage()), "[DONE]"]
        data = ("\r\n".join("data: " + event + "\r\n" for event in frames) + "\r\n").encode()
        parser = SSEParser()
        capture = StreamCapture(3, 4, 10)
        now = 10
        for byte in data:
            for event in parser.feed(bytes([byte])):
                now += 1
                capture.event(event, now)
        parser.feed(b"", final=True)
        result = capture.result()
        self.assertEqual(result["ttft_visible_ms"], 2000)
        self.assertEqual(result["latency_ms"], 5000)
        self.assertEqual(result["delta_intervals_ms"], [1000])
        self.assertAlmostEqual(result["tpot_client_ms"], 1000 / 3)
        self.assertEqual(result["completion_tokens"], 4)

    def test_no_visible_tokens_is_null_not_zero(self):
        capture = StreamCapture(3, 4, 10)
        capture.event(self.event({"role": "assistant"}), 11)
        capture.event(self.event({}, "length", self.usage()), 12)
        capture.event("[DONE]", 13)
        result = capture.result()
        self.assertIsNone(result["ttft_visible_ms"])
        self.assertIsNone(result["tpot_client_ms"])
        self.assertEqual(result["visible_null_reason"], "no_nonempty_delta")
        self.assertEqual(result["latency_ms"], 3000)

    def test_cached_details_are_not_silently_dropped(self):
        capture = StreamCapture(3, 4, 10)
        usage = {**self.usage(), "prompt_tokens_details": {"cached_tokens": 0}}
        capture.event(self.event({}, "length", usage), 11)
        capture.event("[DONE]", 12)
        self.assertEqual(capture.usage["prompt_tokens_details"]["cached_tokens"], 0)
        self.assertEqual(capture.result()["completion_tokens"], 4)

    def test_usage_and_termination_fail_closed(self):
        for event in ("[DONE]", self.event({}, "stop", self.usage()),
                      self.event({}, "length", {**self.usage(), "completion_tokens": 3})):
            capture = StreamCapture(3, 4, 0)
            with self.subTest(event=event), self.assertRaises(BenchError):
                capture.event(event, 1)
                capture.event("[DONE]", 2)
        capture = StreamCapture(3, 4, 0)
        capture.event(self.event({}, "length", self.usage()), 1)
        with self.assertRaisesRegex(BenchError, "without terminal"):
            capture.result()

    def test_truncated_utf8_and_sse_rejected(self):
        with self.assertRaises(UnicodeDecodeError):
            SSEParser().feed(b"data: \xf0", final=True)
        with self.assertRaises(BenchError):
            SSEParser().feed(b"data: {}", final=True)

    def test_multiline_comments_and_event_metadata(self):
        parser = SSEParser()
        self.assertEqual(parser.feed(b": heartbeat\nevent: message\ndata: first\ndata: second\n\n"), ["first\nsecond"])

    def test_data_after_terminal_rejected(self):
        capture = StreamCapture(3, 4, 0)
        capture.event(self.event({}, "length", self.usage()), 1)
        capture.event("[DONE]", 2)
        with self.assertRaises(BenchError):
            capture.event(self.event({"content": "late"}), 3)


class MetricTests(unittest.TestCase):
    def row(self):
        return {"prompt_tokens": 3, "completion_tokens": 4}

    def test_llama_cold_cache_and_native_counts_required(self):
        capture = StreamCapture(3, 4, 0)
        capture.usage = {"prompt_tokens_details": {"cached_tokens": 0}}
        capture.timings = {"prompt_n": 3, "predicted_n": 4, "prompt_ms": 10, "predicted_ms": 15}
        row = self.row()
        bench.verify_metrics(row, capture, "llama", None, "id", 0)
        self.assertEqual(row["reused_tokens"], 0)
        self.assertEqual(row["prefilled_tokens"], 3)
        for field, value in (("prompt_n", 2), ("predicted_n", 3)):
            bad = copy.deepcopy(capture)
            bad.timings[field] = value
            with self.assertRaises(BenchError):
                bench.verify_metrics(self.row(), bad, "llama", None, "id", 0)
        capture.usage = {}
        with self.assertRaises(BenchError):
            bench.verify_metrics(self.row(), capture, "llama", None, "id", 0)

    def test_fns_exact_output_hash_and_disabled_counters(self):
        log = {"request_id": "id", "reason": "length", "prompt_tokens": 3, "completion_tokens": 4,
               "reused_prefix": 0, "prefill_ms": 5, "decode_ms": 6, "spec_windows": 0,
               "spec_drafted": 0, "spec_accepted": 0, "spec_emitted": 0, "decode_iterations": 3,
               "token_ids": [1, 2, 3, 4]}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "metrics.jsonl"
            path.write_text(json.dumps(log) + "\n", encoding="utf-8")
            row = self.row()
            bench.verify_metrics(row, None, "fns", path, "id", 0)
            self.assertEqual(row["output_tokens_sha256"], bench.content_hash(log["token_ids"]))
            for key, value in (("spec_accepted", 1), ("reused_prefix", 1), ("token_ids", [1, 2])):
                path.write_text(json.dumps({**log, key: value}) + "\n", encoding="utf-8")
                with self.subTest(key=key), self.assertRaises(BenchError):
                    bench.verify_metrics(self.row(), None, "fns", path, "id", 0)


class ConfigurationTests(unittest.TestCase):
    def test_explicit_calibration_and_no_hidden_bit_changes(self):
        argv = ["--n-gpu-layers", "99", "--flash-attn", "on", "--cache-type-k", "f16", "--cache-type-v", "f16"]
        bench.validate_extra_args("llama", argv)
        for bad in (argv + ["--cache-type-k", "q8_0"], ["--n-gpu-layers", "99"], argv + ["--unknown", "yes"]):
            with self.assertRaises(BenchError):
                bench.validate_extra_args("llama", bad)
        with self.assertRaises(BenchError):
            bench.validate_extra_args("fns", ["--fa-int8"])

    def test_unresolved_environment_path_fails_instead_of_defaulting(self):
        with patch.dict("os.environ", {}, clear=True):
            with self.assertRaises(KeyError):
                bench.path_value("${SPEED_MISSING_ROOT}/model.gguf", Path.cwd())


class GPUProcessPolicyTests(unittest.TestCase):
    desktop_path = r"C:\Windows\Desktop Fixture\shell.exe"

    def config(self, mode="wddm_desktop"):
        return {"gpu_process_policy": {"mode": mode, "allowed_graphics_paths": [self.desktop_path] if mode == "wddm_desktop" else []},
                "environment_constraints": {"min_free_vram_mib": 20000, "max_gpu_temperature_c": 80,
                                            "max_idle_gpu_utilization_percent": 5}, "machine": {}}

    def inspect(self, cfg, records=None, apps=None, driver="WDDM", fields=None, xml=None):
        records = [(1234, self.desktop_path, "C+G")] if records is None else records
        gpu_fields = ["NVIDIA GeForce RTX 3090", "580.88", "22000", "24576", "40", "0", "P8", "350", "1", "16", driver, "Enabled"]
        for key, value in (fields or {}).items():
            gpu_fields[key] = value
        root = bench.ET.Element("nvidia_smi_log")
        bench.ET.SubElement(root, "driver_version").text = "580.88"
        gpu = bench.ET.SubElement(root, "gpu")
        bench.ET.SubElement(gpu, "product_name").text = gpu_fields[0]
        bench.ET.SubElement(bench.ET.SubElement(gpu, "driver_model"), "current_dm").text = driver
        processes = bench.ET.SubElement(gpu, "processes")
        for pid, path, kind in records:
            node = bench.ET.SubElement(processes, "process_info")
            for key, value in (("pid", str(pid)), ("process_name", path), ("type", kind)):
                if value is not None:
                    bench.ET.SubElement(node, key).text = value
        if apps is None:
            apps = "\n".join(f"{pid}, {path}" for pid, path, kind in records if kind in ("C", "C+G"))
        def command(argv):
            if "--query-gpu=" in argv[1]:
                return ", ".join(gpu_fields)
            if "--query-compute-apps=" in argv[1]:
                return apps
            if argv[1:] == ["-q", "-x"]:
                return xml if xml is not None else bench.ET.tostring(root, encoding="unicode")
            if argv[0] == "powershell.exe":
                return '{"ram_bytes":68719476736}'
            if argv[0] == "powercfg":
                return "fixture"
            raise AssertionError(argv)
        with patch.object(bench.shutil, "which", return_value="nvidia-smi"), \
             patch.object(bench, "command_output", side_effect=command):
            return bench.inspect_machine(cfg)

    def test_exactly_listed_graphical_contexts_are_recorded_privately(self):
        cfg = self.config()
        cfg["gpu_process_policy"]["allowed_graphics_paths"] = [self.desktop_path.upper().replace("\\", "/")]
        observed = self.inspect(cfg, [(1234, self.desktop_path, "C+G"), (4321, self.desktop_path, "G")])
        self.assertEqual(observed["gpu_process_policy"], cfg["gpu_process_policy"])
        self.assertEqual(observed["permitted_graphics_processes"], [
            {"pid": 1234, "path": self.desktop_path, "type": "C+G"},
            {"pid": 4321, "path": self.desktop_path, "type": "G"}])

    def test_attempt_records_configured_policy_before_preflight(self):
        policy = self.config()["gpu_process_policy"]
        manifest = {"run_id": "fixture", "quant": "q3", "comparison": "service", "mode": "greedy",
                    "effective_options": {}, "gpu_process_policy": policy}
        request = {"request_id": "fixture", "config": "fixture", "category": "fixture"}
        row = bench.empty_attempt(manifest, request, 0, 0, "A", 0, None)
        self.assertEqual(row["gpu_process_policy"], policy)

    def test_unlisted_compute_unknown_and_missing_records_fail_closed(self):
        for records in ([(1234, r"C:\Other\shell.exe", "C+G")], [(1234, self.desktop_path, "C")],
                        [(1234, self.desktop_path, "unknown")], [(1234, None, "G")],
                        [(1234, self.desktop_path, None)], [(1234, "shell.exe", "G")],
                        [(1234, self.desktop_path, "G"), (1234, self.desktop_path, "G")]):
            with self.subTest(records=records), self.assertRaises(BenchError):
                self.inspect(self.config(), records)
        for xml in ("<unknown/>", "not xml", "<nvidia_smi_log><gpu/></nvidia_smi_log>",
                    '<nvidia_smi_log><driver_version>580.88</driver_version><gpu><product_name>NVIDIA GeForce RTX 3090</product_name>'
                    '<driver_model><current_dm>WDDM</current_dm></driver_model><processes><unexpected/></processes></gpu></nvidia_smi_log>'):
            with self.subTest(xml=xml), self.assertRaises(BenchError):
                self.inspect(self.config(), xml=xml)

    def test_compute_query_must_match_typed_pid_path_and_membership(self):
        for apps in ("1235, " + self.desktop_path, "1234, C:\\Other\\shell.exe", "1234, shell.exe", "", "N/A"):
            with self.subTest(apps=apps), self.assertRaises(BenchError):
                self.inspect(self.config(), apps=apps)

    def test_desktop_policy_rejects_non_wddm_and_preserves_all_thresholds(self):
        for driver in ("TCC", "N/A"):
            with self.subTest(driver=driver), self.assertRaises(BenchError):
                self.inspect(self.config(), driver=driver)
        for fields in ({2: "19999"}, {4: "81"}, {5: "6"}):
            with self.subTest(fields=fields), self.assertRaises(BenchError):
                self.inspect(self.config(), fields=fields)

    def test_legacy_and_explicit_strict_keep_original_compute_rejection(self):
        for cfg in (self.config("strict"), {key: value for key, value in self.config().items() if key != "gpu_process_policy"}):
            with self.subTest(cfg=cfg):
                with self.assertRaisesRegex(BenchError, "Other GPU compute"):
                    self.inspect(cfg)
                observed = self.inspect(cfg, records=[], apps="")
                self.assertEqual(observed["gpu_process_policy"]["mode"], "strict")
                self.assertEqual(observed["permitted_graphics_processes"], [])

    def test_policy_validation_precedes_artifacts_and_launch(self):
        bad = [{"mode": "unknown"}, {"mode": "strict", "allowed_graphics_paths": [self.desktop_path]},
               {"mode": "wddm_desktop"}, {"mode": "wddm_desktop", "allowed_graphics_paths": "shell.exe"}]
        for path in ("shell.exe", "C:relative.exe", r"C:\Windows\..\shell.exe", r"C:\Windows\*.exe",
                     r"C:\${ROOT}\shell.exe", r"C:\%ROOT%\shell.exe", " C:/Windows/shell.exe"):
            bad.append({"mode": "wddm_desktop", "allowed_graphics_paths": [path]})
        bad.append({"mode": "wddm_desktop", "allowed_graphics_paths": [self.desktop_path, self.desktop_path.upper()]})
        for policy in bad:
            with self.subTest(policy=policy), patch.object(bench, "load_json", return_value={"schema_version": 1, "gpu_process_policy": policy}), \
                 patch.object(bench, "digest", return_value="a" * 64), patch.object(bench, "artifact") as artifact, \
                 self.assertRaises(BenchError):
                bench.validate_campaign(Path("fixture.json"))
            artifact.assert_not_called()


class GPUTelemetryTests(unittest.TestCase):
    def test_only_active_measured_samples_contribute_ranges(self):
        from speed_runtime import GPUStateSampler
        sampler = GPUStateSampler("unused", "unused")
        sampler.samples = [
            {"monotonic": 9, "values": ["1", "23000", "1", "1", "30", "99", "P0", "100"]},
            {"monotonic": 11, "values": ["18000", "6000", "4", "16", "70", "90", "P0", "300"]},
            {"monotonic": 12, "values": ["18000", "6000", "1", "8", "65", "0", "P8", "50"]},
            {"monotonic": 13, "values": ["bad"]},
            {"monotonic": 15, "values": ["1", "23000", "1", "1", "30", "99", "P0", "100"]},
        ]
        result = sampler.summary(10, 14)
        self.assertEqual((result["samples"], result["valid_samples"], result["active_samples"]), (3, 2, 1))
        self.assertEqual(result["pcie_gen_min"], 4)
        self.assertEqual(result["pcie_width_min"], 16)
        self.assertEqual(result["power_draw_watts_max"], 300)
        self.assertEqual(result["memory_used_mib_max"], 18000)
        missing = sampler.summary(20, 21)
        self.assertIsNone(missing["pcie_gen_min"])
        self.assertEqual(missing["null_reason"], "no_active_sample_in_measured_window")


class PairLifecycleTests(unittest.TestCase):
    def config(self, tmp):
        return {"output_dir": str(Path(tmp) / "runs"), "campaign_path": "fixture", "campaign_sha256": "a" * 64,
                "prepared_dir": str(Path(tmp) / "prepared"), "source_lock": {"path": "fixture/source-lock.json"},
                "sources": {}, "pairs": 1, "requests": [{"request_id": "b" * 64, "config": "fixture", "category": "fixture"}]}

    def manifest(self, directory):
        return {"run_id": directory.name, "quant": "q3", "comparison": "service", "mode": "greedy", "seeds": [0],
                "campaign_sha256": "a" * 64, "arms": {"A": "llama", "B": "fns"}}

    def test_failed_pair_preserved_and_explicit_resume_links_both_arms(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict("os.environ", {"TEMP": tmp}):
            cfg = self.config(tmp)
            def create(cfg, directory, *args):
                return self.manifest(directory)
            def measure(cfg, manifest, request, seed, pair, arm, attempt, replaces, *args):
                return {"request_id": request["request_id"], "seed": seed, "pair": pair, "arm": arm,
                        "attempt": attempt, "replaces": replaces, "status": "error" if arm == "B" and attempt == 0 else "ok",
                        "error_code": "usage" if arm == "B" and attempt == 0 else None}
            with patch.object(bench, "create_manifest", side_effect=create), \
                 patch.object(bench, "verify_local_inputs", return_value=3), \
                 patch.object(bench, "verify_llama_inputs"), \
                 patch.object(bench, "measure_arm", side_effect=measure), \
                 patch.object(bench.shutil, "copyfile"):
                directory, ok = bench.run_campaign(cfg, "q3", "service", "greedy")
                self.assertFalse(ok)
                initial = bench.jsonl(directory / "attempts.jsonl")
                self.assertEqual([r["status"] for r in initial], ["ok", "error"])
                _, ok = bench.run_campaign(cfg, "q3", "service", "greedy", directory)
                self.assertTrue(ok)
                history = bench.jsonl(directory / "attempts.jsonl")
                self.assertEqual(len(history), 4)
                self.assertEqual([r["replaces"] for r in history], [None, None, 0, 0])
                self.assertEqual([r["pair_generation"] for r in history], [0, 0, 1, 1])
                bench.run_campaign(cfg, "q3", "service", "greedy", directory)
                self.assertEqual(len(bench.jsonl(directory / "attempts.jsonl")), 4)

    def test_owned_shutdown_failure_keeps_attempt_and_stops_campaign(self):
        cfg = {"models": {}, "timeouts": {"request": 5}}
        manifest = {"run_id": "fixture", "quant": "q3", "comparison": "service", "mode": "greedy",
                    "arms": {"A": "llama"}, "effective_options": {}}
        request = {"request_id": "fixture", "config": "fixture", "category": "fixture"}
        class FailingServer:
            argv = ["fixture"]
            startup_ms = None
            shutdown_ms = 2
            def start(self):
                raise BenchError("startup", "fixture startup failure")
            def stop(self):
                raise BenchError("process", "fixture native child remains")
        with patch.object(bench, "make_server", return_value=(FailingServer(), "metrics")), \
             patch.object(bench, "inspect_machine", return_value={}):
            row = bench.measure_arm(cfg, manifest, request, 0, 0, "A", 0, None, Path("fixture"), 3)
        self.assertEqual(row["status"], "error")
        self.assertEqual(row["error_code"], "process")
        self.assertEqual(row["shutdown_ms"], 2)
        self.assertIsNone(row["latency_ms"])


if __name__ == "__main__":
    unittest.main()
