"""Authored SPEED fixtures: no corpus, dataset package, server binary or network access."""
import contextlib
import copy
import json
import io
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bench" / "scripts"))
import speed_bench as bench
from speed_runtime import BenchError, SSEParser, StreamCapture
import speed_runtime as runtime


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

    def test_connection_closes_when_capture_evidence_write_fails(self):
        class Connection:
            sock = SimpleNamespace(settimeout=lambda timeout: None)
            closed = False
            def connect(self): pass
            def putrequest(self, *args): pass
            def putheader(self, *args): pass
            def endheaders(self): pass
            def send(self, payload): pass
            def getresponse(self):
                return SimpleNamespace(status=500, read=lambda: b"error")
            def close(self):
                self.closed = True
        conn = Connection()
        with tempfile.TemporaryDirectory() as tmp, \
             patch.object(runtime.http.client, "HTTPConnection", return_value=conn), \
             patch.object(Path, "write_bytes", side_effect=OSError("disk full")), \
             self.assertRaises(OSError):
            runtime.stream_request("http://127.0.0.1:1", {}, "id", 1, 1, 1, Path(tmp) / "raw")
        self.assertTrue(conn.closed)


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

    def inspect(self, cfg, records=None, apps=None, driver="WDDM", fields=None, xml=None, owned_pids=None):
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
            return bench.inspect_machine(cfg) if owned_pids is None else bench.inspect_live_gpu(cfg, owned_pids)

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

    def test_live_monitor_allows_only_owned_native_tree_without_idle_limits(self):
        engine = r"C:\Fixture\fns-serve.exe"
        cfg = self.config()
        observed = self.inspect(cfg, records=[(1234, self.desktop_path, "C+G"), (5678, engine, "C")],
                                fields={2: "500", 5: "100"}, owned_pids={5678})
        self.assertEqual([r["pid"] for r in observed["gpu_processes"]], [1234, 5678])
        with self.assertRaises(BenchError):
            self.inspect(cfg, records=[(5678, engine, "C")], owned_pids={9999})
        with self.assertRaises(BenchError):
            self.inspect(cfg, records=[(5678, engine, "C")], fields={4: "81"}, owned_pids={5678})

    def test_live_strict_allows_graphics_and_owned_compute_but_not_other_compute(self):
        engine = r"C:\Fixture\fns-serve.exe"
        graphics = r"C:\Fixture\desktop.exe"
        cfg = self.config("strict")
        records = [(1234, graphics, "G"), (5678, engine, "C")]
        for driver in ("WDDM", "TCC"):
            observed = self.inspect(cfg, records=records, driver=driver, owned_pids={5678})
            self.assertEqual([r["pid"] for r in observed["gpu_processes"]], [1234, 5678])
            with self.assertRaises(BenchError):
                self.inspect(cfg, records=[(1234, graphics, "C+G"), (5678, engine, "C")],
                             driver=driver, owned_pids={5678})
        with self.assertRaises(BenchError):
            self.inspect(cfg, records=[], owned_pids=set())

    def test_policy_validation_precedes_artifacts_and_launch(self):
        bad = [{"mode": "unknown"}, {"mode": "strict", "allowed_graphics_paths": [self.desktop_path]},
               {"mode": "wddm_desktop"}, {"mode": "wddm_desktop", "allowed_graphics_paths": "shell.exe"}]
        for path in ("shell.exe", "C:relative.exe", r"C:\Windows\..\shell.exe", r"C:\Windows\*.exe",
                     r"C:\${ROOT}\shell.exe", r"C:\%ROOT%\shell.exe", " C:/Windows/shell.exe"):
            bad.append({"mode": "wddm_desktop", "allowed_graphics_paths": [path]})
        bad.append({"mode": "wddm_desktop", "allowed_graphics_paths": [self.desktop_path, self.desktop_path.upper()]})
        for policy in bad:
            with self.subTest(policy=policy), patch.object(bench, "load_json", return_value={"schema_version": 2, "gpu_process_policy": policy}), \
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

    def test_live_process_monitor_records_initial_evidence_and_stops_once(self):
        class Process:
            stdout = io.StringIO("18000,6000,4,16,70,90,P0,300\n")
            stopped = False
            def poll(self):
                return 0 if self.stopped else None
            def terminate(self):
                self.stopped = True
            def wait(self, timeout):
                return 0
        with tempfile.TemporaryDirectory() as tmp:
            process = Process()
            with patch.object(runtime.subprocess, "Popen", return_value=process):
                sampler = runtime.GPUStateSampler("unused", Path(tmp) / "gpu.jsonl",
                                                  monitor=lambda: {"owned_pids": [123]}, max_temperature=80)
                try:
                    sampler.start()
                    sampler.check()
                    self.assertEqual(sampler.monitor_samples[0]["result"], {"owned_pids": [123]})
                finally:
                    sampler.stop()
            self.assertTrue(process.stopped)
            self.assertFalse(sampler.monitor_thread.is_alive())

    def test_monitor_failure_stops_owned_server_telemetry_and_surfaces_error(self):
        class Process:
            stdout = io.StringIO("")
            stopped = False
            def poll(self):
                return 0 if self.stopped else None
            def terminate(self):
                self.stopped = True
            def wait(self, timeout):
                return 0
        def fail():
            raise BenchError("environment", "unapproved GPU process")
        with tempfile.TemporaryDirectory() as tmp:
            process = Process()
            with patch.object(runtime.subprocess, "Popen", return_value=process):
                sampler = runtime.GPUStateSampler("unused", Path(tmp) / "gpu.jsonl", monitor=fail)
                try:
                    with self.assertRaisesRegex(BenchError, "unapproved GPU process"):
                        sampler.start()
                finally:
                    sampler.stop()
            self.assertTrue(process.stopped)


class OwnedServerCleanupTests(unittest.TestCase):
    def test_unassigned_suspended_process_is_killed_even_if_job_termination_fails(self):
        class Process:
            killed = False
            waited = False
            def poll(self):
                return None if not self.killed else 1
            def kill(self):
                self.killed = True
            def wait(self, timeout):
                self.waited = True
        process = Process()
        server = runtime.OwnedServer([], {}, ".", "http://127.0.0.1:1", "unused", 1, 1)
        server.proc, server.job = process, 1
        server.kernel = SimpleNamespace(TerminateJobObject=lambda *args: 0, CloseHandle=lambda job: None)
        with self.assertRaises(BenchError):
            server.stop()
        self.assertTrue(process.killed)
        self.assertTrue(process.waited)
        self.assertIsNone(server.job)

    def test_process_group_children_are_killed_after_launcher_exits(self):
        process = SimpleNamespace(pid=123, wait=lambda timeout: 0)
        server = runtime.OwnedServer([], {}, ".", "http://127.0.0.1:1", "unused", 1, 1)
        server.proc = process
        signals = []
        with patch.object(runtime.os, "killpg", side_effect=lambda pid, sig: signals.append(sig), create=True), \
             patch.object(runtime.signal, "SIGKILL", 9, create=True):
            server.stop()
        self.assertEqual(signals, [runtime.signal.SIGTERM, 9])


class GPULockCleanupTests(unittest.TestCase):
    def test_lock_released_if_owner_metadata_write_fails(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict("os.environ", {"TEMP": tmp}), \
             patch.object(bench.os, "write", side_effect=OSError("disk full")):
            with self.assertRaises(OSError):
                bench.acquire_gpu_lock({"campaign_path": "fixture"})
            self.assertFalse((Path(tmp) / "fns-speed-gpu.lock").exists())

    def test_check_releases_lock_if_result_write_fails(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict("os.environ", {"TEMP": tmp}), \
             patch.object(bench, "verify_local_inputs", return_value=1), \
             patch.object(bench, "verify_llama_inputs"), \
             patch.object(bench, "save_json", side_effect=OSError("disk full")):
            with self.assertRaises(OSError):
                bench.check_campaign({"output_dir": tmp, "campaign_path": "fixture", "campaign_sha256": "a", "requests": []})
            self.assertFalse((Path(tmp) / "fns-speed-gpu.lock").exists())


class BlockLifecycleTests(unittest.TestCase):
    def config(self, tmp, count=3, pairs=1):
        return {"output_dir": str(Path(tmp) / "runs"), "campaign_path": "fixture", "campaign_sha256": "a" * 64,
                "prepared_dir": str(Path(tmp) / "prepared"), "source_lock": {"path": "fixture/source-lock.json"},
                "sources": {}, "pairs": pairs, "max_tokens": 4, "sampling": copy.deepcopy(bench.MODES),
                "timeouts": {"request": 5}, "environment_constraints": {"max_gpu_temperature_c": 80},
                "warmup": {"messages": [{"role": "user", "content": "warmup"}], "max_tokens": 2},
                "execution_protocol": {"sha256": "e" * 64}, "execution_protocol_data": bench.EXECUTION_PROTOCOL,
                "requests": [{"request_id": f"{i:064x}", "config": "fixture", "category": "fixture",
                              "prompt_tokens": 3, "messages": [{"role": "user", "content": f"request-{i}"}]}
                             for i in range(count)]}

    @contextlib.contextmanager
    def harness(self, cfg, fail_request=None, fail_start=False,
                fail_shutdown=False, fail_monitor=False, mismatch=False):
        state = SimpleNamespace(servers=[], samplers=[], streams=[], prefixes=[], waits=0, fail_request=fail_request,
                                fail_start=fail_start, fail_shutdown=fail_shutdown, fail_monitor=fail_monitor, mismatch=mismatch)
        def create(cfg, directory, quant, comparison, mode):
            return {"schema_version": 2, "run_id": directory.name, "quant": quant, "comparison": comparison,
                    "mode": mode, "seeds": [0], "campaign_sha256": cfg["campaign_sha256"],
                    "execution_protocol": cfg["execution_protocol_data"], "execution_protocol_sha256": "e" * 64,
                    "effective_options": {}, "arms": {"A": "llama" if comparison == "service" else "fns", "B": "fns"}}
        def make(cfg, quant, engine, spec, sample, prefix):
            server = SimpleNamespace(argv=[engine], url="fixture", started=0, stopped=0, spec=spec)
            def start():
                server.started += 1
                if state.fail_start:
                    raise BenchError("startup", "fixture startup failure")
            def stop():
                server.stopped += 1
                if state.fail_shutdown:
                    raise BenchError("process", "fixture native child remains")
            server.start, server.stop = start, stop
            server.owned_pids = lambda: {123}
            state.servers.append(server)
            state.prefixes.append(str(prefix))
            return server, "metrics"
        def sampler(*args, **kwargs):
            instance = SimpleNamespace(error=None, monitor_samples=[], started=0, stopped=0, windows=[])
            def start():
                instance.started += 1
            def stop():
                instance.stopped += 1
                if state.fail_monitor:
                    raise BenchError("environment", "fixture monitor stop failure")
            def summary(started, done):
                instance.windows.append((started, done))
                return {"samples": 1}
            instance.start, instance.stop, instance.summary = start, stop, summary
            instance.check = lambda: None
            state.samplers.append(instance)
            return instance
        def stream(url, body, correlation, prompt_tokens, max_tokens, timeout, raw_path):
            state.streams.append((body, correlation, raw_path))
            text = body["messages"][0]["content"]
            if text == state.fail_request:
                raise BenchError("usage", "fixture request failure")
            return ({"prompt_tokens": prompt_tokens, "completion_tokens": max_tokens, "latency_ms": 10},
                    SimpleNamespace(started=10, done=20))
        def metrics(row, capture, engine, path, correlation, spec):
            row.update(reused_tokens=0, prefilled_tokens=row["prompt_tokens"],
                       output_tokens_sha256="different" if state.mismatch and spec else "same")
        def wait(cfg):
            state.waits += 1
            return {"fixture": "idle"}
        with contextlib.ExitStack() as stack:
            for name, callback in (("create_manifest", create), ("make_server", make),
                                   ("GPUStateSampler", sampler), ("stream_request", stream),
                                   ("verify_metrics", metrics), ("wait_for_idle", wait)):
                stack.enter_context(patch.object(bench, name, side_effect=callback))
            stack.enter_context(patch.object(bench, "verify_local_inputs", return_value=3))
            stack.enter_context(patch.object(bench, "verify_llama_inputs"))
            stack.enter_context(patch.object(bench.shutil, "copyfile"))
            stack.enter_context(patch.object(bench.shutil, "which", return_value="nvidia-smi"))
            yield state

    def test_one_server_warmup_sampler_per_arm_and_identical_request_order(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict("os.environ", {"TEMP": tmp}):
            cfg = self.config(tmp, pairs=2)
            with self.harness(cfg) as state:
                directory, ok = bench.run_campaign(cfg, "q3", "service", "greedy")
            self.assertTrue(ok)
            sessions = bench.jsonl(directory / "sessions.jsonl")
            rows = bench.jsonl(directory / "attempts.jsonl")
            self.assertEqual([s["arm"] for s in sessions], ["A", "B", "B", "A"])
            self.assertEqual(len(state.servers), 4)
            self.assertEqual(state.waits, 4)
            self.assertTrue(all(s.started == s.stopped == 1 for s in state.servers + state.samplers))
            self.assertEqual(len(state.streams), 16)
            self.assertEqual(sum(x[0]["messages"] == cfg["warmup"]["messages"] for x in state.streams), 4)
            self.assertTrue(all(not body["cache_prompt"] for body, _, _ in state.streams))
            self.assertEqual(len({path for _, _, path in state.streams}), 16)
            for session in sessions:
                members = [r for r in rows if r["session_id"] == session["session_id"]]
                self.assertEqual([r["request_id"] for r in members], session["request_ids"])
                self.assertTrue(all(r["gpu_during_request"] == {"samples": 1} for r in members))
                for key in ("startup_ms", "warmup_ms", "shutdown_ms"):
                    self.assertGreaterEqual(session[key], 0)
                    self.assertTrue(all(key not in r for r in members))
            self.assertEqual([r["observed_order"] for r in rows], list(range(12)))
            self.assertTrue(all(s.windows == [(10, 20)] * 3 for s in state.samplers))

    def test_failed_block_resume_repeats_successes_and_links_history(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict("os.environ", {"TEMP": tmp}):
            cfg = self.config(tmp)
            with self.harness(cfg, fail_request="request-1") as state:
                directory, ok = bench.run_campaign(cfg, "q3", "service", "greedy")
                self.assertFalse(ok)
                first = bench.jsonl(directory / "attempts.jsonl")
                self.assertEqual([r["status"] for r in first], ["ok", "error"])
                self.assertEqual(len(state.servers), 1)
                state.fail_request = None
                _, ok = bench.run_campaign(cfg, "q3", "service", "greedy", directory)
                self.assertTrue(ok)
                rows = bench.jsonl(directory / "attempts.jsonl")
                self.assertEqual([r["block_generation"] for r in rows], [0, 0] + [1] * 6)
                self.assertEqual([r["replaces"] for r in rows[2:]], [0, 0, None, None, None, None])
                self.assertEqual([r["pair_generation"] for r in rows], [r["block_generation"] for r in rows])
                before = len(state.servers)
                bench.run_campaign(cfg, "q3", "service", "greedy", directory)
                self.assertEqual(len(state.servers), before)
                self.assertEqual(len(bench.jsonl(directory / "attempts.jsonl")), 8)
                self.assertEqual(len(set(state.prefixes)), len(state.prefixes))

    def test_resume_refuses_missing_or_incomplete_successful_session_evidence(self):
        for damage in ("one_row", "all_rows", "missing_file", "failed_row", "wrong_order"):
            with self.subTest(damage=damage), tempfile.TemporaryDirectory() as tmp, patch.dict("os.environ", {"TEMP": tmp}):
                cfg = self.config(tmp)
                with self.harness(cfg) as state:
                    directory, ok = bench.run_campaign(cfg, "q3", "service", "greedy")
                    self.assertTrue(ok)
                    path = directory / "attempts.jsonl"
                    records = bench.jsonl(path)
                    if damage == "one_row":
                        records.pop(0)
                    elif damage == "all_rows":
                        records = records[3:]
                    elif damage == "missing_file":
                        path.unlink()
                    elif damage == "failed_row":
                        records[0]["status"] = "error"
                    else:
                        records[0], records[1] = records[1], records[0]
                    if damage != "missing_file":
                        path.write_bytes(b"".join(bench.canonical(r) + b"\n" for r in records))
                    before_attempts = path.read_bytes() if path.exists() else None
                    before_sessions = (directory / "sessions.jsonl").read_bytes()
                    before_servers = len(state.servers)
                    with self.assertRaisesRegex(BenchError, "successful session lacks complete successful request evidence"):
                        bench.run_campaign(cfg, "q3", "service", "greedy", directory)
                    self.assertEqual(len(state.servers), before_servers)
                    self.assertEqual(path.read_bytes() if path.exists() else None, before_attempts)
                    self.assertEqual((directory / "sessions.jsonl").read_bytes(), before_sessions)

    def test_missing_session_can_be_resumed(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict("os.environ", {"TEMP": tmp}):
            cfg = self.config(tmp)
            with self.harness(cfg):
                directory, _ = bench.run_campaign(cfg, "q3", "service", "greedy")
                path = directory / "sessions.jsonl"
                records = bench.jsonl(path)
                path.write_bytes(b"".join(bench.canonical(r) + b"\n" for r in records[:-1]))
                _, ok = bench.run_campaign(cfg, "q3", "service", "greedy", directory)
            self.assertTrue(ok)
            self.assertEqual([r["block_generation"] for r in bench.jsonl(directory / "attempts.jsonl")[-6:]], [1] * 6)

    def test_session_only_generation_is_resumed_without_reusing_raw_paths(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict("os.environ", {"TEMP": tmp}):
            cfg = self.config(tmp)
            with self.harness(cfg):
                directory, _ = bench.run_campaign(cfg, "q3", "service", "greedy")
                sessions = bench.jsonl(directory / "sessions.jsonl")
                orphan = {**sessions[0], "session_id": "orphan", "block_generation": 1, "status": "error",
                          "error_code": "interrupted", "error": "Interrupted before first row"}
                bench.append_attempts(directory / "sessions.jsonl", [orphan])
                _, ok = bench.run_campaign(cfg, "q3", "service", "greedy", directory)
            self.assertTrue(ok)
            self.assertEqual([r["block_generation"] for r in bench.jsonl(directory / "attempts.jsonl")[-6:]], [2] * 6)

    def test_interrupt_stops_after_current_request_and_finalizes_session(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict("os.environ", {"TEMP": tmp}):
            cfg = self.config(tmp)
            with self.harness(cfg) as state, patch.object(bench, "stream_request", side_effect=KeyboardInterrupt):
                directory, ok = bench.run_campaign(cfg, "q3", "service", "greedy")
            self.assertFalse(ok)
            self.assertEqual(len(state.servers), 1)
            self.assertEqual(state.servers[0].stopped, 1)
            self.assertEqual(bench.jsonl(directory / "sessions.jsonl")[0]["error_code"], "interrupted")
            self.assertEqual(bench.jsonl(directory / "attempts.jsonl")[0]["error_code"], "interrupted")

    def test_spec_parity_invalidates_linked_rows_then_resume_whole_block(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict("os.environ", {"TEMP": tmp}):
            cfg = self.config(tmp)
            with self.harness(cfg, mismatch=True) as state:
                directory, ok = bench.run_campaign(cfg, "q3", "speculation", "greedy")
                self.assertFalse(ok)
                rows = bench.jsonl(directory / "attempts.jsonl")
                self.assertEqual(len(rows), 12)
                self.assertTrue(all(r["status"] == "ok" for r in rows[:6]))
                self.assertTrue(all(r["error_code"] == "spec_parity" and r["replaces"] == 0 for r in rows[6:]))
                self.assertTrue(all(s["status"] == "ok" for s in bench.jsonl(directory / "sessions.jsonl")))
                state.mismatch = False
                _, ok = bench.run_campaign(cfg, "q3", "speculation", "greedy", directory)
                self.assertTrue(ok)
                self.assertTrue(all(r["attempt"] == 2 and r["block_generation"] == 1
                                    for r in bench.jsonl(directory / "attempts.jsonl")[-6:]))
                path = directory / "attempts.jsonl"
                rows = bench.jsonl(path)
                path.write_bytes(b"".join(bench.canonical(row) + b"\n" for row in rows[1:]))
                before_servers = len(state.servers)
                with self.assertRaisesRegex(BenchError, "successful session lacks complete successful request evidence"):
                    bench.run_campaign(cfg, "q3", "speculation", "greedy", directory)
                self.assertEqual(len(state.servers), before_servers)

    def test_start_shutdown_and_monitor_errors_finalize_and_stop(self):
        for failure, code in (("fail_start", "startup"), ("fail_shutdown", "process"), ("fail_monitor", "environment")):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as tmp, patch.dict("os.environ", {"TEMP": tmp}):
                cfg = self.config(tmp)
                with self.harness(cfg, fail_start=failure == "fail_start", fail_shutdown=failure == "fail_shutdown",
                                  fail_monitor=failure == "fail_monitor") as state:
                    directory, ok = bench.run_campaign(cfg, "q3", "service", "greedy")
                self.assertFalse(ok)
                self.assertEqual(len(state.servers), 1)
                self.assertEqual(state.servers[0].stopped, 1)
                session = bench.jsonl(directory / "sessions.jsonl")[0]
                self.assertEqual((session["status"], session["error_code"]), ("error", code))
                self.assertTrue(all(isinstance(session[k], (int, float)) and session[k] >= 0
                                    for k in ("startup_ms", "warmup_ms", "shutdown_ms")))
                self.assertTrue(bench.jsonl(directory / "attempts.jsonl"))

    def test_preflight_error_retains_failed_session_and_first_row(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict("os.environ", {"TEMP": tmp}):
            cfg = self.config(tmp)
            with self.harness(cfg) as state, patch.object(bench, "wait_for_idle", side_effect=BenchError("environment", "busy")):
                directory, ok = bench.run_campaign(cfg, "q3", "service", "greedy")
            self.assertFalse(ok)
            self.assertEqual(state.servers, [])
            session = bench.jsonl(directory / "sessions.jsonl")[0]
            self.assertEqual(session["environment_before"], {})
            self.assertEqual(session["request_ids"], [r["request_id"] for r in cfg["requests"]])
            self.assertEqual(session["error_code"], "environment")
            self.assertEqual(len(bench.jsonl(directory / "attempts.jsonl")), 1)

    def test_schema_one_resume_rejected_before_preflight(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict("os.environ", {"TEMP": tmp}):
            cfg = self.config(tmp)
            directory = Path(tmp) / "old"
            directory.mkdir()
            bench.save_json(directory / "run-manifest.json", {"schema_version": 1})
            with patch.object(bench, "verify_local_inputs") as preflight, self.assertRaises(BenchError):
                bench.run_campaign(cfg, "q3", "service", "greedy", directory)
            preflight.assert_not_called()


class BlockOrderingTests(unittest.TestCase):
    def test_groups_in_first_occurrence_order_and_slices_at_thirty_two(self):
        def request(i, category):
            return {"request_id": str(i), "config": "fixture", "category": category}
        requests = [request(0, "b"), request(1, "a")] + [request(i, "b") for i in range(2, 35)]
        blocks = bench.deterministic_blocks(requests)
        self.assertEqual([len(b) for b in blocks], [32, 2, 1])
        self.assertEqual([[r["request_id"] for r in b] for b in blocks],
                         [[str(i) for i in [0] + list(range(2, 33))], ["33", "34"], ["1"]])

    def test_idle_requires_two_stable_seconds_and_cooldown_after_rejection(self):
        now = [0.0]
        waits = []
        def sleep(seconds):
            waits.append(seconds)
            now[0] += seconds
        calls = [0]
        def gpu(cfg):
            calls[0] += 1
            if calls[0] == 2:
                raise BenchError("environment", "busy")
        with patch.object(bench.time, "perf_counter", side_effect=lambda: now[0]), \
             patch.object(bench.time, "sleep", side_effect=sleep), patch.object(bench, "inspect_live_gpu", side_effect=gpu), \
             patch.object(bench, "inspect_machine", return_value={"idle": True}) as fingerprint:
            self.assertEqual(bench.wait_for_idle({}), {"idle": True})
        self.assertIn(10, waits)
        self.assertGreaterEqual(now[0], 12)
        fingerprint.assert_called_once()

    def test_idle_deadline_is_bounded_and_never_fingerprints_busy_machine(self):
        now = [0.0]
        def sleep(seconds):
            now[0] += seconds
        with patch.object(bench.time, "perf_counter", side_effect=lambda: now[0]), \
             patch.object(bench.time, "sleep", side_effect=sleep), \
             patch.object(bench, "inspect_live_gpu", side_effect=BenchError("environment", "busy")), \
             patch.object(bench, "inspect_machine") as fingerprint:
            with self.assertRaisesRegex(BenchError, "Idle/cooldown deadline"):
                bench.wait_for_idle({})
        self.assertEqual(now[0], 1800)
        fingerprint.assert_not_called()


if __name__ == "__main__":
    unittest.main()
