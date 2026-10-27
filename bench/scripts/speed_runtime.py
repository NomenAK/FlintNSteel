"""Private SPEED runtime: strict SSE capture and owned, serial server lifetimes.

Only standard-library modules are imported here; fixture tests never load a dataset.
"""
from __future__ import annotations

import codecs
import ctypes
import http.client
import json
import math
import os
import signal
import socket
import subprocess
import time
import threading
from pathlib import Path
from urllib.parse import urlsplit


class BenchError(RuntimeError):
    def __init__(self, code, message):
        super().__init__(message)
        self.code = code


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


class SSEParser:
    """Incremental UTF-8/SSE decoder, including split CRLF and multiline data."""
    def __init__(self):
        self.decoder = codecs.getincrementaldecoder("utf-8")("strict")
        self.buffer = ""
        self.data = []

    def feed(self, chunk, final=False):
        self.buffer += self.decoder.decode(chunk, final=final)
        events = []
        while True:
            positions = [p for p in (self.buffer.find("\n"), self.buffer.find("\r")) if p >= 0]
            if not positions:
                break
            end = min(positions)
            if self.buffer[end] == "\r" and end == len(self.buffer) - 1 and not final:
                break
            width = 2 if self.buffer[end:end + 2] == "\r\n" else 1
            line, self.buffer = self.buffer[:end], self.buffer[end + width:]
            if not line:
                if self.data:
                    events.append("\n".join(self.data))
                    self.data.clear()
            elif not line.startswith(":"):
                field, _, value = line.partition(":")
                if field == "data":
                    self.data.append(value[1:] if value.startswith(" ") else value)
        if final and (self.buffer or self.data):
            raise BenchError("stream", "Truncated SSE event")
        return events


class StreamCapture:
    def __init__(self, prompt_tokens, max_tokens, started):
        self.prompt_tokens, self.max_tokens, self.started = prompt_tokens, max_tokens, started
        self.visible = []
        self.usage = None
        self.finish = None
        self.done = None
        self.timings = None
        self.extensions = {}

    def event(self, text, now):
        if self.done is not None:
            raise BenchError("stream", "Event after terminal marker")
        if text == "[DONE]":
            if self.usage is None or self.finish != "length":
                raise BenchError("usage", "Terminal marker without fixed-budget finish and usage")
            self.done = now
            return
        try:
            obj = json.loads(text)
        except (ValueError, TypeError) as exc:
            raise BenchError("stream", "Invalid SSE JSON") from exc
        if not isinstance(obj, dict) or obj.get("error"):
            raise BenchError("stream", "Invalid or error SSE event")
        if isinstance(obj.get("timings"), dict):
            self.timings = obj["timings"]
        for key in ("tokens_cached", "tokens_evaluated", "fns"):
            if key in obj:
                self.extensions[key] = obj[key]
        usage = obj.get("usage")
        if usage is not None:
            if self.usage is not None and usage != self.usage:
                raise BenchError("usage", "Conflicting usage events")
            if not isinstance(usage, dict) or any(type(usage.get(k)) is not int for k in
                                                     ("prompt_tokens", "completion_tokens", "total_tokens")):
                raise BenchError("usage", "Missing integer terminal usage")
            if {k: usage[k] for k in ("prompt_tokens", "completion_tokens", "total_tokens")} != {
                    "prompt_tokens": self.prompt_tokens, "completion_tokens": self.max_tokens,
                    "total_tokens": self.prompt_tokens + self.max_tokens}:
                raise BenchError("usage", "Usage differs from exact prepared input/output budget")
            self.usage = usage
        choices = obj.get("choices")
        if not isinstance(choices, list) or len(choices) > 1:
            raise BenchError("stream", "Expected one completion choice")
        visible = False
        for choice in choices:
            if choice.get("index") != 0:
                raise BenchError("stream", "Unexpected completion index")
            delta = choice.get("delta", {})
            if not isinstance(delta, dict) or delta.get("tool_calls"):
                raise BenchError("stream", "Unexpected tool calls or malformed delta")
            for key in ("content", "reasoning_content"):
                value = delta.get(key)
                if value is not None and not isinstance(value, str):
                    raise BenchError("stream", "Non-text completion delta")
                visible |= bool(value)
            finish = choice.get("finish_reason")
            if finish is not None:
                if self.finish is not None:
                    raise BenchError("stream", "Duplicate completion finish")
                self.finish = finish
            elif self.finish is not None and visible:
                raise BenchError("stream", "Content after completion finish")
        if visible:
            self.visible.append(now)

    def result(self):
        if self.done is None:
            raise BenchError("stream", "Connection closed without terminal SSE marker")
        first = self.visible[0] if self.visible else None
        last = self.visible[-1] if self.visible else None
        return {"latency_ms": (self.done - self.started) * 1000,
                "ttft_visible_ms": (first - self.started) * 1000 if first is not None else None,
                "tpot_client_ms": (last - first) * 1000 / (self.max_tokens - 1)
                if first is not None and self.max_tokens > 1 else None,
                "visible_null_reason": None if first is not None else "no_nonempty_delta",
                "delta_intervals_ms": [(b - a) * 1000 for a, b in zip(self.visible, self.visible[1:])],
                "completion_tokens": self.max_tokens, "prompt_tokens": self.prompt_tokens,
                "finish_reason": self.finish}


def stream_request(base_url, body, request_id, prompt_tokens, max_tokens, timeout, raw_path):
    """Time immediately before body transmission; never count SSE chunks as tokens."""
    url = urlsplit(base_url)
    if url.scheme != "http" or url.hostname != "127.0.0.1" or not url.port:
        raise BenchError("http", "Benchmark transport must be explicit HTTP loopback")
    conn = http.client.HTTPConnection(url.hostname, url.port, timeout=min(timeout, 1.0))
    payload = canonical(body)
    parser = SSEParser()
    capture = None
    try:
        conn.connect()
        conn.putrequest("POST", "/v1/chat/completions")
        conn.putheader("Content-Type", "application/json")
        conn.putheader("Content-Length", str(len(payload)))
        conn.putheader("X-Request-ID", request_id)
        conn.endheaders()
        started = time.perf_counter()
        capture = StreamCapture(prompt_tokens, max_tokens, started)
        transport = conn.sock
        transport.settimeout(timeout)
        conn.send(payload)
        response = conn.getresponse()
        with Path(raw_path).open("xb") as raw:
            if response.status != 200:
                raw.write(response.read())
                raise BenchError("http", f"HTTP {response.status}")
            if response.getheader("Content-Type", "").split(";")[0].strip() != "text/event-stream":
                raise BenchError("stream", "Response is not SSE")
            while capture.done is None:
                remaining = timeout - (time.perf_counter() - started)
                if remaining <= 0:
                    raise BenchError("timeout", "Request deadline exceeded")
                # HTTP/1.0 streaming detaches conn.sock when headers declare connection-close.
                transport.settimeout(remaining)
                chunk = response.read1(4096)
                now = time.perf_counter()
                if now - started > timeout:
                    raise BenchError("timeout", "Request deadline exceeded")
                raw.write(chunk)
                if not chunk:
                    parser.feed(b"", final=True)
                    break
                for event in parser.feed(chunk):
                    capture.event(event, now)
        return capture.result(), capture
    except (TimeoutError, socket.timeout) as exc:
        raise BenchError("timeout", "HTTP request timed out") from exc
    except (UnicodeError, http.client.HTTPException, ConnectionError) as exc:
        raise BenchError("stream", str(exc)) from exc
    finally:
        try:
            if capture is not None:
                evidence = {"started_monotonic": capture.started, "done_monotonic": capture.done,
                            "visible_monotonic": capture.visible, "usage": capture.usage,
                            "finish_reason": capture.finish, "timings": capture.timings,
                            "extensions": capture.extensions}
                Path(str(raw_path) + ".capture.json").write_bytes(canonical(evidence) + b"\n")
        finally:
            conn.close()


def json_request(base_url, endpoint, body=None, timeout=10):
    url = urlsplit(base_url)
    conn = http.client.HTTPConnection(url.hostname, url.port, timeout=timeout)
    try:
        conn.request("GET" if body is None else "POST", endpoint,
                     None if body is None else canonical(body), {"Content-Type": "application/json"})
        response = conn.getresponse()
        data = response.read()
        if response.status != 200:
            raise BenchError("http", f"{endpoint}: HTTP {response.status}")
        return json.loads(data)
    finally:
        conn.close()


class OwnedServer:
    """Windows job kills native descendants, even if the Python parent exits early."""
    def __init__(self, argv, env, cwd, url, log, startup_timeout, shutdown_timeout):
        self.argv, self.env, self.cwd, self.url = argv, env, cwd, url
        self.log, self.startup_timeout, self.shutdown_timeout = log, startup_timeout, shutdown_timeout
        self.proc = None
        self.job = None
        self.assigned = False
        self.handle = None
        self.startup_ms = self.shutdown_ms = None

    def start(self):
        url = urlsplit(self.url)
        with socket.socket() as probe:
            if probe.connect_ex((url.hostname, url.port)) == 0:
                raise BenchError("process", "Server port already occupied; refusing another GPU process")
        started = time.perf_counter()
        self.handle = Path(self.log).open("xb")
        kwargs = {"stdout": self.handle, "stderr": subprocess.STDOUT, "stdin": subprocess.DEVNULL,
                  "cwd": self.cwd, "env": self.env}
        if os.name == "nt":
            # Assign before the parent starts executing, so no native child can escape the job.
            kernel = ctypes.WinDLL("kernel32", use_last_error=True)
            kernel.CreateJobObjectW.restype = ctypes.c_void_p
            kernel.AssignProcessToJobObject.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
            kernel.TerminateJobObject.argtypes = [ctypes.c_void_p, ctypes.c_uint]
            kernel.CloseHandle.argtypes = [ctypes.c_void_p]
            self.kernel = kernel
            self.job = kernel.CreateJobObjectW(None, None)
            if not self.job:
                raise BenchError("process", "CreateJobObject failed")
            class BasicLimits(ctypes.Structure):
                _fields_ = [("process_time", ctypes.c_longlong), ("job_time", ctypes.c_longlong),
                            ("flags", ctypes.c_uint), ("minimum", ctypes.c_size_t), ("maximum", ctypes.c_size_t),
                            ("active_limit", ctypes.c_uint), ("affinity", ctypes.c_size_t),
                            ("priority", ctypes.c_uint), ("scheduling", ctypes.c_uint)]
            class ExtendedLimits(ctypes.Structure):
                _fields_ = [("basic", BasicLimits), ("io", ctypes.c_ulonglong * 6),
                            ("process_memory", ctypes.c_size_t), ("job_memory", ctypes.c_size_t),
                            ("peak_process", ctypes.c_size_t), ("peak_job", ctypes.c_size_t)]
            limits = ExtendedLimits()
            limits.basic.flags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
            kernel.SetInformationJobObject.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_uint]
            if not kernel.SetInformationJobObject(self.job, 9, ctypes.byref(limits), ctypes.sizeof(limits)):
                raise BenchError("process", "Cannot enable owned-child kill-on-close")
            self.proc = subprocess.Popen(self.argv, creationflags=0x00000004 | 0x00000200, **kwargs)
            if not kernel.AssignProcessToJobObject(self.job, int(getattr(self.proc, "_handle"))):
                raise BenchError("process", "Cannot own process tree through Windows job")
            self.assigned = True
            ntdll = ctypes.WinDLL("ntdll")
            ntdll.NtResumeProcess.argtypes = [ctypes.c_void_p]
            if ntdll.NtResumeProcess(int(getattr(self.proc, "_handle"))) != 0:
                raise BenchError("process", "Cannot resume owned process")
        else:
            self.proc = subprocess.Popen(self.argv, start_new_session=True, **kwargs)
        deadline = started + self.startup_timeout
        while time.perf_counter() < deadline:
            if self.proc.poll() is not None:
                raise BenchError("startup", f"Server exited with {self.proc.returncode}")
            try:
                ready = json_request(self.url, "/v1/models", timeout=min(1, self.startup_timeout))
                if isinstance(ready.get("data"), list) and ready["data"]:
                    self.startup_ms = (time.perf_counter() - started) * 1000
                    return
            except (OSError, ValueError, BenchError):
                pass
            time.sleep(0.1)
        raise BenchError("startup", "Readiness deadline exceeded")

    def owned_pids(self):
        """Query the owned Windows job, including the native inference child."""
        if self.job is None:
            return {self.proc.pid} if self.proc is not None else set()
        size = 4096
        while size <= 1024 * 1024:
            buffer = ctypes.create_string_buffer(size)
            self.kernel.QueryInformationJobObject.argtypes = [ctypes.c_void_p, ctypes.c_int,
                                                             ctypes.c_void_p, ctypes.c_uint, ctypes.c_void_p]
            if self.kernel.QueryInformationJobObject(self.job, 3, buffer, size, None):
                count = ctypes.c_uint.from_buffer(buffer, 4).value
                pids = (ctypes.c_size_t * count).from_buffer(buffer, 8)
                return set(pids)
            if ctypes.get_last_error() != 234:  # ERROR_MORE_DATA
                raise BenchError("environment", "Cannot inspect owned GPU process tree")
            size *= 2
        raise BenchError("environment", "Owned process tree exceeds inspection budget")

    def stop(self):
        started = time.perf_counter()
        try:
            if self.job is not None:
                if not self.kernel.TerminateJobObject(self.job, 1):
                    raise BenchError("process", "Failed to terminate owned Windows job")
                if self.proc is not None and self.assigned:
                    self.proc.wait(timeout=self.shutdown_timeout)
                # An empty job, not just a dead launcher, is required before the next arm starts.
                class Accounting(ctypes.Structure):
                    _fields_ = [("times", ctypes.c_longlong * 4), ("faults", ctypes.c_uint),
                                ("total", ctypes.c_uint), ("active", ctypes.c_uint), ("terminated", ctypes.c_uint)]
                info = Accounting()
                self.kernel.QueryInformationJobObject.argtypes = [ctypes.c_void_p, ctypes.c_int,
                                                                 ctypes.c_void_p, ctypes.c_uint, ctypes.c_void_p]
                deadline = started + self.shutdown_timeout
                while True:
                    if not self.kernel.QueryInformationJobObject(self.job, 1, ctypes.byref(info), ctypes.sizeof(info), None):
                        raise BenchError("process", "Cannot verify native child shutdown")
                    if info.active == 0:
                        break
                    if time.perf_counter() >= deadline:
                        raise BenchError("process", "Native children remain alive")
                    time.sleep(0.05)
            elif self.proc is not None:
                try:
                    getattr(os, "killpg")(self.proc.pid, signal.SIGTERM)
                    self.proc.wait(timeout=self.shutdown_timeout)
                except ProcessLookupError:
                    pass
                except subprocess.TimeoutExpired:
                    pass
                finally:
                    try:
                        getattr(os, "killpg")(self.proc.pid, getattr(signal, "SIGKILL"))
                    except ProcessLookupError:
                        pass
                    self.proc.wait(timeout=self.shutdown_timeout)
        finally:
            try:
                if self.proc is not None and not self.assigned and self.job is not None:
                    if self.proc.poll() is None:
                        self.proc.kill()
                    self.proc.wait(timeout=self.shutdown_timeout)
            finally:
                if self.job is not None:
                    self.kernel.CloseHandle(self.job)
                    self.job = None
                if self.handle is not None:
                    self.handle.close()
                self.shutdown_ms = (time.perf_counter() - started) * 1000


class GPUStateSampler:
    """CPU-only NVML CLI sampling; never synchronizes CUDA or alters GPU settings."""
    FIELDS = ("memory.used", "memory.free", "pcie.link.gen.current", "pcie.link.width.current",
              "temperature.gpu", "utilization.gpu", "pstate", "power.draw")

    def __init__(self, executable, path, monitor=None, max_temperature=None):
        self.executable, self.path = executable, Path(path)
        self.samples = []
        self.proc = None
        self.thread = None
        self.error = None
        self.monitor, self.max_temperature = monitor, max_temperature
        self.monitor_thread = None
        self.monitor_samples = []
        self.stopping = threading.Event()
        self.monitor_ready = threading.Event()

    def start(self):
        self.proc = subprocess.Popen([self.executable, "--query-gpu=" + ",".join(self.FIELDS),
                                      "--format=csv,noheader,nounits", "--loop-ms=200"],
                                     stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                     text=True, encoding="utf-8", errors="replace", stdin=subprocess.DEVNULL)
        stdout = self.proc.stdout
        if stdout is None:
            raise BenchError("environment", "GPU telemetry stdout unavailable")
        def collect():
            try:
                with self.path.open("x", encoding="utf-8") as output:
                    for line in stdout:
                        values = [v.strip() for v in line.strip().split(",")]
                        record = {"monotonic": time.perf_counter(), "values": values}
                        self.samples.append(record)
                        if self.max_temperature is not None:
                            if len(values) != len(self.FIELDS):
                                raise BenchError("environment", "Malformed live GPU telemetry")
                            temperature = float(values[self.FIELDS.index("temperature.gpu")])
                            if not math.isfinite(temperature) or temperature > self.max_temperature:
                                raise BenchError("environment", "GPU temperature threshold exceeded during session")
                        output.write(canonical(record).decode("utf-8") + "\n")
                        output.flush()
            except Exception as exc:
                self.error = str(exc)
        self.thread = threading.Thread(target=collect, name="speed-gpu-telemetry", daemon=True)
        self.thread.start()
        if self.monitor is not None:
            monitor = self.monitor
            def monitor_processes():
                while not self.stopping.is_set():
                    try:
                        result = monitor()
                        self.monitor_samples.append({"monotonic": time.perf_counter(), "result": result})
                    except Exception as exc:
                        self.error = str(exc)
                        self.monitor_ready.set()
                        return
                    self.monitor_ready.set()
                    self.stopping.wait(2)
            self.monitor_thread = threading.Thread(target=monitor_processes, name="speed-gpu-processes", daemon=True)
            self.monitor_thread.start()
            if not self.monitor_ready.wait(timeout=30):
                raise BenchError("environment", "Initial GPU process inspection deadline exceeded")
            self.check()

    def check(self):
        if self.error is not None:
            raise BenchError("environment", "GPU monitoring failed: " + self.error)
        if self.proc is not None and self.proc.poll() is not None:
            raise BenchError("environment", "GPU telemetry process exited during session")

    def stop(self):
        self.stopping.set()
        try:
            if self.proc is not None:
                if self.proc.poll() is None:
                    self.proc.terminate()
                try:
                    self.proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    self.proc.kill()
                    self.proc.wait(timeout=10)
        finally:
            for thread in (self.thread, self.monitor_thread):
                if thread is not None:
                    thread.join(timeout=10)
                    if thread.is_alive():
                        raise BenchError("environment", "GPU monitoring reader did not stop")

    def summary(self, started, done):
        samples = [r for r in self.samples if started <= r["monotonic"] <= done]
        valid = []
        for record in samples:
            try:
                values = record["values"]
                if len(values) != len(self.FIELDS):
                    continue
                parsed = dict(zip(self.FIELDS, values))
                numbers = {key: float(value) for key, value in parsed.items() if key != "pstate"}
                if not all(math.isfinite(n) and n >= 0 for n in numbers.values()):
                    continue
                valid.append(numbers)
            except (ValueError, TypeError):
                continue
        active = [r for r in valid if r["utilization.gpu"] > 0]
        result = {"samples": len(samples), "valid_samples": len(valid), "active_samples": len(active),
                  "interval_ms": 200, "error": self.error,
                  "null_reason": None if active else "no_active_sample_in_measured_window"}
        names = {"memory.used": "memory_used_mib", "memory.free": "memory_free_mib",
                 "pcie.link.gen.current": "pcie_gen", "pcie.link.width.current": "pcie_width",
                 "temperature.gpu": "temperature_c", "utilization.gpu": "utilization_percent", "power.draw": "power_w"}
        names["power.draw"] = "power_draw_watts"
        for field, name in names.items():
            values = [r[field] for r in active]
            result[name + "_min"] = min(values) if values else None
            result[name + "_max"] = max(values) if values else None
        return result


def finite_number(value, label, integer=False):
    if type(value) not in ((int,) if integer else (int, float)) or not math.isfinite(value) or value < 0:
        raise BenchError("metrics", f"Invalid {label}")
    return value
