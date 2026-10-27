"""FlintNSteel CLI: serve, status, stop, doctor, launch pi.

  py -3.14 serve/fns.py doctor                 checks GPU, driver, free VRAM, executables, models, Python, pi
  py -3.14 serve/fns.py serve [--quant q3|q4]  starts fns_server.py detached, waits for /v1/models, writes the state
  py -3.14 serve/fns.py status                 running server (pid, port, model) and its /v1/models response
  py -3.14 serve/fns.py stop                   stops the server started by `serve`
  py -3.14 serve/fns.py launch pi [-- args]    starts the server if needed, then pi with an isolated config

rec's Go CLI (cli/*.go) is limited to darwin/linux (`//go:build darwin || linux`); this module covers the useful part
on Windows, without download or installation. pi must already be installed (`pi` on PATH, or --pi). pi's config is
written to a FlintNSteel-specific directory (PI_CODING_AGENT_DIR), as `piUser` does in rec, with the
`openai-completions` provider pointing at the local server.
"""

import argparse
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
import uuid
import urllib.request
from pathlib import Path

from flint_assets import CHAT_TEMPLATE, build_dir, cached_hf_tokenizer, local_tokenizer, models_dir, profile_dir, vocab_path

ROOT = Path(__file__).resolve().parents[1]
MODELS = models_dir()
STATE = Path(os.environ.get("FNS_STATE", Path(os.environ.get("LOCALAPPDATA", Path.home())) / "FlintNSteel"))
QUANTS = {"q3": "flint-UD-Q3_K_XL.gguf", "q4": "flint-UD-Q4_K_XL.gguf"}
Q4_PROFILE = profile_dir() / "q4-code-train-8k.route"  # expert-routing profile for Q4
VOCAB = vocab_path()
EXES = ["fns-serve.exe"]
MIN_VRAM_FREE_MIB = {"q3": 20000, "q4": 20000}  # [estimate] Q3 weights 18.5 GB + KV/buffers; Q4: experts take the rest
CTX = 32768


def state_file() -> Path:
    return STATE / "server.json"


def read_state() -> dict | None:
    try:
        return json.loads(state_file().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def process_identity(pid: object) -> str | None:
    """Return a process birth identity, or None when it cannot be verified."""
    if type(pid) is not int or pid <= 0:
        return None
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes

        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
        kernel.OpenProcess.restype = wintypes.HANDLE
        kernel.GetProcessTimes.argtypes = (wintypes.HANDLE, ctypes.POINTER(wintypes.FILETIME),
                                          ctypes.POINTER(wintypes.FILETIME), ctypes.POINTER(wintypes.FILETIME),
                                          ctypes.POINTER(wintypes.FILETIME))
        kernel.GetProcessTimes.restype = wintypes.BOOL
        kernel.CloseHandle.argtypes = (wintypes.HANDLE,)
        handle = kernel.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
        if not handle:
            return None
        try:
            created, exited, kernel_time, user_time = (wintypes.FILETIME() for _ in range(4))
            if not kernel.GetProcessTimes(handle, ctypes.byref(created), ctypes.byref(exited),
                                          ctypes.byref(kernel_time), ctypes.byref(user_time)):
                return None
            return f"win:{created.dwHighDateTime:08x}{created.dwLowDateTime:08x}"
        finally:
            kernel.CloseHandle(handle)
    if sys.platform.startswith("linux"):
        try:
            stat = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
            start = stat[stat.rindex(")") + 2:].split()[19]
            boot = Path("/proc/sys/kernel/random/boot_id").read_text(encoding="utf-8").strip()
            return f"linux:{boot}:{start}"
        except (OSError, ValueError, IndexError):
            return None
    return None


def owned_state() -> dict | None:
    s = read_state()
    if s is None:
        return None
    if (not isinstance(s, dict) or not isinstance(s.get("identity"), str)
            or not isinstance(s.get("launch_token"), str) or not s["launch_token"]
            or type(s.get("port")) is not int or not 0 < s["port"] < 65536
            or process_identity(s.get("pid")) != s["identity"]):
        state_file().unlink(missing_ok=True)
        return None
    return s


def models_endpoint(port: int, timeout: float = 2.0) -> dict | None:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=timeout) as r:
            return json.loads(r.read())
    except (OSError, ValueError):
        return None


def health_endpoint(port: int, timeout: float = 2.0) -> dict | None:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=timeout) as r:
            return json.loads(r.read())
    except (OSError, ValueError):
        return None


def running() -> dict | None:
    s = owned_state()
    if s and health_endpoint(s["port"]) == {"pid": s["pid"], "launch_token": s.get("launch_token")}:
        return s
    return None


# --- doctor -----------------------------------------------------------------------------------------------------------
def gpu_facts() -> dict | None:
    smi = shutil.which("nvidia-smi")
    if not smi:
        return None
    out = subprocess.run([smi, "--query-gpu=name,driver_version,memory.free,memory.total,compute_cap",
                          "--format=csv,noheader,nounits"], capture_output=True, text=True)
    if out.returncode or not out.stdout.strip():
        return None
    name, driver, free, total, cc = [x.strip() for x in out.stdout.splitlines()[0].split(",")]
    return {"name": name, "driver": driver, "free_mib": int(free), "total_mib": int(total), "cc": cc}


def doctor(o) -> int:
    checks: list[tuple[str, bool, str]] = []

    def check(name: str, ok: bool, detail: str) -> None:
        checks.append((name, ok, detail))

    g = gpu_facts()
    check("GPU", bool(g) and g["cc"] == "8.6", f"{g['name']}, sm_{g['cc'].replace('.', '')}" if g else "nvidia-smi not found")
    if g:
        check("driver", True, g["driver"])
        s = running()
        # A FlintNSteel server already loaded holds the VRAM: that is not a failure.
        need = MIN_VRAM_FREE_MIB[o.quant]
        check("free VRAM", g["free_mib"] >= need or s is not None,
              f"{g['free_mib']} / {g['total_mib']} MiB" + (" (FlintNSteel server loaded)" if s else f", {need} required"))
    for exe in EXES:
        p = build_dir() / exe
        check(exe, p.is_file(), str(p) if p.is_file() else f"{p} missing: run build.bat")
    model = MODELS / QUANTS[o.quant]
    check("model", model.is_file(), f"{model} ({model.stat().st_size / 1e9:.2f} GB)" if model.is_file() else
          f"{model} missing")
    if o.quant == "q4":
        check("Q4 profile", Q4_PROFILE.is_file(), str(Q4_PROFILE) if Q4_PROFILE.is_file() else
              f"{Q4_PROFILE} missing: py -3.14 bench/scripts/route_profile.py profile")
    check("vocabulary", VOCAB.is_file(), str(VOCAB) if VOCAB.is_file() else
          f"{VOCAB} missing: py -3.14 tools/xgrammar/export_vocab.py")
    check("chat template", CHAT_TEMPLATE.is_file(), str(CHAT_TEMPLATE))
    missing = []
    for mod in ("jinja2", "tokenizers", "huggingface_hub"):
        try:
            __import__(mod)
        except ImportError:
            missing.append(mod)
    check("Python", not missing and sys.version_info >= (3, 10),
          f"{sys.version.split()[0]}" + (f", missing: {', '.join(missing)}" if missing else ""))
    if not missing:
        try:
            tok = local_tokenizer() or cached_hf_tokenizer()
            check("tokenizer", tok is not None, str(tok) if tok is not None else
                  "not local and not in the HF cache (downloaded on first start)")
        except FileNotFoundError as e:
            check("tokenizer", False, str(e))
    pi = o.pi or shutil.which("pi")
    if pi:
        v = subprocess.run([pi, "--version"], capture_output=True, text=True, shell=sys.platform == "win32")
        check("pi", v.returncode == 0, f"{pi} {v.stdout.strip()}")
    else:
        check("pi", False, "not on PATH (npm install -g @mariozechner/pi-coding-agent, or --pi)")
    width = max(len(n) for n, _, _ in checks)
    for name, ok, detail in checks:
        print(f"{'ok  ' if ok else 'FAIL'} {name:<{width}}  {detail}")
    bad = [n for n, ok, _ in checks if not ok]
    print("all checks passed" if not bad else f"{len(bad)} problem(s): {', '.join(bad)}")
    return 1 if bad else 0


# --- serve / status / stop --------------------------------------------------------------------------------------------
def serve(o) -> int:
    s = running()
    if s:
        print(f"already running: pid {s['pid']}, http://127.0.0.1:{s['port']}/v1 ({s['model']})")
        return 0
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("127.0.0.1", o.port))
    except OSError as e:
        print(f"port {o.port} is unavailable: {e}", file=sys.stderr)
        return 1
    model = MODELS / QUANTS[o.quant]
    if o.quant == "q4" and not Q4_PROFILE.is_file():
        print(f"Q4 profile missing: {Q4_PROFILE}\n"
              "create it with: py -3.14 bench/scripts/route_profile.py profile", file=sys.stderr)
        return 1
    token = uuid.uuid4().hex
    cmd = [sys.executable, str(ROOT / "serve" / "fns_server.py"), "--model", str(model), "--port", str(o.port),
           "--ctx", str(o.ctx), "--log", str(STATE / "requests.jsonl"), "--launch-token", token]
    if o.quant == "q4":  # adaptive expert cache (same bits, x1.045 under agent load)
        cmd += ["--expert-profile", str(Q4_PROFILE), "--rebalance", "64"]
    STATE.mkdir(parents=True, exist_ok=True)
    log = STATE / "server.log"
    flags = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS if sys.platform == "win32" else 0
    with log.open("w", encoding="utf-8") as lf:
        proc = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                creationflags=flags, start_new_session=sys.platform != "win32")
    identity = process_identity(proc.pid)
    if identity is None:
        proc.terminate()
        print("cannot verify server process identity", file=sys.stderr)
        return 1
    t0 = time.time()
    while time.time() - t0 < o.timeout:
        if proc.poll() is not None:
            print(f"fns_server stopped (code {proc.returncode}), see {log}", file=sys.stderr)
            return 1
        if health_endpoint(o.port, 1.0) == {"pid": proc.pid, "launch_token": token}:
            st = {"pid": proc.pid, "identity": identity, "launch_token": token, "port": o.port,
                  "model": str(model), "quant": o.quant, "ctx": o.ctx, "started": time.time(), "log": str(log)}
            state_file().write_text(json.dumps(st), encoding="utf-8")
            print(f"ready in {time.time() - t0:.1f} s: http://127.0.0.1:{o.port}/v1 ({model.name}, pid {proc.pid})")
            return 0
        time.sleep(0.5)
    print(f"not ready after {o.timeout} s, see {log}", file=sys.stderr)
    if proc.poll() is None and process_identity(proc.pid) == identity:
        kill(proc.pid)
    return 1


def kill(pid: int) -> None:
    if sys.platform == "win32":
        subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True)
    else:
        os.killpg(pid, signal.SIGTERM)


def status(o) -> int:
    s = owned_state()
    if not s:
        print("no server started by fns serve")
        return 1
    health = health_endpoint(s["port"])
    m = models_endpoint(s["port"]) if health == {"pid": s["pid"], "launch_token": s.get("launch_token")} else None
    print(f"pid {s['pid']}, http://127.0.0.1:{s['port']}/v1, {Path(s['model']).name}, "
          f"context {s['ctx']}" + (f", models {[d['id'] for d in m['data']]}" if m else ", /v1/models no response"))
    return 0 if m else 1


def stop(o) -> int:
    s = owned_state()
    if not s:
        print("no server started by fns serve")
        return 0
    if process_identity(s["pid"]) == s["identity"]:
        kill(s["pid"])
    state_file().unlink(missing_ok=True)
    print(f"stopped (pid {s['pid']})")
    return 0


# --- launch pi --------------------------------------------------------------------------------------------------------
def pi_config(port: int, ctx: int) -> Path:
    """FlintNSteel-specific pi agent directory: provider `fns`, model `flint`, no key (pi's models.json)."""
    d = STATE / "pi-agent"
    d.mkdir(parents=True, exist_ok=True)
    models = {"providers": {"fns": {
        "baseUrl": f"http://127.0.0.1:{port}/v1", "api": "openai-completions", "apiKey": "none",
        "models": [{"id": "flint", "name": "Flint (FlintNSteel)", "reasoning": False, "input": ["text"],
                    "contextWindow": ctx, "maxTokens": 8192,
                    "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0},
                    "compat": {"supportsStore": False, "supportsDeveloperRole": False,
                               "supportsReasoningEffort": False, "supportsStrictMode": False,
                               "maxTokensField": "max_tokens"}}]}}}
    (d / "models.json").write_text(json.dumps(models, indent=1), encoding="utf-8")
    for name, empty in (("settings.json", "{}"), ("auth.json", "{}")):
        if not (d / name).exists():
            (d / name).write_text(empty, encoding="utf-8")
    return d


def launch(o) -> int:
    if o.client != "pi":
        print("only `launch pi` is supported", file=sys.stderr)
        return 2
    pi = o.pi or shutil.which("pi")
    if not pi:
        print("pi not found on PATH (or --pi)", file=sys.stderr)
        return 1
    s = running()
    if not s:
        if serve(o):
            return 1
        s = running()
        if not s:
            print("server started but unreachable", file=sys.stderr)
            return 1
    agent = pi_config(s["port"], s["ctx"])
    env = {**os.environ, "PI_CODING_AGENT_DIR": str(agent), "PI_SKIP_VERSION_CHECK": "1"}
    env.pop("PI_CODING_AGENT_SESSION_DIR", None)
    rest = o.rest[1:] if o.rest[:1] == ["--"] else o.rest
    argv = [pi, "--provider", "fns", "--model", "flint", *rest]
    print(f"-> pi on Flint, http://127.0.0.1:{s['port']}/v1, {s['ctx']} tokens", file=sys.stderr)
    return subprocess.run(argv, env=env, shell=sys.platform == "win32").returncode


def main() -> None:
    ap = argparse.ArgumentParser(prog="fns", description="FlintNSteel: local Flint server and clients")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def server_opts(p):
        p.add_argument("--quant", choices=sorted(QUANTS), default="q3", help="UD-Q3_K_XL (all in VRAM) or UD-Q4_K_XL")
        p.add_argument("--port", type=int, default=8091)
        p.add_argument("--ctx", type=int, default=CTX)
        p.add_argument("--timeout", type=float, default=300.0, help="server wait (s)")

    p = sub.add_parser("doctor")
    p.add_argument("--quant", choices=sorted(QUANTS), default="q3")
    p.add_argument("--pi")
    p = sub.add_parser("serve")
    server_opts(p)
    sub.add_parser("status")
    sub.add_parser("stop")
    p = sub.add_parser("launch")
    p.add_argument("client", choices=["pi"])
    p.add_argument("--pi", help="pi executable (default: PATH)")
    server_opts(p)
    p.add_argument("rest", nargs=argparse.REMAINDER, help="arguments passed to pi (after --)")
    o = ap.parse_args()
    sys.exit({"doctor": doctor, "serve": serve, "status": status, "stop": stop, "launch": launch}[o.cmd](o))


if __name__ == "__main__":
    main()
