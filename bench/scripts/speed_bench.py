"""FNS-SPEED v1: private, sequential paired HTTP campaigns and reviewed exports.

Run ``check``, ``run`` and ``report`` with --help. This module never downloads data.
Campaign paths are relative to the campaign JSON; argv are arrays, never shell commands.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.metadata
import json
import os
import platform
import ntpath
import re
import shutil
import subprocess
import sys
import time
import uuid
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path, PureWindowsPath

from speed_runtime import (BenchError, GPUStateSampler, OwnedServer, canonical, finite_number,
                           json_request, stream_request)

ROOT = Path(__file__).resolve().parents[2]
HASH = re.compile(r"^[0-9a-f]{64}$")
COMMIT = re.compile(r"^[0-9a-f]{40}$")
MODES = {"greedy": {"temperature": 0, "top_p": 1, "top_k": 0, "seeds": [0]},
         "sample": {"temperature": 1, "top_p": 0.95, "top_k": 20, "seeds": [0, 7]}}
NEUTRAL = {"presence_penalty": 0, "frequency_penalty": 0, "repeat_penalty": 1,
           "min_p": 0, "typical_p": 1, "tfs_z": 1, "mirostat": 0,
           "dynatemp_range": 0, "xtc_probability": 0, "top_n_sigma": -1}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def content_hash(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def jsonl(path):
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def save_json(path, value):
    Path(path).write_bytes(canonical(value) + b"\n")


def require(ok, message, code="other"):
    if not ok:
        raise BenchError(code, message)


def path_value(value, base):
    require(isinstance(value, str) and bool(value), "Path must be a nonempty string")
    expanded = re.sub(r"\$\{([A-Za-z_][A-Za-z_0-9]*)\}",
                      lambda m: os.environ[m[1]], value)
    require("${" not in expanded, "Unresolved environment variable")
    path = Path(expanded)
    return (base / path).resolve() if not path.is_absolute() else path.resolve()


def artifact(value, base):
    require(isinstance(value, dict) and HASH.fullmatch(value.get("sha256", "")),
            "Artifact requires path and lowercase SHA-256")
    path = path_value(value["path"], base)
    require(path.is_file(), f"Missing artifact: {path}")
    actual = digest(path)
    require(actual == value["sha256"], f"Artifact hash mismatch: {path}")
    return {"path": str(path), "sha256": actual}


def command_output(argv, cwd=None):
    result = subprocess.run(argv, cwd=cwd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    require(result.returncode == 0, f"Command failed: {argv[0]}: {result.stderr}")
    return result.stdout.strip()


def validate_campaign(path):
    """Validate all comparison-affecting settings before touching either engine."""
    path = Path(path).resolve()
    cfg = load_json(path)
    base = path.parent
    require(cfg.get("schema_version") == 1, "Unsupported campaign schema")
    cfg["campaign_path"] = str(path)
    cfg["campaign_sha256"] = digest(path)
    cfg["gpu_process_policy"] = validate_gpu_process_policy(cfg.get("gpu_process_policy"))
    for key in ("protocol", "dependencies", "tokenizer", "template", "vocab", "fns_binary", "llama_binary"):
        cfg[key] = artifact(cfg[key], base)
    protocol = load_json(cfg["protocol"]["path"])
    require(protocol.get("schema_version") == 1 and protocol.get("name") == "FNS-SPEED v1", "Unknown protocol")
    require(protocol.get("dataset", {}).get("revision") == "fbd6dc64353a3153bf24252a113ebd6d2cff9dec"
            and protocol.get("resolver", {}).get("revision") == "a4dfb916c8d4f958e11ff4dea8fd594f4ecc4260",
            "Unpinned dataset or resolver")
    cfg["protocol_data"] = protocol
    require(type(cfg.get("ctx")) is int and cfg["ctx"] > 0 and type(cfg.get("max_tokens")) is int
            and cfg["max_tokens"] > 0, "Explicit positive context and output budgets required")
    require(cfg["ctx"] == protocol["ctx"] and cfg["max_tokens"] == protocol["max_tokens"], "Campaign budgets differ from protocol")
    require(type(cfg.get("pairs")) is int and cfg["pairs"] > 0, "Positive integer pairs required")
    require(cfg.get("pair_order") == ["A", "B"], "Protocol v1 requires adjacent fixed AB pairs")
    require(cfg.get("generation") == {"ignore_eos": True, "cache_prompt": False, "grammar": False,
                                      "penalties": "disabled", "llama_filters": "neutral"},
            "Explicit fixed-budget generation and neutral sampling filters required")
    require(cfg.get("sampling") == MODES, "Explicit protocol greedy/sample settings required")
    for key in ("startup", "request", "shutdown"):
        value = cfg["timeouts"][key]
        require(type(value) in (int, float) and 0 < value <= 86400, f"Invalid {key} timeout")
    require(cfg.get("warmup", {}).get("messages") and type(cfg["warmup"].get("max_tokens")) is int
            and 0 < cfg["warmup"]["max_tokens"] <= cfg["max_tokens"], "Fixed warmup required")
    cfg["warmup_sha256"] = content_hash(cfg["warmup"])
    cfg["output_dir"] = str(path_value(cfg["output_dir"], base))
    cfg["prepared_dir"] = str(path_value(cfg["prepared_dir"], base))
    cfg["source_lock"] = artifact(cfg["source_lock"], base)
    # Reuse the preparation owner's offline source inventory validator: every provenance file is hashed.
    from speed_prepare import verify_lock
    locked = verify_lock(Path(cfg["source_lock"]["path"]).parent, protocol)
    require(locked.get("protocol_sha256") == cfg["protocol"]["sha256"], "Source lock protocol hash mismatch")
    prep_dir = Path(cfg["prepared_dir"])
    prep = load_json(prep_dir / "preparation.json")
    for key, expected in (("protocol", cfg["protocol"]["sha256"]), ("source_lock", cfg["source_lock"]["sha256"]),
                          ("tokenizer", cfg["tokenizer"]["sha256"]), ("template", cfg["template"]["sha256"])):
        require(prep.get(key + "_sha256") == expected, f"Preparation {key} hash mismatch")
    for key in ("requests", "selection"):
        require(digest(prep_dir / (key + ".jsonl")) == prep.get(key + "_sha256"), f"Prepared {key} hash mismatch")
    require(prep.get("ctx") == cfg["ctx"] and prep.get("max_tokens") == cfg["max_tokens"], "Preparation budget mismatch")
    require(prep.get("dependency_lock_sha256") == cfg["dependencies"]["sha256"], "Preparation dependency lock mismatch")
    profile = prep.get("profile")
    require(profile in ("smoke", "release", "full"), "Unknown selection profile")
    require(type(prep.get("reproducible")) is bool and prep["reproducible"] == locked["reproducible"],
            "Preparation/source-lock reproducibility declaration differs")
    if profile != "smoke":
        require(cfg["pairs"] >= 5, "Publication requires at least five pairs")
        require(locked["reproducible"] is True,
                "Release/full require independently reacquirable locked sources; only smoke permits local-only replay")
    if profile == "release":
        require(cfg["ctx"] == 32768 and cfg["max_tokens"] == 256, "Release v1 budgets are fixed")
        require(set(protocol["configs"]) == {c["config"] for c in prep["coverage"]}, "Release requires all six configurations")
    cfg["preparation"] = prep
    cfg["preparation_sha256"] = digest(prep_dir / "preparation.json")
    requests = jsonl(prep_dir / "requests.jsonl")
    require(bool(requests), "No prepared requests")
    seen = set()
    for request in requests:
        rid = request["request_id"]
        require(HASH.fullmatch(rid) is not None and rid not in seen, "Duplicate or malformed request ID")
        seen.add(rid)
        require(rid == content_hash([request["config"], str(request["question_id"])]), "Request ID hash mismatch")
        tokens = request["tokens"]
        require(isinstance(tokens, list) and tokens and all(type(t) is int and t >= 0 for t in tokens), "Invalid prepared tokens")
        require(request["prompt_tokens"] == len(tokens) and request["prompt_sha256"] == content_hash(tokens),
                "Prepared token content hash mismatch")
        require(len(tokens) + cfg["max_tokens"] <= cfg["ctx"], "Prepared input exceeds context; no truncation allowed")
        messages = request["messages"]
        require(isinstance(messages, list) and len(messages) == 1 and messages[0].get("role") == "user"
                and isinstance(messages[0].get("content"), str) and bool(messages[0]["content"]), "Expected one nonempty user turn")
        require(hashlib.sha256(messages[0]["content"].encode()).hexdigest() == request["normalized_text_sha256"],
                "Normalized text hash mismatch")
    selected_rows = jsonl(prep_dir / "selection.jsonl")
    require(len({r["request_id"] for r in selected_rows}) == len(selected_rows), "Duplicate selection rows")
    require({r["request_id"] for r in selected_rows if r["selected"]} == seen, "Selection/request membership differs")
    request_map = {r["request_id"]: r for r in requests}
    for selection in selected_rows:
        admissible = selection["prompt_tokens"] + cfg["max_tokens"] <= cfg["ctx"]
        require(selection.get("config") in protocol["configs"] and selection.get("category") in protocol["categories"][selection["config"]],
                "Selection category/configuration outside protocol")
        require(content_hash([selection["config"], str(selection["question_id"])]) == selection["request_id"], "Selection ID hash mismatch")
        require(hashlib.sha256(selection["source_text"].encode()).hexdigest() == selection["source_text_sha256"]
                and hashlib.sha256(selection["normalized_text"].encode()).hexdigest() == selection["normalized_text_sha256"],
                "Selection original/normalized text hash mismatch")
        require(selection["admissible"] is admissible, "Selection context decision mismatch")
        if selection["selected"]:
            row = request_map[selection["request_id"]]
            require(all(selection[k] == row[k] for k in ("config", "category", "prompt_tokens", "prompt_sha256")),
                    "Selected row content differs from prepared request")
    for coverage in prep["coverage"]:
        members = [r for r in selected_rows if (r["config"], r["category"]) == (coverage["config"], coverage["category"])]
        eligible = [r for r in members if r["admissible"]]
        expected = 1 if profile == "smoke" else 32 if profile == "release" and coverage["config"] != "qualitative" else len(eligible)
        ordered = sorted(eligible, key=lambda r: (content_hash(["fns-speed-v1", r["config"], str(r["question_id"])]), str(r["question_id"])))
        require({r["request_id"] for r in ordered[:expected]} == {r["request_id"] for r in members if r["selected"]},
                "Prepared selection violates deterministic recipe")
        require((coverage["total"], coverage["admissible"], coverage["selected"], coverage["excluded_context"]) ==
                (len(members), len(eligible), min(expected, len(eligible)), len(members) - len(eligible)), "Coverage counts mismatch")
    included = set(prep.get("configs", []))
    require({(c["config"], c["category"]) for c in prep["coverage"]} ==
            {(c, k) for c in included for k in protocol["categories"][c]}, "Coverage omits an included category")
    require(len(prep["coverage"]) == len({(c["config"], c["category"]) for c in prep["coverage"]}), "Duplicate coverage rows")
    cfg["requests"] = requests
    cfg["sources"] = {name: {**source, "path": str(path_value(source["path"], base))}
                      for name, source in cfg["sources"].items()}
    require(set(cfg["sources"]) == {"fns", "llama"}, "Both source recipes required")
    for source in cfg["sources"].values():
        require(COMMIT.fullmatch(source.get("commit", "")) is not None and source.get("build_recipe")
                and source.get("repository"), "Source requires commit, repository, and build recipe")
        actual = command_output(["git", "rev-parse", "HEAD"], source["path"])
        require(actual == source["commit"], "Source commit mismatch")
    for quant in ("q3", "q4"):
        model = cfg["models"][quant]
        model["artifact"] = artifact(model["artifact"], base)
        if quant == "q4":
            model["profile"] = artifact(model["profile"], base)
            require(model.get("profile_recipe") and model.get("profile_corpus") == "outside-speed",
                    "Q4 routing profile requires independent calibration recipe")
        require(model.get("llama_calibration", {}).get("corpus") == "outside-speed"
                and model["llama_calibration"].get("recipe"), "llama settings require independent calibration")
        for name in ("llama", "fns"):
            argv = model[name + "_argv"]
            require(isinstance(argv, list) and all(isinstance(s, str) and s for s in argv), "Engine options must be argv arrays")
            validate_extra_args(name, argv)
    require(set(cfg["launchers"]) == {"fns", "llama"}, "Explicit engine launcher argv required")
    for name, launcher in cfg["launchers"].items():
        require(isinstance(launcher, list) and launcher and all(isinstance(s, str) for s in launcher), "Launcher must be argv")
        if name == "fns":
            require(launcher == ["{python}", "{fns_server}"], "FNS launcher must select only the fingerprinted Python service")
        else:
            require(launcher == ["{llama_binary}"], "llama launcher must select fingerprinted executable")
    cfg["dlls"] = [artifact(a, base) for a in cfg["dlls"]]
    cfg["fns_server"] = artifact(cfg["fns_server"], base)
    require(cfg["fns_server"]["sha256"] == digest(ROOT / "serve" / "fns_server.py")
            and cfg["template"]["sha256"] == digest(ROOT / "serve" / "chat_template.jinja"),
            "FNS production service and template must match parity checker artifacts")
    declared_dlls = {Path(a["path"]).resolve() for a in cfg["dlls"]}
    adjacent_dlls = {p.resolve() for key in ("fns_binary", "llama_binary") for p in Path(cfg[key]["path"]).parent.glob("*.dll")}
    require(adjacent_dlls <= declared_dlls, "Fingerprint every DLL beside the engine executables in dlls")
    cfg["environment"] = dict(cfg["environment"])
    require(all(isinstance(k, str) and isinstance(v, str) for k, v in cfg["environment"].items()), "Environment must contain strings")
    require(all(type(cfg.get("ports", {}).get(e)) is int and 1024 <= cfg["ports"][e] <= 65535 for e in ("fns", "llama"))
            and cfg["ports"]["fns"] != cfg["ports"]["llama"], "Distinct explicit loopback ports required")
    for key in ("FNS_MODELS", "FNS_TOKENIZER"):
        if key in os.environ:
            cfg["environment"][key] = os.environ[key]
    if "FNS_TOKENIZER" in cfg["environment"]:
        require(Path(cfg["environment"]["FNS_TOKENIZER"]).resolve() == Path(cfg["tokenizer"]["path"]),
                "Explicit FNS_TOKENIZER differs from campaign tokenizer")
    cfg["environment"]["FNS_TOKENIZER"] = cfg["tokenizer"]["path"]
    require(isinstance(cfg.get("environment_constraints"), dict), "Environmental invalidation rules required")
    for key in ("min_free_vram_mib", "max_gpu_temperature_c", "max_idle_gpu_utilization_percent"):
        require(type(cfg["environment_constraints"].get(key)) in (int, float), f"Missing environmental rule: {key}")
    for key in ("cuda", "msvc", "build_architecture", "display_state", "memory_settings"):
        require(cfg.get("machine", {}).get(key), f"Missing declared machine state: {key}")
    return cfg


def validate_extra_args(engine, argv):
    allowed = {"fns": {"--ubatch": "int", "--rebalance": "int"},
               "llama": {"--n-gpu-layers": "int", "--flash-attn": "str", "--cache-type-k": "str",
                         "--cache-type-v": "str", "--n-cpu-moe": "int", "--moe-cache-mib": "int",
                         "--threads": "int", "--threads-batch": "int", "--batch-size": "int",
                         "--ubatch-size": "int", "--swa-full": None, "--no-mmap": None,
                         "--mlock": None, "--fit": "str", "--device": "str", "--main-gpu": "int"}}[engine]
    index = 0
    found = set()
    while index < len(argv):
        flag = argv[index]
        require(flag in allowed and flag not in found, f"Unsupported/duplicate comparison-affecting option: {flag}")
        found.add(flag)
        kind = allowed[flag]
        index += 1
        if kind is not None:
            require(index < len(argv), f"Missing value for {flag}")
            value = argv[index]
            require(not value.startswith("--"), f"Missing value for {flag}")
            if kind == "int":
                require(re.fullmatch(r"[0-9]+", value) is not None, f"Expected nonnegative integer for {flag}")
            if flag in ("--flash-attn", "--fit"):
                require(value in ("on", "off"), f"Explicit on/off required for {flag}")
            if flag == "--device":
                require(re.fullmatch(r"CUDA[0-9]+(?:,CUDA[0-9]+)*", value) is not None, "Explicit CUDA devices required")
            index += 1
    if engine == "llama":
        require({"--n-gpu-layers", "--flash-attn", "--cache-type-k", "--cache-type-v"} <= found,
                "llama offload, flash attention and KV formats must be explicit")
        for flag in ("--cache-type-k", "--cache-type-v"):
            require(argv[argv.index(flag) + 1] == "f16", "Main campaign requires F16 KV")


@contextlib.contextmanager
def local_chat(cfg):
    # Import lazily, with explicit local artifacts: no dataset or HF network imports in tests.
    sys.path.insert(0, str(ROOT / "serve"))
    previous = os.environ.get("FNS_TOKENIZER")
    os.environ["FNS_TOKENIZER"] = cfg["tokenizer"]["path"]
    import importlib
    fns_server = importlib.import_module("fns_server")
    old_template = fns_server.CHAT_TEMPLATE
    setattr(fns_server, "CHAT_TEMPLATE", Path(cfg["template"]["path"]))
    try:
        yield fns_server.Chat(), fns_server.normalize_messages
    finally:
        setattr(fns_server, "CHAT_TEMPLATE", old_template)
        if previous is None:
            os.environ.pop("FNS_TOKENIZER", None)
        else:
            os.environ["FNS_TOKENIZER"] = previous
        sys.path.remove(str(ROOT / "serve"))


def verify_local_inputs(cfg):
    with local_chat(cfg) as (chat, normalize):
        for row in cfg["requests"]:
            require(normalize(row["messages"]) == row["messages"] == normalize(normalize(row["messages"])),
                    "Prepared normalization is not idempotent", "token_parity")
            require(chat.prompt_ids(row["messages"], None) == row["tokens"],
                    "Prepared input differs from current FNS template/tokenizer", "token_parity")
        warmup_tokens = chat.prompt_ids(cfg["warmup"]["messages"], None)
        require(len(warmup_tokens) + cfg["warmup"]["max_tokens"] <= cfg["ctx"], "Warmup exceeds context")
        return len(warmup_tokens)


def server_spec(cfg, quant, engine, spec, spec_sample, raw_prefix, portable=False):
    model = cfg["models"][quant]
    slots = {key: cfg[key]["path"] for key in ("fns_binary", "llama_binary", "fns_server", "tokenizer", "template", "vocab")}
    slots.update({"python": sys.executable, "model": model["artifact"]["path"], "ctx": str(cfg["ctx"]),
                  "profile": model.get("profile", {}).get("path", ""), "metrics": str(raw_prefix) + ".metrics.jsonl",
                  "port": str(cfg["ports"][engine])})
    argv = list(cfg["launchers"][engine])
    if engine == "fns":
        argv += ["--model", "{model}", "--ctx", "{ctx}", "--engine", "{fns_binary}", "--vocab", "{vocab}",
                 "--port", "{port}", "--spec", str(spec), "--spec-sample", str(spec_sample),
                 "--kv-global", "f16", "--log", "{metrics}", "--log-tokens"]
        if quant == "q4":
            argv += ["--expert-profile", "{profile}"]
    else:
        argv += ["--model", "{model}", "--ctx-size", "{ctx}", "--parallel", "1", "--port", "{port}",
                 "--host", "127.0.0.1", "--jinja", "--chat-template-file", "{template}", "--metrics", "--slots"]
    argv += model[engine + "_argv"]
    if not portable:
        argv = [a.format_map(slots) for a in argv]
    env = {**os.environ, **cfg["environment"]}
    # Engine inherits explicit caller model/tokenizer paths; never replace FNS_MODELS with a repo default.
    env["FNS_TOKENIZER"] = cfg["tokenizer"]["path"]
    url = f"http://127.0.0.1:{cfg['ports'][engine]}"
    return argv, env, url, slots["metrics"]


def make_server(cfg, quant, engine, spec, spec_sample, prefix):
    argv, env, url, metrics = server_spec(cfg, quant, engine, spec, spec_sample, prefix)
    return OwnedServer(argv, env, str(ROOT), url, str(prefix) + ".server.log",
                       cfg["timeouts"]["startup"], cfg["timeouts"]["shutdown"]), metrics


def verify_llama_inputs(cfg, quant, raw_dir):
    server, _ = make_server(cfg, quant, "llama", 0, 0, raw_dir / f"parity-{quant}")
    records = []
    try:
        server.start()
        with local_chat(cfg) as (chat, _):
            for row in cfg["requests"]:
                body = {"messages": row["messages"], "add_generation_prompt": True}
                rendered = json_request(server.url, "/apply-template", body, cfg["timeouts"]["request"])
                text = rendered.get("prompt")
                expected = chat.template.render(messages=row["messages"], tools=None, add_generation_prompt=True)
                require(text == expected, "llama/FNS rendered template mismatch", "token_parity")
                tokenized = json_request(server.url, "/tokenize", {"content": text, "add_special": False,
                                          "parse_special": True}, cfg["timeouts"]["request"])
                require(tokenized.get("tokens") == row["tokens"], "llama/FNS exact token-ID mismatch", "token_parity")
                records.append({"request_id": row["request_id"], "prompt_sha256": row["prompt_sha256"], "identical": True})
    finally:
        server.stop()
        save_json(raw_dir / f"parity-{quant}.json", {"requests": records, "startup_ms": server.startup_ms,
                                                      "shutdown_ms": server.shutdown_ms})


def effective_options(cfg, mode):
    sampling = cfg["sampling"][mode]
    return {"ctx": cfg["ctx"], "max_tokens": cfg["max_tokens"], "ignore_eos": True,
            "cache_prompt": False, "grammar": False, **sampling, **NEUTRAL,
            "spec": {"A": 0, "B": 3}, "spec_sample": {"A": 0, "B": 2}}


def request_body(cfg, messages, mode, seed, max_tokens=None):
    return {"messages": messages, "stream": True, "stream_options": {"include_usage": True},
            "max_tokens": cfg["max_tokens"] if max_tokens is None else max_tokens,
            "ignore_eos": True, "cache_prompt": False, "seed": seed,
            **{k: v for k, v in cfg["sampling"][mode].items() if k != "seeds"}, **NEUTRAL}


def _graphics_path(value):
    require(isinstance(value, str) and value == value.strip() and bool(value)
            and not any(c in value for c in '*?%${}\x00"')
            and PureWindowsPath(value).is_absolute()
            and not any(part in (".", "..") for part in re.split(r"[\\/]", value)),
            "GPU graphical allowlist requires literal full Windows executable paths", "environment")
    return ntpath.normcase(ntpath.normpath(value))


def validate_gpu_process_policy(raw=None):
    if raw is None:
        return {"mode": "strict", "allowed_graphics_paths": []}
    require(isinstance(raw, dict) and set(raw) <= {"mode", "allowed_graphics_paths"}
            and raw.get("mode") in ("strict", "wddm_desktop"), "Unknown GPU process policy", "environment")
    paths = raw.get("allowed_graphics_paths", [])
    require(isinstance(paths, list), "GPU graphical allowlist must be an array", "environment")
    normalized = [_graphics_path(path) for path in paths]
    require(len(set(normalized)) == len(paths), "Duplicate GPU graphical allowlist path", "environment")
    require(bool(paths) if raw["mode"] == "wddm_desktop" else not paths,
            "Desktop policy requires explicit paths; strict policy forbids exceptions", "environment")
    return {"mode": raw["mode"], "allowed_graphics_paths": list(paths)}


def _desktop_processes(smi, apps, fields, policy):
    require(fields[10] == "WDDM", "Desktop GPU process policy requires WDDM", "environment")
    xml = command_output([smi, "-q", "-x"])
    try:
        root = ET.fromstring(xml)
    except ET.ParseError as exc:
        raise BenchError("environment", "Invalid nvidia-smi process XML") from exc
    gpus = root.findall("gpu")
    require(root.tag == "nvidia_smi_log" and len(gpus) == 1,
            "Unknown nvidia-smi GPU XML schema", "environment")
    gpu = gpus[0]
    require(gpu.findtext("product_name") == fields[0] and root.findtext("driver_version") == fields[1]
            and gpu.findtext("driver_model/current_dm") == "WDDM",
            "GPU XML identity or driver model differs from current query", "environment")
    groups = gpu.findall("processes")
    require(len(groups) == 1 and not (groups[0].text or "").strip(),
            "Missing or unknown GPU process XML schema", "environment")
    allowed = {_graphics_path(path) for path in policy["allowed_graphics_paths"]}
    records, by_pid = [], {}
    for node in groups[0]:
        require(node.tag == "process_info" and all(len(node.findall(key)) == 1 for key in ("pid", "process_name", "type"))
                and all(child.tag in {"pid", "process_name", "type", "gpu_instance_id", "compute_instance_id", "used_memory"} for child in node),
                "Unknown GPU process record schema", "environment")
        pid, path, kind = (node.findtext(key) or "" for key in ("pid", "process_name", "type"))
        require(isinstance(pid, str) and re.fullmatch(r"[0-9]+", pid) is not None and int(pid) > 0
                and int(pid) not in by_pid, "Missing or duplicate GPU process PID", "environment")
        require(kind in ("G", "C+G"), "Pure compute or unknown GPU process type", "environment")
        require(_graphics_path(path) in allowed, "Unlisted GPU graphical process", "environment")
        record = {"pid": int(pid), "path": path, "type": kind}
        records.append(record)
        by_pid[int(pid)] = record
    queried = {}
    if apps.strip() and apps.strip().lower() != "no running processes found":
        for line in apps.splitlines():
            parts = line.split(",", 1)
            require(len(parts) == 2 and re.fullmatch(r"[0-9]+", parts[0].strip()) is not None,
                    "Unknown GPU compute-process query schema", "environment")
            pid, path = int(parts[0].strip()), parts[1].strip()
            require(pid > 0 and pid not in queried and pid in by_pid
                    and _graphics_path(path) == _graphics_path(by_pid[pid]["path"]),
                    "GPU process query and XML PID/path mismatch", "environment")
            queried[pid] = path
    require(set(queried) == {record["pid"] for record in records if record["type"] == "C+G"},
            "GPU compute-process query and typed XML membership differ", "environment")
    return records


def inspect_machine(cfg):
    """Observed state and predeclared invalidation thresholds, before every arm."""
    policy = validate_gpu_process_policy(cfg.get("gpu_process_policy"))
    smi = shutil.which("nvidia-smi")
    require(smi is not None, "nvidia-smi unavailable", "environment")
    query = "name,driver_version,memory.free,memory.total,temperature.gpu,utilization.gpu,pstate,power.limit,pcie.link.gen.current,pcie.link.width.current,driver_model.current,display_active"
    raw = command_output([smi, "--query-gpu=" + query, "--format=csv,noheader,nounits"])
    lines = raw.splitlines()
    require(len(lines) == 1, "Exactly one visible GPU is required", "environment")
    fields = [s.strip() for s in lines[0].split(",")]
    require(len(fields) == 12, "Unexpected GPU telemetry schema", "environment")
    rules = cfg["environment_constraints"]
    require(float(fields[2]) >= rules["min_free_vram_mib"], "Insufficient free VRAM", "environment")
    require(float(fields[4]) <= rules["max_gpu_temperature_c"], "GPU temperature threshold exceeded", "environment")
    require(float(fields[5]) <= rules["max_idle_gpu_utilization_percent"], "Concurrent GPU load detected", "environment")
    # Strict keeps the original compute-process rule; desktop opt-in additionally
    # authenticates every graphical context against typed XML and current PIDs.
    apps = command_output([smi, "--query-compute-apps=pid,process_name", "--format=csv,noheader,nounits"])
    permitted = []
    if policy["mode"] == "strict":
        require(not apps.strip() or apps.strip().lower() == "no running processes found", "Other GPU compute processes present", "environment")
    else:
        permitted = _desktop_processes(smi, apps, fields, policy)
    result = {"os": platform.platform(), "python": sys.version, "architecture": platform.machine(),
              "cpu": platform.processor(), "gpu": fields[0], "driver": fields[1], "gpu_state": dict(zip(query.split(","), fields)),
              "compute_processes": apps, "declared": cfg["machine"], "gpu_process_policy": policy,
              "permitted_graphics_processes": permitted}
    if os.name == "nt":
        state = command_output(["powershell.exe", "-NoProfile", "-NonInteractive", "-Command",
                                "$c=Get-CimInstance Win32_ComputerSystem; $o=Get-CimInstance Win32_OperatingSystem; "
                                "$p=Get-CimInstance Win32_Processor; @{ram_bytes=[long]$c.TotalPhysicalMemory; "
                                "free_ram_kib=[long]$o.FreePhysicalMemory;cpu=$p.Name;os=$o.Caption;version=$o.Version} | ConvertTo-Json -Compress"])
        result.update(json.loads(state))
        result["power_plan"] = command_output(["powercfg", "/getactivescheme"])
    else:
        result["ram_bytes"] = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    return result


def source_snapshot(source, destination):
    root = Path(source["path"])
    files = subprocess.run(["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"], cwd=root,
                           capture_output=True, check=True).stdout.decode("utf-8").split("\0")
    entries = []
    with zipfile.ZipFile(destination, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for relative in sorted(set(files)):
            if not relative or Path(relative).parts[0] in {"perso", "external", "models", "build", ".venv", ".git"}:
                continue
            path = root / relative
            if path.name.lower() in {".env", "credentials.txt"} or any(p.startswith(".venv") for p in path.parts):
                continue
            if not path.is_file():
                continue
            data = path.read_bytes()
            entries.append([relative, hashlib.sha256(data).hexdigest()])
            info = zipfile.ZipInfo(relative, (1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, data)
    dirty = command_output(["git", "status", "--porcelain", "--untracked-files=normal"], root)
    patch = subprocess.run(["git", "diff", "--binary", "HEAD"], cwd=root, capture_output=True, check=True).stdout
    return {**source, "dirty": bool(dirty), "status": dirty, "patch_sha256": hashlib.sha256(patch).hexdigest(),
            "build_recipe_sha256": content_hash(source["build_recipe"]),
            "snapshot_sha256": digest(destination), "source_tree_sha256": content_hash(entries),
            "snapshot": str(destination), "snapshot_files": len(entries)}


def create_manifest(cfg, directory, quant, comparison, mode):
    raw = directory / "raw"
    sources = {name: source_snapshot(source, raw / (name + "-source.zip")) for name, source in cfg["sources"].items()}
    machine = inspect_machine(cfg)
    arms = {"A": "llama" if comparison == "service" else "fns", "B": "fns"}
    options = {arm: server_spec(cfg, quant, engine, 0 if arm == "A" else 3,
                               0 if arm == "A" else 2, raw / "symbolic", portable=True)[0]
               for arm, engine in arms.items()}
    hashes = {name: cfg[name]["sha256"] for name in ("tokenizer", "template", "protocol", "dependencies", "fns_binary", "llama_binary")}
    hashes.update({"model": cfg["models"][quant]["artifact"]["sha256"],
                   "profile": cfg["models"][quant].get("profile", {}).get("sha256"), "source_lock": cfg["source_lock"]["sha256"]})
    expected = [{k: r[k] for k in ("request_id", "config", "category", "prompt_tokens")} for r in cfg["requests"]]
    return {"schema_version": 1, "run_id": directory.name, "protocol_name": "FNS-SPEED v1", "profile": cfg["preparation"]["profile"],
            "quant": quant, "comparison": comparison, "mode": mode, "seeds": cfg["sampling"][mode]["seeds"],
            "pairs": cfg["pairs"], "max_tokens": cfg["max_tokens"], "ctx": cfg["ctx"],
            "coverage": cfg["preparation"]["coverage"], "expected_request_ids": [r["request_id"] for r in expected],
            "expected_requests": expected, "protocol_sha256": cfg["protocol"]["sha256"],
            "preparation_sha256": cfg["preparation_sha256"], "requests_sha256": cfg["preparation"]["requests_sha256"],
            "source_lock_sha256": cfg["source_lock"]["sha256"], "campaign_sha256": cfg["campaign_sha256"],
            "selection_sha256": cfg["preparation"]["selection_sha256"],
            "reproducible": bool(cfg["preparation"]["reproducible"] and all(
                not s["dirty"] or (s.get("snapshot_repository") and s.get("published_snapshot_sha256") == s["snapshot_sha256"])
                for s in sources.values())),
            "effective_options": effective_options(cfg, mode), "warmup": cfg["warmup"], "warmup_sha256": cfg["warmup_sha256"],
            "pair_order": ["A", "B"], "arms": arms, "machine": machine, "sources": sources,
            "artifacts": {**{key: cfg[key] for key in ("tokenizer", "template", "vocab", "fns_binary", "llama_binary", "fns_server", "dependencies", "dlls")},
                          "model": cfg["models"][quant]},
            "environment": {key: value for key, value in os.environ.items() if key.startswith(("FNS_", "CUDA_", "HF_"))},
            "configured_environment": cfg["environment"], "environment_constraints": cfg["environment_constraints"],
            "gpu_process_policy": machine["gpu_process_policy"],
            "python": {"executable": sys.executable, "version": sys.version,
                       "packages": {d.metadata["Name"]: d.version for d in importlib.metadata.distributions() if d.metadata["Name"]}},
            "public_provenance": {"commits": {k: v["commit"] for k, v in sources.items()}, "hashes": hashes,
                                  "sources": {k: {"dirty": s["dirty"], **{h: s[h] for h in ("patch_sha256", "snapshot_sha256", "source_tree_sha256")},
                                                  "build_recipe_sha256": content_hash(s["build_recipe"])} for k, s in sources.items()},
                                  "machine": {**{k: machine[k] for k in ("os", "gpu", "cpu", "ram_bytes", "driver")},
                                              **{k: cfg["machine"][k] for k in ("cuda", "msvc")}},
                                  "options": {**options, "fns": options["B"], "llama": server_spec(cfg, quant, "llama", 0, 0, raw / "symbolic", portable=True)[0]}}}


def verify_metrics(row, capture, engine, metrics_path, correlation, spec):
    if engine == "fns":
        logs = [r for r in jsonl(metrics_path) if r.get("request_id") == correlation]
        require(len(logs) == 1, "Missing or duplicate correlated native FNS record", "metrics")
        log = logs[0]
        require(log.get("prompt_tokens") == row["prompt_tokens"] and log.get("completion_tokens") == row["completion_tokens"],
                "Native FNS usage mismatch", "usage")
        require(log.get("reason") == "length", "Native FNS fixed budget did not terminate by length", "usage")
        reused = log.get("reused_prefix")
        tokens = log.get("token_ids")
        require(isinstance(tokens, list) and len(tokens) == row["completion_tokens"]
                and all(type(t) is int and t >= 0 for t in tokens), "Missing complete native output token IDs", "metrics")
        row["output_tokens_sha256"] = content_hash(tokens)
        metrics = {k: finite_number(log.get(k), k, integer=k.startswith("spec_") or k == "decode_iterations")
                   for k in ("prefill_ms", "decode_ms", "spec_windows", "spec_drafted", "spec_accepted", "spec_emitted", "decode_iterations")}
        require(metrics["spec_accepted"] <= metrics["spec_drafted"] and metrics["spec_emitted"] <= row["completion_tokens"] - 1,
                "Invalid speculation counters", "metrics")
        if spec == 0:
            require(all(metrics[k] == 0 for k in ("spec_windows", "spec_drafted", "spec_accepted", "spec_emitted")),
                    "Speculative counters nonzero in disabled arm", "metrics")
        row["prefilled_tokens"] = log.get("prefilled_tokens", log["prompt_tokens"] - reused if type(reused) is int else None)
        require(metrics["decode_iterations"] <= row["completion_tokens"] - 1,
                "Decode iteration count exceeds generated-token steps", "metrics")
    else:
        usage = capture.usage
        reused = usage.get("prompt_tokens_details", {}).get("cached_tokens")
        timing = capture.timings
        require(isinstance(timing, dict), "Missing llama terminal native timing counters", "metrics")
        prefilled = timing.get("prompt_n")
        require(type(prefilled) is int and prefilled == row["prompt_tokens"], "llama prefilled token count mismatch", "kv_reuse")
        require(timing.get("predicted_n") == row["completion_tokens"], "llama generated count mismatch", "usage")
        row["prefilled_tokens"] = prefilled
        metrics = {"prefill_ms": finite_number(timing.get("prompt_ms"), "prompt_ms"),
                   "decode_ms": finite_number(timing.get("predicted_ms"), "predicted_ms"),
                   **{k: None for k in ("spec_windows", "spec_drafted", "spec_accepted", "spec_emitted", "decode_iterations")}}
    require(type(reused) is int and reused == 0, "KV reuse must be explicitly observed zero", "kv_reuse")
    require(row["prefilled_tokens"] == row["prompt_tokens"], "Full prefill count mismatch", "kv_reuse")
    row["reused_tokens"] = reused
    row["metrics"] = metrics


def empty_attempt(manifest, request, seed, pair, arm, attempt, replaces):
    row = {"schema_version": 1, "run_id": manifest["run_id"], "request_id": request["request_id"],
           "config": request["config"], "category": request["category"], "quant": manifest["quant"],
           "comparison": manifest["comparison"], "mode": manifest["mode"], "seed": seed, "pair": pair,
           "arm": arm, "attempt": attempt, "replaces": replaces, "status": "error", "error_code": None,
           "effective_options": manifest["effective_options"]}
    row["gpu_process_policy"] = manifest.get("gpu_process_policy", {"mode": "strict", "allowed_graphics_paths": []})
    for key in ("latency_ms", "ttft_visible_ms", "tpot_client_ms", "completion_tokens", "prompt_tokens", "reused_tokens",
                "finish_reason", "output_tokens_sha256", "startup_ms", "warmup_ms", "shutdown_ms"):
        row[key] = None
    row["delta_intervals_ms"] = []
    row["metrics"] = {}
    return row


def measure_arm(cfg, manifest, request, seed, pair, arm, attempt, replaces, directory, warmup_prompt_tokens):
    row = empty_attempt(manifest, request, seed, pair, arm, attempt, replaces)
    engine = manifest["arms"][arm]
    spec = 0 if arm == "A" else 3
    spec_sample = 0 if arm == "A" else 2
    key = f"{request['request_id']}-{seed}-{pair}-{arm}-{attempt}"
    prefix = directory / "raw" / key
    server, metrics = make_server(cfg, manifest["quant"], engine, spec, spec_sample, prefix)
    sampler = None
    measured_capture = None
    try:
        row["environment_before"] = inspect_machine(cfg)
        row["argv"] = server.argv
        server.start()
        row["startup_ms"] = server.startup_ms
        smi = shutil.which("nvidia-smi")
        if smi:
            sampler = GPUStateSampler(smi, str(prefix) + ".gpu.jsonl")
            sampler.start()
        warmup = cfg["warmup"]
        began = time.perf_counter()
        stream_request(server.url, request_body(cfg, warmup["messages"], "greedy", 0, warmup["max_tokens"]),
                       key + ".warmup", warmup_prompt_tokens, warmup["max_tokens"], cfg["timeouts"]["request"], str(prefix) + ".warmup.sse")
        row["warmup_ms"] = (time.perf_counter() - began) * 1000
        result, capture = stream_request(server.url, request_body(cfg, request["messages"], manifest["mode"], seed),
                                         key, request["prompt_tokens"], cfg["max_tokens"],
                                         cfg["timeouts"]["request"], str(prefix) + ".sse")
        row.update(result)
        measured_capture = capture
        verify_metrics(row, capture, engine, metrics, key, spec)
        smi = shutil.which("nvidia-smi")
        if smi:
            row["gpu_after_request"] = command_output([smi, "--query-gpu=memory.used,memory.free,pcie.link.gen.current,pcie.link.width.current,temperature.gpu,pstate,power.draw", "--format=csv,noheader,nounits"])
        row["status"] = "ok"
    except KeyboardInterrupt:
        row.update(error_code="interrupted", error="Interrupted by operator")
    except Exception as exc:
        row.update(error_code=exc.code if isinstance(exc, BenchError) else "other", error=f"{type(exc).__name__}: {exc}")
    finally:
        try:
            server.stop()
        except Exception as exc:
            row.update(status="error", error_code="process", error=f"Shutdown: {exc}")
        row["shutdown_ms"] = server.shutdown_ms
        if sampler is not None:
            try:
                sampler.stop()
                if sampler.error is not None:
                    raise BenchError("environment", "GPU telemetry reader failed: " + sampler.error)
                if measured_capture is not None and measured_capture.done is not None:
                    row["gpu_during_request"] = sampler.summary(measured_capture.started, measured_capture.done)
            except Exception as exc:
                if row["error_code"] not in ("process", "interrupted"):
                    row.update(status="error", error_code="environment", error=f"GPU telemetry: {exc}")
                else:
                    row["telemetry_error"] = str(exc)
    return row


def append_attempts(path, rows):
    with Path(path).open("ab") as stream:
        for row in rows:
            stream.write(canonical(row) + b"\n")
        stream.flush()
        os.fsync(stream.fileno())


def run_campaign(cfg, quant, comparison, mode, resume=None):
    root = Path(cfg["output_dir"])
    root.mkdir(parents=True, exist_ok=True)
    # Check and run share one process-wide GPU ownership lock across output directories.
    lock = acquire_gpu_lock(cfg)
    directory = Path(resume).resolve() if resume else root / (time.strftime("%Y%m%dT%H%M%S") + f"-{quant}-{comparison}-{mode}-" + uuid.uuid4().hex[:8])
    try:
        if resume:
            manifest = load_json(directory / "run-manifest.json")
            require((manifest["quant"], manifest["comparison"], manifest["mode"], manifest["campaign_sha256"]) ==
                    (quant, comparison, mode, cfg["campaign_sha256"]), "Resume configuration differs from original campaign")
            previous = jsonl(directory / "attempts.jsonl") if (directory / "attempts.jsonl").exists() else []
            for name, source in cfg["sources"].items():
                snapshot = source_snapshot(source, directory / "raw" / (name + "-resume-" + uuid.uuid4().hex + ".zip"))
                require(snapshot["source_tree_sha256"] == manifest["sources"][name]["source_tree_sha256"],
                        "Cannot resume with modified engine source")
        else:
            directory.mkdir()
            (directory / "raw").mkdir()
            manifest = create_manifest(cfg, directory, quant, comparison, mode)
            save_json(directory / "run-manifest.json", manifest)
            (directory / "attempts.jsonl").touch(exist_ok=False)
            previous = []
            shutil.copyfile(cfg["campaign_path"], directory / "campaign.json")
            shutil.copyfile(cfg["source_lock"]["path"], directory / "source-lock.json")
            for name in ("requests.jsonl", "selection.jsonl", "preparation.json"):
                shutil.copyfile(Path(cfg["prepared_dir"]) / name, directory / name)
        try:
            warmup_prompt_tokens = verify_local_inputs(cfg)
            # Exact parity covers every selected input on the fingerprinted llama instance.
            parity_dir = directory / "raw" / ("check-" + uuid.uuid4().hex)
            parity_dir.mkdir()
            verify_llama_inputs(cfg, quant, parity_dir)
        except BaseException as exc:
            manifest.setdefault("preflight_failures", []).append({"error_code": getattr(exc, "code", "other"), "error": str(exc)})
            save_json(directory / "run-manifest.json", manifest)
            raise
        had_error = False
        observed_order = len(previous)
        for request in cfg["requests"]:
            for seed in manifest["seeds"]:
                for pair in range(cfg["pairs"]):
                    existing = [r for r in previous if (r["request_id"], r["seed"], r["pair"]) == (request["request_id"], seed, pair)]
                    latest = {arm: max((r for r in existing if r["arm"] == arm), key=lambda r: r["attempt"], default=None) for arm in ("A", "B")}
                    if all(r and r["status"] == "ok" for r in latest.values()):
                        continue
                    # Retry both adjacent arms explicitly; never mix a new arm with an old thermal state.
                    rows = []
                    generation = max((r.get("pair_generation", 0) for r in existing), default=-1) + 1
                    for arm in ("A", "B"):
                        old = latest[arm]
                        row = measure_arm(cfg, manifest, request, seed, pair, arm, old["attempt"] + 1 if old else 0,
                                          old["attempt"] if old else None, directory, warmup_prompt_tokens)
                        row["pair_generation"] = generation
                        rows.append(row)
                        row["observed_order"] = observed_order
                        observed_order += 1
                        # Persist A before starting B: interrupted pairs remain visible and resumable.
                        append_attempts(directory / "attempts.jsonl", [row])
                        if row["error_code"] in ("process", "interrupted"):
                            return directory, False
                    if comparison == "speculation" and all(r["status"] == "ok" for r in rows):
                        if rows[0]["output_tokens_sha256"] != rows[1]["output_tokens_sha256"]:
                            # Append linked invalidations, preserving the original evidence and exact outputs.
                            for row in rows:
                                invalid = {**row, "attempt": row["attempt"] + 1, "replaces": row["attempt"],
                                           "status": "error", "error_code": "spec_parity", "error": "FNS sequential/speculative token sequence mismatch"}
                                invalid["observed_order"] = observed_order
                                append_attempts(directory / "attempts.jsonl", [invalid])
                                observed_order += 1
                            had_error = True
                    had_error |= any(r["status"] != "ok" for r in rows)
                    previous.extend(rows)
        return directory, not had_error
    finally:
        lock.unlink()


def acquire_gpu_lock(cfg):
    lock = Path(os.environ.get("TEMP", os.environ.get("TMP", str(ROOT / "perso")))) / "fns-speed-gpu.lock"
    lock.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise BenchError("process", f"Campaign lock exists: {lock}; inspect owner before removing stale lock") from exc
    try:
        os.write(descriptor, canonical({"pid": os.getpid(), "campaign": cfg["campaign_path"]}))
    finally:
        os.close(descriptor)
    return lock


def check_campaign(cfg):
    root = Path(cfg["output_dir"])
    directory = root / ("check-" + time.strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:8])
    directory.mkdir(parents=True)
    lock = acquire_gpu_lock(cfg)
    results = {"schema_version": 1, "campaign_sha256": cfg["campaign_sha256"],
               "requests": len(cfg["requests"]), "status": "error"}
    try:
        results["warmup_prompt_tokens"] = verify_local_inputs(cfg)
        inspect_machine(cfg)
        for quant in ("q3", "q4"):
            verify_llama_inputs(cfg, quant, directory)
        results["status"] = "ok"
    except BaseException as exc:
        results.update(error_code=getattr(exc, "code", "other"), error=str(exc))
        raise
    finally:
        save_json(directory / "check.json", results)
        lock.unlink()
    return directory


def main(argv=None):
    parser = argparse.ArgumentParser(description="FNS-SPEED v1 single-turn, concurrency-one paired HTTP benchmark")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("check", "run"):
        sub = commands.add_parser(name)
        sub.add_argument("--campaign", required=True, type=Path)
        if name == "run":
            sub.add_argument("--quant", choices=("q3", "q4"), required=True)
            sub.add_argument("--comparison", choices=("service", "speculation"), required=True)
            sub.add_argument("--mode", choices=("greedy", "sample"), default="greedy")
            sub.add_argument("--resume", type=Path, help="Explicitly retry failed/missing adjacent pairs, retaining linked history")
    report = commands.add_parser("report")
    report.add_argument("--run-dir", required=True, type=Path)
    report.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == "report":
            from speed_report import generate_report
            generate_report(args.run_dir, args.out)
            return 0
        cfg = validate_campaign(args.campaign)
        if args.command == "check":
            print(check_campaign(cfg))
            return 0
        directory, ok = run_campaign(cfg, args.quant, args.comparison, args.mode, args.resume)
        print(directory)
        return 0 if ok else 1
    except (BenchError, OSError, ValueError, KeyError) as exc:
        print(f"SPEED {getattr(exc, 'code', 'configuration')}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
