"""Validate private FNS-SPEED campaigns and export aggregate, positive-schema reports.

No dataset rows, identifiers, raw argv, arbitrary provenance text, or timestamps are
copied into the public result. A report is a local review artifact, not an upload.
"""
from __future__ import annotations

import hashlib
import json
import math
import ntpath
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path, PureWindowsPath
from typing import NoReturn

CONFIGS = ("qualitative", "throughput_1k", "throughput_2k", "throughput_8k", "throughput_16k", "throughput_32k")
CATEGORIES = {config: (("coding", "stem", "humanities", "math", "multilingual", "qa", "rag", "roleplay", "reasoning", "summarization", "writing") if config == "qualitative" else ("high_entropy", "mixed", "low_entropy")) for config in CONFIGS}
ERROR_CODES = frozenset(("startup", "timeout", "http", "stream", "usage", "kv_reuse", "token_parity", "spec_parity", "metrics", "environment", "process", "interrupted", "other"))
COUNTERS = ("spec_windows", "spec_drafted", "spec_accepted", "spec_emitted", "decode_iterations")
HEX64 = re.compile(r"[0-9a-f]{64}\Z")
HEX40 = re.compile(r"[0-9a-f]{40}\Z")
EXECUTION_PROTOCOL = {"schema_version": 2, "name": "FNS-SPEED v2", "block_size": 32,
                      "block_order": "config-category-prepared-order", "arm_order": "alternating-block-repetition",
                      "server_lifetime": "block-arm", "warmup": "once-per-session", "cache_prompt": False}


class ReportError(ValueError):
    """The private inputs cannot support a complete, trustworthy report."""


def _fail(message) -> NoReturn:
    raise ReportError(message)


def _require(condition, message):
    if not condition:
        _fail(message)


def _integer(value, minimum=0):
    return type(value) is int and value >= minimum


def _number(value, minimum=0, positive=False):
    return type(value) in (int, float) and math.isfinite(value) and (value > minimum if positive else value >= minimum)


def _hash(value):
    return isinstance(value, str) and HEX64.fullmatch(value) is not None


def _canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def _load(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"), parse_constant=lambda _: (_ for _ in ()).throw(ValueError()))
    except (OSError, UnicodeError, ValueError) as exc:
        raise ReportError("Unreadable or invalid campaign JSON") from exc


def _nearest(values, percentile):
    return sorted(values)[max(0, math.ceil(len(values) * percentile) - 1)] if values else None


def _ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def _nullable_ratio(numerator, denominator):
    return {"value": _ratio(numerator, denominator), "reason": None if denominator else "zero_denominator"}


# Values are parsed and reserialized; arbitrary option strings never reach output.
_PATH_FLAGS = {"--model": "{model}", "-m": "{model}", "--engine": "{fns_binary}", "--vocab": "{vocab}", "--expert-profile": "{profile}", "--chat-template-file": "{template}", "--log": "{metrics}"}
_INT_FLAGS = frozenset(("--ctx", "--ctx-size", "-c", "--spec", "--spec-sample", "--parallel", "--n-gpu-layers", "-ngl", "--n-cpu-moe", "--moe-cache-mib", "--ubatch", "--rebalance", "--threads", "--threads-batch", "-t", "-tb", "--batch-size", "--ubatch-size", "-b", "-ub", "--gpu-layers", "--cache-ram", "--main-gpu"))
_BOOL_FLAGS = frozenset(("--jinja", "--metrics", "--slots", "--log-tokens", "--no-mmap", "--mlock", "--no-warmup", "--no-webui", "--no-context-shift", "--swa-full"))
_ENUM_FLAGS = {"--kv-global": ("f16",), "--cache-type-k": ("f16",), "--cache-type-v": ("f16",), "-ctk": ("f16",), "-ctv": ("f16",), "--flash-attn": ("on", "off", "auto"), "-fa": ("on", "off", "auto"), "--host": ("127.0.0.1", "localhost"), "--fit": ("on", "off")}


def _portable_options(argv):
    _require(isinstance(argv, list) and all(isinstance(x, str) for x in argv), "Engine options must be argv arrays")
    result, seen, index = [], set(), 0
    while index < len(argv):
        flag = argv[index]
        _require(flag not in seen, "Duplicate engine option")
        seen.add(flag)
        index += 1
        if flag in _BOOL_FLAGS:
            result.append(flag)
            continue
        _require(index < len(argv), "Engine option missing its value")
        value = argv[index]
        index += 1
        if flag in _PATH_FLAGS:
            _require(bool(value), "Missing engine artifact option")
            safe = _PATH_FLAGS[flag]
        elif flag == "--port":
            _require(value == "{port}" or (value.isascii() and value.isdigit() and 0 < int(value) < 65536), "Invalid engine port")
            safe = "{port}"
        elif flag in _INT_FLAGS:
            if value == "{ctx}" and flag in ("--ctx", "--ctx-size", "-c"):
                safe = "{ctx}"
            else:
                _require(re.fullmatch(r"-?[0-9]{1,12}", value) is not None, "Non-numeric engine option")
                safe = str(int(value))
        elif flag == "--device":
            _require(re.fullmatch(r"CUDA[0-9]{1,2}(?:,CUDA[0-9]{1,2})*", value) is not None, "Unsupported engine device")
            safe = ",".join("CUDA" + str(int(part[4:])) for part in value.split(","))
        elif flag in _ENUM_FLAGS:
            _require(value in _ENUM_FLAGS[flag], "Unsupported engine option value")
            safe = value
        else:
            raise ReportError("Unknown engine option; add an explicit typed export rule")
        result.extend((flag, safe))
    return result


def _option_map(argv):
    result, index = {}, 0
    aliases = {"-m": "--model", "-c": "--ctx-size", "-ngl": "--n-gpu-layers", "--gpu-layers": "--n-gpu-layers", "-ctk": "--cache-type-k", "-ctv": "--cache-type-v", "-fa": "--flash-attn", "-t": "--threads", "-tb": "--threads-batch", "-b": "--batch-size", "-ub": "--ubatch-size"}
    while index < len(argv):
        flag = argv[index]
        index += 1
        key = aliases.get(flag, flag)
        _require(key not in result, "Duplicate engine option alias")
        result[key] = True if flag in _BOOL_FLAGS else argv[index]
        index += flag not in _BOOL_FLAGS
    return result


def _arm_options(manifest, raw):
    portable, maps, launchers = {}, {}, {}
    for arm in ("A", "B"):
        argv = raw[arm]
        _require(isinstance(argv, list) and all(isinstance(value, str) for value in argv), "Engine options must be string arrays")
        launcher = ["{python}", "{fns_server}"] if arm == "B" or manifest["comparison"] == "speculation" else ["{llama_binary}"]
        prefix = launcher if argv[:len(launcher)] == launcher else []
        _require(bool(prefix) or not argv or argv[0].startswith("--"), "Nonportable engine launcher")
        portable[arm] = _portable_options(argv[len(prefix):])
        maps[arm] = _option_map(portable[arm])
        launchers[arm] = prefix
    for arm, options in maps.items():
        _require(options.get("--model") == "{model}", "Missing model option")
        if arm == "B" or manifest["comparison"] == "speculation":
            expected = {"--ctx": "{ctx}", "--spec": "0" if arm == "A" else "3", "--spec-sample": "0" if arm == "A" else "2", "--kv-global": "f16", "--engine": "{fns_binary}", "--vocab": "{vocab}", "--log": "{metrics}", "--log-tokens": True}
            for key, value in expected.items():
                actual = options.get(key)
                _require(actual == value or (key == "--ctx" and actual == str(manifest["ctx"])), "Missing or incompatible FNS option")
            _require(manifest["quant"] != "q4" or options.get("--expert-profile") == "{profile}", "Missing Q4 routing profile option")
        else:
            _require(options.get("--ctx-size") in ("{ctx}", str(manifest["ctx"])), "Missing llama context option")
            _require(options.get("--parallel") == "1" and options.get("--jinja") is True and options.get("--chat-template-file") == "{template}", "Missing llama template or concurrency option")
            _require(options.get("--cache-type-k") == "f16" and options.get("--cache-type-v") == "f16", "Missing explicit llama cache precision")
    if manifest["comparison"] == "speculation":
        invariant = {arm: {key: value for key, value in options.items() if key not in ("--spec", "--spec-sample")} for arm, options in maps.items()}
        _require(invariant["A"] == invariant["B"], "Ablation changes options other than speculation")
    return {arm: launchers[arm] + portable[arm] for arm in ("A", "B")}


def _version(value):
    if isinstance(value, str) and re.fullmatch(r"[0-9]{1,6}(?:\.[0-9]{1,6}){0,4}", value):
        return [int(part) for part in value.split(".")]
    return None


def _machine(raw, observed=None):
    _require(isinstance(raw, dict), "Missing machine provenance")
    observed = observed if isinstance(observed, dict) else {}
    ram = raw.get("ram_bytes")
    _require(_integer(ram, 1), "Missing machine RAM size")
    gpu = raw.get("gpu", "")
    match = re.fullmatch(r"(?:NVIDIA )?(?:GeForce )?(RTX|GTX) ([0-9]{3,4})(?: (Ti|SUPER))?", gpu) if isinstance(gpu, str) else None
    safe_gpu = {"vendor": "NVIDIA", "series": match[1], "model": int(match[2]), "variant": match[3]} if match else None
    cpu = raw.get("cpu", "")
    amd = re.fullmatch(r"AMD Ryzen ([3579]) ([0-9]{4,5})(X3D|XT|X|G|GE|U|HS|HX|H)?(?: [0-9]{1,2}-Core Processor)?", cpu) if isinstance(cpu, str) else None
    intel = re.fullmatch(r"(?:[0-9]{1,2}th Gen )?Intel(?:\(R\))? Core(?:\(TM\))? (i[3579])[- ]([0-9]{4,5})(K|KF|F|KS|T|H|HK|HX|U|P)?(?: CPU)?(?: @ [0-9]\.[0-9]{1,2}GHz)?", cpu) if isinstance(cpu, str) else None
    safe_cpu = ({"vendor": "AMD", "family": "Ryzen", "tier": int(amd[1]), "model": int(amd[2]), "suffix": amd[3]} if amd else
                {"vendor": "Intel", "family": "Core", "tier": int(intel[1][1:]), "model": int(intel[2]), "suffix": intel[3]} if intel else None)
    os_value = raw.get("os", "")
    os_match = re.fullmatch(r"(?:Microsoft )?Windows(?: ([0-9]{1,2})(?: (?:Pro(?: for Workstations)?|Home|Enterprise|Education))?)?", os_value) if isinstance(os_value, str) else None
    platform_match = re.fullmatch(r"Windows-([0-9]{1,2})-([0-9]+\.[0-9]+\.[0-9]+)(?:-SP[0-9]+)?", os_value) if isinstance(os_value, str) else None
    os_version = _version(observed.get("version")) or (_version(platform_match[2]) if platform_match else None)
    os_public = {"family": "Windows", "version": int(os_match[1]) if os_match and os_match[1] else int(platform_match[1]) if platform_match else None, "build_version": os_version} if os_match or platform_match else None
    state = observed.get("gpu_state", {})
    state = state if isinstance(state, dict) else {}
    def numeric(key):
        value = state.get(key)
        return float(value) if isinstance(value, str) and re.fullmatch(r"[0-9]{1,9}(?:\.[0-9]{1,6})?", value) else None
    driver_model = state.get("driver_model.current")
    display = state.get("display_active")
    plan = observed.get("power_plan", "")
    plans = {"381b4222-f694-41f0-9685-ff5bb260df2e": "balanced", "8c5e7fda-e8bf-4a96-9a85-a6e23a8c635c": "high_performance", "a1841308-3541-4fab-bc81-f71556f20b4a": "power_saver", "e9a42b02-d5df-448d-aa00-03f14749eb61": "ultimate_performance"}
    plan_kind = next((kind for guid, kind in plans.items() if isinstance(plan, str) and guid in plan.lower()), "custom" if plan else None)
    architecture = observed.get("architecture")
    declared = observed.get("declared", {})
    declared = declared if isinstance(declared, dict) else {}
    build = declared.get("build_architecture")
    memory = declared.get("memory_settings", {})
    memory = memory if isinstance(memory, dict) else {}
    return {"os": os_public, "gpu": safe_gpu, "cpu": safe_cpu, "ram_bytes": ram,
            "architecture": "x64" if architecture in ("AMD64", "x86_64", "x64") else None,
            "build_architecture": "x64" if build in ("AMD64", "x86_64", "x64") else {"cuda_sm": int(build[3:])} if isinstance(build, str) and re.fullmatch(r"sm_[0-9]{2,3}", build) else None,
            "memory_settings": {key: "f16" if memory.get(key) == "f16" else None for key in ("fns_kv", "llama_k", "llama_v")},
            "driver_version": _version(raw.get("driver")), "cuda_version": _version(raw.get("cuda")), "msvc_version": _version(raw.get("msvc")),
            "driver_model": driver_model if driver_model in ("WDDM", "TCC") else None,
            "display_active": True if display == "Enabled" else False if display == "Disabled" else None,
            "power_plan": plan_kind, "vram_free_mib": numeric("memory.free"), "vram_total_mib": numeric("memory.total"),
            "pcie_observed_generation": numeric("pcie.link.gen.current"), "pcie_observed_width": numeric("pcie.link.width.current"),
            "pcie_observation": "pre_request_not_verified_under_load", "power_limit_watts": numeric("power.limit"),
            "gpu_temperature_c": numeric("temperature.gpu"), "gpu_utilization_percent": numeric("utilization.gpu"),
            "pstate": int(state["pstate"][1:]) if isinstance(state.get("pstate"), str) and re.fullmatch(r"P[0-9]{1,2}", state["pstate"]) else None}


def _provenance(manifest):
    raw = manifest.get("public_provenance")
    _require(isinstance(raw, dict), "Missing structured provenance")
    commits = raw.get("commits", {})
    _require(isinstance(commits, dict) and all(isinstance(commits.get(k), str) and HEX40.fullmatch(commits[k]) for k in ("fns", "llama")), "Missing source commits")
    hashes = raw.get("hashes", {})
    required = ("model", "tokenizer", "template", "protocol", "execution_protocol", "source_lock", "dependencies", "fns_binary", "llama_binary")
    _require(isinstance(hashes, dict) and all(_hash(hashes.get(k)) for k in required), "Missing artifact hashes")
    _require(hashes["protocol"] == manifest["protocol_sha256"] and hashes["source_lock"] == manifest["source_lock_sha256"], "Inconsistent provenance hashes")
    _require(hashes["execution_protocol"] == manifest["execution_protocol_sha256"], "Inconsistent execution protocol provenance hash")
    _require(manifest["quant"] != "q4" or _hash(hashes.get("profile")), "Missing Q4 routing profile hash")
    public_hashes = {key: hashes[key] for key in required}
    for key in ("profile", "fns_patch", "llama_patch", "fns_source", "llama_source", "warmup", "selection"):
        value = hashes.get(key)
        _require(value is None or _hash(value), "Invalid optional artifact hash")
        public_hashes[key] = value
    sources = raw.get("sources")
    _require(isinstance(sources, dict), "Missing recoverable source provenance")
    safe_sources = {}
    for engine in ("fns", "llama"):
        source = sources.get(engine)
        _require(isinstance(source, dict) and type(source.get("dirty")) is bool, "Missing source modification status")
        _require(_hash(source.get("source_tree_sha256")) and _hash(source.get("build_recipe_sha256")), "Missing source tree or build recipe hash")
        safe_sources[engine] = {"dirty": source["dirty"], "source_tree_sha256": source["source_tree_sha256"], "build_recipe_sha256": source["build_recipe_sha256"]}
        for key in ("patch_sha256", "snapshot_sha256"):
            value = source.get(key)
            _require(value is None or _hash(value), "Invalid source recovery hash")
            safe_sources[engine][key] = value
        _require(not source["dirty"] or _hash(source.get("snapshot_sha256")), "Modified sources require a recovery snapshot")
    options = raw.get("options", {})
    _require(isinstance(options, dict) and "A" in options and "B" in options, "Missing arm-specific engine options")
    artifacts = manifest.get("artifacts", {})
    _require(isinstance(artifacts, dict), "Invalid artifacts")
    dlls = artifacts.get("dlls", [])
    _require(isinstance(dlls, list) and all(isinstance(item, dict) and _hash(item.get("sha256")) for item in dlls), "Invalid DLL artifact hashes")
    python = manifest.get("python", {})
    _require(isinstance(python, dict) and isinstance(python.get("version", ""), str), "Invalid Python provenance")
    return {"commits": {key: commits[key] for key in ("fns", "llama")}, "hashes": public_hashes, "sources": safe_sources,
            "dll_sha256": sorted(item["sha256"] for item in dlls), "python_version": _version(python.get("version", "").split(" ")[0]),
            "machine": _machine(raw.get("machine"), manifest.get("machine")), "options": _arm_options(manifest, options), "build_recipe_reference": "docs/BENCHMARKING.md"}


def _effective(manifest):
    options = manifest.get("effective_options")
    _require(isinstance(options, dict), "Missing effective generation options")
    mode = manifest["mode"]
    expected = {"ctx": manifest["ctx"], "max_tokens": manifest["max_tokens"], "ignore_eos": True, "cache_prompt": False, "grammar": False,
                "temperature": 0 if mode == "greedy" else 1, "top_p": 1 if mode == "greedy" else .95, "top_k": 0 if mode == "greedy" else 20,
                "presence_penalty": 0, "frequency_penalty": 0, "repeat_penalty": 1, "min_p": 0, "typical_p": 1, "tfs_z": 1,
                "mirostat": 0, "dynatemp_range": 0, "xtc_probability": 0, "top_n_sigma": -1,
                "seeds": manifest["seeds"], "spec": {"A": 0, "B": 3}, "spec_sample": {"A": 0, "B": 2}}
    _require(set(options) == set(expected), "Unexpected or missing generation options")
    # bool and int equality must not make cache_prompt=0 look like explicit false.
    _require(_canonical(options) == _canonical(expected), "Incompatible generation options")
    return expected


def _coverage(manifest):
    rows = manifest.get("coverage")
    _require(isinstance(rows, list) and rows, "Missing coverage")
    result, seen = [], set()
    for row in rows:
        _require(isinstance(row, dict), "Invalid coverage row")
        config, category = row.get("config"), row.get("category")
        _require(config in CONFIGS and category in CATEGORIES[config], "Invalid coverage category")
        key = config, category
        _require(key not in seen, "Duplicate coverage category")
        seen.add(key)
        counts = {key: row.get(key) for key in ("total", "admissible", "selected", "excluded_context")}
        _require(all(_integer(value) for value in counts.values()), "Invalid coverage counts")
        _require(counts["total"] == counts["admissible"] + counts["excluded_context"], "Inconsistent context exclusions")
        limit = 1 if manifest["profile"] == "smoke" else (32 if manifest["profile"] == "release" and config != "qualitative" else counts["admissible"])
        _require(counts["selected"] == min(limit, counts["admissible"]), "Incomplete profile selection")
        lengths = row.get("lengths")
        _require(isinstance(lengths, dict), "Missing length distribution")
        safe_lengths = {key: lengths.get(key) for key in ("min", "max", "p50", "p95")}
        if counts["total"]:
            _require(all(_number(v) for v in safe_lengths.values()), "Invalid length distribution")
            _require(safe_lengths["min"] <= safe_lengths["p50"] <= safe_lengths["p95"] <= safe_lengths["max"], "Inconsistent length distribution")
        else:
            _require(all(v is None for v in safe_lengths.values()), "Nonempty lengths for empty category")
        result.append({"config": config, "category": category, **counts, "lengths": safe_lengths})
    configs = {key[0] for key in seen}
    _require(manifest["profile"] != "release" or configs == set(CONFIGS), "Release requires all six configurations")
    _require(seen == {(config, category) for config in configs for category in CATEGORIES[config]}, "Missing category coverage")
    return sorted(result, key=lambda r: (CONFIGS.index(r["config"]), CATEGORIES[r["config"]].index(r["category"])))


def _gpu_policy(raw=None):
    if raw is None:
        return {"mode": "strict", "allowed_graphics_paths": []}
    _require(isinstance(raw, dict) and set(raw) <= {"mode", "allowed_graphics_paths"}
             and raw.get("mode") in ("strict", "wddm_desktop"), "Unknown GPU process policy")
    paths = raw.get("allowed_graphics_paths", [])
    _require(isinstance(paths, list), "Invalid GPU process allowlist")
    normalized = []
    for value in paths:
        _require(isinstance(value, str) and value == value.strip() and bool(value)
                 and not any(c in value for c in '*?%${}\x00"')
                 and PureWindowsPath(value).is_absolute()
                 and not any(part in (".", "..") for part in re.split(r"[\\/]", value)),
                 "GPU process allowlist requires literal full Windows paths")
        normalized.append(ntpath.normcase(ntpath.normpath(value)))
    _require(len(set(normalized)) == len(paths), "Duplicate GPU process allowlist path")
    _require(bool(paths) if raw["mode"] == "wddm_desktop" else not paths, "Invalid GPU process policy exceptions")
    return {"mode": raw["mode"], "allowed_graphics_paths": list(paths)}


def _gpu_observation(observed, policy):
    if not isinstance(observed, dict):
        _fail("Missing observed GPU process evidence")
    _require(_gpu_policy(observed.get("gpu_process_policy")) == policy,
             "Missing or inconsistent observed GPU process policy")
    records = observed.get("permitted_graphics_processes")
    if not isinstance(records, list):
        _fail("Missing permitted GPU process evidence")
    if policy["mode"] == "strict":
        _require(not records, "Strict GPU policy carries exceptions")
        return 0
    state = observed.get("gpu_state")
    _require(isinstance(state, dict) and state.get("driver_model.current") == "WDDM", "Desktop policy requires observed WDDM")
    allowed = {ntpath.normcase(ntpath.normpath(path)) for path in policy["allowed_graphics_paths"]}
    pids = set()
    for record in records:
        _require(isinstance(record, dict) and _integer(record.get("pid"), 1)
                 and record["pid"] not in pids and record.get("type") in ("G", "C+G")
                 and isinstance(record.get("path"), str)
                 and ntpath.normcase(ntpath.normpath(record["path"])) in allowed,
                 "Invalid or unlisted permitted GPU graphical context")
        pids.add(record["pid"])
    return len(records)


def _public_gpu_policy(manifest, sessions):
    policy = _gpu_policy(manifest.get("gpu_process_policy"))
    counts = []
    if "gpu_process_policy" in manifest or policy["mode"] == "wddm_desktop":
        counts.append(_gpu_observation(manifest.get("machine"), policy))
        counts.extend(_gpu_observation(session.get("environment_before"), policy) for session in sessions if session["status"] == "ok")
    desktop = policy["mode"] == "wddm_desktop"
    return {"mode": policy["mode"], "measurement_environment": "desktop-session" if desktop else "strict-preflight",
            "allowed_graphics_path_count": len(policy["allowed_graphics_paths"]),
            "permitted_context_count_max": max(counts, default=0),
            "limitations": (["Explicitly allowlisted WDDM graphical contexts are permitted; these are desktop-session measurements, not absolute GPU isolation."] if desktop else
                            ["Strict preflight rejects reported compute contexts; it does not establish absolute GPU isolation."])
                           + ["Idle utilization, temperature and free-VRAM thresholds remain enforced before each arm; process observations do not prove absence of interference throughout the request."]}


def _manifest(manifest):
    _require(isinstance(manifest, dict) and type(manifest.get("schema_version")) is int and manifest["schema_version"] == 2, "Invalid manifest schema; FNS-SPEED v2 required")
    _require(manifest.get("protocol_name") == "FNS-SPEED v2", "Unknown protocol")
    _require("pair_order" not in manifest and _canonical(manifest.get("execution_protocol")) == _canonical(EXECUTION_PROTOCOL), "Incompatible execution protocol")
    try:
        execution_hash = hashlib.sha256((Path(__file__).resolve().parents[1] / "speed" / "execution-v2.json").read_bytes()).hexdigest()
    except OSError as exc:
        raise ReportError("Execution protocol artifact is unavailable") from exc
    _require(manifest.get("execution_protocol_sha256") == execution_hash, "Execution protocol hash mismatch")
    for key, allowed in (("profile", ("smoke", "release", "full")), ("quant", ("q3", "q4")), ("comparison", ("service", "speculation")), ("mode", ("greedy", "sample"))):
        _require(manifest.get(key) in allowed, "Invalid campaign dimensions")
    _require(isinstance(manifest.get("run_id"), str) and manifest["run_id"], "Missing private run identity")
    _require(type(manifest.get("reproducible")) is bool, "Missing reproducibility status")
    _require(_integer(manifest.get("pairs"), 1) and _integer(manifest.get("max_tokens"), 1) and _integer(manifest.get("ctx"), 1), "Invalid campaign budgets")
    _require(manifest["max_tokens"] < manifest["ctx"], "Invalid context budget")
    seeds = [0] if manifest["mode"] == "greedy" else [0, 7]
    _require(manifest.get("seeds") == seeds and all(type(v) is int for v in manifest["seeds"]), "Invalid sampling seeds")
    if manifest["profile"] != "smoke":
        _require(manifest["pairs"] >= 5 and manifest["reproducible"], "Publication requires five pairs and reproducible inputs")
    if manifest["profile"] == "release":
        _require(manifest["ctx"] == 32768 and manifest["max_tokens"] == 256, "Release budgets differ from FNS-SPEED v2")
    for key in ("protocol_sha256", "execution_protocol_sha256", "preparation_sha256", "requests_sha256", "source_lock_sha256", "selection_sha256", "warmup_sha256"):
        _require(_hash(manifest.get(key)), "Missing reproducibility hash")
    coverage, options = _coverage(manifest), _effective(manifest)
    _gpu_policy(manifest.get("gpu_process_policy"))
    constraints = manifest.get("environment_constraints")
    _require(isinstance(constraints, dict) and _number(constraints.get("max_gpu_temperature_c")), "Missing or invalid GPU temperature ceiling")
    requests = manifest.get("expected_requests")
    ids = manifest.get("expected_request_ids")
    if not isinstance(requests, list) or not isinstance(ids, list):
        _fail("Missing expected request inventory")
    inventory, counts = {}, Counter()
    for row in requests:
        _require(isinstance(row, dict), "Invalid expected request")
        rid, config, category, count = row.get("request_id"), row.get("config"), row.get("category"), row.get("prompt_tokens")
        _require(isinstance(rid, str) and rid and rid not in inventory, "Duplicate expected request")
        _require(config in CONFIGS and category in CATEGORIES[config], "Invalid expected category")
        _require(_integer(count, 1) and count + manifest["max_tokens"] <= manifest["ctx"], "Invalid expected prompt budget")
        inventory[rid] = (config, category, count)
        counts[(config, category)] += 1
    _require(all(isinstance(rid, str) for rid in ids) and ids == list(inventory), "Inconsistent expected request identities or prepared order")
    _require(bool(inventory), "Campaign has no measurable examples")
    _require(counts == Counter({(row["config"], row["category"]): row["selected"] for row in coverage}), "Expected requests disagree with coverage")
    return coverage, options, inventory, _provenance(manifest)


def _successful(row, manifest, expected, fns):
    _require(row.get("error_code") is None, "Successful attempt carries an error")
    _require(_number(row.get("latency_ms"), positive=True), "Missing successful duration")
    for key in ("ttft_visible_ms", "tpot_client_ms"):
        value = row.get(key)
        _require(key in row and (value is None or (_number(value) and value <= row["latency_ms"])), "Invalid streaming duration")
    intervals = row.get("delta_intervals_ms")
    _require(isinstance(intervals, list) and all(_number(v) for v in intervals) and sum(intervals) <= row["latency_ms"], "Invalid SSE delta intervals")
    _require(row["ttft_visible_ms"] is not None or (row["tpot_client_ms"] is None and not intervals), "Visible timing contradicts absent delta")
    _require(type(row.get("completion_tokens")) is int and row["completion_tokens"] == manifest["max_tokens"], "Incomplete fixed-budget output")
    _require(type(row.get("prompt_tokens")) is int and row["prompt_tokens"] == expected[2], "Prompt token count mismatch")
    _require(type(row.get("prefilled_tokens")) is int and row["prefilled_tokens"] == expected[2], "Full prefill token count missing or inconsistent")
    _require(type(row.get("reused_tokens")) is int and row["reused_tokens"] == 0, "Unexpected KV reuse")
    _require(row.get("finish_reason") == "length", "Output did not terminate at fixed budget")
    output_hash = row.get("output_tokens_sha256")
    _require(_hash(output_hash) if fns else output_hash is None or _hash(output_hash), "Missing or invalid output token hash")
    metrics = row.get("metrics")
    _require(isinstance(metrics, dict), "Missing native metrics container")
    for key in ("prefill_ms", "decode_ms"):
        _require(_number(metrics.get(key)), "Missing native timing")
    for key in COUNTERS:
        _require((not fns and metrics.get(key) is None) or _integer(metrics.get(key)), "Missing native counter")
    if fns:
        n = row["completion_tokens"] - 1
        _require(metrics["spec_accepted"] <= metrics["spec_drafted"] and metrics["spec_accepted"] <= metrics["spec_emitted"] <= n, "Inconsistent speculative counters")
        _require(metrics["decode_iterations"] <= n and (n == 0 or metrics["decode_iterations"] > 0), "Inconsistent decode iterations")
        _require(metrics["spec_windows"] <= metrics["decode_iterations"], "Inconsistent speculative windows")
        if row["arm"] == "A":
            _require(all(metrics[key] == 0 for key in COUNTERS[:-1]), "Disabled speculation has nonzero counters")
            _require(metrics["decode_iterations"] == n, "Sequential decode iterations mismatch")
    if manifest["profile"] != "smoke":
        _require(isinstance(row.get("gpu_during_request"), dict), "Publication requires measured request GPU sampler evidence")
    return row


def _live_gpu_checks(session, policy, max_temperature):
    checks = session.get("gpu_process_checks")
    _require(isinstance(checks, list) and bool(checks), "Successful session lacks live GPU process checks")
    previous = -1
    allowed = {ntpath.normcase(ntpath.normpath(path)) for path in policy["allowed_graphics_paths"]}
    for check in checks:
        _require(isinstance(check, dict) and _number(check.get("monotonic")) and check["monotonic"] > previous,
                 "Invalid live GPU check order")
        previous = check["monotonic"]
        result = check.get("result")
        if not isinstance(result, dict):
            _fail("Missing live GPU process check result")
        _require(check.get("error") is None and result.get("error") is None, "Failed live GPU process check in successful session")
        state, owned, processes = (result.get(key) for key in ("gpu_state", "owned_pids", "gpu_processes"))
        if not isinstance(state, dict):
            _fail("Missing live GPU state")
        try:
            temperature = float(state["temperature.gpu"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ReportError("Missing live GPU temperature") from exc
        _require(math.isfinite(temperature) and 0 <= temperature <= max_temperature, "Live GPU temperature exceeds declared ceiling")
        _require(isinstance(owned, list) and bool(owned) and all(_integer(pid, 1) for pid in owned)
                 and len(set(owned)) == len(owned), "Missing owned process identities")
        if not isinstance(processes, list):
            _fail("Missing live GPU process inventory")
        seen = set()
        for process in processes:
            _require(isinstance(process, dict) and _integer(process.get("pid"), 1) and process["pid"] not in seen,
                     "Invalid live GPU process identity")
            seen.add(process["pid"])
            path, kind = process.get("path"), process.get("type")
            if not isinstance(path, str):
                _fail("Missing live GPU process path")
            _require(isinstance(path, str) and PureWindowsPath(path).is_absolute()
                     and not any(c in path for c in '*?%${}\x00"')
                     and not any(part in (".", "..") for part in re.split(r"[\\/]", path)), "Invalid live GPU process path")
            if process["pid"] in owned:
                _require(kind in ("C", "C+G"), "Invalid owned GPU process type")
            else:
                _require((policy["mode"] == "strict" and kind == "G")
                         or (policy["mode"] == "wddm_desktop" and state.get("driver_model.current") == "WDDM"
                             and kind in ("G", "C+G") and ntpath.normcase(ntpath.normpath(path)) in allowed),
                         "Unapproved live GPU process")


def _jsonl(path, label):
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise ReportError("Unreadable " + label) from exc
    result = []
    for line in lines:
        _require(bool(line.strip()), "Empty " + label + " row")
        try:
            row = json.loads(line, parse_constant=lambda _: (_ for _ in ()).throw(ValueError()))
        except ValueError as exc:
            raise ReportError("Invalid " + label + " JSON") from exc
        _require(isinstance(row, dict) and type(row.get("schema_version")) is int and row["schema_version"] == 2, "Invalid " + label + " schema; FNS-SPEED v2 required")
        result.append(row)
    return result


def _blocks(inventory):
    groups = defaultdict(list)
    for rid, expected in inventory.items():
        groups[expected[:2]].append(rid)
    return [ids[start:start + 32] for ids in groups.values() for start in range(0, len(ids), 32)]


def _attempts(path, manifest, inventory, options):
    blocks = _blocks(inventory)
    membership = {rid: block_id for block_id, ids in enumerate(blocks) for rid in ids}
    schedule = {(block_id, seed, pair): index for index, (block_id, seed, pair) in enumerate(
        (block_id, seed, pair) for block_id in range(len(blocks)) for seed in manifest["seeds"] for pair in range(manifest["pairs"]))}
    sessions = _jsonl(path.with_name("sessions.jsonl"), "session")
    session_ids, arm_ids, evidence, session_order = {}, {}, defaultdict(set), []

    def dimensions(row):
        block_id, generation, seed, pair, arm = (row.get(key) for key in ("block_id", "block_generation", "seed", "pair", "arm"))
        _require(_integer(block_id) and block_id < len(blocks) and _integer(generation)
                 and type(seed) is int and _integer(pair) and (block_id, seed, pair) in schedule and arm in ("A", "B"), "Unexpected block/session dimension")
        sid = row.get("session_id")
        _require(isinstance(sid, str) and bool(sid), "Missing private session identity")
        group = block_id, seed, pair
        arm_key = (*group, generation, arm)
        _require(arm_key not in arm_ids or arm_ids[arm_key] == sid, "Same-generation session mixing")
        _require(sid not in session_ids or session_ids[sid] == arm_key, "Session identity reused across blocks or arms")
        arm_ids[arm_key], session_ids[sid] = sid, arm_key
        evidence[group].add(generation)
        first_arm = "A" if (block_id + pair) % 2 == 0 else "B"
        order = schedule[group], generation, int(arm != first_arm)
        return group, generation, arm, sid, order

    finalized, session_failures = {}, Counter()
    for session in sessions:
        group, generation, arm, sid, order = dimensions(session)
        _require(sid not in finalized, "Duplicate finalized session")
        _require(session.get("request_ids") == blocks[group[0]], "Session membership differs from complete ordered block")
        _require(session.get("status") in ("ok", "error"), "Invalid finalized session status")
        _require(all(_number(session.get(key + "_ms")) for key in ("startup", "warmup", "shutdown")), "Missing finalized session lifecycle duration")
        _require(isinstance(session.get("environment_before"), dict), "Missing session environment evidence")
        if session["status"] == "ok":
            _require(session.get("error_code") is None, "Successful session carries an error")
            _live_gpu_checks(session, _gpu_policy(manifest.get("gpu_process_policy")), manifest["environment_constraints"]["max_gpu_temperature_c"])
            if "gpu_process_policy" in manifest:
                _gpu_observation(session["environment_before"], _gpu_policy(manifest["gpu_process_policy"]))
        else:
            _require(session.get("error_code") in ERROR_CODES, "Invalid session failure code")
            session_failures[session["error_code"]] += 1
        finalized[sid] = session
        session_order.append(order)
    _require(session_order == sorted(session_order), "Sessions violate block, generation or alternating arm order")

    chains, failures, originals, invalidations = defaultdict(list), Counter(), defaultdict(list), defaultdict(list)
    measurement_order, previous_observed, previous_group_order = [], -1, (-1, -1)
    for row in _jsonl(path, "attempt"):
        for key in ("run_id", "quant", "comparison", "mode"):
            _require(row.get(key) == manifest[key], "Attempt campaign mismatch")
        rid = row.get("request_id")
        _require(isinstance(rid, str) and rid in inventory, "Unexpected request")
        expected = inventory[rid]
        _require((row.get("config"), row.get("category")) == expected[:2], "Attempt category mismatch")
        group, generation, arm, sid, order = dimensions(row)
        block_id, seed, pair = group
        _require(order[:2] >= previous_group_order, "Attempt block generation order moved backwards")
        previous_group_order = order[:2]
        _require(membership[rid] == block_id, "Attempt does not belong to its block")
        _require(_integer(row.get("pair_generation")) and row["pair_generation"] == generation, "Pair/block generation mismatch")
        observed = row.get("observed_order")
        _require(_integer(observed) and observed > previous_observed, "Attempt observed order is not globally increasing")
        previous_observed = observed
        _require(not any(key in row for key in ("startup_ms", "warmup_ms", "shutdown_ms")), "Session lifecycle costs must not be copied into requests")
        _require(_canonical(row.get("effective_options")) == _canonical(options), "Attempt options differ from campaign")
        if "gpu_process_policy" in manifest or "gpu_process_policy" in row:
            _require(_gpu_policy(row.get("gpu_process_policy")) == _gpu_policy(manifest.get("gpu_process_policy")), "Attempt GPU process policy differs from campaign")
        _require(_integer(row.get("attempt")) and row.get("status") in ("ok", "error"), "Invalid attempt status")
        key = rid, seed, pair, arm
        chain = chains[key]
        _require(row["attempt"] == len(chain) and row.get("replaces") == (len(chain) - 1 if chain else None), "Duplicate, missing, or unlinked retry")
        invalidation = bool(chain and chain[-1]["block_generation"] == generation)
        if invalidation:
            previous = chain[-1]
            _require(previous["status"] == "ok" and row["status"] == "error" and row.get("error_code") == "spec_parity"
                     and manifest["comparison"] == "speculation" and previous["session_id"] == sid, "Unjustified same-generation replacement")
            invalidations[(rid, seed, pair, generation)].append(row)
        else:
            _require(not chain or generation > chain[-1]["block_generation"], "Retry generation moved backwards")
            originals[sid].append(row)
            measurement_order.append(order)
        if row["status"] == "error":
            _require(row.get("error_code") in ERROR_CODES, "Invalid failure code")
            failures[row["error_code"]] += 1
        else:
            _successful(row, manifest, expected, arm == "B" or manifest["comparison"] == "speculation")
        chain.append(row)
    _require(measurement_order == sorted(measurement_order), "Requests violate block, generation or alternating arm order")
    for rid, seed, pair, generation in invalidations:
        hashes = []
        for arm in ("A", "B"):
            original = [row for row in chains[(rid, seed, pair, arm)] if row["block_generation"] == generation and row["status"] == "ok"]
            _require(len(original) == 1, "Parity invalidation lacks original successful arm evidence")
            hashes.append(original[0]["output_tokens_sha256"])
        _require(hashes[0] != hashes[1], "Parity invalidation does not justify rerunning matching outputs")
    for sid, measured in originals.items():
        block_id = session_ids[sid][0]
        _require([row["request_id"] for row in measured] == blocks[block_id][:len(measured)], "Session requests are missing, duplicated or out of prepared order")
        _require(all(row["status"] == "ok" for row in measured[:-1]), "Session continued after a failed request")
    for sid, session in finalized.items():
        measured = originals[sid]
        if session["status"] == "ok":
            _require([row["request_id"] for row in measured] == session["request_ids"] and all(row["status"] == "ok" for row in measured), "Successful session lacks complete successful request rows")

    def complete(group, generation):
        block_id, seed, pair = group
        output_hashes = {}
        for arm in ("A", "B"):
            sid = arm_ids.get((*group, generation, arm))
            session = finalized.get(sid)
            if session is None or session["status"] != "ok":
                return False
            for rid in blocks[block_id]:
                rows = [row for row in chains[(rid, seed, pair, arm)] if row["block_generation"] == generation]
                if not rows or rows[-1]["status"] != "ok" or rows[-1]["session_id"] != sid:
                    return False
                output_hashes[(rid, arm)] = rows[-1]["output_tokens_sha256"]
        if manifest["comparison"] == "speculation":
            if any(output_hashes[(rid, "A")] != output_hashes[(rid, "B")] for rid in blocks[block_id]):
                return False
        return True

    _require(set(evidence) == set(schedule), "Incomplete expected block/seed/pair coverage")
    for group, generations in evidence.items():
        _require(sorted(generations) == list(range(max(generations) + 1)), "Block generations must advance adjacently from zero")
        for generation in range(max(generations)):
            _require(not complete(group, generation), "Valid block was silently rerun")
        _require(complete(group, max(generations)), "Campaign contains an incomplete, failed or parity-invalid latest block generation")
        for generation in generations:
            first_arm = "A" if (group[0] + group[2]) % 2 == 0 else "B"
            second_arm = "B" if first_arm == "A" else "A"
            second_sid = arm_ids.get((*group, generation, second_arm))
            if second_sid is not None:
                first_sid = arm_ids.get((*group, generation, first_arm))
                _require(first_sid in finalized and finalized[first_sid]["status"] == "ok", "Second arm lacks a successful finalized first session")
    expected_keys = {(rid, seed, pair, arm) for rid in inventory for seed in manifest["seeds"] for pair in range(manifest["pairs"]) for arm in ("A", "B")}
    _require(set(chains) == expected_keys, "Incomplete expected request/seed/pair/arm coverage")
    final = {key: chain[-1] for key, chain in chains.items()}
    if manifest["comparison"] == "speculation":
        for rid, seed, pair, arm in final:
            if arm == "A":
                _require(final[(rid, seed, pair, "A")]["output_tokens_sha256"] == final[(rid, seed, pair, "B")]["output_tokens_sha256"], "FNS exact output parity failed")
    summary = {"total": len(sessions), "successful": sum(session["status"] == "ok" for session in sessions),
               "failed": sum(session_failures.values()), "failed_by_code": dict(sorted(session_failures.items())),
               "unfinalized": len(set(session_ids) - set(finalized)),
               "lifecycle_summed_ms": {key: sum(session[key + "_ms"] for session in sessions) for key in ("startup", "warmup", "shutdown")}}
    return final, failures, sum(len(chain) for chain in chains.values()), sessions, summary


def _gpu_summary(rows):
    summaries = [row["gpu_during_request"] for row in rows if isinstance(row.get("gpu_during_request"), dict)]
    sample_keys = ("samples", "valid_samples", "active_samples")
    for summary in summaries:
        _require(all(_integer(summary.get(key)) for key in sample_keys), "Invalid measured GPU sample counts")
        _require(summary["active_samples"] <= summary["valid_samples"] <= summary["samples"], "Invalid active GPU sample count")
        _require(type(summary.get("interval_ms")) is int and summary["interval_ms"] == 200, "Invalid GPU sampling interval")
        for minimum, maximum in (("pcie_gen_min", "pcie_gen_max"), ("pcie_width_min", "pcie_width_max")):
            low, high = summary.get(minimum), summary.get(maximum)
            _require((low is None and high is None) or (_number(low) and _number(high) and low <= high), "Invalid measured PCIe range")
    result: dict[str, object] = {key: sum(summary[key] for summary in summaries) for key in sample_keys}
    result["requests_observed"] = len(summaries)
    result["requests_unobserved"] = len(rows) - len(summaries)
    for key in ("pcie_gen_min", "pcie_gen_max", "pcie_width_min", "pcie_width_max", "memory_used_mib_max", "temperature_c_max", "power_draw_watts_max", "utilization_percent_max"):
        values = []
        for summary in summaries:
            value = summary.get(key)
            _require(value is None or _number(value), "Invalid measured GPU numeric metric")
            if value is not None:
                _require(summary["active_samples"] > 0, "Under-load GPU metric has no active sample")
                values.append(value)
        result[key] = (min(values) if key.endswith("_min") else max(values)) if values else None
    result["sampling_interval_ms"] = 200
    result["active_definition"] = "utilization_positive_and_arrival_within_measured_request"
    result["null_reason"] = None if result["active_samples"] else "no_active_measured_sample"
    result["observation"] = "measured_request_active_samples" if result["active_samples"] else "no_active_measured_sample"
    return result


def _arm_summary(rows, fns):
    latency = [row["latency_ms"] for row in rows]
    ttft = [row["ttft_visible_ms"] for row in rows if row["ttft_visible_ms"] is not None]
    tpot = [row["tpot_client_ms"] for row in rows if row["tpot_client_ms"] is not None]
    intervals = [value for row in rows for value in row["delta_intervals_ms"]]
    result = {"requests": len(rows), "latency_ms": {"p50": _nearest(latency, .5), "p95": _nearest(latency, .95)},
              "completion_tokens": sum(row["completion_tokens"] for row in rows), "summed_latency_ms": sum(latency),
              "throughput_tokens_per_second": _ratio(1000 * sum(row["completion_tokens"] for row in rows), sum(latency)),
              "ttft_visible_ms": {"p50": _nearest(ttft, .5), "p95": _nearest(ttft, .95), "unobserved": len(rows) - len(ttft), "null_reason": "no_visible_delta" if len(ttft) != len(rows) else None},
              "tpot_client_ms": {"p50": _nearest(tpot, .5), "p95": _nearest(tpot, .95), "unobserved": len(rows) - len(tpot)},
              "sse_nonempty_delta_intervals_ms": {"count": len(intervals), "p50": _nearest(intervals, .5), "p95": _nearest(intervals, .95)}}
    result["gpu_during_request"] = _gpu_summary(rows)
    native = {}
    for key in ("prefill_ms", "decode_ms"):
        values = [row["metrics"].get(key) for row in rows if row["metrics"].get(key) is not None]
        native[key] = {"p50": _nearest(values, .5), "p95": _nearest(values, .95), "observed": len(values)}
    result["native_diagnostics"] = native
    if fns:
        sums = {key: sum(row["metrics"][key] for row in rows) for key in COUNTERS}
        result["speculation"] = {"counts": sums, "acceptance": _nullable_ratio(sums["spec_accepted"], sums["spec_drafted"]),
                                 "emitted_per_window": _nullable_ratio(sums["spec_emitted"], sums["spec_windows"]),
                                 "effective_advance": _nullable_ratio(sum(row["completion_tokens"] - 1 for row in rows), sums["decode_iterations"])}
        result["native_diagnostics"]["after_first_token_tokens_per_second"] = _ratio(1000 * sum(row["completion_tokens"] - 1 for row in rows), sum(row["metrics"]["decode_ms"] for row in rows))
    return result


def _categories(manifest, inventory, coverage, final):
    result = []
    for cover in coverage:
        ids = [rid for rid, expected in inventory.items() if expected[:2] == (cover["config"], cover["category"])]
        ratios, wins, paired_ratios = [], 0, []
        arms = {"A": [], "B": []}
        for rid in ids:
            per_arm = {arm: [final[(rid, seed, pair, arm)] for seed in manifest["seeds"] for pair in range(manifest["pairs"])] for arm in arms}
            ratios.append(statistics.median(row["latency_ms"] for row in per_arm["A"]) / statistics.median(row["latency_ms"] for row in per_arm["B"]))
            for seed in manifest["seeds"]:
                for pair in range(manifest["pairs"]):
                    a, b = final[(rid, seed, pair, "A")], final[(rid, seed, pair, "B")]
                    wins += b["latency_ms"] < a["latency_ms"]
                    paired_ratios.append(a["latency_ms"] / b["latency_ms"])
            for arm in arms:
                arms[arm].extend(per_arm[arm])
        median_ratio = statistics.median(ratios) if ratios else None
        fraction = _ratio(wins, len(paired_ratios))
        qualification = "no_data" if median_ratio is None else ("stable_gain" if median_ratio >= 1.02 and fraction is not None and fraction >= .8 else "not_qualified")
        result.append({"config": cover["config"], "category": cover["category"], "examples": len(ids), "valid_pairs": len(paired_ratios),
                       "median_example_latency_ratio_a_over_b": median_ratio, "paired_wins_b": wins, "paired_win_fraction_b": fraction,
                       "paired_latency_ratio_dispersion": {"p50": _nearest(paired_ratios, .5), "p95": _nearest(paired_ratios, .95)},
                       "qualification": qualification, "arms": {arm: _arm_summary(rows, arm == "B" or manifest["comparison"] == "speculation") for arm, rows in arms.items()}})
    return result


def _markdown(report):
    lines = ["# FNS-SPEED v2", "", "Adaptation of SPEED-Bench, single-turn, concurrency 1.", "", "**" + report["publication_status"] + "**", "",
             "This is not an official SPEED-Bench score or NVIDIA certification. Fixed-budget generation may pass a natural stopping point. No task quality, agent trace, tool-use, or cross-turn KV reuse is evaluated.", "",
             "Latency is measured through validated terminal SSE usage. TTFT is the first visible content/reasoning delta, not GPU prefill. SSE intervals are not per-token ITL. Native timings are engine-specific diagnostics; FNS decode begins after first-token selection.", "",
             "Category ratio: median across examples of median(A latency) / median(B latency), taking repetitions and declared seeds within each example. Quantiles use nearest rank. Throughput is sum(completion tokens) / sum(request seconds), request-duration weighted. No global speedup is reported. Stable gain requires ratio >= 1.02 and at least 80% paired wins; these are not significance tests.", ""]
    for index, run in enumerate(report["runs"], 1):
        lines += [f"## Run {index}: {run['quant']} / {run['comparison']} / {run['mode']} / {run['profile']}", "", f"{run['publication_status']}; {run['pairs_per_example_per_seed']} pairs per example/seed; {run['attempts']['failed']} initial/retried failures retained.", "",
                  "Session lifecycle overhead (counted once per finalized session, including failures): " + json.dumps(run["sessions"], sort_keys=True) + ".", "",
                  "GPU process policy: " + run["gpu_process_policy"]["mode"] + "; " + run["gpu_process_policy"]["measurement_environment"] + "; " + str(run["gpu_process_policy"]["allowed_graphics_path_count"]) + " configured graphical paths; maximum " + str(run["gpu_process_policy"]["permitted_context_count_max"]) + " permitted contexts observed.", "",
                  *run["gpu_process_policy"]["limitations"], "", "| Configuration | Category | Examples | Pairs | A/B median ratio | B wins | Qualification | A p50/p95 ms | B p50/p95 ms | A/B tokens/s |", "|---|---|---:|---:|---:|---:|---|---|---|---|"]
        def fmt(value):
            return "null" if value is None else f"{value:.6g}"
        for row in run["categories"]:
            a, b = row["arms"]["A"], row["arms"]["B"]
            lines.append(f"| {row['config']} | {row['category']} | {row['examples']} | {row['valid_pairs']} | {fmt(row['median_example_latency_ratio_a_over_b'])} | {row['paired_wins_b']} | {row['qualification']} | {fmt(a['latency_ms']['p50'])}/{fmt(a['latency_ms']['p95'])} | {fmt(b['latency_ms']['p50'])}/{fmt(b['latency_ms']['p95'])} | {fmt(a['throughput_tokens_per_second'])}/{fmt(b['throughput_tokens_per_second'])} |")
        lines += ["", "### Coverage and exclusions", "", "| Configuration | Category | Total | Admissible | Selected | Context excluded | Prompt min/p50/p95/max |", "|---|---|---:|---:|---:|---:|---|"]
        for row in run["coverage"]:
            lengths = "/".join(fmt(row["lengths"][key]) for key in ("min", "p50", "p95", "max"))
            lines.append(f"| {row['config']} | {row['category']} | {row['total']} | {row['admissible']} | {row['selected']} | {row['excluded_context']} | {lengths} |")
        lines += ["", "Empty categories preclude a whole-configuration coverage claim. Context exclusions are dataset admissibility decisions, never engine failures.", "", "### Failed attempts", "", "```json", json.dumps(run["attempts"], sort_keys=True, indent=2), "```", "", "### Reproduction parameters and sanitized provenance", "", "Artifact slots in argv must be replaced with locally acquired artifacts matching the hashes. Paths and arbitrary machine text are deliberately not exported. Null hardware/version fields mean the private observation was not representable by the typed public schema.", "", "```json", json.dumps({key: run[key] for key in ("reproducibility", "effective_options", "provenance")}, sort_keys=True, indent=2), "```", ""]
    lines += ["## Publication review", "", "This export is local only. Publication review is separate from statistical validation. Dataset rows, prompts, generated text, token IDs, source metadata, raw failures and private paths are not included.", "", "All detailed aggregate diagnostics, including nullable speculation ratios and streaming dispersion, are in report.json.", ""]
    return "\n".join(lines)


def generate_report(run_dir, out):
    """Return the deterministic report and write report.json/report.md after validation.

    Accept a run directory or a collection whose immediate subdirectories are runs.
    Malformed/incomplete campaigns fail closed, including smoke; a complete smoke
    is always labelled NON-PUBLISHABLE. This function never starts an engine.
    """
    root, destination = Path(run_dir), Path(out)
    _require(root.is_dir(), "Run directory does not exist")
    paths = [root] if (root / "run-manifest.json").is_file() else sorted((path for path in root.iterdir() if path.is_dir() and (path / "run-manifest.json").is_file()), key=lambda path: path.name)
    _require(bool(paths), "No campaigns found")
    runs, dimensions, run_ids, campaign_signature = [], set(), set(), None
    model_signatures, engine_signatures = {}, {}
    for path in paths:
        manifest = _load(path / "run-manifest.json")
        coverage, options, inventory, provenance = _manifest(manifest)
        dimension = tuple(manifest[key] for key in ("profile", "quant", "comparison", "mode"))
        _require(dimension not in dimensions and manifest["run_id"] not in run_ids, "Duplicate campaign dimensions or run identity")
        dimensions.add(dimension)
        run_ids.add(manifest["run_id"])
        signature = _canonical({key: manifest[key] for key in ("profile", "protocol_sha256", "execution_protocol_sha256", "preparation_sha256", "requests_sha256", "source_lock_sha256", "selection_sha256", "warmup_sha256", "ctx", "max_tokens", "pairs")})
        signature += _canonical({"coverage": coverage, "inventory": [(rid, *value) for rid, value in inventory.items()]})
        signature += _canonical(_gpu_policy(manifest.get("gpu_process_policy")))
        _require(campaign_signature is None or campaign_signature == signature, "Incompatible campaign collection")
        campaign_signature = signature
        quant = manifest["quant"]
        model_signature = _canonical({key: provenance["hashes"].get(key) for key in ("model", "tokenizer", "template", "profile")})
        _require(quant not in model_signatures or model_signatures[quant] == model_signature, "Same quant uses incompatible model artifacts")
        model_signatures[quant] = model_signature
        for arm in ("A", "B"):
            engine = "fns_spec" if arm == "B" else ("llama" if manifest["comparison"] == "service" else "fns_sequential")
            key = quant, engine
            machine_identity = {name: provenance["machine"][name] for name in ("os", "gpu", "cpu", "ram_bytes", "driver_version", "cuda_version", "msvc_version", "architecture", "build_architecture", "driver_model", "display_active", "power_plan", "vram_total_mib", "power_limit_watts")}
            arm_signature = _canonical({"options": provenance["options"][arm], "machine": machine_identity, "commits": provenance["commits"], "sources": provenance["sources"]})
            _require(key not in engine_signatures or engine_signatures[key] == arm_signature, "Incompatible engine options or machine within collection")
            engine_signatures[key] = arm_signature
        final, failures, attempted, sessions, session_summary = _attempts(path / "attempts.jsonl", manifest, inventory, options)
        categories = _categories(manifest, inventory, coverage, final)
        gpu_policy = _public_gpu_policy(manifest, sessions)
        if manifest["profile"] != "smoke":
            for category in categories:
                if category["examples"]:
                    for arm in ("A", "B"):
                        telemetry = category["arms"][arm]["gpu_during_request"]
                        _require(telemetry["active_samples"] > 0 and all(telemetry[key] is not None for key in ("pcie_gen_min", "pcie_gen_max", "pcie_width_min", "pcie_width_max")), "Publication requires under-load PCIe evidence for every selected category and arm")
        runs.append({"profile": manifest["profile"], "quant": manifest["quant"], "comparison": manifest["comparison"], "mode": manifest["mode"],
                     "publication_status": "NON-PUBLISHABLE" if manifest["profile"] == "smoke" else "COMPLETE — PUBLICATION REVIEW REQUIRED",
                     "pairs_per_example_per_seed": manifest["pairs"], "effective_options": options,
                     "gpu_process_policy": gpu_policy,
                     "reproducibility": {"inputs_reproducible": manifest["reproducible"], **{key: manifest[key] for key in ("protocol_sha256", "execution_protocol_sha256", "preparation_sha256", "requests_sha256", "source_lock_sha256", "selection_sha256", "warmup_sha256")}},
                     "provenance": provenance, "coverage": coverage, "categories": categories, "sessions": session_summary,
                     "attempts": {"total": attempted, "successful_final": len(final), "failed": sum(failures.values()), "failed_by_code": dict(sorted(failures.items()))}})
    runs.sort(key=lambda run: tuple(run[key] for key in ("profile", "quant", "comparison", "mode")))
    report = {"schema_version": 2, "protocol": "FNS-SPEED v2", "scope": "adaptation of SPEED-Bench, single-turn, concurrency 1",
              "publication_status": "NON-PUBLISHABLE" if any(run["profile"] == "smoke" for run in runs) else "COMPLETE — PUBLICATION REVIEW REQUIRED",
              "statistics": {"latency_ratio": "median_across_examples_of_ratio_of_arm_medians_over_repetitions_and_seeds", "quantiles": "nearest_rank", "throughput": "sum_tokens_divided_by_sum_request_seconds", "stable_gain_min_ratio": 1.02, "stable_gain_min_paired_win_fraction": .8, "significance_test": False},
              "references": {"dataset": {"repository": "nvidia/SPEED-Bench", "revision": "fbd6dc64353a3153bf24252a113ebd6d2cff9dec"}, "resolver": {"repository": "NVIDIA/Model-Optimizer", "revision": "a4dfb916c8d4f958e11ff4dea8fd594f4ecc4260"}},
              "selection": {"salt": "fns-speed-v1", "order": "sha256_compact_json_of_salt_config_string_question_id_then_string_question_id", "release_throughput_per_category": 32, "turn": 0}, "runs": runs}
    encoded = json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n"
    markdown = _markdown(report)
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "report.json").write_text(encoded, encoding="utf-8")
    (destination / "report.md").write_text(markdown, encoding="utf-8")
    return report
