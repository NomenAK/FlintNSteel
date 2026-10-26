"""Acquire and prepare FNS-SPEED v1; all dataset artifacts remain private.

Importing this module performs no acquisition and imports no network libraries.
The official NVIDIA transformations execute only during explicitly consented fetch.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import socket
import sys
from pathlib import Path

from speed_sources import (Acquisition, PLACEHOLDER, ROOT, ResolverSources, SHA1,
                           canonical, digest, immutable_write, official_resolver,
                           private_work_dir, read_rows)

DEFAULT_PROTOCOL = ROOT / "bench" / "speed" / "protocol-v1.json"
DEPENDENCIES = ROOT / "bench" / "speed" / "requirements.lock"
DATASET_REVISION = "fbd6dc64353a3153bf24252a113ebd6d2cff9dec"
RESOLVER_REVISION = "a4dfb916c8d4f958e11ff4dea8fd594f4ecc4260"


def content_hash(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def jsonl(rows):
    return b"".join(canonical(row) + b"\n" for row in rows)


def load_protocol(path):
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if value.get("schema_version") != 1 or value.get("name") != "FNS-SPEED v1":
        raise ValueError("Unsupported protocol")
    if value.get("dataset") != {"repo": "nvidia/SPEED-Bench", "revision": DATASET_REVISION}:
        raise ValueError("FNS-SPEED v1 requires its pinned dataset")
    resolver = value.get("resolver", {})
    if resolver.get("repo") != "NVIDIA/Model-Optimizer" or resolver.get("revision") != RESOLVER_REVISION:
        raise ValueError("FNS-SPEED v1 requires its pinned official resolver")
    for name in ("ctx", "max_tokens", "throughput_per_category", "pairs"):
        if type(value.get(name)) is not int or value[name] <= 0:
            raise ValueError(f"Invalid positive protocol integer: {name}")
    if value["max_tokens"] > value["ctx"]:
        raise ValueError("Output budget exceeds context")
    if len(value["configs"]) != len(set(value["configs"])):
        raise ValueError("Duplicate protocol configurations")
    return value


def dependencies():
    versions = {}
    for line in DEPENDENCIES.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        name, wanted = line.split("==", 1)
        actual = importlib.metadata.version(name)
        if actual != wanted:
            raise ValueError(f"Dependency version mismatch: {name}: expected {wanted}, installed {actual}")
        versions[name] = actual
    return {"python": platform.python_version(), "implementation": platform.python_implementation(), "packages": versions}


def adaptation_hash():
    return content_hash({name: digest(Path(__file__).with_name(name)) for name in ("speed_prepare.py", "speed_sources.py")})


def validate_rows(rows, config, categories, *, resolved=True):
    seen = set()
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("Dataset row must be an object")
        for key in ("question_id", "category", "source", "src_id"):
            if not isinstance(row.get(key), str) or not row[key]:
                raise ValueError(f"Missing or unexpected source schema: {key}")
        if row["question_id"] in seen:
            raise ValueError(f"Duplicate question identity in {config}")
        seen.add(row["question_id"])
        if row["category"] not in categories:
            raise ValueError(f"Unexpected category in {config}")
        turns = row.get("turns")
        if not isinstance(turns, list) or not turns or any(not isinstance(t, str) or not t.strip() for t in turns):
            raise ValueError("Every row requires nonempty string turns")
        if resolved and any(PLACEHOLDER in turn for turn in turns):
            raise ValueError("Unresolved official source placeholder")
    if not rows:
        raise ValueError(f"Missing source rows for {config}")


@contextlib.contextmanager
def offline_network():
    """Enforce offline even if an imported tokenizer/helper gains network code."""
    old = {key: os.environ.get(key) for key in ("HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE")}
    for key in old:
        os.environ[key] = "1"
    original_connect, original_connect_ex, original_create = socket.socket.connect, socket.socket.connect_ex, socket.create_connection
    original_dns = socket.getaddrinfo
    def denied(*args, **kwargs):
        raise RuntimeError("Network access forbidden during offline preparation")
    socket.socket.connect = denied
    socket.socket.connect_ex = denied
    socket.create_connection = denied
    socket.getaddrinfo = denied
    try:
        yield
    finally:
        socket.socket.connect, socket.socket.connect_ex, socket.create_connection = original_connect, original_connect_ex, original_create
        socket.getaddrinfo = original_dns
        for key, value in old.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def load_chat(tokenizer=None, template=None):
    """Use the production normalizer and Chat.prompt_ids, never a second template."""
    sys.path.insert(0, str(ROOT / "serve"))
    import flint_assets
    selected = Path(tokenizer) if tokenizer else flint_assets.local_tokenizer()
    if selected is None:
        raise ValueError("Offline preparation needs a local tokenizer: --tokenizer or FNS_TOKENIZER")
    selected = selected.resolve()
    template = Path(template or flint_assets.CHAT_TEMPLATE).resolve()
    if not selected.is_file() or not template.is_file():
        raise ValueError("Missing local tokenizer or template")
    import fns_server
    old_tokenizer = os.environ.get("FNS_TOKENIZER")
    old_template = fns_server.CHAT_TEMPLATE
    os.environ["FNS_TOKENIZER"] = str(selected)
    fns_server.CHAT_TEMPLATE = template
    try:
        chat = fns_server.Chat()
    finally:
        fns_server.CHAT_TEMPLATE = old_template
        if old_tokenizer is None:
            os.environ.pop("FNS_TOKENIZER", None)
        else:
            os.environ["FNS_TOKENIZER"] = old_tokenizer
    return chat, fns_server.normalize_messages, selected, template


def selection_key(config, question_id, salt="fns-speed-v1"):
    return content_hash([salt, config, str(question_id)]), str(question_id)


def nearest_rank(values, fraction):
    return sorted(values)[max(0, math.ceil(len(values) * fraction) - 1)] if values else None


def select_requests(corpus, protocol, profile, chat, normalize):
    """Account for every row before quota selection, including empty categories."""
    if profile not in ("smoke", "release", "full"):
        raise ValueError("Unknown selection profile")
    if profile == "release" and set(corpus) != set(protocol["configs"]):
        raise ValueError("Release requires all six configurations")
    requests, selection, coverage = [], [], []
    for config in protocol["configs"]:
        if config not in corpus:
            continue
        rows = corpus[config]
        validate_rows(rows, config, protocol["categories"][config])
        by_category = {category: [] for category in protocol["categories"][config]}
        for row in rows:
            source = row["turns"][0]
            messages = normalize([{"role": "user", "content": source}])
            if normalize(messages) != messages:
                raise ValueError("Production message normalization is not idempotent")
            tokens = chat.prompt_ids(messages, None)
            if not isinstance(tokens, list) or not tokens or any(type(t) is not int or t < 0 for t in tokens):
                raise ValueError("Invalid production prompt token IDs")
            normalized = messages[0]["content"]
            request = {"schema_version": 1, "request_id": content_hash([config, str(row["question_id"])]),
                       "config": config, "category": row["category"], "question_id": str(row["question_id"]),
                       "messages": messages, "tokens": tokens, "prompt_tokens": len(tokens),
                       "unused_turns": len(row["turns"]) - 1, "transformed": normalized != source,
                       "source_text_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
                       "normalized_text_sha256": hashlib.sha256(normalized.encode("utf-8")).hexdigest(),
                       "prompt_sha256": content_hash(tokens)}
            by_category[row["category"]].append((request, source))
        for category in protocol["categories"][config]:
            candidates = sorted(by_category[category], key=lambda pair: selection_key(config, pair[0]["question_id"], protocol["selection_salt"]))
            eligible = [request for request, _ in candidates if request["prompt_tokens"] + protocol["max_tokens"] <= protocol["ctx"]]
            quota = 1 if profile == "smoke" else protocol["throughput_per_category"] if profile == "release" and config != "qualitative" else len(eligible)
            chosen = eligible[:quota]
            selected = {request["request_id"] for request in chosen}
            requests.extend(chosen)
            for request, source in candidates:
                admissible = request["prompt_tokens"] + protocol["max_tokens"] <= protocol["ctx"]
                selection.append({key: value for key, value in request.items() if key not in ("messages", "tokens")}
                                 | {"source_text": source, "normalized_text": request["messages"][0]["content"],
                                    "admissible": admissible, "selected": request["request_id"] in selected,
                                    "reason": "context" if not admissible else "selected" if request["request_id"] in selected else "quota"})
            lengths = [request["prompt_tokens"] for request, _ in candidates]
            coverage.append({"config": config, "category": category, "total": len(candidates),
                             "admissible": len(eligible), "selected": len(chosen), "excluded_context": len(candidates) - len(eligible),
                             "lengths": {"min": min(lengths) if lengths else None, "max": max(lengths) if lengths else None,
                                         "p50": nearest_rank(lengths, .5), "p95": nearest_rank(lengths, .95)}})
    return requests, selection, coverage


def fetch(args):
    protocol = load_protocol(args.protocol)
    work = private_work_dir(args.work_dir)
    a = Acquisition(work, args.accept_license)
    a.consent("nvidia-evaluation-dataset", [protocol["licenses"]["nvidia-evaluation-dataset"], protocol["licenses"]["third-party-sources"]])
    if (work / "source-lock.json").exists():
        lock = verify_lock(work, protocol)
        if any(config not in lock["resolved"] for config in requested_configs(args, protocol)):
            raise ValueError("Existing source lock lacks requested configurations; use a new work directory")
        print("Existing immutable source lock verified; no acquisition necessary.")
        return
    environment = dependencies()
    configs = requested_configs(args, protocol)
    corpus, raw_paths = {}, {}
    for config in configs:
        paths = a.hf(protocol["dataset"]["repo"], protocol["dataset"]["revision"], [f"{config}/test-*.parquet"])
        rows = read_rows(paths)
        validate_rows(rows, config, protocol["categories"][config], resolved=False)
        corpus[config], raw_paths[config] = rows, [path.relative_to(work).as_posix() for path in paths]
    source_urls = sorted({row["source"] for rows in corpus.values() for row in rows})
    a.consent("third-party-sources", [protocol["licenses"]["third-party-sources"], *source_urls])
    resolver = protocol["resolver"]
    base = f"https://raw.githubusercontent.com/{resolver['repo']}/{resolver['revision']}/"
    resolver_path = a.blob(base + resolver["path"], "resolver/speed.py")
    a.blob(base + "LICENSE", "resolver/LICENSE")
    # tiktoken's encoding has an upstream embedded expected SHA-256. Warm its
    # private cache only under fetch consent, and lock the downloaded bytes.
    import tiktoken
    tiktoken.get_encoding("o200k_base")
    for path in sorted((work / "tiktoken").rglob("*")):
        if path.is_file():
            relative = path.relative_to(work).as_posix()
            a.records[relative] = {"path": relative, "sha256": digest(path), "bytes": path.stat().st_size,
                                   "terms": "https://github.com/openai/tiktoken/blob/0.14.0/LICENSE"}
    external = ResolverSources(a)
    resolved = {}
    # Preserve original pinned file/row order for upstream stateful HLE RNG.
    # Selection ordering is applied only after official resolution.
    for config, rows in corpus.items():
        instance = official_resolver(resolver_path, external)
        result = []
        for row in rows:
            result.append(instance._fetch_all_turns_data(dict(row), config))
        validate_rows(result, config, protocol["categories"][config])
        target = work / "resolved" / f"{config}.jsonl"
        immutable_write(target, jsonl(result))
        relative = target.relative_to(work).as_posix()
        a.records[relative] = {"path": relative, "sha256": digest(target), "bytes": target.stat().st_size, "derived": True,
                               "canonical_sha256": content_hash(result)}
        resolved[config] = relative
    lock = {"schema_version": 1, "dataset": protocol["dataset"], "resolver": resolver,
            "adaptation_sha256": adaptation_hash(), "dependency_lock_sha256": digest(DEPENDENCIES),
            "protocol_sha256": digest(args.protocol), "environment": environment,
            "licenses_accepted": sorted(a.accepted), "source_review": source_urls,
            "licensing_limitations": {"LiveCodeBench": "Pinned card labels licence cc without specifying a grant; review source terms separately.",
                                       "AdaLEval": "Original and pinned third-party mirror cards do not establish an explicit permissive dataset licence or redistribution grant; byte-exact reacquisition is not legal permission.",
                                       "OPUS-100": "Pinned card labels licence unknown; constituent corpora have separate terms.",
                                       "ChatRAG-Bench": "Pinned card delegates to original dataset licences, including HybriDialogue and SQA."},
            "reproducibility_scope": "Immutable source bytes and preparation only; not a legal permission determination.",
            "raw": raw_paths, "resolved": resolved, "sources": sorted(a.records.values(), key=lambda r: r["path"]),
            "reproducible": all(record.get("independently_reacquirable", True) for record in a.records.values()),
            "resolution_order": "pinned sorted files, original row order; HLE RNG reset per configuration"}
    immutable_write(work / "source-lock.json", canonical(lock) + b"\n")
    print("Immutable private source-lock.json created.")


def requested_configs(args, protocol):
    configs = args.configs or protocol["configs"]
    if not configs or len(configs) != len(set(configs)) or any(c not in protocol["configs"] for c in configs):
        raise ValueError("Invalid or duplicated configurations")
    return [c for c in protocol["configs"] if c in configs]


def locked_path(work, relative):
    path = (work / relative).resolve()
    if work.resolve() not in path.parents:
        raise ValueError("Source lock path escapes private work directory")
    return path


def verify_lock(work, protocol):
    lock = json.loads((work / "source-lock.json").read_text(encoding="utf-8"))
    if lock.get("schema_version") != 1 or type(lock.get("reproducible")) is not bool:
        raise ValueError("Missing or invalid source lock")
    if lock.get("dataset") != protocol["dataset"] or lock.get("resolver") != protocol["resolver"]:
        raise ValueError("Source lock protocol pins differ")
    if lock.get("adaptation_sha256") != adaptation_hash() or lock.get("dependency_lock_sha256") != digest(DEPENDENCIES):
        raise ValueError("Source resolver adaptation or dependencies changed; acquire a new work directory")
    sources = lock.get("sources", [])
    paths = set()
    for source in sources:
        relative = source["path"]
        if relative in paths:
            raise ValueError("Duplicate locked source file")
        paths.add(relative)
        path = locked_path(work, relative)
        if not path.is_file() or digest(path) != source["sha256"]:
            raise ValueError(f"Missing or modified locked source: {relative}")
        if "revision" in source and not SHA1.fullmatch(source["revision"]):
            raise ValueError("Unstable source revision")
    required = [p for files in lock["raw"].values() for p in files] + list(lock["resolved"].values()) + ["resolver/speed.py", "resolver/LICENSE"]
    if not sources or any(p not in paths for p in required):
        raise ValueError("Source lock does not inventory all required artifacts")
    return lock


def prepare(args):
    if not args.offline:
        raise ValueError("Preparation requires --offline; acquisition belongs to fetch")
    protocol = load_protocol(args.protocol)
    work = private_work_dir(args.work_dir)
    configs = requested_configs(args, protocol)
    if args.profile == "release" and configs != protocol["configs"]:
        raise ValueError("Release requires all six configurations")
    with offline_network():
        environment = dependencies()
        lock = verify_lock(work, protocol)
        if args.profile != "smoke" and lock["reproducible"] is not True:
            raise ValueError("Release/full preparation requires independently reacquirable immutable sources; only nonpublishable smoke is allowed")
        if environment != lock["environment"]:
            raise ValueError("Preparation environment differs from immutable source lock")
        if lock.get("protocol_sha256") != digest(args.protocol):
            raise ValueError("Protocol changed after source locking")
        corpus = {}
        for config in configs:
            if config not in lock["resolved"]:
                raise ValueError(f"Missing locked configuration: {config}")
            corpus[config] = read_rows([locked_path(work, lock["resolved"][config])])
        chat, normalize, tokenizer, template = load_chat(args.tokenizer, args.template)
        requests, selection, coverage = select_requests(corpus, protocol, args.profile, chat, normalize)
        output = work / "prepared" / args.profile
        request_bytes, selection_bytes = jsonl(requests), jsonl(selection)
        manifest = {"schema_version": 1, "profile": args.profile, "configs": configs,
                    "protocol_sha256": digest(args.protocol), "source_lock_sha256": digest(work / "source-lock.json"),
                    "requests_sha256": hashlib.sha256(request_bytes).hexdigest(),
                    "selection_sha256": hashlib.sha256(selection_bytes).hexdigest(),
                    "tokenizer_sha256": digest(tokenizer), "template_sha256": digest(template),
                    "dependency_lock_sha256": digest(DEPENDENCIES), "adaptation_sha256": adaptation_hash(),
                    "ctx": protocol["ctx"], "max_tokens": protocol["max_tokens"], "coverage": coverage,
                    "reproducible": lock["reproducible"], "environment": environment,
                    "unused_turns": sum(row["unused_turns"] for row in selection),
                    "publishable_profile": args.profile != "smoke"}
        # No timestamps or personal paths: identical inputs produce identical bytes.
        for name, data in (("requests.jsonl", request_bytes), ("selection.jsonl", selection_bytes),
                           ("preparation.json", canonical(manifest) + b"\n")):
            immutable_write(output / name, data)
    print(f"Prepared {len(requests)} requests; profile={args.profile}; private files verified or created.")


def main(argv=None):
    parser = argparse.ArgumentParser(description="FNS-SPEED v1 private acquisition and deterministic offline preparation")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("fetch", "prepare"):
        command = sub.add_parser(name)
        command.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
        command.add_argument("--work-dir", type=Path, default=ROOT / "perso" / "bench" / "speed")
        command.add_argument("--configs", nargs="+", help="Explicit configuration subset; release requires all six")
        if name == "fetch":
            command.add_argument("--accept-license", action="append", default=[],
                                 choices=("nvidia-evaluation-dataset", "third-party-sources"),
                                 help="Explicit, repeatable licence acceptance; both tiers required for full acquisition")
        else:
            command.add_argument("--profile", choices=("smoke", "release", "full"), required=True)
            command.add_argument("--offline", action="store_true", required=True, help="Forbid all network connections")
            command.add_argument("--tokenizer", type=Path, help="Existing local Flint tokenizer; never downloads")
            command.add_argument("--template", type=Path, help="Existing local Flint Jinja template")
    args = parser.parse_args(argv)
    try:
        (fetch if args.command == "fetch" else prepare)(args)
    except (ValueError, RuntimeError, OSError, importlib.metadata.PackageNotFoundError) as error:
        # This is a private acquisition/preparation command; no prompts are printed.
        parser.exit(1, f"{type(error).__name__}: {error}\n")


if __name__ == "__main__":
    main()
