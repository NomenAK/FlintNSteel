"""Private, consent-gated acquisition for the pinned NVIDIA SPEED resolver.

The upstream Apache-2.0 resolver is retained verbatim in the private cache. Only
its class and enum are compiled; none of its network loaders or package startup
code is executed. Its prompt transformations are not reimplemented here.
"""
from __future__ import annotations

import ast
import hashlib
import json
import os
import re
from pathlib import Path
from urllib.parse import quote, urlparse

PLACEHOLDER = "FULL BENCHMARK DATA SHOULD BE FETCHED FROM THE SOURCE USING SPECDEC_BENCH"
SHA1 = re.compile(r"[0-9a-f]{40}\Z")
ROOT = Path(__file__).resolve().parents[2]

# Audited against the original artifacts in adaleval-mirror-comparison.json.
# Only these ten original URLs are approved; the mutable host is never fetched.
ADALEVAL_SOURCE_REVISION = "2154258d5fa3969ac5429b3132d505570ef8a57a"
ADALEVAL_MIRROR_REPO = "AmamiSora/LOOM_Data"
ADALEVAL_MIRROR_REVISION = "73c39cc596c1d69d009a07e41a3af57c819fbbb7"
ADALEVAL_MIRROR_PINS = {
    "http://opencompass.openxlab.space/utils/AdaLEval/textsort_8k.json": {
        "original_sha256": "34baabbbb06a88efbeb64eb227575464031cf61ce37ecab27d14ce521b34f7dd",
        "mirror_sha256": "9adf23eafceedff5388b4a44188b8886db5d5e2bcf08e1e5d5854eab4fadcf06",
    },
    "http://opencompass.openxlab.space/utils/AdaLEval/stackselect_32k.json": {
        "original_sha256": "b4721166c1c41d67aa76c8ed0ab764c8db57ac774ed7ba0d7226c88131870cc7",
        "mirror_sha256": "0e87db334dbec9df18b6bb2aa7a8c91bd769efbc116d3c925683c074a8f75677",
    },
    "http://opencompass.openxlab.space/utils/AdaLEval/textsort_16k.json": {
        "original_sha256": "6029ca7c22326d83c150036e6d50d9cb41649e4adcf539cdf164de59605efc25",
        "mirror_sha256": "c1242b332aae379526c4db5b2b26234ab141149bd185cfb55020f08347fad9e4",
    },
    "http://opencompass.openxlab.space/utils/AdaLEval/stackselect_2k.json": {
        "original_sha256": "653d6e0f31e00aa984e15bd1280443f2a7ecc137daa005afa4d26eeb36af958e",
        "mirror_sha256": "baa0a7650c534ea70b27e5de6a8b968f52ebd3ef7624248110922e916cb40f07",
    },
    "http://opencompass.openxlab.space/utils/AdaLEval/textsort_32k.json": {
        "original_sha256": "9fd0ed77957653f230867b83f772444ad6bf1078433aa20ff6d1f9daad57865d",
        "mirror_sha256": "abc14b86b4de7c850c4305fc7a7d12b78c711766d737abbf6212c95ebd18b194",
    },
    "http://opencompass.openxlab.space/utils/AdaLEval/stackselect_16k.json": {
        "original_sha256": "8ce5662e8d4a73993ee9191a930a03ccd68769df9533f5c4673231f3d5af7f72",
        "mirror_sha256": "0b555730c2ad7f9c50bfdd60e39df3500e54752c058063709f18a99439ff4f28",
    },
    "http://opencompass.openxlab.space/utils/AdaLEval/textsort_2k.json": {
        "original_sha256": "09a041614cfae833e18b9434535dbe6fdc29277b78eacaff9c400c1f86c6b0a5",
        "mirror_sha256": "cfe7715ef7d52cdf8ec36e5cf05d60a1b4890d6cdc8d5a438e260281805e6cf3",
    },
    "http://opencompass.openxlab.space/utils/AdaLEval/stackselect_1k.json": {
        "original_sha256": "22b4dd4983c863d3bd6df206c1dd33ce2584b71b7d911edc768ab7e583d1d574",
        "mirror_sha256": "1be72790a5077fc6c49253b9b28c76448d51429507070cfcff363a96b6c85724",
    },
    "http://opencompass.openxlab.space/utils/AdaLEval/textsort_1k.json": {
        "original_sha256": "34bffa1e49aeb7e76e042a94ce20f63a42785041cc8f13fcbb5e410eda21bee7",
        "mirror_sha256": "823c7e67e88202f5ea2bcb00177097383e194f142a239579cb2c7857886aa58e",
    },
    "http://opencompass.openxlab.space/utils/AdaLEval/stackselect_8k.json": {
        "original_sha256": "f760da1a62b614f18f4ab09b813e54608c0657fba9cb3091daeb833e198b688b",
        "mirror_sha256": "41d6e0713823862e4a5302d9d409ca79066ca124658a686ce5fb0b153ea35549",
    },
}


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def immutable_write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != data:
            raise ValueError(f"Refusing to overwrite a different immutable artifact: {path.name}")
        return
    temporary = path.with_name(path.name + ".partial")
    temporary.write_bytes(data)
    temporary.replace(path)


def private_work_dir(path):
    path = Path(path).resolve()
    private = (ROOT / "perso").resolve()
    if path == ROOT or (ROOT in path.parents and private != path and private not in path.parents):
        raise ValueError("Data/cache directory must be under perso/ or outside the repository")
    path.mkdir(parents=True, exist_ok=True)
    # Redirect data/cache storage, not the user's authentication credentials.
    # HF_TOKEN_PATH is resolved before changing HF_HOME and is never copied.
    original_home = Path(os.environ.get("HF_HOME") or Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache") / "huggingface")
    os.environ.setdefault("HF_TOKEN_PATH", str(original_home / "token"))
    for key, child in (("HF_HOME", "hf"), ("HF_DATASETS_CACHE", "hf/datasets"),
                       ("HUGGINGFACE_HUB_CACHE", "hf/hub"), ("TIKTOKEN_CACHE_DIR", "tiktoken"),
                       ("TMP", "tmp"), ("TEMP", "tmp"), ("TMPDIR", "tmp")):
        target = path / child
        target.mkdir(parents=True, exist_ok=True)
        os.environ[key] = str(target)
    return path


def read_rows(paths):
    """Read only explicitly listed local files, with no remote dataset dispatch."""
    rows = []
    for path in paths:
        path = Path(path)
        if path.suffix == ".parquet":
            import pyarrow.parquet as pq
            rows.extend(pq.read_table(path).to_pylist())
        elif path.suffix == ".jsonl":
            with path.open(encoding="utf-8") as stream:
                rows.extend(json.loads(line) for line in stream if line.strip())
        else:
            value = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(value, list):
                raise ValueError("Expected a JSON row array")
            rows.extend(value)
    return rows


class Acquisition:
    def __init__(self, work, accepted):
        self.work = Path(work)
        self.accepted = set(accepted)
        self.records = {}
        self.plan_path = self.work / "acquisition-plan.json"
        self.plan = json.loads(self.plan_path.read_text(encoding="utf-8")) if self.plan_path.exists() else {}

    def consent(self, key, links):
        print(f"License review [{key}]:", flush=True)
        for link in sorted(set(links)):
            print(f"  {link}", flush=True)
        if key not in self.accepted:
            raise ValueError(f"Explicit acceptance required: --accept-license {key}")

    def remember(self, key, value):
        old = self.plan.get(key)
        if old is not None and old != value:
            raise ValueError("Immutable acquisition plan conflict")
        self.plan[key] = json.loads(canonical(value))
        self.plan_path.write_bytes(canonical(self.plan) + b"\n")
        return value

    def _json(self, url):
        from urllib.request import Request, urlopen
        request = Request(url, headers={"User-Agent": "FlintNSteel-SPEED/1", "Accept": "application/vnd.github+json"})
        with urlopen(request, timeout=120) as response:
            return json.load(response)

    def blob(self, url, relative):
        target = self.work / relative
        if not target.exists():
            from urllib.request import Request, urlopen
            target.parent.mkdir(parents=True, exist_ok=True)
            request = Request(url, headers={"User-Agent": "FlintNSteel-SPEED/1"})
            temporary = target.with_name(target.name + ".partial")
            with urlopen(request, timeout=300) as response, temporary.open("wb") as stream:
                import shutil
                shutil.copyfileobj(response, stream)
            temporary.replace(target)
        record = {"url": url, "path": target.relative_to(self.work).as_posix(), "sha256": digest(target), "bytes": target.stat().st_size}
        key = "blob:" + url
        if key in self.plan and self.plan[key] != record:
            raise ValueError("Cached download differs from its acquisition hash")
        self.remember(key, record)
        self.records[record["path"]] = record
        return target

    def hf(self, repo, revision, patterns):
        from huggingface_hub import HfApi, hf_hub_download
        api = HfApi()
        key = f"hf:{repo}:{revision or 'resolved'}"
        pinned = revision or self.plan.get(key)
        if pinned is None:
            pinned = api.dataset_info(repo).sha
        if not isinstance(pinned, str) or not SHA1.fullmatch(pinned):
            raise ValueError(f"Source has no immutable revision: {repo}")
        if revision and revision != pinned:
            raise ValueError("Source revision conflict")
        self.remember(key, pinned)
        listing_key = f"listing:{repo}@{pinned}"
        files = self.plan.get(listing_key)
        if files is None:
            files = api.list_repo_files(repo, repo_type="dataset", revision=pinned)
            self.remember(listing_key, files)
        import fnmatch
        selected = [name for name in files if any(fnmatch.fnmatchcase(name, p) for p in patterns)]
        if any(not any(fnmatch.fnmatchcase(name, pattern) for name in files) for pattern in patterns):
            raise ValueError(f"A required source file or test split is missing at the pinned revision: {repo}")
        legal = [name for name in files if Path(name).name.lower().startswith(("readme", "license", "licence", "copying", "notice"))]
        paths = []
        for name in sorted(set(selected + legal)):
            relative = Path("sources") / repo / pinned / name
            target = self.work / relative
            if not target.exists():
                cached = Path(hf_hub_download(repo, name, repo_type="dataset", revision=pinned,
                                             cache_dir=str(self.work / "hf" / "hub")))
                immutable_write(target, cached.read_bytes())
            url = f"https://huggingface.co/datasets/{repo}/resolve/{pinned}/{name}"
            record = {"url": url, "repo": repo, "revision": pinned, "path": relative.as_posix(),
                      "sha256": digest(target), "bytes": target.stat().st_size,
                      "terms": f"https://huggingface.co/datasets/{repo}/blob/{pinned}/README.md"}
            record_key = "file:" + record["path"]
            self.remember(record_key, record)
            self.records[record["path"]] = record
            if name in selected:
                paths.append(target)
        return paths

    def pin_url(self, url):
        key = "url:" + url
        if key in self.plan:
            return self.plan[key]
        parsed = urlparse(url)
        parts = parsed.path.strip("/").split("/")
        if parsed.netloc in ("github.com", "raw.githubusercontent.com"):
            owner, repo = parts[:2]
            if parsed.netloc == "github.com":
                if len(parts) < 4 or parts[2] not in ("tree", "blob", "raw"):
                    raise ValueError("Source GitHub URL must identify a file and revision")
                ref, rest = parts[3], parts[4:]
            else:
                ref, rest = parts[2], parts[3:]
            commit = ref if SHA1.fullmatch(ref) else self._json(f"https://api.github.com/repos/{owner}/{repo}/commits/{quote(ref, safe='')}")["sha"]
            pinned = f"https://raw.githubusercontent.com/{owner}/{repo}/{commit}/{'/'.join(rest)}"
        elif parsed.netloc == "huggingface.co" and parts[0] == "datasets":
            repo = "/".join(parts[1:3])
            if parts[3] not in ("raw", "resolve", "tree"):
                raise ValueError("Source Hugging Face URL has no revision")
            ref = parts[4]
            from huggingface_hub import HfApi
            commit = ref if SHA1.fullmatch(ref) else HfApi().dataset_info(repo, revision=ref).sha
            pinned = f"https://huggingface.co/datasets/{repo}/resolve/{commit}/{'/'.join(parts[5:])}"
        else:
            raise ValueError(f"Source is not independently immutable: {parsed.netloc}")
        if not isinstance(commit, str) or not SHA1.fullmatch(commit):
            raise ValueError("Source revision is unstable")
        return self.remember(key, pinned)

    def _adaleval_source(self, url):
        pin = ADALEVAL_MIRROR_PINS.get(url)
        if pin is None:
            raise ValueError(f"Unsupported AdaLEval source: {url}")
        original_card = f"https://raw.githubusercontent.com/open-compass/Ada-LEval/{ADALEVAL_SOURCE_REVISION}/README.md"
        mirror_root = f"https://huggingface.co/datasets/{ADALEVAL_MIRROR_REPO}"
        mirror_card = f"{mirror_root}/resolve/{ADALEVAL_MIRROR_REVISION}/README.md"
        self.consent("third-party-sources", [url, original_card, mirror_root, mirror_card])
        cards = []
        for card_url, relative in (
            (original_card, "source-terms/AdaLEval/README.md"),
            (mirror_card, f"source-terms/{ADALEVAL_MIRROR_REPO}/{ADALEVAL_MIRROR_REVISION}/README.md"),
        ):
            self.blob(card_url, relative)
            self.records[relative]["license_status"] = "unknown"
            cards.append(relative)
        name = Path(urlparse(url).path).stem
        mirror_name = f"Ada_LEval/{name}.jsonl"
        mirror_url = f"{mirror_root}/resolve/{ADALEVAL_MIRROR_REVISION}/{mirror_name}"
        mirror_relative = f"sources/{ADALEVAL_MIRROR_REPO}/{ADALEVAL_MIRROR_REVISION}/{mirror_name}"
        mirror = self.blob(mirror_url, mirror_relative)
        if digest(mirror) != pin["mirror_sha256"]:
            raise ValueError(f"AdaLEval mirror SHA256 mismatch: {name}")
        # Read physical JSONL lines, retaining input order, duplicate records,
        # object key order, and Unicode separators embedded inside strings.
        rows = read_rows([mirror])
        original = json.dumps(rows, ensure_ascii=False, indent=4).encode("utf-8")
        if hashlib.sha256(original).hexdigest() != pin["original_sha256"]:
            raise ValueError(f"AdaLEval reconstructed original SHA256 mismatch: {name}")
        relative = "external/" + hashlib.sha256(url.encode()).hexdigest() + "/" + Path(urlparse(url).path).name
        path = self.work / relative
        immutable_write(path, original)
        record = {"url": url, "path": relative, "sha256": pin["original_sha256"], "bytes": len(original)}
        # Preserve the former original URL/path and base plan record identity.
        # Provenance must not alias mutable records or acquisition-plan values.
        self.remember("blob:" + url, record)
        mirror_record = dict(self.records[mirror_relative])
        record.update({
            "independently_reacquirable": True,
            "terms": original_card,
            "license_status": "unknown",
            "license_limitation": "Neither the original nor mirror card establishes an explicit dataset licence; mirror availability is not a grant.",
            "acquisition": {
                "method": "pinned_mirror_exact_original_reconstruction",
                "source_repository": "open-compass/Ada-LEval",
                "source_revision": ADALEVAL_SOURCE_REVISION,
                "mirror": {**mirror_record, "repo": ADALEVAL_MIRROR_REPO,
                           "revision": ADALEVAL_MIRROR_REVISION, "file": mirror_name},
                "expected_original_sha256": pin["original_sha256"],
                "source_rows": len(rows),
                "serialization": {"parser": "JSONL physical lines in file order; skip blank lines",
                                  "serializer": "json.dumps", "encoding": "utf-8",
                                  "ensure_ascii": False, "indent": 4, "sort_keys": False,
                                  "separators": [",", ": "], "allow_nan": True,
                                  "trailing_newline": False},
                "cards": cards,
            },
        })
        self.records[relative] = record
        self.records[mirror_relative].update({"repo": ADALEVAL_MIRROR_REPO,
                                               "revision": ADALEVAL_MIRROR_REVISION,
                                               "terms": mirror_card, "license_status": "unknown",
                                               "independently_reacquirable": True})
        self.remember("adaleval-mirror:" + url, record)
        return path

    def json_source(self, url, *, pandas=False):
        parsed = urlparse(url)
        if parsed.netloc == "opencompass.openxlab.space" and parsed.path.startswith("/utils/AdaLEval/"):
            path = self._adaleval_source(url)
        else:
            pinned = self.pin_url(url)
            relative = "external/" + hashlib.sha256(pinned.encode()).hexdigest() + "/" + Path(urlparse(pinned).path).name
            path = self.blob(pinned, relative)
            parsed = urlparse(pinned)
            parts = parsed.path.strip("/").split("/")
            revision = parts[2] if parsed.netloc == "raw.githubusercontent.com" else parts[4]
            self.records[relative]["revision"] = revision
            self.records[relative]["independently_reacquirable"] = True
            self.records[relative]["terms"] = url
            # Retain actual source licensing and card at the same commit.
            if parsed.netloc == "raw.githubusercontent.com":
                owner, repo = parts[:2]
                tree = self._json(f"https://api.github.com/repos/{owner}/{repo}/git/trees/{revision}")
                legal = [entry["path"] for entry in tree.get("tree", []) if entry["type"] == "blob"
                         and entry["path"].lower().startswith(("license", "licence", "copying", "notice", "readme"))]
                if not legal:
                    raise ValueError(f"Source has no reviewable licensing/card: {owner}/{repo}")
                for name in sorted(legal):
                    self.blob(f"https://raw.githubusercontent.com/{owner}/{repo}/{revision}/{name}",
                              f"source-terms/{owner}/{repo}/{revision}/{name}")
            else:
                repo = "/".join(parts[1:3])
                self.hf(repo, revision, ["README.md"])
        from datasets import load_dataset
        if pandas:
            import pandas as pd
            return pd.read_json(path, lines=True)
        return load_dataset("json", data_files={"test": str(path)}, split="test", cache_dir=str(self.work / "hf" / "datasets"))


class ResolverSources:
    """Adapter for every external loader in the pinned official resolver."""
    def __init__(self, acquisition):
        self.acquisition = acquisition
        self.cache = {}

    def __call__(self, dataset_name, config_name="default"):
        key = (dataset_name, json.dumps(config_name, sort_keys=True))
        if key in self.cache:
            return self.cache[key]
        try:
            result = self.load(dataset_name, config_name)
        except Exception as error:
            gate = " Accept the source's gated terms and run hf auth login (or provide HF_TOKEN); the pinned revision is not substituted." if dataset_name == "cais/hle" else " Inspect source access/terms and the private exception chain."
            raise RuntimeError(f"Required source unavailable: {dataset_name} ({type(error).__name__})." + gate) from error
        # Keep the upstream AdaLEval first-match policy in pinned file order.
        # Do not deduplicate or reorder: the official transform selects iloc[0].
        if dataset_name == "AdaLEval/stackselect":
            if isinstance(result, tuple):
                raise ValueError("Unexpected AdaLEval split tuple")
            values = list(result["question_id"])
            for record in self.acquisition.records.values():
                if record.get("url") == config_name:
                    record["duplicate_identity_policy"] = "official_first_occurrence_in_locked_file_order"
                    record["source_rows"] = len(values)
                    record["distinct_question_ids"] = len(set(values))
        # Other keyed sources must remain unambiguous.
        column = {"abisee/cnn_dailymail": "id", "cais/hle": "id",
                  "livecodebench/code_generation_lite": "question_id", "deepmind/code_contests": "name"}.get(dataset_name)
        if column:
            for part in result if isinstance(result, tuple) else (result,):
                values = list(part[column])
                if len(values) != len(set(values)):
                    raise ValueError(f"Duplicate source identities: {dataset_name}")
        self.cache[key] = result
        return result

    def load(self, name, config):
        from datasets import Dataset, concatenate_datasets
        a = self.acquisition
        revisions = {
            "cais/hle": "021a3d71f516a7ac28ceb8d284969902edf1edeb",
            "abisee/cnn_dailymail": "96df5e686bee6baa90b8bee7c28b81fa3fa6223d",
            "deepmind/code_contests": "802411c3010cb00d1b05bad57ca77365a3c699d6",
            "Helsinki-NLP/opus-100": "805090dc28bf78897da9641cdf08b61287580df9",
            "TIGER-Lab/MMLU-Pro": "30527804ea8854662078e457808040d872ecdf29",
        }
        if name in ("RUCAIBox/BAMBOO", "mtbench101/mt-bench-101", "AdaLEval/stackselect", "AdaLEval/textsort", "ZenMoore/RoleBench/roles"):
            return a.json_source(config)
        if name == "ZenMoore/RoleBench":
            return a.json_source(config, pandas=True)
        if name == "Neph0s/CoSER":
            return a.json_source(config.replace("tree", "raw") + "/test/test_set.json")
        if name == "livecodebench/code_generation_lite":
            paths = a.hf(name, "0fe84c3912ea0c4d4a78037083943e8f0c4dd505", [f"test{suffix}.jsonl" for suffix in ("", "2", "3", "4", "5", "6")])
            # Match the official JSON loader on local immutable files. A giant
            # Dataset.from_list materializes multi-GB private test-case strings
            # into one Arrow chunk and overflows 32-bit offsets during its
            # fingerprint; the official loader writes bounded Arrow batches.
            from datasets import load_dataset
            return load_dataset("json", data_files={"test": [str(path) for path in paths]},
                                split="test", cache_dir=str(a.work / "hf" / "datasets"))
        if name == "nvidia/ChatRAG-Bench":
            # Upstream requests "hybridial"; actual pinned directory is hybridial.
            return concatenate_datasets([Dataset.from_list(read_rows(a.hf(name, "af6c7d420ddddf21f54f8ab3394bbf462aad2577", [f"data/{c}/test.json"]))) for c in config])
        if name not in revisions and name != "abisee/cnn_dailymail":
            raise ValueError(f"Unsupported source in pinned resolver: {name}")
        pattern = f"{config}/test*.parquet" if name == "Helsinki-NLP/opus-100" else "3.0.0/test*.parquet" if name == "abisee/cnn_dailymail" else "**test*.parquet"
        result = Dataset.from_list(read_rows(a.hf(name, revisions.get(name), [pattern])))
        if name == "cais/hle" and config == "train_test_split":
            split = result.train_test_split(test_size=0.5, shuffle=True, seed=42)
            return split["train"], split["test"]
        return result


def official_resolver(path, sources):
    """Compile original transformations, replacing only acquisition boundaries."""
    payload = Path(path).read_bytes()
    git_blob = hashlib.sha1(b"blob " + str(len(payload)).encode("ascii") + b"\0" + payload).hexdigest()
    if git_blob != "552a537a176f266e3f2a4dcb3de23cdb05575b8d":
        raise ValueError("Official resolver bytes do not match the authenticated pinned Git blob")
    import random
    from enum import Enum
    from typing import Literal, get_args
    import numpy as np
    import pandas as pd
    import tiktoken
    tree = ast.parse(payload.decode("utf-8"))
    names = {"BenchmarkDataset", "SPEEDBench"}
    nodes: list[ast.stmt] = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name in names]
    if len(nodes) != 2:
        raise ValueError("Unexpected pinned official resolver structure")
    namespace = {"__name__": "fns_pinned_speed_resolver", "Dataset": object, "Enum": Enum,
                 "random": random, "re": re, "Path": Path, "Any": object, "Literal": Literal,
                 "get_args": get_args, "np": np, "pd": pd, "tiktoken": tiktoken,
                 "TURNS_PLACEHOLDER": PLACEHOLDER, "not_installed": False,
                 "config_type": Literal["qualitative", "throughput_1k", "throughput_2k", "throughput_8k", "throughput_16k", "throughput_32k"]}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    cls = namespace["SPEEDBench"]
    instance = object.__new__(cls)
    instance.external_datasets = {}
    instance._get_external_dataset = sources
    return instance
