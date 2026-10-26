#!/usr/bin/env python3
"""Opt-in real-GPU native service contract checks; never downloads assets or runs in unit discovery.

python tests/speed_native_smoke.py --engine build/fns-serve.exe --q3 MODEL --q4 MODEL \
    --expert-profile PROFILE --tokenizer TOKENIZER --out-dir perso/bench/speed/native-smoke

Synthetic authored prompts, not SPEED-Bench data. Processes run sequentially. Exact sequential/speculative token
parity is checked at matching cache states, not across cold/cached prefill geometries. All requests and token IDs
are private files in the fresh output directory. No expected generated wording is hard-coded. Coverage of rare
rejection/acceptance boundaries is reported explicitly, never claimed when the model did not exercise it.
"""

import argparse
import json
import os
import queue
import subprocess
import sys
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "serve"))

from flint_assets import CHAT_TEMPLATE, local_tokenizer  # noqa: E402
from fns_server import Chat  # noqa: E402

COUNTERS = ("spec_windows", "spec_drafted", "spec_accepted", "spec_emitted", "decode_iterations")
PROMPTS = (
    "Count upward from one, separating each number with a comma. Continue the sequence.",
    "Continue this exact repeated pattern for many lines: red blue green red blue green red blue green",
    "Repeat this sentence several times: The small clock keeps time.",
    "Write a Python list containing the integers from zero to fifty in order.",
    "Continue: alpha beta gamma delta alpha beta gamma delta alpha beta gamma delta",
    "Explain in a few sentences why water freezes. Then summarize your explanation.",
    "Continue this repeated list exactly, one entry per line:\n" + "".join(
        f"item_{i:03d}: value_{i:03d}\n" for i in range(50)) * 3,
)


def lookup_draft(history, want):
    """Independent protocol oracle matching the public n=4..2 lookup rule (no native execution)."""
    for n in (4, 3, 2):
        for start in range(len(history) - n - 1, -1, -1):
            if history[start:start + n] == history[-n:]:
                return history[start + n:start + n + want]
    return []


def expected_counters(prompt, output, width, budget=None):
    """Reconstruct every verifier boundary from actual emitted IDs, without guessing model output or logits.

    No grammar in this smoke. Each lookup's drafts are submitted, and accepted choices follow those drafts until
    a mismatch, bonus, output budget or stop. Unemitted candidates still count as drafted, not as accepted.
    """
    counts = dict.fromkeys(COUNTERS, 0)
    coverage = set()
    windows = []
    if not output:
        raise AssertionError("native output is empty")
    if width == 0:
        counts["decode_iterations"] = len(output) - 1
        return counts, coverage, windows
    history = list(prompt) + output[:1]
    budget = len(output) if budget is None else budget
    pos = 1
    while pos < len(output):
        counts["decode_iterations"] += 1
        draft = lookup_draft(history, min(width - 1, budget - pos))
        if not draft:
            coverage.add("no_draft")
            history.append(output[pos])
            pos += 1
            continue
        counts["spec_windows"] += 1
        counts["spec_drafted"] += len(draft)
        start, accepted, last_accepted = pos, 0, False
        for column in range(len(draft) + 1):
            if pos == len(output):
                break
            token = output[pos]
            history.append(token)
            pos += 1
            counts["spec_emitted"] += 1
            last_accepted = column < len(draft) and token == draft[column]
            if column == len(draft) or token != draft[column]:
                break
            accepted += 1
            counts["spec_accepted"] += 1
        if accepted == 0:
            coverage.add("immediate_rejection")
        elif accepted < len(draft):
            coverage.add("partial_acceptance")
        else:
            coverage.add("full_acceptance")
        windows.append({"start": start, "end": pos, "drafted": len(draft), "accepted": accepted,
                        "last_accepted": last_accepted})
    return counts, coverage, windows


class Native:
    def __init__(self, argv, directory, timeout):
        self.timeout = timeout
        self.stderr = (directory / "stderr.log").open("w", encoding="utf-8")
        self.trace = (directory / "requests.jsonl").open("w", encoding="utf-8")
        self.process = subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.stderr,
                                        text=True, encoding="utf-8")
        if self.process.stdin is None or self.process.stdout is None:
            self.process.kill()
            self.process.wait()
            self.stderr.close()
            self.trace.close()
            raise RuntimeError("native process pipes are missing")
        self.stdin, self.stdout = self.process.stdin, self.process.stdout
        self.lines = queue.Queue()

        def consume():
            try:
                for line in self.stdout:
                    self.lines.put(line)
            finally:
                self.lines.put(None)

        self.reader = threading.Thread(target=consume, daemon=True)
        self.reader.start()
        try:
            if self.read().get("ready") is not True:
                raise AssertionError("native service did not become ready")
        except BaseException:
            self.close()
            raise

    def read(self):
        try:
            line = self.lines.get(timeout=self.timeout)
        except queue.Empty as exc:
            raise TimeoutError("native response timeout") from exc
        if line is None:
            raise RuntimeError("native process closed stdout")
        return json.loads(line)

    def request(self, body, expect_error=False):
        self.stdin.write(json.dumps(body) + "\n")
        self.stdin.flush()
        tokens = []
        while True:
            event = self.read()
            if "token" in event:
                tokens.append(event["token"])
                continue
            self.trace.write(json.dumps({"request": body, "token_ids": tokens, "native": event}) + "\n")
            self.trace.flush()
            if expect_error:
                if "error" not in event or tokens:
                    raise AssertionError("invalid native cache_prompt was not rejected")
                return tokens, event
            if not event.get("done") or "error" in event:
                raise AssertionError(f"native request failed; see private trace: {event}")
            return tokens, event

    def close(self):
        self.stdin.close()
        try:
            self.process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait()
        self.reader.join(timeout=5)
        self.stdout.close()
        self.stderr.close()
        self.trace.close()


def validate(prompt, tokens, done, width, budget, stops=()):
    expected, coverage, windows = expected_counters(prompt, tokens, width, budget)
    if done.get("prompt_tokens") != len(prompt) or done.get("generated") != len(tokens):
        raise AssertionError("native prompt/generated counts differ from consumed IDs")
    for field, value in expected.items():
        if type(done.get(field)) is not int or done[field] != value:
            raise AssertionError(f"{field}: native={done.get(field)}, reconstructed={value}")
    if not 0 <= done["spec_accepted"] <= done["spec_drafted"] or done["spec_emitted"] > len(tokens) - 1:
        raise AssertionError("native speculative counter invariants failed")
    reason = "stop" if tokens[-1] in stops else "length"
    if done.get("reason") != reason:
        raise AssertionError("native stop/budget reason disagrees with emitted IDs")
    if len(tokens) > budget or (len(tokens) < budget and tokens[-1] not in stops):
        raise AssertionError("unexpected native early termination")
    if windows and windows[-1]["end"] == len(tokens):
        coverage.add("budget_in_window" if reason == "length" else "stop_in_window")
        if reason == "stop" and windows[-1]["last_accepted"]:
            coverage.add("accepted_stop")
    return coverage, windows


def run_arm(args, quant, model, spec, cases, mode, seed, root):
    name = f"{quant}-{mode}-{seed}-spec{spec}"
    directory = root / name
    directory.mkdir()
    argv = [str(args.engine), "--model", str(model), "--ctx", str(args.ctx), "--ubatch", str(args.ubatch),
            "--spec", str(spec), "--spec-sample", str(min(spec, 2))]
    if quant == "q4":
        argv += ["--expert-profile", str(args.expert_profile)]
    native = Native(argv, directory, args.timeout)
    rows, covered, probes = [], set(), []
    width = spec if mode == "greedy" else min(spec, 2)
    sampling = {"temperature": 0 if mode == "greedy" else 1, "top_p": 1 if mode == "greedy" else 0.95,
                "top_k": 0 if mode == "greedy" else 20, "seed": seed, "presence_penalty": 0}
    try:
        for invalid in (None, 0, 1, "false", [], {}):
            native.request({"tokens": cases[0], "max_tokens": 1, "cache_prompt": invalid}, expect_error=True)
        for index, prompt in enumerate(cases):
            base = {"tokens": prompt, "max_tokens": args.max_tokens, "stop": [], **sampling}
            state_outputs = {}
            for state, cache in (("cold", False), ("cold_repeat", False), ("cached", True), ("default", None)):
                body = {**base, **({} if cache is None else {"cache_prompt": cache})}
                tokens, done = native.request(body)
                coverage, windows = validate(prompt, tokens, done, width, args.max_tokens)
                covered.update(coverage)
                if cache is False and done.get("reused") != 0:
                    raise AssertionError("cache_prompt=false reused native KV")
                if cache is not False and done.get("reused") != len(prompt) - 1:
                    raise AssertionError("true/default cache did not reuse the expected native prefix")
                state_outputs[state] = tokens
                rows.append({"case": index, "state": state, "token_ids": tokens, "metrics": done})
                if state == "cold" and spec:
                    for window in windows:
                        if window["accepted"]:
                            # Re-run the same cold state with an accepted token as the stop; skip earlier occurrences.
                            position = window["start"] + window["accepted"] - 1
                            target = tokens[position]
                            if target not in tokens[:position]:
                                probes.append((index, position, target))
                                break
            if state_outputs["cold"] != state_outputs["cold_repeat"]:
                raise AssertionError("repeated cold output changed")
            if state_outputs["cached"] != state_outputs["default"]:
                raise AssertionError("explicit/default cached output changed")
        # A one-token budget checks zero post-first counters even if its ID is a declared stop.
        first, done = native.request({"tokens": cases[0], "max_tokens": 1, "cache_prompt": False, "stop": [], **sampling})
        validate(cases[0], first, done, width, 1)
        token, done = native.request({"tokens": cases[0], "max_tokens": 1, "cache_prompt": False,
                                      "stop": first, **sampling})
        validate(cases[0], token, done, width, 1, first)
        if token != first:
            raise AssertionError("stop list changed token choice")
        stopped, done = native.request({"tokens": cases[0], "max_tokens": args.max_tokens,
                                        "cache_prompt": False, "stop": first, **sampling})
        validate(cases[0], stopped, done, width, args.max_tokens, first)
        if stopped != first:
            raise AssertionError("first-token stop did not terminate immediately")
        for index, position, target in probes[:3]:
            for budget, stops in ((position + 1, []), (args.max_tokens, [target])):
                body = {"tokens": cases[index], "max_tokens": budget, "cache_prompt": False, "stop": stops, **sampling}
                tokens, done = native.request(body)
                reference = rows[index * 4]["token_ids"]
                if tokens != reference[:len(tokens)]:
                    raise AssertionError("stop/budget probe changed the generated token prefix")
                coverage, _ = validate(cases[index], tokens, done, width, budget, stops)
                covered.update(coverage)
        return rows, covered
    finally:
        native.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", type=Path, required=True)
    parser.add_argument("--q3", type=Path, required=True)
    parser.add_argument("--q4", type=Path, required=True)
    parser.add_argument("--expert-profile", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, help="existing local tokenizer; no network fallback")
    parser.add_argument("--ctx", type=int, default=4096)
    parser.add_argument("--ubatch", type=int, default=2048)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--out-dir", type=Path, default=Path("perso/bench/speed/native-smoke"))
    args = parser.parse_args()
    for artifact in (args.engine, args.q3, args.q4, args.expert_profile, CHAT_TEMPLATE):
        if not artifact.is_file():
            parser.error(f"missing local artifact: {artifact}")
    tokenizer = args.tokenizer or local_tokenizer()
    if tokenizer is None or not tokenizer.is_file():
        parser.error("provide an existing --tokenizer (this smoke never downloads)")
    os.environ["FNS_TOKENIZER"] = str(tokenizer.resolve())
    if args.max_tokens < 4 or args.ctx < args.max_tokens + 8:
        parser.error("max-tokens must be >=4 with room in ctx for prompts")
    args.out_dir.mkdir(parents=True, exist_ok=False)
    chat = Chat()
    cases = [chat.prompt_ids([{"role": "user", "content": prompt}], None) for prompt in PROMPTS]
    if any(len(prompt) + args.max_tokens > args.ctx for prompt in cases):
        parser.error("a synthetic prompt exceeds the configured context")
    summary = {"synthetic_only": True, "release_benchmark": False, "quants": {}}
    for quant in ("q3", "q4"):
        coverage = set()
        for mode, seeds in (("greedy", (0,)), ("sample", (0, 7))):
            for seed in seeds:
                sequential, _ = run_arm(args, quant, getattr(args, quant), 0, cases, mode, seed, args.out_dir)
                speculative, observed = run_arm(args, quant, getattr(args, quant), 3, cases, mode, seed, args.out_dir)
                coverage.update(observed)
                for a, b in zip(sequential, speculative, strict=True):
                    if (a["case"], a["state"], a["token_ids"]) != (b["case"], b["state"], b["token_ids"]):
                        raise AssertionError(f"{quant}/{mode}/{seed}: sequential/speculative token divergence")
        required = {"no_draft", "full_acceptance", "budget_in_window"}
        optional = {"immediate_rejection", "partial_acceptance", "stop_in_window", "accepted_stop"}
        summary["quants"][quant] = {"covered": sorted(coverage), "missing_required": sorted(required - coverage),
                                     "unexercised_boundaries": sorted(optional - coverage)}
        if required - coverage:
            (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
            raise AssertionError(f"{quant}: required real native coverage missing: {sorted(required - coverage)}")
        print(f"PASS {quant}: cold KV, cached/default stability, greedy/sample sequential-spec parity; "
              f"boundaries={','.join(sorted(coverage))}", flush=True)
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
