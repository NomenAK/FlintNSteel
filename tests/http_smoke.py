#!/usr/bin/env python3
"""Live HTTP smoke test for the FlintNSteel OpenAI-compatible server (serve/fns_server.py).

Run against a server that is already started, for example:

    python serve/fns_server.py --model model.gguf --port 8080
    python tests/http_smoke.py --url http://127.0.0.1:8080/v1

No mock server is used. Checks cover response structure and usage, cached-to-cached output equality, SSE framing and
reconstruction compared with the non-streamed response for the same prompt, a read_file tool call (non-streamed and
streamed), and a client disconnect followed by a new request.
Free-text output is never compared against a fixed string. Cold and cached generations are not compared, because the
prefill geometry differs between them.

Exit status is 0 when every check passes, 1 otherwise.
"""

import argparse
import http.client
import json
import sys
import time
import urllib.parse
import urllib.request

PROMPT = "Use read_file to read README.md. Do not guess its content."
SENTENCE_PROMPT = "Reply with one short sentence."
READ_FILE_TOOL = {
    "type": "function",
    "function": {
        "name": "read_file",
        "description": "Read a UTF-8 text file from the workspace.",
        "parameters": {
            "type": "object",
            "properties": {"path": {"type": "string", "description": "Path of the file to read."}},
            "required": ["path"],
        },
    },
}
FINISH_REASONS = {"stop", "length", "tool_calls"}
GREEDY = {"temperature": 0, "top_k": 1, "max_tokens": 256}

failures = []


def check(condition, message):
    print(("PASS " if condition else "FAIL ") + message, flush=True)
    if not condition:
        failures.append(message)


def endpoint(base, path):
    return base.rstrip("/") + path


def get_json(base, timeout):
    with urllib.request.urlopen(endpoint(base, "/models"), timeout=timeout) as r:
        return json.loads(r.read())


def post_json(base, body, timeout):
    req = urllib.request.Request(endpoint(base, "/chat/completions"), data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def open_stream(base, body, timeout):
    parts = urllib.parse.urlsplit(base)
    conn = http.client.HTTPConnection(parts.hostname, parts.port, timeout=timeout)
    conn.request("POST", endpoint(parts.path, "/chat/completions"), body=json.dumps(body).encode(),
                 headers={"Content-Type": "application/json", "Accept": "text/event-stream"})
    return conn, conn.getresponse()


def close_stream(conn, response):
    # The response holds the socket's file object: close it first, then the connection, so the socket really closes.
    response.close()
    conn.close()


def read_events(response):
    """Yield the decoded JSON payloads of SSE data lines; the terminal [DONE] payload is yielded as the string."""
    for raw in response:
        line = raw.decode("utf-8").rstrip("\r\n")
        if not line.startswith("data: "):
            continue
        payload = line[len("data: "):]
        if payload == "[DONE]":
            yield "[DONE]"
            return
        yield json.loads(payload)


def check_usage(usage, label):
    keys = ("prompt_tokens", "completion_tokens", "total_tokens")
    check(isinstance(usage, dict) and all(isinstance(usage.get(k), int) for k in keys),
          f"{label}: usage has integer prompt/completion/total tokens")
    if isinstance(usage, dict) and all(isinstance(usage.get(k), int) for k in keys):
        check(usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"],
              f"{label}: total_tokens equals prompt plus completion tokens")
        check(usage["completion_tokens"] >= 1, f"{label}: at least one completion token generated")


def check_nonstream_structure(resp, label):
    check(resp.get("object") == "chat.completion", f"{label}: object is chat.completion")
    check(isinstance(resp.get("id"), str) and resp["id"].startswith("chatcmpl-"), f"{label}: id has chatcmpl prefix")
    choices = resp.get("choices")
    check(isinstance(choices, list) and len(choices) == 1, f"{label}: exactly one choice")
    if not choices:
        return None
    choice = choices[0]
    message = choice.get("message", {})
    check(choice.get("index") == 0 and message.get("role") == "assistant", f"{label}: choice 0 is assistant message")
    check(choice.get("finish_reason") in FINISH_REASONS, f"{label}: finish_reason is one of {sorted(FINISH_REASONS)}")
    check_usage(resp.get("usage"), label)
    return choice


def parse_tool_arguments(call, label):
    arguments = call.get("function", {}).get("arguments")
    check(isinstance(arguments, str), f"{label}: tool arguments are a JSON string")
    try:
        parsed = json.loads(arguments)
    except (TypeError, ValueError):
        check(False, f"{label}: tool arguments parse as JSON")
        return None
    check(isinstance(parsed, dict), f"{label}: tool arguments are a JSON object")
    return parsed


def check_read_file_call(calls, label):
    check(bool(calls) and calls[0].get("function", {}).get("name") == "read_file",
          f"{label}: first tool call is read_file")
    if not calls:
        return
    args = parse_tool_arguments(calls[0], label)
    if args is not None:
        check(isinstance(args.get("path"), str) and args["path"].strip() != "",
              f"{label}: read_file arguments include a non-empty string path")


def tool_body(stream=False):
    return {"messages": [{"role": "user", "content": PROMPT}], "tools": [READ_FILE_TOOL], "stream": stream,
            **GREEDY}


def collect_stream(base, body, timeout):
    """POST a streamed request and return (status, events) after the body is fully read and the socket closed."""
    conn, response = open_stream(base, body, timeout)
    try:
        status = response.status
        content_type = response.getheader("Content-Type", "")
        events = list(read_events(response))
    finally:
        close_stream(conn, response)
    return status, content_type, events


def stream_text(chunks):
    """Reconstruct (content, reasoning_content, finish_reasons) from the chunk deltas."""
    content = reasoning = ""
    finishes = []
    for c in chunks:
        for choice in c.get("choices", []):
            delta = choice.get("delta", {})
            content += delta.get("content") or ""
            reasoning += delta.get("reasoning_content") or ""
            if choice.get("finish_reason") is not None:
                finishes.append(choice["finish_reason"])
    return content, reasoning, finishes


def run_nonstream(base, model, timeout):
    print("-- nonstream structure and usage", flush=True)
    body = {"model": model, "messages": [{"role": "user", "content": SENTENCE_PROMPT}],
            "stream": False, **GREEDY, "max_tokens": 64}
    resp = post_json(base, body, timeout)
    check_nonstream_structure(resp, "nonstream")


def run_cache_equality(base, model, timeout):
    """The first request warms the prefix cache; the next two are cached and must agree with each other."""
    print("-- repeated cache equality (cached-to-cached only)", flush=True)
    body = {"model": model, "messages": [{"role": "user", "content": "Count from one to five."}],
            "stream": False, **GREEDY, "max_tokens": 64}
    post_json(base, body, timeout)  # warm-up: cold generation, not compared
    second = post_json(base, body, timeout)
    third = post_json(base, body, timeout)
    a = check_nonstream_structure(second, "cached call 2")
    b = check_nonstream_structure(third, "cached call 3")
    if a is None or b is None:
        return
    check(a["message"].get("content") == b["message"].get("content"),
          "cached calls 2 and 3 return identical content")
    check(a["message"].get("reasoning_content") == b["message"].get("reasoning_content"),
          "cached calls 2 and 3 return identical reasoning_content")
    check(a["finish_reason"] == b["finish_reason"],
          "cached calls 2 and 3 return the same finish_reason")
    stats = third.get("fns") or {}
    if isinstance(stats.get("reused_prefix"), int):
        check(stats["reused_prefix"] > 0, "cached call 3 reuses a non-empty prefix")


def run_stream_matches_nonstream(base, model, timeout):
    """Same prompt and sampling: the streamed deltas must reconstruct exactly the non-streamed message text."""
    print("-- SSE reconstruction equals cached nonstream response", flush=True)
    body = {"model": model, "messages": [{"role": "user", "content": SENTENCE_PROMPT}], **GREEDY, "max_tokens": 64}
    post_json(base, {**body, "stream": False}, timeout)  # warm-up for this exact prompt, not compared
    reference = post_json(base, {**body, "stream": False}, timeout)  # cached nonstream reference
    ref_choice = check_nonstream_structure(reference, "nonstream reference")
    status, content_type, events = collect_stream(base, {**body, "stream": True}, timeout)
    check(status == 200, "stream reconstruction: HTTP 200")
    check(content_type.startswith("text/event-stream"), "stream reconstruction: text/event-stream type")
    check(bool(events) and events[-1] == "[DONE]", "stream reconstruction: last SSE event is [DONE]")
    chunks = [e for e in events if isinstance(e, dict)]
    check(bool(chunks) and all(c.get("object") == "chat.completion.chunk" for c in chunks),
          "stream reconstruction: every data event is a chat.completion.chunk")
    content, reasoning, finishes = stream_text(chunks)
    check(len(finishes) == 1 and finishes[0] in FINISH_REASONS,
          "stream reconstruction: exactly one finish_reason")
    if ref_choice is None:
        return
    message = ref_choice["message"]
    check(content == (message.get("content") or ""),
          "stream reconstruction: content deltas equal nonstream content exactly")
    check(reasoning == (message.get("reasoning_content") or ""),
          "stream reconstruction: reasoning deltas equal nonstream reasoning_content exactly")
    check(finishes == [ref_choice["finish_reason"]],
          "stream reconstruction: finish_reason equals nonstream finish_reason")
    if chunks and "usage" in chunks[-1]:
        check_usage(chunks[-1]["usage"], "stream reconstruction")


def run_nonstream_tool(base, model, timeout):
    print("-- nonstream read_file tool call", flush=True)
    resp = post_json(base, {"model": model, **tool_body(stream=False)}, timeout)
    choice = check_nonstream_structure(resp, "tool nonstream")
    if choice is None:
        return
    message = choice["message"]
    check(choice["finish_reason"] == "tool_calls", "tool nonstream: finish_reason is tool_calls")
    check_read_file_call(message.get("tool_calls") or [], "tool nonstream")
    for call in message.get("tool_calls") or []:
        check(isinstance(call.get("id"), str) and call.get("type") == "function",
              "tool nonstream: call has id and type function")


def run_stream_tool(base, model, timeout):
    print("-- streamed read_file tool call", flush=True)
    status, _, events = collect_stream(base, {"model": model, **tool_body(stream=True)}, timeout)
    check(status == 200, "stream tool: HTTP 200")
    check(bool(events) and events[-1] == "[DONE]", "stream tool: last SSE event is [DONE]")
    chunks = [e for e in events if isinstance(e, dict)]
    calls = []
    for c in chunks:
        for choice in c.get("choices", []):
            calls.extend(choice.get("delta", {}).get("tool_calls") or [])
    finish = [c["choices"][0]["finish_reason"] for c in chunks if c.get("choices")
              and c["choices"][0].get("finish_reason") is not None]
    check(finish == ["tool_calls"], "stream tool: finish_reason is tool_calls")
    check_read_file_call(calls, "stream tool")


def run_disconnect(base, model, timeout, server_log):
    """Start a long ignore_eos stream, drop it after the first data event, then require the next request to succeed.

    ignore_eos keeps generation running to max_tokens, so fewer completion tokens in the server log means the server
    cancelled the request rather than stopping naturally at EOS.
    """
    print("-- client disconnect then next request", flush=True)
    before = count_lines(server_log)
    body = {"model": model, "messages": [{"role": "user", "content": "Write a long story about a river."}],
            "stream": True, **GREEDY, "max_tokens": 256, "ignore_eos": True}
    conn, response = open_stream(base, body, timeout)
    try:
        check(response.status == 200, "disconnect: stream started with HTTP 200")
        for raw in response:
            if raw.decode("utf-8").startswith("data: "):
                break
    finally:
        close_stream(conn, response)  # really drops the socket, so the server sees the client leave mid-generation
    follow = post_json(base, {"model": model, "messages": [{"role": "user", "content": "Say ok."}], "stream": False,
                              **GREEDY, "max_tokens": 16}, timeout)
    check_nonstream_structure(follow, "follow-up after disconnect")
    if server_log:
        wait_for_lines(server_log, before + 2, timeout)
        new = read_log(server_log)[before:]
        check(len(new) >= 2, "disconnect: server logged the cancelled request and the follow-up")
        if len(new) >= 2:
            cancelled = new[0]
            check(cancelled.get("stream") is True, "disconnect: logged cancelled request was streamed")
            check(cancelled.get("completion_tokens", 0) < 256,
                  "disconnect: cancelled request stopped before max_tokens (ignore_eos forces length otherwise)")


def count_lines(path):
    return len(read_log(path)) if path else 0


def read_log(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def wait_for_lines(path, count, timeout):
    deadline = time.monotonic() + timeout
    while len(read_log(path)) < count and time.monotonic() < deadline:
        time.sleep(0.5)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://127.0.0.1:8080/v1",
                    help="base URL of the running server, including /v1 (default: %(default)s)")
    ap.add_argument("--timeout", type=float, default=600.0, help="per-request timeout in seconds")
    ap.add_argument("--server-log", help="JSONL file written by fns_server --log; enables native cancel checks")
    args = ap.parse_args()

    models = get_json(args.url, args.timeout)
    model_ids = [m.get("id") for m in models.get("data", [])]
    check(bool(model_ids), "models: /v1/models lists at least one model")
    if not model_ids:
        return 1
    model = model_ids[0]

    run_nonstream(args.url, model, args.timeout)
    run_cache_equality(args.url, model, args.timeout)
    run_stream_matches_nonstream(args.url, model, args.timeout)
    run_nonstream_tool(args.url, model, args.timeout)
    run_stream_tool(args.url, model, args.timeout)
    run_disconnect(args.url, model, args.timeout, args.server_log)

    if failures:
        print(f"{len(failures)} check(s) failed", flush=True)
        return 1
    print("all HTTP smoke checks passed", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
