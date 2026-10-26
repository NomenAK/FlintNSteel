"""FlintNSteel HTTP server: /v1/chat/completions on top of fns-serve.

  python serve/fns_server.py --model <file.gguf> [--port 8080] [--ctx 32768]
      [--ubatch 2048] [--expert-profile F] [--vocab flint-vocab.json]

Pipeline: the chat template (serve/chat_template.jinja, rendered with jinja2 as transformers would), message
normalization and tool-call parsing (normalize_messages and parse_tool_call, adapted from rec), tokenization with
`tokenizers`, the structural-tag grammar (serve/flint_grammar.py) enforced token by token by the xgrammar matcher in
fns-serve, reasoning split into `reasoning_content`, and an engine subprocess that keeps its context between requests
(the common prefix is reused). One request at a time.
Streaming (SSE, "stream": true): reasoning and text as tokens arrive; from the first <tool_call>, text is held and tool
calls are sent complete in the last delta.
Sampling: temperature, top_p, top_k, seed, presence_penalty from the request; defaults from rec (engine.py, Sampling):
1.0 / 0.95 / 20 / 0 / 0; temperature 0 is greedy.
Model files: FNS_MODELS (default ROOT/models). Tokenizer and vocabulary resolution: see serve/flint_assets.py.

Code adapted from rec (https://github.com/recursive-oss/rec, commit 8af99abaf50b33903147d89eed91d1bb3ac642dd,
engine/src/rec_mlx/chat.py and chat_template.jinja), licensed under Apache-2.0.
"""

import argparse
import bisect
import itertools
import json
import os
import re
import subprocess
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import IO

import jinja2
from tokenizers import Tokenizer
from tokenizers.decoders import DecodeStream

from flint_assets import CHAT_TEMPLATE, build_dir, tokenizer_path, vocab_path
from flint_grammar import build_flint_glm_grammar

EOS = [150001, 150003, 150005, 150018]  # Flint config.json eos_token_id
TOOL_CALL = re.compile(r"<tool_call>(.*?)</tool_call>", re.S)


# --- Adapted from rec, engine/src/rec_mlx/chat.py (Apache-2.0) -----------------------------------------------------
def normalize_messages(messages):
    normalized = []
    for original in messages:
        message = original
        if message.get("role") != "assistant" and isinstance(message.get("content"), str):
            text = message["content"].replace("\x00", "").replace("<think>", "").replace("</think>", "")
            text = re.sub(
                r"<\|(?:begin_of_text|end_of_text|system|developer|user|assistant|observation)\|>",
                lambda match: match.group().replace("<", "&lt;").replace(">", "&gt;"),
                text,
            )
            message = {**message, "content": text}
        calls = message.get("tool_calls")
        if calls:
            normalized_calls = []
            for call in calls:
                function = call.get("function", call)
                arguments = function.get("arguments")
                if isinstance(arguments, str):
                    function = {**function, "arguments": json.loads(arguments) if arguments.strip() else {}}
                normalized_calls.append({**call, "function": function} if "function" in call else function)
            message = {**message, "tool_calls": normalized_calls}
        normalized.append(message)
    return normalized


_CALL = re.compile(r"\s*([^<\s]+)\s*((?:<arg_key>.*?</arg_key>\s*<arg_value>.*?</arg_value>\s*)*)", re.S)
_ARGUMENT = re.compile(r"<arg_key>(.*?)</arg_key>\s*<arg_value>(.*?)</arg_value>", re.S)


def _accepts(schema, kind):
    types = schema.get("type")
    if types == kind or (isinstance(types, list) and kind in types):
        return True
    if kind == "string" and any(isinstance(value, str) for value in schema.get("enum", ())):
        return True
    return any(_accepts(option, kind) for key in ("anyOf", "oneOf") for option in schema.get(key, ()))


def parse_tool_call(text, tools):
    """Invert the template's tool-call rendering: string arguments are raw, others are JSON."""
    call = _CALL.fullmatch(text)
    if call is None:
        raise ValueError(f"Malformed Flint tool call: {text!r}")
    name = call[1]
    properties = {}
    for tool in tools or ():
        if tool["function"]["name"] == name:
            properties = tool["function"].get("parameters", {}).get("properties", {})
    arguments = {}
    for key, value in _ARGUMENT.findall(call[2]):
        schema = properties.get(key, {})
        if value == "null" and _accepts(schema, "null"):
            value = None
        elif not _accepts(schema, "string"):
            try:
                value = json.loads(value)
            except json.JSONDecodeError:
                pass
        arguments[key] = value
    return {"name": name, "arguments": arguments}
# ---------------------------------------------------------------------------------------------------------------------


def tojson(value, ensure_ascii=True):
    return json.dumps(value, ensure_ascii=ensure_ascii)


class Chat:
    def __init__(self):
        env = jinja2.Environment(trim_blocks=False, lstrip_blocks=False, keep_trailing_newline=True)
        env.filters["tojson"] = tojson
        self.template = env.from_string(CHAT_TEMPLATE.read_text(encoding="utf-8"))
        self.tok = Tokenizer.from_file(str(tokenizer_path()))
        # Special tokens: `tokenizers` splits them out before pre-tokenization, so the text after the end of one of
        # them is tokenized independently of what precedes it.
        self._special = re.compile("|".join(
            re.escape(t.content) for t in self.tok.get_added_tokens_decoder().values() if t.special))
        self._last = ("", [], [])  # (text, ids, end offset of each token in text) of the last prompt

    def prompt_ids(self, messages, tools):
        text = self.template.render(messages=normalize_messages(messages), tools=tools, add_generation_prompt=True)
        return self.encode_prompt(text)

    def encode_prompt(self, text):
        """Encodes text, reusing the previous prompt: an agent resends the same prefix every turn. The common prefix is
        cut after the last special token it contains; only the rest is tokenized (same result as the full text, see
        __init__)."""
        old, old_ids, old_ends = self._last
        common = len(os.path.commonprefix([old, text]))
        cut = max((m.end() for m in self._special.finditer(text, 0, common)), default=0)
        n = bisect.bisect_right(old_ends, cut)
        if n and old_ends[n - 1] != cut:  # no token boundary at cut: encode the full text
            cut = n = 0
        enc = self.tok.encode(text[cut:], add_special_tokens=False)
        ids = old_ids[:n] + enc.ids
        self._last = (text, ids, old_ends[:n] + [cut + e for _, e in enc.offsets])
        return ids

    def decode(self, ids):
        return self.tok.decode(ids, skip_special_tokens=False)


class Engine:
    """fns-serve as a subprocess; one request at a time (lock)."""

    def __init__(self, args):
        tok = str(tokenizer_path())
        vocab = Path(args.vocab)
        cmd = [args.engine or str(build_dir() / "fns-serve.exe"), "--model", args.model, "--ctx", str(args.ctx),
               "--ubatch", str(args.ubatch), "--tokenizer", tok, "--vocab", str(vocab)]
        if args.expert_profile:
            cmd += ["--expert-profile", args.expert_profile]
        cmd += ["--spec", str(args.spec)]
        if getattr(args, "spec_sample", None) is not None:
            cmd += ["--spec-sample", str(args.spec_sample)]
        if args.kv_global != "f16":
            cmd += ["--kv-global", args.kv_global]
        if args.fa_int8:
            cmd += ["--fa-int8"]
        if args.fa_swa:
            cmd += ["--fa-swa"]
        if args.attention_mma:
            cmd += ["--attention-mma"]
        if args.rebalance:
            cmd += ["--rebalance", str(args.rebalance)]
        self.p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, encoding="utf-8")
        if self.p.stdin is None or self.p.stdout is None:
            raise RuntimeError("fns-serve pipes missing despite stdin=PIPE, stdout=PIPE")
        self.stdin: IO[str] = self.p.stdin
        self.stdout: IO[str] = self.p.stdout
        ready = json.loads(self.stdout.readline())
        if ready.get("ready") is not True:
            raise SystemExit("fns-serve failed to start")
        self.grammar = bool(ready.get("grammar"))
        self.lock = threading.Lock()
        self.ctx = args.ctx
        self.next_id = 0
        self.current = None  # id of the request in progress (cancellation)

    def stream(self, ids, max_tokens, grammar=None, sampling=None, ignore_eos=False, cache_prompt=True):
        """Yields tokens as they are generated; the last item is the final fns-serve message (dict)."""
        if type(cache_prompt) is not bool:
            raise ValueError("cache_prompt must be a boolean")
        # With a grammar, the turn ends with <|end_of_text|> (text) or <|observation|> (tool calls), and the matcher
        # alone decides: <|user|> is an allowed end before <|end_of_text|>.
        stop = [] if ignore_eos else [150001, 150005] if grammar is not None else EOS
        req = {"tokens": ids, "max_tokens": max_tokens, "stop": stop, **(sampling or {}), "cache_prompt": cache_prompt}
        if grammar is not None:
            req["grammar"] = grammar
        with self.lock:
            self.next_id += 1
            self.current = req["id"] = self.next_id
            self.stdin.write(json.dumps(req) + "\n")
            self.stdin.flush()
            try:
                while True:
                    m = json.loads(self.stdout.readline())
                    if "token" in m:
                        yield m["token"]
                    elif "error" in m:
                        raise RuntimeError(m["error"])
                    else:
                        yield m
                        return
            finally:
                self.current = None

    def cancel(self):
        """Stops the current generation at the next token; call from the thread that consumes stream()."""
        if self.current is not None:
            self.stdin.write(json.dumps({"cancel": self.current}) + "\n")
            self.stdin.flush()


def finish_message(chat, out, tools):
    """Final assistant message (text, reasoning, tool calls) and finish reason, from the generated tokens."""
    stopped = bool(out) and out[-1] in EOS
    body = out[:-1] if stopped else out
    text = chat.decode(body)
    # <|user|> and <|observation|> (turn ends allowed by the grammar) are not content.
    for marker in ("<|observation|>", "<|user|>"):
        if text.endswith(marker):
            text = text[: -len(marker)]
    reasoning = None
    if text.startswith("<think>") and "</think>" in text:
        reasoning, text = text[len("<think>"):].split("</think>", 1)
    message = {"role": "assistant", "content": text}
    if reasoning is not None:
        message["reasoning_content"] = reasoning
    # A malformed call (possible without grammar, or with ignore_eos which generates past the end of the turn) stays
    # in the text instead of failing the request.
    calls = []
    for c in TOOL_CALL.findall(text):
        try:
            calls.append(parse_tool_call(c, tools))
        except ValueError:
            calls = []
            break
    if calls:
        message["content"] = TOOL_CALL.sub("", text).strip() or None
        message["tool_calls"] = [{"id": f"call_{uuid.uuid4().hex[:24]}", "type": "function",
                                  "function": {"name": call["name"],
                                               "arguments": json.dumps(call["arguments"], ensure_ascii=False)}}
                                 for call in calls]
    finish = "tool_calls" if calls else ("stop" if stopped else "length")
    return message, finish


class StreamSplitter:
    """Splits generated text into SSE deltas: reasoning (<think>...</think>) then content. Text from the first
    <tool_call> (or an end marker) is held back: tool calls are sent whole in the last delta."""

    HOLD = ("<tool_call>", "<|observation|>", "<|user|>", "<|end_of_text|>")

    def __init__(self):
        self.text = ""
        self.sent = 0          # characters of self.text already sent or consumed
        self.state = "start"   # start -> think -> content -> held

    def feed(self, piece):
        self.text += piece
        deltas = []
        while True:
            rest = self.text[self.sent:]
            if self.state == "start":
                if len(rest) < len("<think>") and "<think>".startswith(rest):
                    break  # possibly the start of <think>
                if rest.startswith("<think>"):
                    self.sent += len("<think>")
                    self.state = "think"
                else:
                    self.state = "content"
                continue
            if self.state == "think":
                end = rest.find("</think>")
                if end >= 0:
                    if end:
                        deltas.append(("reasoning_content", rest[:end]))
                    self.sent += end + len("</think>")
                    self.state = "content"
                    continue
                safe = len(rest) - (len("</think>") - 1)  # do not split a </think> in progress
                if safe > 0:
                    deltas.append(("reasoning_content", rest[:safe]))
                    self.sent += safe
                break
            if self.state == "content":
                cut = min([i for i in (rest.find(h) for h in self.HOLD) if i >= 0], default=-1)
                if cut >= 0:
                    if cut:
                        deltas.append(("content", rest[:cut]))
                    self.sent += cut
                    self.state = "held"
                    break
                # Keep a tail that could start a held marker.
                keep = max((k for h in self.HOLD for k in range(1, len(h)) if rest.endswith(h[:k])), default=0)
                if len(rest) - keep > 0:
                    deltas.append(("content", rest[:len(rest) - keep]))
                    self.sent += len(rest) - keep
                break
            break  # held
        return deltas


def make_handler(chat, engine, model_name, log=None, record=None, log_tokens=False):
    log_lock = threading.Lock()
    if log_tokens and log is None:
        raise ValueError("log_tokens requires a local log")

    class Handler(BaseHTTPRequestHandler):
        def reply(self, status, value):
            body = json.dumps(value, ensure_ascii=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/v1/models":
                self.reply(200, {"object": "list", "data": [{"id": model_name, "object": "model"}]})
            else:
                self.reply(404, {"error": "Unknown endpoint"})

        def do_POST(self):
            if self.path != "/v1/chat/completions":
                self.reply(404, {"error": "Unknown endpoint"})
                return
            try:
                req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
                cache_prompt = req.get("cache_prompt", True)
                if type(cache_prompt) is not bool:
                    raise ValueError("cache_prompt must be a boolean")
                request_id = self.headers.get("X-Request-ID")
                if request_id is not None and re.fullmatch(r"[A-Za-z0-9._:-]{1,128}", request_id) is None:
                    raise ValueError("X-Request-ID must contain 1 to 128 ASCII letters, digits, '.', '_', ':', or '-'")
                if record is not None:
                    with log_lock:
                        record.write(json.dumps(req, ensure_ascii=False) + "\n")
                        record.flush()
                tools = req.get("tools")
                ids = chat.prompt_ids(req["messages"], tools)
                # ignore_eos (as llama-server, to replay a trace at fixed length): no stop and no grammar.
                ignore_eos = bool(req.get("ignore_eos"))
                grammar = build_flint_glm_grammar(chat.tok, tools) if engine.grammar and not ignore_eos else None
                sampling = {
                    "temperature": float(req.get("temperature", 1.0)), "top_p": float(req.get("top_p", 0.95)),
                    "top_k": int(req.get("top_k", 20)), "seed": int(req.get("seed", 0)),
                    "presence_penalty": float(req.get("presence_penalty", 0.0)),
                }
                if (not 0 <= sampling["temperature"] <= 2 or not 0 < sampling["top_p"] <= 1 or sampling["top_k"] < 0
                        or not -2 <= sampling["presence_penalty"] <= 2):
                    raise ValueError("temperature must be in [0, 2], top_p in (0, 1], top_k nonnegative, "
                                     "presence_penalty in [-2, 2]")
                # As rec: absent or too large max_tokens means the rest of the context.
                remaining = engine.ctx - len(ids)
                if remaining < 1:
                    raise ValueError(f"prompt of {len(ids)} tokens exceeds the {engine.ctx}-token context")
                asked = req.get("max_tokens", req.get("max_completion_tokens"))
                max_tokens = remaining if asked is None else min(int(asked), remaining)
                if max_tokens < 1:
                    raise ValueError("max_tokens must be positive")
                gen = engine.stream(ids, max_tokens, grammar, sampling, ignore_eos, cache_prompt=cache_prompt)
                first = next(gen)  # prefill and first token: request errors are raised here
            except (ValueError, KeyError, RuntimeError) as e:
                self.reply(400, {"error": str(e)})
                return
            cid, created = f"chatcmpl-{uuid.uuid4().hex}", int(time.time())
            stream = bool(req.get("stream"))
            gone = []  # client disconnected: cancel generation (fns-serve stops at the next token)

            def chunk(delta, finish=None, usage=None):
                if gone:
                    return
                c = {"id": cid, "object": "chat.completion.chunk", "created": created, "model": model_name,
                     "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}
                if usage is not None:
                    c["usage"] = usage
                try:
                    self.wfile.write(b"data: " + json.dumps(c, ensure_ascii=False).encode() + b"\n\n")
                    self.wfile.flush()
                except OSError:
                    gone.append(True)
                    engine.cancel()

            out, info = [], {}
            split, dec = StreamSplitter(), DecodeStream(skip_special_tokens=False)
            sent = {"content": "", "reasoning_content": ""}
            if stream:
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.end_headers()
                chunk({"role": "assistant"})
            for item in itertools.chain([first], gen):
                if isinstance(item, dict):
                    info = item
                    break
                out.append(item)
                if stream:
                    for field, text in split.feed(dec.step(chat.tok, item) or ""):
                        sent[field] += text
                        chunk({field: text})
            message, finish = finish_message(chat, out, tools)
            if info.get("reason") == "length":
                finish = "length"  # budget takes precedence, including when its last token happens to be EOS
            usage = {"prompt_tokens": len(ids), "completion_tokens": len(out), "total_tokens": len(ids) + len(out)}
            stats = {"reused_prefix": info.get("reused"), "prefill_ms": info.get("prefill_ms"),
                     "decode_ms": info.get("decode_ms"), "grammar": grammar is not None}
            if log is not None:
                with log_lock:
                    entry = {"t": time.time(), "request_id": request_id, "prompt_tokens": len(ids),
                             "completion_tokens": len(out), "finish": finish, "reason": info.get("reason"),
                             "stream": stream, **stats}
                    for name in ("spec_windows", "spec_drafted", "spec_accepted", "spec_emitted", "decode_iterations"):
                        entry[name] = info.get(name)
                    if log_tokens:
                        entry["token_ids"] = out
                    log.write(json.dumps(entry) + "\n")
                    log.flush()
            if stream:
                # What the splitter held back is sent before the complete tool calls.
                for field in ("reasoning_content", "content"):
                    final = message.get(field) or ""
                    if final.startswith(sent[field]) and len(final) > len(sent[field]):
                        chunk({field: final[len(sent[field]):]})
                if message.get("tool_calls"):
                    chunk({"tool_calls": [{"index": i, **c} for i, c in enumerate(message["tool_calls"])]})
                chunk({}, finish, usage)
                if not gone:
                    try:
                        self.wfile.write(b"data: [DONE]\n\n")
                        self.wfile.flush()
                    except OSError:
                        pass
                return
            self.reply(200, {
                "id": cid, "object": "chat.completion", "created": created, "model": model_name,
                "choices": [{"index": 0, "message": message, "finish_reason": finish}],
                "usage": usage, "fns": stats,
            })

        def log_message(self, format, *args):
            pass

    return Handler


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--ctx", type=int, default=32768)
    ap.add_argument("--ubatch", type=int, default=2048)
    ap.add_argument("--expert-profile")
    ap.add_argument("--engine", help="fns-serve executable (default: build/fns-serve.exe); for binary A/B tests")
    ap.add_argument("--spec", type=int, default=3,
                    help="prompt-lookup speculation (same tokens): window of W tokens in greedy mode, 2 when sampling "
                         "(2 to 8; 0: disabled)")
    ap.add_argument("--spec-sample", type=int,
                    help="sampling speculation width (0 or 2 to --spec; default min(2, --spec))")
    ap.add_argument("--kv-global", choices=["f16", "q8_0"], default="f16",
                    help="KV cache of the global layers (q8_0: opt-in, changes the bits)")
    ap.add_argument("--fa-int8", action="store_true",
                    help="INT8 FlashAttention for prefill, global layers (opt-in, changes the bits)")
    ap.add_argument("--fa-swa", action="store_true",
                    help="FlashAttention for prefill of sliding-window layers, visible tiles only (opt-in, "
                         "changes the bits)")
    ap.add_argument("--attention-mma", action="store_true",
                    help="MMA global attention for decode/verification (opt-in, changes the bits; F16 KV)")
    ap.add_argument("--rebalance", type=int, default=0,
                    help="adaptive expert cache every N tokens (hybrid Q4, same bits); 0: static")
    ap.add_argument("--vocab", default=str(vocab_path()),
                    help="Flint token list (tools/xgrammar/export_vocab.py), used by the grammar")
    ap.add_argument("--log", help="local per-request JSONL metrics (no generated token IDs by default)")
    ap.add_argument("--log-tokens", action="store_true", help="include private generated token IDs in --log")
    ap.add_argument("--record", help="record each request body (JSONL), for bench/scripts/agent_replay.py")
    o = ap.parse_args()
    if o.log_tokens and not o.log:
        ap.error("--log-tokens requires --log")
    if o.spec_sample is not None and (o.spec_sample < 0 or o.spec_sample == 1 or o.spec_sample > o.spec):
        ap.error("--spec-sample must be 0 or 2 to --spec")
    chat = Chat()
    engine = Engine(o)
    log = open(o.log, "a", encoding="utf-8") if o.log else None
    record = open(o.record, "a", encoding="utf-8") if o.record else None
    server = ThreadingHTTPServer(("127.0.0.1", o.port),
                                make_handler(chat, engine, Path(o.model).stem, log, record, o.log_tokens))
    print(f"fns_server : http://127.0.0.1:{o.port}/v1", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
