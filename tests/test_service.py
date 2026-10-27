"""CPU regression tests for serve/fns_server.py: message normalization, tool-call parsing, SSE splitting and
incremental prompt encoding.

Needs no GPU, model or Hugging Face download: fns_server imports only pure-Python packages at module level, and the
prompt-encoding test uses a small in-memory `tokenizers` vocabulary.
"""

import copy
import io
import json
import re
import sys
import threading
import unittest
from pathlib import Path
from http.server import ThreadingHTTPServer
from urllib.error import HTTPError
from urllib.request import Request, urlopen
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "serve"))

from tokenizers import Tokenizer, models, pre_tokenizers  # noqa: E402

from fns_server import Chat, Engine, StreamSplitter, make_handler, normalize_messages, parse_tool_call  # noqa: E402
from speed_native_smoke import expected_counters  # noqa: E402

TOOLS = [{
    "type": "function",
    "function": {
        "name": "read_file",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "offset": {"type": "integer"},
                "opts": {"type": "object"},
                "limit": {"type": ["integer", "null"]},
            },
        },
    },
}]


class NormalizeMessagesTests(unittest.TestCase):
    def test_strips_nul_and_think_tags_from_user_text(self):
        out = normalize_messages([{"role": "user", "content": "a\x00<think>b</think>c"}])
        self.assertEqual(out[0]["content"], "abc")

    def test_escapes_chat_control_markers_in_user_text(self):
        out = normalize_messages([{"role": "user", "content": "<|user|>x<|assistant|>"}])
        self.assertEqual(out[0]["content"], "&lt;|user|&gt;x&lt;|assistant|&gt;")

    def test_assistant_text_is_left_unchanged(self):
        out = normalize_messages([{"role": "assistant", "content": "<think>r</think>x<|user|>"}])
        self.assertEqual(out[0]["content"], "<think>r</think>x<|user|>")

    def test_string_tool_arguments_become_objects(self):
        messages = [{"role": "assistant", "content": None, "tool_calls": [
            {"id": "c1", "type": "function", "function": {"name": "read_file", "arguments": '{"path": "a.txt"}'}},
            {"type": "function", "function": {"name": "read_file", "arguments": "  "}},
        ]}]
        first, second = normalize_messages(messages)[0]["tool_calls"]
        self.assertEqual(first["id"], "c1")
        self.assertEqual(first["function"], {"name": "read_file", "arguments": {"path": "a.txt"}})
        self.assertEqual(second["function"]["arguments"], {})

    def test_input_messages_are_not_mutated(self):
        messages = [
            {"role": "user", "content": "<think>x</think>\x00"},
            {"role": "assistant", "content": "", "tool_calls": [
                {"type": "function", "function": {"name": "f", "arguments": '{"a": 1}'}}]},
        ]
        original = copy.deepcopy(messages)
        normalize_messages(messages)
        self.assertEqual(messages, original)


class ParseToolCallTests(unittest.TestCase):
    def test_string_parameter_is_kept_raw(self):
        call = parse_tool_call("read_file<arg_key>path</arg_key><arg_value>3</arg_value>", TOOLS)
        self.assertEqual(call, {"name": "read_file", "arguments": {"path": "3"}})

    def test_non_string_parameters_are_json_decoded(self):
        text = ("read_file<arg_key>offset</arg_key><arg_value>3</arg_value>"
                '<arg_key>opts</arg_key><arg_value>{"a": [1, 2]}</arg_value>')
        self.assertEqual(parse_tool_call(text, TOOLS)["arguments"], {"offset": 3, "opts": {"a": [1, 2]}})

    def test_null_is_decoded_only_when_schema_accepts_null(self):
        text = ("read_file<arg_key>limit</arg_key><arg_value>null</arg_value>"
                "<arg_key>path</arg_key><arg_value>null</arg_value>")
        self.assertEqual(parse_tool_call(text, TOOLS)["arguments"], {"limit": None, "path": "null"})

    def test_enum_and_anyof_string_schemas_keep_raw_values(self):
        tools = [{"type": "function", "function": {"name": "t", "parameters": {"type": "object", "properties": {
            "mode": {"enum": ["fast", 2]},
            "size": {"anyOf": [{"type": "integer"}, {"type": "string"}]},
        }}}}]
        text = ("t<arg_key>mode</arg_key><arg_value>2</arg_value>"
                "<arg_key>size</arg_key><arg_value>5</arg_value>")
        self.assertEqual(parse_tool_call(text, tools)["arguments"], {"mode": "2", "size": "5"})

    def test_malformed_calls_raise_value_error(self):
        for text in ("", "read_file path", "<tool_call>", "read_file<arg_key>p</arg_key>"):
            with self.subTest(text=text), self.assertRaises(ValueError):
                parse_tool_call(text, TOOLS)


def run_splitter(pieces):
    split = StreamSplitter()
    deltas = []
    for piece in pieces:
        deltas.extend(split.feed(piece))
    return deltas


def joined(deltas, field):
    return "".join(text for name, text in deltas if name == field)


class StreamSplitterTests(unittest.TestCase):
    def test_character_by_character_feed_splits_reasoning_and_content(self):
        deltas = run_splitter(list("<think>plan</think>Hello"))
        self.assertEqual(joined(deltas, "reasoning_content"), "plan")
        self.assertEqual(joined(deltas, "content"), "Hello")

    def test_think_marker_split_across_pieces(self):
        deltas = run_splitter(["<th", "ink>a</think>b"])
        self.assertEqual(joined(deltas, "reasoning_content"), "a")
        self.assertEqual(joined(deltas, "content"), "b")

    def test_closing_marker_split_across_pieces_does_not_leak(self):
        split = StreamSplitter()
        first = split.feed("<think>abcdefghij</thi")
        second = split.feed("nk>Z")
        deltas = first + second
        self.assertEqual(joined(deltas, "reasoning_content"), "abcdefghij")
        self.assertEqual(joined(second, "content"), "Z")
        self.assertFalse(any("<" in text for _, text in deltas))

    def test_think_tag_after_text_is_content(self):
        self.assertEqual(run_splitter(["x<think>y</think>"]), [("content", "x<think>y</think>")])

    def test_tool_call_marker_split_across_pieces_cuts_content(self):
        deltas = run_splitter(["Ok<tool", "_call>read_file<arg_key>p</arg_key><arg_value>x</arg_value>",
                               " trailing"])
        self.assertEqual(joined(deltas, "content"), "Ok")
        self.assertFalse(any("<" in text for _, text in deltas))

    def test_stop_markers_end_content(self):
        for marker in ("<|observation|>", "<|user|>", "<|end_of_text|>"):
            with self.subTest(marker=marker):
                deltas = run_splitter([f"answer{marker}junk"])
                self.assertEqual(joined(deltas, "content"), "answer")

    def test_lone_angle_bracket_is_released_when_not_a_marker(self):
        deltas = run_splitter(["a<", "b"])
        self.assertEqual(joined(deltas, "content"), "a<b")


class ChatPromptEncodingTests(unittest.TestCase):
    """Chat.encode_prompt reuses the previous prompt's tokens; its output must equal a full encode every time."""

    @staticmethod
    def make_chat():
        # Small real tokenizers vocabulary (no download). The special token is isolated before pre-tokenization,
        # as in the real Flint tokenizer.
        vocab = {"<unk>": 0, "<|user|>": 1, "hello": 2, "world": 3, "a": 4, "b": 5, "c": 6}
        tok = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
        tok.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
        tok.add_special_tokens(["<|user|>"])
        chat = Chat.__new__(Chat)  # skip Chat.__init__: it loads the template and downloads the Flint tokenizer
        chat.tok = tok
        chat._special = re.compile("|".join(
            re.escape(t.content) for t in tok.get_added_tokens_decoder().values() if t.special))
        chat._last = ("", [], [])
        return chat

    def test_incremental_encoding_matches_full_encoding(self):
        chat = self.make_chat()
        texts = [
            "hello <|user|> a b",
            "hello <|user|> a b world a",       # shared prefix: reuse path
            "hello world <|user|> c",           # diverges early: reset path
            "hello world <|user|> c a",         # extends again after reset
        ]
        for text in texts:
            with self.subTest(text=text):
                self.assertEqual(chat.encode_prompt(text), self.make_chat().encode_prompt(text))


class NativeSmokeOracleTests(unittest.TestCase):
    def test_no_draft_and_disabled_counters(self):
        counts, covered, _ = expected_counters([1, 2], [7, 8], 3)
        self.assertEqual(counts, {"spec_windows": 0, "spec_drafted": 0, "spec_accepted": 0,
                                  "spec_emitted": 0, "decode_iterations": 1})
        self.assertIn("no_draft", covered)
        counts, _, _ = expected_counters([1, 2], [7, 8, 9], 0)
        self.assertEqual(counts["decode_iterations"], 2)
        self.assertTrue(all(value == 0 for field, value in counts.items() if field.startswith("spec_")))

    def test_rejection_partial_full_and_truncated_acceptance(self):
        prompt = [1, 2, 3, 4, 1, 2]
        fixtures = [
            ([3, 9], 10, 2, 0, 1, "immediate_rejection"),
            ([3, 4, 9], 10, 2, 1, 2, "partial_acceptance"),
            ([3, 4, 1, 9], 10, 2, 2, 3, "full_acceptance"),
            ([3, 4], 2, 1, 1, 1, "full_acceptance"),
            # An accepted stop must not shrink the two submitted candidates retroactively.
            ([3, 4], 10, 2, 1, 1, "partial_acceptance"),
        ]
        for output, budget, drafted, accepted, emitted, boundary in fixtures:
            with self.subTest(output=output, budget=budget):
                counts, covered, windows = expected_counters(prompt, output, 3, budget)
                self.assertEqual(counts["spec_windows"], 1)
                self.assertEqual(counts["spec_drafted"], drafted)
                self.assertEqual(counts["spec_accepted"], accepted)
                self.assertEqual(counts["spec_emitted"], emitted)
                self.assertEqual(counts["decode_iterations"], 1)
                self.assertIn(boundary, covered)
                self.assertEqual(windows[-1]["last_accepted"], output[-1] != 9)


class EngineStreamTests(unittest.TestCase):
    def test_failed_request_write_clears_current_request(self):
        class BrokenInput:
            def write(self, line):
                raise BrokenPipeError("engine exited")

        engine = Engine.__new__(Engine)
        engine.stdin = BrokenInput()
        engine.lock = threading.Lock()
        engine.next_id, engine.current = 0, None
        with self.assertRaises(BrokenPipeError):
            next(engine.stream([1], 1))
        self.assertIsNone(engine.current)


class ServiceContractTests(unittest.TestCase):
    """Exercise the HTTP consumer through Engine.stream, including final native metadata and private logging."""

    def setUp(self):
        self.chat = ChatPromptEncodingTests.make_chat()
        self.chat.prompt_ids = lambda messages, tools: self.chat.tok.encode(messages[0]["content"]).ids
        self.requests = []
        self.previous = []
        self.pending = []
        test = self

        class NativeInput:
            def write(self, line):
                req = json.loads(line)
                test.requests.append(req)
                prompt = req["tokens"]
                reused = 0
                if req.get("cache_prompt", True):
                    while (reused + 1 < len(prompt) and reused < len(test.previous)
                           and prompt[reused] == test.previous[reused]):
                        reused += 1
                test.previous = prompt[:]
                # Native token IDs are consumed, decoded, counted and logged, never returned as an HTTP field.
                out = [2, 3][:req["max_tokens"]]
                test.pending.extend(json.dumps({"token": t}) + "\n" for t in out)
                test.pending.append(json.dumps({"done": True, "reason": "length", "generated": len(out),
                                               "prompt_tokens": len(prompt), "reused": reused,
                                               "prefill_ms": 2.0, "decode_ms": 1.0, "spec_windows": 0,
                                               "spec_drafted": 0, "spec_accepted": 0, "spec_emitted": 0,
                                               "decode_iterations": len(out) - 1}) + "\n")

            def flush(self):
                pass

        class NativeOutput:
            def readline(self):
                return test.pending.pop(0)

        self.engine = Engine.__new__(Engine)
        self.engine.stdin, self.engine.stdout = NativeInput(), NativeOutput()
        self.engine.lock = threading.Lock()
        self.engine.next_id, self.engine.current = 0, None
        self.engine.ctx, self.engine.grammar = 128, False
        self.log = io.StringIO()
        self.flushed = []

        class Log:
            def write(self, text):
                return test.log.write(text)

            def flush(self):
                test.flushed.append(test.log.getvalue())

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(self.chat, self.engine, "test", Log(),
                                                                     log_tokens=True))
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.url = f"http://127.0.0.1:{self.server.server_port}/v1/chat/completions"

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()

    def post(self, **options):
        body = {"messages": [{"role": "user", "content": "hello world a"}], "max_tokens": 2, **options}
        request = Request(self.url, data=json.dumps(body).encode(),
                          headers={"Content-Type": "application/json", "X-Request-ID": "test:case-1"})
        return urlopen(request, timeout=5)

    def test_cache_false_then_true_and_default_consume_native_result(self):
        for cache, expected in ((False, 0), (False, 0), (True, 2), (None, 2)):
            with self.post(**({} if cache is None else {"cache_prompt": cache})) as response:
                result = json.load(response)
            self.assertEqual(result["fns"]["reused_prefix"], expected)
            self.assertEqual(result["usage"]["completion_tokens"], 2)
            self.assertEqual(result["choices"][0]["message"]["content"], "hello world")
            self.assertNotIn("token_ids", result)
        self.assertIs(self.requests[-1]["cache_prompt"], True)

    def test_cache_requires_json_boolean_before_engine_consumption(self):
        for value in (None, 0, 1, "false", [], {}):
            with self.subTest(value=value), self.assertRaises(HTTPError) as error:
                self.post(cache_prompt=value)
            with error.exception as response:
                self.assertEqual(response.code, 400)
        self.assertEqual(self.requests, [])

    def test_invalid_request_shapes_are_rejected(self):
        for body in (None, [], "hello", {"messages": None}, {"messages": [None]},
                     {"messages": [{"content": "hello"}], "max_tokens": []}):
            with self.subTest(body=body):
                request = Request(self.url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
                with self.assertRaises(HTTPError) as error:
                    urlopen(request, timeout=5)
                with error.exception as response:
                    self.assertEqual(response.code, 400)
        self.assertEqual(self.requests, [])

    def test_stream_has_terminal_usage_and_flushed_private_correlation(self):
        with self.post(stream=True, cache_prompt=False) as response:
            payloads = []
            for line in response:
                if line.startswith(b"data: "):
                    data = line[6:].strip()
                    if data == b"[DONE]":
                        self.assertTrue(self.flushed)
                        break
                    payloads.append(json.loads(data))
                    if payloads[-1]["choices"][0]["finish_reason"] is not None:
                        self.assertTrue(self.flushed)
        terminal = payloads[-1]
        self.assertEqual(terminal["usage"]["completion_tokens"], 2)
        self.assertEqual(terminal["choices"][0]["finish_reason"], "length")
        self.assertTrue(all("token_ids" not in event for event in payloads))
        entry = json.loads(self.flushed[-1].splitlines()[-1])
        self.assertEqual(entry["request_id"], "test:case-1")
        self.assertEqual(entry["token_ids"], [2, 3])
        self.assertEqual(entry["reason"], "length")
        self.assertEqual(entry["decode_iterations"], 1)
        for field in ("spec_windows", "spec_drafted", "spec_accepted", "spec_emitted"):
            self.assertEqual(entry[field], 0)

    def test_default_logging_omits_generated_ids(self):
        handler = make_handler(self.chat, self.engine, "test", self.log)
        self.server.RequestHandlerClass = handler
        with self.post() as response:
            json.load(response)
        entry = json.loads(self.log.getvalue())
        self.assertNotIn("token_ids", entry)
        self.assertEqual(entry["completion_tokens"], 2)

    def test_token_logging_requires_explicit_log(self):
        with self.assertRaises(ValueError):
            make_handler(self.chat, self.engine, "test", log_tokens=True)

    def test_invalid_correlation_is_rejected(self):
        request = Request(self.url, data=json.dumps({"messages": []}).encode(), headers={"X-Request-ID": "bad id"})
        with self.assertRaises(HTTPError) as error:
            urlopen(request, timeout=5)
        with error.exception as response:
            self.assertEqual(response.code, 400)
        self.assertEqual(self.requests, [])

    def test_native_length_overrides_eos_surface_finish(self):
        with patch("fns_server.finish_message", return_value=({"role": "assistant", "content": ""}, "stop")):
            with self.post(ignore_eos=True) as response:
                result = json.load(response)
        self.assertEqual(result["choices"][0]["finish_reason"], "length")


if __name__ == "__main__":
    unittest.main()
