"""CPU regression tests for serve/fns_server.py: message normalization, tool-call parsing, SSE splitting and
incremental prompt encoding.

Needs no GPU, model or Hugging Face download: fns_server imports only pure-Python packages at module level, and the
prompt-encoding test uses a small in-memory `tokenizers` vocabulary.
"""

import copy
import re
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "serve"))

from tokenizers import Tokenizer, models, pre_tokenizers  # noqa: E402

from fns_server import Chat, StreamSplitter, normalize_messages, parse_tool_call  # noqa: E402

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


if __name__ == "__main__":
    unittest.main()
