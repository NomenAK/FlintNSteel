"""Flint structural tag for xgrammar, as JSON, without importing xgrammar (its import loads torch).

Adapted from rec, engine/src/rec_mlx/grammar_builder.py (commit 8af99abaf50b33903147d89eed91d1bb3ac642dd, Apache-2.0),
`build_flint_glm_grammar`, and from the `tool_choice="required", reasoning=False` branch of
`get_glm_4_7_structural_tag` in xgrammar v0.2.1 (python/xgrammar/builtin_structural_tag.py). The dictionary written
here is the one `StructuralTag.model_dump()` would produce (types and fields from xgrammar's structural_tag.py).

Assistant output: optional reasoning `<think>...</think>`, then free text without protocol markers, then either an
optional `<|user|>` (end of turn, before the end token), or one or more tool calls
`<tool_call>name<arg_key>...</arg_key><arg_value>...</arg_value>...</tool_call>` matching the tool schema, followed by
`<|observation|>`.
"""

PROTOCOL_MARKERS = [
    "<think>", "</think>", "<tool_call>", "</tool_call>", "<arg_key>", "</arg_key>", "<arg_value>", "</arg_value>",
    "<|begin_of_text|>", "<|end_of_text|>", "<|system|>", "<|developer|>", "<|user|>", "<|assistant|>",
    "<|observation|>",
]
STOP_MARKERS = ["<|end_of_text|>", "<|user|>", "<|observation|>"]


def _token(tokenizer, marker):
    ids = tokenizer.encode(marker, add_special_tokens=False).ids
    if len(ids) != 1 or tokenizer.id_to_token(ids[0]) != marker:
        raise ValueError(f"Flint grammar requires {marker!r} to be a dedicated vocabulary token")
    return {"type": "token", "token": ids[0]}


def _parameters(function):
    # xgrammar accepts any object when strict is false; callers require an object parameter schema.
    if function.get("strict") is False:
        return True
    return function["parameters"]


def build_flint_glm_grammar(tokenizer, tools=None):
    """JSON structural tag, passed as-is to fns-serve in the "grammar" field."""
    endings = [{"type": "optional", "content": _token(tokenizer, "<|user|>")}]
    if tools:
        tags = []
        for definition in tools:
            function = definition.get("function", definition)
            parameters = function.get("parameters")
            if parameters is None or parameters.get("type") != "object":
                raise ValueError("Flint grammar requires object parameter schemas")
            tags.append({
                "type": "tag",
                "begin": "<tool_call>" + function["name"],
                "content": {"type": "json_schema", "json_schema": _parameters(function), "style": "glm_xml"},
                "end": "</tool_call>",
            })
        calls = {"type": "tags_with_separator", "tags": tags, "separator": "", "at_least_one": True,
                 "stop_after_first": False}
        endings.append({"type": "sequence", "elements": [calls, _token(tokenizer, "<|observation|>")]})
    reasoning = {
        "type": "tag",
        "begin": _token(tokenizer, "<think>"),
        "content": {"type": "any_text", "excludes": ["</think>", *STOP_MARKERS]},
        "end": _token(tokenizer, "</think>"),
    }
    return {
        "type": "structural_tag",
        "format": {
            "type": "sequence",
            "elements": [
                {"type": "optional", "content": reasoning},
                {"type": "any_text", "excludes": list(PROTOCOL_MARKERS)},
                {"type": "or", "elements": endings},
            ],
        },
    }
