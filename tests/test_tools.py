import json

from vserve.tools import (
    _read_chat_template,
    detect_reasoning_parser,
    detect_tool_parser,
    supports_tools,
)


def test_read_chat_template_ignores_non_mapping_tokenizer_config(tmp_path):
    model_dir = tmp_path / "neutral-model"
    model_dir.mkdir()
    (model_dir / "tokenizer_config.json").write_text(json.dumps(["bad"]))

    assert _read_chat_template(model_dir) is None
    assert detect_tool_parser(model_dir) is None
    assert detect_reasoning_parser(model_dir) is None
    assert supports_tools(model_dir) is False


def test_read_chat_template_ignores_non_string_template_entries(tmp_path):
    model_dir = tmp_path / "plain-model"
    model_dir.mkdir()
    (model_dir / "tokenizer_config.json").write_text(json.dumps({
        "chat_template": [
            {"name": "tool_use", "template": 123},
            {"name": "default", "template": {"jinja": True}},
        ]
    }))

    assert _read_chat_template(model_dir) is None
    assert detect_tool_parser(model_dir) is None
    assert detect_reasoning_parser(model_dir) is None
    assert supports_tools(model_dir) is False


def test_read_chat_template_ignores_non_utf8_file(tmp_path):
    model_dir = tmp_path / "broken-template-model"
    model_dir.mkdir()
    (model_dir / "tokenizer_config.json").write_bytes(b"\x80\x81\x82")

    assert _read_chat_template(model_dir) is None
    assert detect_tool_parser(model_dir) is None
    assert detect_reasoning_parser(model_dir) is None
    assert supports_tools(model_dir) is False
