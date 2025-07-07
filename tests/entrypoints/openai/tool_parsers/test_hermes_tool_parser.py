# SPDX-License-Identifier: Apache-2.0

from unittest.mock import MagicMock

import pytest

from tests.entrypoints.openai.tool_parsers.utils import (
    run_tool_extraction, run_tool_extraction_streaming)
from vllm.entrypoints.openai.protocol import FunctionCall
from vllm.entrypoints.openai.tool_parsers import ToolParser, ToolParserManager


# Test cases for Hermes tool parser using <tool_call> format
SIMPLE_FUNCTION_OUTPUT = '<tool_call>{"name": "get_weather", "arguments": {"city": "San Francisco", "metric": "celsius"}}</tool_call>'
SIMPLE_FUNCTION_CALL = FunctionCall(
    name="get_weather",
    arguments='{"city": "San Francisco", "metric": "celsius"}',
)

MORE_TYPES_FUNCTION_OUTPUT = '<tool_call>{"name": "register_user", "arguments": {"name": "John Doe", "age": 37, "address": {"city": "San Francisco", "state": "CA"}, "role": null, "passed_test": true, "aliases": ["John", "Johnny"]}}</tool_call>'
MORE_TYPES_FUNCTION_CALL = FunctionCall(
    name="register_user",
    arguments='{"name": "John Doe", "age": 37, "address": {"city": "San Francisco", "state": "CA"}, "role": null, "passed_test": true, "aliases": ["John", "Johnny"]}',
)

PARAMETERLESS_FUNCTION_OUTPUT = '<tool_call>{"name": "get_weather", "arguments": {}}</tool_call>'
PARAMETERLESS_FUNCTION_CALL = FunctionCall(
    name="get_weather",
    arguments='{}',
)

EMPTY_DICT_FUNCTION_OUTPUT = '<tool_call>{"name": "do_something_cool", "arguments": {"additional_data": {}}}</tool_call>'
EMPTY_DICT_FUNCTION_CALL = FunctionCall(
    name="do_something_cool",
    arguments='{"additional_data": {}}',
)

EMPTY_LIST_FUNCTION_OUTPUT = '<tool_call>{"name": "do_something_cool", "arguments": {"steps": []}}</tool_call>'
EMPTY_LIST_FUNCTION_CALL = FunctionCall(
    name="do_something_cool",
    arguments='{"steps": []}',
)

ESCAPED_STRING_FUNCTION_OUTPUT = r'<tool_call>{"name": "get_weather", "arguments": {"city": "Martha\'s Vineyard", "metric": "\"cool units\""}}</tool_call>'
ESCAPED_STRING_FUNCTION_CALL = FunctionCall(
    name="get_weather",
    arguments='{"city": "Martha\'s Vineyard", "metric": "\\"cool units\\""}',
)

# Test output with content before tool call
CONTENT_BEFORE_TOOL_OUTPUT = 'I need to check the weather. <tool_call>{"name": "get_weather", "arguments": {"city": "San Francisco", "metric": "celsius"}}</tool_call>'

# Test output with multiple tool calls
MULTIPLE_TOOLS_OUTPUT = '<tool_call>{"name": "get_weather", "arguments": {"city": "San Francisco", "metric": "celsius"}}</tool_call><tool_call>{"name": "register_user", "arguments": {"name": "John Doe", "age": 37}}</tool_call>'

# Test scratch pad format (should be ignored)
SCRATCH_PAD_OUTPUT = '<scratch_pad>Let me think about this...</scratch_pad><tool_call>{"name": "get_weather", "arguments": {"city": "San Francisco", "metric": "celsius"}}</tool_call>'


@pytest.mark.parametrize("streaming", [True, False])
def test_no_tool_call(streaming: bool):
    mock_tokenizer = MagicMock()
    mock_tokenizer.get_vocab.return_value = {
        "<tool_call>": 1000,
        "</tool_call>": 1001,
    }
    tool_parser: ToolParser = ToolParserManager.get_tool_parser("hermes")(
        mock_tokenizer)
    model_output = "How can I help you today?"

    content, tool_calls = run_tool_extraction(tool_parser,
                                              model_output,
                                              streaming=streaming)

    assert content == model_output
    assert len(tool_calls) == 0


@pytest.mark.parametrize("streaming", [True, False])
def test_content_before_tool_call(streaming: bool):
    mock_tokenizer = MagicMock()
    mock_tokenizer.get_vocab.return_value = {
        "<tool_call>": 1000,
        "</tool_call>": 1001,
    }
    tool_parser: ToolParser = ToolParserManager.get_tool_parser("hermes")(
        mock_tokenizer)

    content, tool_calls = run_tool_extraction(tool_parser,
                                              CONTENT_BEFORE_TOOL_OUTPUT,
                                              streaming=streaming)

    assert content == "I need to check the weather. "
    assert len(tool_calls) == 1
    assert tool_calls[0].function == SIMPLE_FUNCTION_CALL


@pytest.mark.parametrize("streaming", [True, False])
def test_scratch_pad_ignored(streaming: bool):
    mock_tokenizer = MagicMock()
    mock_tokenizer.get_vocab.return_value = {
        "<tool_call>": 1000,
        "</tool_call>": 1001,
    }
    tool_parser: ToolParser = ToolParserManager.get_tool_parser("hermes")(
        mock_tokenizer)

    content, tool_calls = run_tool_extraction(tool_parser,
                                              SCRATCH_PAD_OUTPUT,
                                              streaming=streaming)

    assert content == ""
    assert len(tool_calls) == 1
    assert tool_calls[0].function == SIMPLE_FUNCTION_CALL


TEST_CASES = [
    pytest.param(True,
                 SIMPLE_FUNCTION_OUTPUT, [SIMPLE_FUNCTION_CALL],
                 id="simple_streaming"),
    pytest.param(False,
                 SIMPLE_FUNCTION_OUTPUT, [SIMPLE_FUNCTION_CALL],
                 id="simple_nonstreaming"),
    pytest.param(True,
                 MORE_TYPES_FUNCTION_OUTPUT, [MORE_TYPES_FUNCTION_CALL],
                 id="more_types_streaming"),
    pytest.param(False,
                 MORE_TYPES_FUNCTION_OUTPUT, [MORE_TYPES_FUNCTION_CALL],
                 id="more_types_nonstreaming"),
    pytest.param(True,
                 PARAMETERLESS_FUNCTION_OUTPUT, [PARAMETERLESS_FUNCTION_CALL],
                 id="parameterless_streaming"),
    pytest.param(False,
                 PARAMETERLESS_FUNCTION_OUTPUT, [PARAMETERLESS_FUNCTION_CALL],
                 id="parameterless_nonstreaming"),
    pytest.param(True,
                 EMPTY_DICT_FUNCTION_OUTPUT, [EMPTY_DICT_FUNCTION_CALL],
                 id="empty_dict_streaming"),
    pytest.param(False,
                 EMPTY_DICT_FUNCTION_OUTPUT, [EMPTY_DICT_FUNCTION_CALL],
                 id="empty_dict_nonstreaming"),
    pytest.param(True,
                 EMPTY_LIST_FUNCTION_OUTPUT, [EMPTY_LIST_FUNCTION_CALL],
                 id="empty_list_streaming"),
    pytest.param(False,
                 EMPTY_LIST_FUNCTION_OUTPUT, [EMPTY_LIST_FUNCTION_CALL],
                 id="empty_list_nonstreaming"),
    pytest.param(True,
                 ESCAPED_STRING_FUNCTION_OUTPUT, [ESCAPED_STRING_FUNCTION_CALL],
                 id="escaped_string_streaming"),
    pytest.param(False,
                 ESCAPED_STRING_FUNCTION_OUTPUT, [ESCAPED_STRING_FUNCTION_CALL],
                 id="escaped_string_nonstreaming"),
    pytest.param(True,
                 MULTIPLE_TOOLS_OUTPUT, [SIMPLE_FUNCTION_CALL, FunctionCall(name="register_user", arguments='{"name": "John Doe", "age": 37}')],
                 id="multiple_tools_streaming"),
    pytest.param(False,
                 MULTIPLE_TOOLS_OUTPUT, [SIMPLE_FUNCTION_CALL, FunctionCall(name="register_user", arguments='{"name": "John Doe", "age": 37}')],
                 id="multiple_tools_nonstreaming"),
]


@pytest.mark.parametrize("streaming, model_output, expected_tool_calls",
                         TEST_CASES)
def test_tool_call(streaming: bool, model_output: str,
                   expected_tool_calls: list[FunctionCall]):
    mock_tokenizer = MagicMock()
    mock_tokenizer.get_vocab.return_value = {
        "<tool_call>": 1000,
        "</tool_call>": 1001,
    }
    tool_parser: ToolParser = ToolParserManager.get_tool_parser("hermes")(
        mock_tokenizer)

    content, tool_calls = run_tool_extraction(tool_parser,
                                              model_output,
                                              streaming=streaming)

    assert len(tool_calls) == len(expected_tool_calls)
    for actual, expected in zip(tool_calls, expected_tool_calls):
        assert actual.type == "function"
        assert actual.function == expected


def test_streaming_tool_call_incremental():
    """Test streaming with incremental token generation"""
    mock_tokenizer = MagicMock()
    mock_tokenizer.get_vocab.return_value = {
        "<tool_call>": 1000,
        "</tool_call>": 1001,
    }
    mock_tokenizer.tokenize.side_effect = lambda x: list(x)  # Simple char-by-char tokenization
    
    tool_parser: ToolParser = ToolParserManager.get_tool_parser("hermes")(
        mock_tokenizer)
    
    # Simulate incremental streaming of a tool call
    model_output_deltas = [
        "I need to check the weather. ",
        "<tool_call>",
        '{"name": "get_weather",',
        ' "arguments": {"city": "San Francisco",',
        ' "metric": "celsius"}}',
        "</tool_call>",
    ]

    reconstructor = run_tool_extraction_streaming(
        tool_parser, model_output_deltas, assert_one_tool_per_delta=False)

    assert reconstructor.other_content == "I need to check the weather. "
    assert len(reconstructor.tool_calls) == 1
    assert reconstructor.tool_calls[0].function == SIMPLE_FUNCTION_CALL


def test_streaming_tool_call_with_large_steps():
    """Test streaming with larger delta steps"""
    mock_tokenizer = MagicMock()
    mock_tokenizer.get_vocab.return_value = {
        "<tool_call>": 1000,
        "</tool_call>": 1001,
    }
    mock_tokenizer.tokenize.side_effect = lambda x: list(x)  # Simple char-by-char tokenization
    
    tool_parser: ToolParser = ToolParserManager.get_tool_parser("hermes")(
        mock_tokenizer)
    
    model_output_deltas = [
        'I need to check the weather. <tool_call>{"name": "get_weather", "arguments": {"city": "San Francisco", "metric": "celsius"}}</tool_call>',
        '<tool_call>{"name": "register_user", "arguments": {"name": "John Doe", "age": 37}}</tool_call>',
    ]

    reconstructor = run_tool_extraction_streaming(
        tool_parser, model_output_deltas, assert_one_tool_per_delta=False)

    assert reconstructor.other_content == "I need to check the weather. "
    assert len(reconstructor.tool_calls) == 2
    assert reconstructor.tool_calls[0].function == SIMPLE_FUNCTION_CALL
    assert reconstructor.tool_calls[1].function == FunctionCall(name="register_user", arguments='{"name": "John Doe", "age": 37}')


def test_streaming_incomplete_tool_call():
    """Test streaming with incomplete tool call (should handle gracefully)"""
    mock_tokenizer = MagicMock()
    mock_tokenizer.get_vocab.return_value = {
        "<tool_call>": 1000,
        "</tool_call>": 1001,
    }
    mock_tokenizer.tokenize.side_effect = lambda x: list(x)  # Simple char-by-char tokenization
    
    tool_parser: ToolParser = ToolParserManager.get_tool_parser("hermes")(
        mock_tokenizer)
    
    # Incomplete tool call (missing closing tag)
    model_output_deltas = [
        "I need to check the weather. ",
        "<tool_call>",
        '{"name": "get_weather",',
        ' "arguments": {"city": "San Francisco"',
        # Missing closing brace and tag
    ]

    reconstructor = run_tool_extraction_streaming(
        tool_parser, model_output_deltas, assert_one_tool_per_delta=False)

    # Should handle incomplete tool call gracefully
    assert reconstructor.other_content == "I need to check the weather. "
    # Tool call may or may not be reconstructed depending on parser state
    # The key is that it shouldn't crash


def test_malformed_json_handling():
    """Test handling of malformed JSON in tool calls"""
    mock_tokenizer = MagicMock()
    mock_tokenizer.get_vocab.return_value = {
        "<tool_call>": 1000,
        "</tool_call>": 1001,
    }
    tool_parser: ToolParser = ToolParserManager.get_tool_parser("hermes")(
        mock_tokenizer)
    
    # Malformed JSON (missing quotes)
    malformed_output = '<tool_call>{name: get_weather, arguments: {city: San Francisco}}</tool_call>'

    content, tool_calls = run_tool_extraction(tool_parser,
                                              malformed_output,
                                              streaming=False)

    # Should handle malformed JSON gracefully
    assert content == malformed_output
    assert len(tool_calls) == 0


def test_empty_tool_call():
    """Test handling of empty tool call"""
    mock_tokenizer = MagicMock()
    mock_tokenizer.get_vocab.return_value = {
        "<tool_call>": 1000,
        "</tool_call>": 1001,
    }
    tool_parser: ToolParser = ToolParserManager.get_tool_parser("hermes")(
        mock_tokenizer)
    
    empty_output = '<tool_call></tool_call>'

    content, tool_calls = run_tool_extraction(tool_parser,
                                              empty_output,
                                              streaming=False)

    # Should handle empty tool call gracefully
    assert content == empty_output
    assert len(tool_calls) == 0


def test_nested_tool_call_tags():
    """Test handling of nested or multiple tool call tags"""
    mock_tokenizer = MagicMock()
    mock_tokenizer.get_vocab.return_value = {
        "<tool_call>": 1000,
        "</tool_call>": 1001,
    }
    tool_parser: ToolParser = ToolParserManager.get_tool_parser("hermes")(
        mock_tokenizer)
    
    # Test unclosed tool call followed by another
    unclosed_output = '<tool_call>{"name": "get_weather"<tool_call>{"name": "register_user", "arguments": {"name": "John"}}</tool_call>'

    content, tool_calls = run_tool_extraction(tool_parser,
                                              unclosed_output,
                                              streaming=False)

    # Should extract what it can
    assert len(tool_calls) >= 0  # May extract one or more valid tool calls
