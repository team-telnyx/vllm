# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import os
import re
from collections.abc import Sequence
from dataclasses import dataclass

from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest,
)
from vllm.entrypoints.openai.engine.protocol import (
    DeltaFunctionCall,
    DeltaMessage,
    DeltaToolCall,
    ExtractedToolCallInformation,
    FunctionCall,
    ToolCall,
)
from vllm.logger import init_logger
from vllm.tokenizers import TokenizerLike
from vllm.tool_parsers.abstract_tool_parser import (
    ToolParser,
)

logger = init_logger(__name__)

SECTION_MAX_CHARS: int = int(os.getenv("KIMI_PARSER_SECTION_MAX", "524288"))
MARKER_BUFFER_MAX_CHARS = 256


@dataclass
class _StreamState:
    in_tool_section: bool = False
    section_char_count: int = 0
    marker_buffer: str = ""
    current_args: str = ""

    def reset(self) -> None:
        self.__init__()

    def enter_section(self) -> None:
        self.in_tool_section = True
        self.section_char_count = 0
        self.marker_buffer = ""

    def exit_section(self) -> None:
        self.in_tool_section = False
        self.section_char_count = 0
        self.marker_buffer = ""


class KimiK2ToolParser(ToolParser):
    _SECTION_BEGIN_VARIANTS: tuple[str, ...] = (
        "<|tool_calls_section_begin|>",
        "<|tool_call_section_begin|>",
    )
    _SECTION_END_VARIANTS: tuple[str, ...] = (
        "<|tool_calls_section_end|>",
        "<|tool_call_section_end|>",
    )
    _CALL_BEGIN: str = "<|tool_call_begin|>"
    _CALL_END: str = "<|tool_call_end|>"
    _ARG_BEGIN: str = "<|tool_call_argument_begin|>"

    _ALL_MARKERS: tuple[str, ...] = (
        *_SECTION_BEGIN_VARIANTS,
        *_SECTION_END_VARIANTS,
        _CALL_BEGIN,
        _CALL_END,
        _ARG_BEGIN,
    )

    _CALL_ID_PATTERN: str = r"[^<\s]+(?::\d+|_\d+)"
    _RE_FULL: re.Pattern[str] = re.compile(
        rf"<\|tool_call_begin\|>\s*(?P<call_id>{_CALL_ID_PATTERN})\s*"
        + rf"<\|tool_call_argument_begin\|>\s*(?P<args>(?:(?!<\|tool_call_begin\|>).)*?)\s*"
        + rf"<\|tool_call_end\|>",
        re.DOTALL,
    )
    _RE_STREAM_ID: re.Pattern[str] = re.compile(
        rf"^\s*(?P<call_id>{_CALL_ID_PATTERN})\s*$"
    )
    _RE_UNDERSCORE_SUFFIX: re.Pattern[str] = re.compile(r"_\d+$")

    def __init__(self, tokenizer: TokenizerLike):
        self._state: _StreamState = _StreamState()
        super().__init__(tokenizer)

        self.prev_tool_call_arr: list[dict[str, str | None]] = []
        self.streamed_args_for_tool: list[str] = []
        self.current_tool_id: int = -1
        self.current_tool_name_sent: bool = False

        if not self.model_tokenizer:
            raise ValueError(
                "The model tokenizer must be passed to the ToolParser "
                + "constructor during construction."
            )

        self.tool_calls_start_token: str = self._SECTION_BEGIN_VARIANTS[0]
        self.tool_calls_end_token: str = self._SECTION_END_VARIANTS[0]
        self.tool_calls_start_token_variants: list[str] = list(
            self._SECTION_BEGIN_VARIANTS
        )
        self.tool_calls_end_token_variants: list[str] = list(
            self._SECTION_END_VARIANTS
        )
        self.tool_call_start_token: str = self._CALL_BEGIN
        self.tool_call_end_token: str = self._CALL_END
        self.tool_call_argument_begin_token: str = self._ARG_BEGIN
        self.buffer_max_size: int = MARKER_BUFFER_MAX_CHARS
        self.max_section_chars: int = SECTION_MAX_CHARS

        self.tool_calls_start_token_id: int | None = self.vocab.get(
            self.tool_calls_start_token
        )
        self.tool_calls_end_token_id: int | None = self.vocab.get(
            self.tool_calls_end_token
        )
        self.tool_calls_start_token_ids: list[int] = [
            tid
            for variant in self._SECTION_BEGIN_VARIANTS
            if (tid := self.vocab.get(variant)) is not None
        ]
        self.tool_calls_end_token_ids: list[int] = [
            tid
            for variant in self._SECTION_END_VARIANTS
            if (tid := self.vocab.get(variant)) is not None
        ]
        self.tool_call_start_token_id: int | None = self.vocab.get(
            self.tool_call_start_token
        )
        self.tool_call_end_token_id: int | None = self.vocab.get(
            self.tool_call_end_token
        )

        if not self.tool_calls_start_token_ids or not self.tool_calls_end_token_ids:
            raise RuntimeError(
                "KimiK2ToolParser could not locate tool section begin/end "
                + "tokens in the tokenizer!"
            )
        if self.tool_call_start_token_id is None or self.tool_call_end_token_id is None:
            raise RuntimeError(
                "KimiK2ToolParser could not locate tool call begin/end "
                + "tokens in the tokenizer!"
            )

    @property
    def in_tool_section(self) -> bool:
        return self._state.in_tool_section

    @in_tool_section.setter
    def in_tool_section(self, value: bool) -> None:
        self._state.in_tool_section = value

    @property
    def token_buffer(self) -> str:
        return self._state.marker_buffer

    @token_buffer.setter
    def token_buffer(self, value: str) -> None:
        self._state.marker_buffer = value

    @property
    def section_char_count(self) -> int:
        return self._state.section_char_count

    @section_char_count.setter
    def section_char_count(self, value: int) -> None:
        self._state.section_char_count = value

    def reset_streaming_state(self) -> None:
        self._state.reset()
        self.prev_tool_call_arr.clear()
        self.streamed_args_for_tool.clear()
        self.current_tool_id = -1
        self.current_tool_name_sent = False
        logger.debug("KimiK2ToolParser: state reset")

    def extract_tool_calls(
        self,
        model_output: str,
        request: ChatCompletionRequest,
    ) -> ExtractedToolCallInformation:
        section_begin = next(
            (variant for variant in self._SECTION_BEGIN_VARIANTS if variant in model_output),
            None,
        )
        if section_begin is None:
            return ExtractedToolCallInformation(
                tools_called=False,
                tool_calls=[],
                content=model_output,
            )

        try:
            tool_calls = [
                ToolCall(
                    id=match.group("call_id"),
                    type="function",
                    function=FunctionCall(
                        name=self._call_id_to_name(match.group("call_id")),
                        arguments=match.group("args"),
                    ),
                )
                for match in self._RE_FULL.finditer(model_output)
            ]
            content = model_output[: model_output.index(section_begin)]
            return ExtractedToolCallInformation(
                tools_called=bool(tool_calls),
                tool_calls=tool_calls,
                content=content or None,
            )
        except Exception:
            logger.exception("KimiK2ToolParser.extract_tool_calls failed")
            return ExtractedToolCallInformation(
                tools_called=False,
                tool_calls=[],
                content=model_output,
            )

    def extract_tool_calls_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
        request: ChatCompletionRequest,
    ) -> DeltaMessage | None:
        del previous_text, request

        state = self._state
        state.marker_buffer = (state.marker_buffer + delta_text)[-self.buffer_max_size :]

        found_begin = any(
            marker in state.marker_buffer for marker in self._SECTION_BEGIN_VARIANTS
        )
        found_end = any(
            marker in state.marker_buffer for marker in self._SECTION_END_VARIANTS
        )
        call_end_in_delta = self.tool_call_end_token_id in delta_token_ids

        if found_begin and not state.in_tool_section:
            state.enter_section()

        if (
            not state.in_tool_section
            and not any(token_id in current_token_ids for token_id in self.tool_calls_start_token_ids)
        ):
            return DeltaMessage(content=delta_text)

        if state.in_tool_section:
            state.section_char_count += len(delta_text)
            if state.section_char_count > self.max_section_chars:
                logger.warning(
                    "KimiK2ToolParser: section length exceeded %d max limit.",
                    self.max_section_chars,
                )
                state.exit_section()
                return DeltaMessage(content=delta_text if delta_text.strip() else "")

        prev_begin_count = previous_token_ids.count(self.tool_call_start_token_id)
        cur_begin_count = current_token_ids.count(self.tool_call_start_token_id)
        cur_end_count = current_token_ids.count(self.tool_call_end_token_id)

        if cur_begin_count > prev_begin_count:
            self.current_tool_id += 1
            self.current_tool_name_sent = False
            state.current_args = ""
            self._reset_tool_slot(self.current_tool_id)

        if state.in_tool_section and cur_begin_count == 0:
            if found_end:
                return self._finish_section(delta=None, delta_text=delta_text)
            return DeltaMessage(content="")

        parsed = None
        if cur_begin_count > 0:
            parsed = self._parse_call_portion(self._extract_call_portion(current_text))
            self._sync_current_tool(parsed)

        if not self.current_tool_name_sent:
            name_delta = self._try_emit_name(
                parsed,
                include_arguments=call_end_in_delta,
            )
            if name_delta is not None:
                if found_end and state.in_tool_section and call_end_in_delta:
                    return self._finish_section(name_delta, delta_text)
                return name_delta

        if call_end_in_delta:
            args_delta = None
            if parsed is not None:
                args_delta = self._diff_and_emit_args(parsed.get("arguments") or "")
            if found_end and state.in_tool_section:
                return self._finish_section(args_delta, delta_text)
            return args_delta

        if cur_begin_count > cur_end_count:
            if parsed is not None:
                return self._diff_and_emit_args(parsed.get("arguments") or "")
            return None

        if found_end and state.in_tool_section:
            return self._finish_section(delta=None, delta_text=delta_text)

        if state.in_tool_section:
            return DeltaMessage(content="")

        return DeltaMessage(content=self._strip_all_markers(delta_text))

    @classmethod
    def _call_id_to_name(cls, call_id: str) -> str:
        if ":" in call_id:
            return call_id.split(":", 1)[0].rsplit(".", 1)[-1]
        name = cls._RE_UNDERSCORE_SUFFIX.sub("", call_id)
        return name[10:] if name.startswith("functions_") else name

    def _ensure_tool_slot(self, index: int) -> None:
        while len(self.prev_tool_call_arr) <= index:
            self.prev_tool_call_arr.append(
                {"id": None, "name": None, "arguments": ""}
            )
        while len(self.streamed_args_for_tool) <= index:
            self.streamed_args_for_tool.append("")

    def _reset_tool_slot(self, index: int) -> None:
        self._ensure_tool_slot(index)
        self.prev_tool_call_arr[index] = {
            "id": None,
            "name": None,
            "arguments": "",
        }
        self.streamed_args_for_tool[index] = ""

    def _strip_all_markers(self, text: str) -> str:
        clean_text = text
        for marker in self._ALL_MARKERS:
            clean_text = clean_text.replace(marker, "")
        return clean_text

    def _text_after_section_end(self, delta_text: str) -> str:
        for marker in self._SECTION_END_VARIANTS:
            if marker in delta_text:
                return delta_text.split(marker, 1)[1]
        return ""

    def _extract_call_portion(self, current_text: str) -> str:
        idx = current_text.rfind(self._CALL_BEGIN)
        if idx == -1:
            return ""
        return current_text[idx + len(self._CALL_BEGIN) :].lstrip()

    def _strip_trailing_markers(self, text: str) -> str:
        stripped = text
        if self._CALL_END in stripped:
            stripped = stripped.split(self._CALL_END, 1)[0]
        for marker in self._SECTION_END_VARIANTS:
            if marker in stripped:
                stripped = stripped.split(marker, 1)[0]
        return stripped.rstrip()

    def _parse_call_portion(self, portion: str) -> dict[str, str | None] | None:
        if self._ARG_BEGIN in portion:
            id_part, _, args_part = portion.partition(self._ARG_BEGIN)
            call_id = id_part.strip()
            if not self._RE_STREAM_ID.match(call_id):
                return None
            return {
                "id": call_id,
                "name": self._call_id_to_name(call_id),
                "arguments": self._strip_trailing_markers(args_part),
            }

        candidate = self._strip_trailing_markers(portion)
        match = self._RE_STREAM_ID.match(candidate)
        if match is None:
            return None

        call_id = match.group("call_id")
        return {
            "id": call_id,
            "name": self._call_id_to_name(call_id),
            "arguments": None,
        }

    def _sync_current_tool(self, parsed: dict[str, str | None] | None) -> None:
        if parsed is None or self.current_tool_id < 0:
            return

        self._ensure_tool_slot(self.current_tool_id)
        current = self.prev_tool_call_arr[self.current_tool_id]
        current["id"] = parsed["id"]
        current["name"] = parsed["name"]
        if parsed["arguments"] is not None:
            current["arguments"] = parsed["arguments"]

    def _try_emit_name(
        self,
        parsed: dict[str, str | None] | None,
        *,
        include_arguments: bool,
    ) -> DeltaMessage | None:
        if parsed is None or not parsed.get("name") or self.current_tool_id < 0:
            return None

        self._sync_current_tool(parsed)
        self.current_tool_name_sent = True

        function_delta = DeltaFunctionCall(name=parsed["name"])
        if include_arguments and parsed["arguments"] is not None:
            self._ensure_tool_slot(self.current_tool_id)
            self._state.current_args = parsed["arguments"]
            self.prev_tool_call_arr[self.current_tool_id]["arguments"] = parsed[
                "arguments"
            ]
            self.streamed_args_for_tool[self.current_tool_id] = parsed["arguments"]
            function_delta.arguments = parsed["arguments"]

        return DeltaMessage(
            tool_calls=[
                DeltaToolCall(
                    index=self.current_tool_id,
                    type="function",
                    id=parsed["id"],
                    function=function_delta,
                )
            ]
        )

    def _diff_and_emit_args(self, cur_args: str) -> DeltaMessage | None:
        if self.current_tool_id < 0:
            return None

        already = self._state.current_args
        if not cur_args or cur_args == already:
            return None

        if cur_args.startswith(already):
            new_part = cur_args[len(already) :]
        else:
            new_part = cur_args

        self._ensure_tool_slot(self.current_tool_id)
        self._state.current_args = cur_args
        self.prev_tool_call_arr[self.current_tool_id]["arguments"] = cur_args
        self.streamed_args_for_tool[self.current_tool_id] = cur_args

        if not new_part:
            return None

        return DeltaMessage(
            tool_calls=[
                DeltaToolCall(
                    index=self.current_tool_id,
                    function=DeltaFunctionCall(arguments=new_part),
                )
            ]
        )

    def _finish_section(
        self,
        delta: DeltaMessage | None,
        delta_text: str,
    ) -> DeltaMessage | None:
        post_marker = self._text_after_section_end(delta_text)
        self._state.exit_section()

        if delta is None:
            if post_marker.strip():
                return DeltaMessage(content=post_marker)
            return DeltaMessage(content="")

        if post_marker.strip():
            delta.content = post_marker
        return delta
