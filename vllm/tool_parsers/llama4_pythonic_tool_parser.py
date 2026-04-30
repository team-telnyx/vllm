# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import ast
from collections.abc import Mapping, Sequence
from typing import Final, override

import regex as re

import vllm.envs as envs
from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest,
)
from vllm.entrypoints.openai.engine.protocol import (
    DeltaMessage,
    DeltaToolCall,
    ExtractedToolCallInformation,
    ToolCall,
)
from vllm.logger import init_logger
from vllm.tokenizers import TokenizerLike
from vllm.tool_parsers.abstract_tool_parser import (
    Tool,
    ToolParser,
)
from vllm.tool_parsers.utils import (
    UnexpectedAstError,
    compute_tool_delta,
    handle_single_tool,
    make_valid_python,
)

logger = init_logger(__name__)


class Llama4PythonicToolParser(ToolParser):
    """
    Toolcall parser for Llama4 that produce tool calls in a pythonic style
    Use --enable-auto-tool-choice --tool-call-parser llama4_pythonic
    """

    # TODO(mdepinet): Possible future improvements:
    #   1. Support text + tools separated by either <|python_tag|> or \n\n
    #   2. Support tools outside of a list (or separated by a semicolon).
    #      This depends on item 1 for consistent streaming.
    # Neither of these are necessary for e.g. ToolACE, but both would help make
    # Llama3.2 models more reliable.

    TOOL_CALL_REGEX: Final = re.compile(
        r"\[([a-zA-Z]+\w*\(([a-zA-Z]+\w*=.*,\s*)*([a-zA-Z]+\w*=.*\s)?\),\s*)*([a-zA-Z]+\w*\(([a-zA-Z]+\w*=.*,\s*)*([a-zA-Z]+\w*=.*\s*)?\)\s*)+\]",
        re.DOTALL,
    )

    def __init__(
        self,
        tokenizer: TokenizerLike,
        tools: list[Tool] | None = None,
    ):
        self.pythonic_to_tool_name: dict[str, str] = {}
        self.pythonic_to_tool_name_dict_created: bool = False
        super().__init__(tokenizer, tools)

    # Rename for readability. This is NOT a tool id.
    @property
    def current_tool_index(self) -> int:
        return self.current_tool_id

    @current_tool_index.setter
    def current_tool_index(self, value: int) -> None:
        self.current_tool_id = value

    @staticmethod
    def _get_tool_name(tool: Tool | Mapping[str, object]) -> str | None:
        if isinstance(tool, Mapping):
            function = tool.get("function")
            if isinstance(function, Mapping):
                name = function.get("name")
                return name if isinstance(name, str) else None
            name = tool.get("name")
            return name if isinstance(name, str) else None

        function = getattr(tool, "function", None)
        if function is not None:
            name = getattr(function, "name", None)
            return name if isinstance(name, str) else None
        name = getattr(tool, "name", None)
        return name if isinstance(name, str) else None

    def create_pythonic_tool_name_dict(self, request: ChatCompletionRequest) -> None:
        self.pythonic_to_tool_name = {
            replace_non_letters(tool_name): tool_name
            for tool in (request.tools or self.tools)
            if (tool_name := self._get_tool_name(tool)) is not None
        }

    @staticmethod
    def _get_call_nodes(parsed: object) -> list[ast.Call]:
        if not isinstance(parsed, ast.List):
            raise UnexpectedAstError("Tool output must be a list of function calls")

        call_nodes: list[ast.Call] = []
        for element in parsed.elts:
            if not isinstance(element, ast.Call):
                raise UnexpectedAstError(
                    "Tool output must be a list of function calls"
                )
            call_nodes.append(element)
        return call_nodes

    @override
    def extract_tool_calls(
        self, model_output: str, request: ChatCompletionRequest
    ) -> ExtractedToolCallInformation:
        """
        Extract the tool calls from a complete model response.
        """
        self.create_pythonic_tool_name_dict(request)

        # remove <|python_start|> and <|python_end|>
        # as Llama 4 model sometime will output those tokens
        if model_output.startswith("<|python_start|>"):
            model_output = model_output[len("<|python_start|>") :]
            model_output = model_output.replace("<|python_end|>", "")
        sanitized_model_output = sanitize_function_names(model_output)

        is_tool_call_pattern = False
        try:
            is_tool_call_pattern = (
                self.TOOL_CALL_REGEX.match(
                    sanitized_model_output,
                    timeout=envs.VLLM_TOOL_PARSE_REGEX_TIMEOUT_SECONDS,
                )
                is not None
            )
        except TimeoutError:
            logger.warning("Regex timeout occurred when matching tool call pattern.")
            logger.debug(
                "Regex timeout occurred when matching user input: %s", model_output
            )

        if not is_tool_call_pattern:
            return ExtractedToolCallInformation(
                tools_called=False, tool_calls=[], content=model_output
            )

        try:
            module = ast.parse(sanitized_model_output)
            parsed = getattr(module.body[0], "value", None)
            call_nodes = self._get_call_nodes(parsed)
            return ExtractedToolCallInformation(
                tools_called=True,
                tool_calls=[
                    handle_single_tool_with_name_map(
                        call_node, self.pythonic_to_tool_name
                    )
                    for call_node in call_nodes
                ],
                content=None,
            )
        except Exception:
            logger.exception("Error in extracting tool call from response.")
            # Treat as regular text
            return ExtractedToolCallInformation(
                tools_called=False, tool_calls=[], content=model_output
            )

    @override
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
        if not self.pythonic_to_tool_name_dict_created:
            self.create_pythonic_tool_name_dict(request)
            self.pythonic_to_tool_name_dict_created = True

        tool_start_index = current_text.find("\n\n[")
        if tool_start_index >= 0:
            current_text = current_text[tool_start_index + 2 :]

        previous_tool_start_index = previous_text.find("\n\n[")
        if previous_tool_start_index >= 0:
            previous_text = previous_text[previous_tool_start_index + 2 :]

        if not current_text.startswith("[") and not current_text.startswith(
            "<|python_start|>"
        ):
            return DeltaMessage(content=delta_text)

        try:
            # remove <|python_start|> and <|python_end|>
            if current_text.startswith("<|python_start|>"):
                current_text = current_text[len("<|python_start|>") :]
            if current_text.endswith("<|python_end|>"):
                current_text = current_text[: current_text.rfind("<|python_end|>")]
            valid_and_added_text = make_valid_python(current_text)
            if valid_and_added_text is None:
                return None
            valid_text, added_text = valid_and_added_text
            valid_text = sanitize_function_names(valid_text)

            module = ast.parse(valid_text)
            parsed = getattr(module.body[0], "value", None)
            call_nodes = self._get_call_nodes(parsed)
            tool_calls = [
                handle_single_tool_with_name_map(
                    call_node, self.pythonic_to_tool_name
                )
                for call_node in call_nodes
            ]

            tool_deltas: list[DeltaToolCall] = []
            for index, new_call in enumerate(tool_calls):
                if index < self.current_tool_index:
                    continue

                self.current_tool_index = index
                if len(self.streamed_args_for_tool) == index:
                    self.streamed_args_for_tool.append("")

                new_call_complete = (
                    index < len(tool_calls) - 1 or ")]" not in added_text
                )
                if new_call_complete:
                    self.current_tool_index += 1

                withheld_suffix = added_text[:-2] if not new_call_complete else ""
                if not new_call_complete and added_text[-2] == ")":
                    # Function call is incomplete. Withhold the closing bracket.
                    withheld_suffix = withheld_suffix + "}"
                # Strings get single quotes in the model-produced string.
                # JSON requires double quotes.
                withheld_suffix = withheld_suffix.replace("'", '"')
                delta = compute_tool_delta(
                    self.streamed_args_for_tool[index], new_call, index, withheld_suffix
                )

                if delta is not None:
                    tool_deltas.append(delta)
                    if (
                        delta.function is not None
                        and delta.function.arguments is not None
                    ):
                        self.streamed_args_for_tool[index] += delta.function.arguments

            # HACK: serving_chat.py inspects the internal state of tool parsers
            # when determining its final streaming delta, automatically
            # adding autocompleted JSON.
            # These two lines avoid that nonsense while ensuring finish_reason
            # is set to tool_calls when at least one tool is called.
            if tool_deltas and not self.prev_tool_call_arr:
                self.prev_tool_call_arr = [{"arguments": {}}]

            if tool_deltas:
                return DeltaMessage(tool_calls=tool_deltas)
            elif not added_text and self.current_tool_id > 0:
                # Return an empty DeltaMessage once the tool calls are all done
                # so that finish_reason gets set.
                return DeltaMessage(content="")
            else:
                return None
        except Exception:
            logger.exception("Error trying to handle streaming tool call.")
            logger.debug(
                "Skipping chunk as a result of tool streaming extraction error"
            )
            return None


def handle_single_tool_with_name_map(
    call: ast.Call, pythonic_to_tool_name: dict[str, str]
) -> ToolCall:
    tool_call = handle_single_tool(call)
    tool_call.function.name = pythonic_to_tool_name.get(
        tool_call.function.name, tool_call.function.name
    )
    return tool_call


def replace_non_letters(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]", "_", text)


def sanitize_function_names(text: str) -> str:
    pattern = r"([\[,]\s*)([^(]+)\("
    return re.sub(
        pattern,
        lambda match: match.group(1) + replace_non_letters(match.group(2)) + "(",
        text,
    )
