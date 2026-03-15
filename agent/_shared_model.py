import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from smolagents import Model
from smolagents.models import (
    ChatMessage,
    ChatMessageToolCall,
    ChatMessageToolCallFunction,
    MessageRole,
    TokenUsage,
)

_env_path = Path(__file__).parent.parent / ".env"
load_dotenv(dotenv_path=_env_path)


_SMOLAGENTS_TO_JSONSCHEMA = {
    "string": "string",
    "str": "string",
    "integer": "integer",
    "int": "integer",
    "number": "number",
    "float": "number",
    "boolean": "boolean",
    "bool": "boolean",
    "array": "array",
    "list": "array",
    "object": "object",
    "dict": "object",
    "any": None,
}


def _tool_to_anthropic_schema(tool) -> dict:
    properties = {}
    required = []

    for param_name, param_info in tool.inputs.items():
        smolagents_type = param_info.get("type", "any")
        json_type = _SMOLAGENTS_TO_JSONSCHEMA.get(smolagents_type)

        prop = {"description": param_info.get("description", "")}
        if json_type:
            prop["type"] = json_type
        properties[param_name] = prop
        required.append(param_name)

    return {
        "name": tool.name,
        "description": tool.description,
        "input_schema": {
            "type": "object",
            "properties": properties,
            "required": required,
        },
    }


class AnthropicModel(Model):
    def __init__(
        self,
        model_id: str = "claude-haiku-4-5-20251001",
        max_tokens: int = 4096,
        temperature: float = 0.5,
    ):
        super().__init__(model_id=model_id)
        self.max_tokens = max_tokens
        self.temperature = temperature

        import anthropic
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError(
                "ANTHROPIC_API_KEY not found. "
                "Make sure your .env file exists in the parent directory."
            )
        self._client = anthropic.Anthropic(api_key=api_key)

    def generate(
        self,
        messages: list,
        stop_sequences: list[str] | None = None,
        response_format: dict | None = None,
        tools_to_call_from: list | None = None,
        **kwargs,
    ) -> ChatMessage:
        system_prompt = ""
        anthropic_messages = []

        for msg in messages:
            if hasattr(msg, "role"):
                role = msg.role.value if hasattr(msg.role, "value") else str(msg.role)
                content = msg.content
                tool_calls = getattr(msg, "tool_calls", None)
            else:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                tool_calls = msg.get("tool_calls", None)

            if role == "system":
                system_prompt = content or ""

            elif role in ("user", "tool-response"):
                if role == "tool-response" and isinstance(content, list):
                    anthropic_messages.append({"role": "user", "content": content})
                else:
                    anthropic_messages.append({
                        "role": "user",
                        "content": str(content) if content is not None else "",
                    })

            elif role in ("assistant", "tool-call"):
                if tool_calls:
                    content_blocks = []
                    if content:
                        content_blocks.append({"type": "text", "text": str(content)})
                    for tc in tool_calls:
                        name = tc.function.name if hasattr(tc.function, "name") else tc["function"]["name"]
                        args = tc.function.arguments if hasattr(tc.function, "arguments") else tc["function"]["arguments"]
                        content_blocks.append({
                            "type": "tool_use",
                            "id": tc.id if hasattr(tc, "id") else tc.get("id", "tool_0"),
                            "name": name,
                            "input": args if isinstance(args, dict) else json.loads(args),
                        })
                    anthropic_messages.append({"role": "assistant", "content": content_blocks})
                else:
                    anthropic_messages.append({
                        "role": "assistant",
                        "content": str(content) if content is not None else "",
                    })

        if not anthropic_messages:
            anthropic_messages.append({"role": "user", "content": "Hello"})

        call_kwargs: dict[str, Any] = {
            "model": self.model_id,
            "max_tokens": self.max_tokens,
            "messages": anthropic_messages,
        }
        if system_prompt:
            call_kwargs["system"] = system_prompt
        if stop_sequences:
            call_kwargs["stop_sequences"] = stop_sequences

        if tools_to_call_from:
            call_kwargs["tools"] = [_tool_to_anthropic_schema(t) for t in tools_to_call_from]
            call_kwargs["tool_choice"] = {"type": "auto"}

        response = self._client.messages.create(**call_kwargs)

        text_content = ""
        tool_calls_out = None

        for block in response.content:
            if block.type == "text":
                text_content += block.text
            elif block.type == "tool_use":
                if tool_calls_out is None:
                    tool_calls_out = []
                tool_calls_out.append(
                    ChatMessageToolCall(
                        id=block.id,
                        type="function",
                        function=ChatMessageToolCallFunction(
                            name=block.name,
                            arguments=block.input,
                            description=None,
                        ),
                    )
                )

        return ChatMessage(
            role=MessageRole.ASSISTANT,
            content=text_content if text_content else None,
            tool_calls=tool_calls_out,
            raw=response,
            token_usage=TokenUsage(
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
            ),
        )


def get_model(model_id: str = "claude-haiku-4-5-20251001") -> AnthropicModel:
    return AnthropicModel(model_id=model_id)
