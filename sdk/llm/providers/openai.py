from typing import List, Dict, Any, Optional
import json
from openai import OpenAI
from sdk.memory import Memory
from sdk.models import (
    LLMResponse,
    ToolCall,
    Message,
    UserMessage,
    AIMessage,
    SystemMessage,
    ToolMessage
)
from sdk.llm.llm_interface import LLMInterface
from enum import Enum


class OpenAIResponse(Enum):
    MESSAGE = "message"
    FUNCTION_CALL = "function_call"


class OpenAIProvider(LLMInterface):
    """
    OpenAI provider implementation for LLM interactions.
    """

    def __init__(self, api_key: str, model: str = "gpt-4o-mini"):
        """
        Initialize OpenAI provider.

        Args:
            api_key: OpenAI API key
            model: Model name to use (default: gpt-4o-mini)
        """
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def generate_text(self,
                      messages: Memory,
                      tools: Optional[List[Any]] = None) -> LLMResponse:
        """
        Generate text response from OpenAI.

        Args:
            messages: List of Message objects representing conversation history
            tools: Optional list of tool schemas

        Returns:
            LLMResponse with text and/or tool calls
        """
        # Convert messages to OpenAI format
        openai_messages = self._parse_history(messages)
        # Add tools if provided
        openai_tools = None
        if tools:
            openai_tools = self._parse_tools(tools)

        response = self.client.chat.completions.create(
            model=self.model,
            messages=openai_messages,
            tools=openai_tools,
            temperature=0,
            )
        # Parse and return response
        return self._parse_response(response)

    def _parse_response(self, response) -> LLMResponse:
        """
        Convert OpenAI API response into LLMResponse object.

        Args:
            response: OpenAI ChatCompletion response

        Returns:
            LLMResponse object with text and/or tool calls
        """
        text_content = None
        tool_calls_list = []

        message = response.choices[0].message

        # Get text content
        if message.content:
            text_content = message.content

        # Get tool calls
        if message.tool_calls:
            for tc in message.tool_calls:
                tool_call = ToolCall(
                    name=tc.function.name,
                    tool_call_id=tc.id,
                    arguments=json.loads(tc.function.arguments)
                )
                tool_calls_list.append(tool_call)

        # Return list of tool calls or None
        final_tool_calls = None
        if len(tool_calls_list) > 0:
            final_tool_calls = tool_calls_list

        return LLMResponse(
            text=text_content,
            tool_calls=final_tool_calls
        )

    def _parse_history(self, history: List[Message]) -> List[Dict[str, Any]]:
        """
        Convert internal message objects into OpenAI chat format.

        Args:
            history: List of Message objects

        Returns:
            List of dictionaries in OpenAI message format
        """
        result = []

        for msg in history:
            # System Message
            if isinstance(msg, SystemMessage):
                result.append({
                    "role": "system",
                    "content": msg.content
                })

            # User Message
            elif isinstance(msg, UserMessage):
                result.append({
                    "role": "user",
                    "content": msg.content
                })

            # Assistant Message
            elif isinstance(msg, AIMessage):
                openai_msg = {"role": "assistant"}

                # Case 1: Plain string content
                if isinstance(msg.content, str):
                    openai_msg["content"] = msg.content

                # Case 2: LLMResponse object
                elif isinstance(msg.content, LLMResponse):
                    # Add text content if present
                    if msg.content.text:
                        openai_msg["content"] = msg.content.text

                    # Add tool calls if present
                    if msg.content.tool_calls:
                        tool_calls = (
                            msg.content.tool_calls
                            if isinstance(msg.content.tool_calls, list)
                            else [msg.content.tool_calls]
                        )

                        openai_msg["tool_calls"] = []
                        for call in tool_calls:
                            openai_msg["tool_calls"].append({
                                "id": call.tool_call_id,
                                "type": "function",
                                "function": {
                                    "name": call.name,
                                    "arguments": json.dumps(call.arguments)
                                }
                            })

                result.append(openai_msg)


            # Tool Message
            elif isinstance(msg, ToolMessage):
                content = ""
                if msg.content.success:
                    content = str(msg.content.result)
                else:
                    content = f"ERROR: {msg.content.error}"

                result.append({
                    "role": "tool",
                    "tool_call_id": msg.tool_call_id,
                    "content": content
                })
            else:
                raise ValueError(f"Unknown message type: {type(msg)}")

        return result

    def _parse_tools(self, tools: List[Any]) -> List[Dict[str, Any]]:
        """
        Convert tool schemas to OpenAI Chat Completions function calling format.

        Args:
            tools: List of tool objects or schemas

        Returns:
            List of tool schemas in OpenAI Chat Completions format
        """
        openai_tools = []

        for tool in tools:
            # If tool has get_schema method (BaseTool), call it
            if hasattr(tool, 'get_schema'):
                schema = tool.get_schema()
            # Otherwise assume it's already a schema dict
            elif isinstance(tool, dict):
                schema = tool
            else:
                raise ValueError(f"Invalid tool type: {type(tool)}")

            # Convert to OpenAI Chat Completions format
            # Always wrap in the correct format with nested "function" key
            openai_tools.append({
                "type": "function",
                "function": {
                    "name": schema.get("name"),
                    "description": schema.get("description"),
                    "parameters": schema.get("parameters", {})
                }
            })

        return openai_tools
