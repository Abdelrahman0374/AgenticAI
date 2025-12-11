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
    """OpenAI GPT model provider implementation.

    Implements the LLMInterface for OpenAI's Chat Completions API, supporting:
    - GPT-4 and GPT-3.5 models
    - Function calling / tool use
    - Conversation history management
    - Message format conversion between internal and OpenAI formats

    This provider handles:
    - Authentication via API key
    - Message format conversion (SystemMessage, UserMessage, etc. -> OpenAI format)
    - Tool schema conversion to OpenAI function calling format
    - Response parsing from OpenAI to LLMResponse objects

    Attributes:
        client: Configured OpenAI client instance
        model: Model identifier (e.g., 'gpt-4o-mini', 'gpt-4', 'gpt-3.5-turbo')

    Example:
        provider = OpenAIProvider(
            api_key="sk-...",
            model="gpt-4o-mini"
        )

        messages = [UserMessage(content="Hello")]
        response = provider.generate_text(messages, tools=None)
        print(response.text)
    """

    def __init__(self, api_key: str, model: str = "gpt-4o-mini"):
        """Initialize OpenAI provider with authentication and model selection.

        Args:
            api_key: OpenAI API key for authentication
            model: OpenAI model identifier. Defaults to "gpt-4o-mini".
                  Common options: "gpt-4", "gpt-4o-mini", "gpt-3.5-turbo"

        Raises:
            OpenAIError: If API key is invalid or authentication fails
        """
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def generate_text(self,
                      messages: Memory,
                      tools: Optional[List[Any]] = None) -> LLMResponse:
        """Generate text response from OpenAI with optional function calling.

        Converts internal message format to OpenAI's Chat Completions format,
        sends the request with optional tool schemas, and parses the response
        back into a standardized LLMResponse.

        Args:
            messages: List of Message objects (UserMessage, AIMessage, SystemMessage,
                     ToolMessage) representing conversation history
            tools: Optional list of tool schemas for function calling. Each tool
                  should have 'name', 'description', and 'parameters' fields.

        Returns:
            LLMResponse containing:
                - text: The model's text response (if any)
                - tool_calls: List of ToolCall objects if model requests tool use

        Raises:
            OpenAIError: If API request fails
            JSONDecodeError: If tool call arguments are malformed
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
        """Convert OpenAI ChatCompletion response to internal LLMResponse format.

        Extracts text content and tool calls from the OpenAI response object
        and structures them into a standardized LLMResponse that can be used
        by the agent framework.

        Args:
            response: OpenAI ChatCompletion response object from the API

        Returns:
            LLMResponse with:
                - text: String content from the assistant (None if no text)
                - tool_calls: List of ToolCall objects (None if no tool calls)

        Note:
            Returns None for tool_calls if the list is empty, rather than an
            empty list, to simplify downstream conditional logic.
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
        """Convert internal Message objects to OpenAI Chat Completions format.

        Transforms the SDK's message types (UserMessage, AIMessage, SystemMessage,
        ToolMessage) into the dictionary format expected by OpenAI's API. Handles
        both simple text messages and complex tool call messages.

        Args:
            history: List of Message objects representing the conversation

        Returns:
            List of dictionaries in OpenAI message format with 'role' and 'content'
            fields, plus additional fields for tool calls and tool responses

        Raises:
            ValueError: If an unknown message type is encountered

        Note:
            - AIMessage content can be either a string or LLMResponse object
            - Tool calls are serialized to JSON in the OpenAI format
            - Tool results are formatted with ERROR prefix if unsuccessful
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
        """Convert tool schemas to OpenAI Chat Completions function calling format.

        Transforms tool definitions into the format expected by OpenAI's function
        calling API. Handles both BaseTool instances (with get_schema method) and
        pre-formatted schema dictionaries.

        Args:
            tools: List of tool objects (with get_schema method) or schema dictionaries
                  containing 'name', 'description', and 'parameters' fields

        Returns:
            List of tool schemas in OpenAI Chat Completions format:
            [
                {
                    "type": "function",
                    "function": {
                        "name": str,
                        "description": str,
                        "parameters": dict  # JSON Schema
                    }
                },
                ...
            ]

        Raises:
            ValueError: If tool type is invalid (not a schema dict or tool object)

        Note:
            Always wraps schemas in the OpenAI format with "type": "function"
            and nested "function" key, even if input is a raw schema dict.
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
