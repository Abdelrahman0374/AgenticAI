from pydantic import BaseModel, Field
from typing import Literal, Union, Dict, Any, List, TypedDict, Optional
from enum import Enum


class LLMEnums(Enum):
    """Enumeration of supported LLM providers.

    Defines the available Large Language Model providers that can be used
    with the agent framework.
    """
    OPENAI = "OPENAI"
    GEMINI = "GEMINI"

class MessageRole(Enum):
    """Enumeration of message roles in conversations.

    Defines the different roles that can participate in agent conversations:
    - SYSTEM: System instructions and context
    - USER: Human user input
    - ASSISTANT: AI assistant responses
    - TOOL: Tool execution results
    """
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"

class OpenAIMessageRole(Enum):
    """OpenAI-specific message role enumeration.

    Mirrors the message roles used by OpenAI's Chat Completions API.
    Used for format conversion between internal and OpenAI representations.
    """
    SYSTEM = "system"
    USER =  "user"
    ASSISTANT = "assistant"
    TOOL = "tool"

#-----------------------------------------------------------
class ToolCall(BaseModel):
    """Represents a request from the LLM to execute a tool/function.

    When an LLM wants to perform an action (like reading a file or calling
    an API), it generates a ToolCall specifying which tool to use and what
    arguments to pass.

    Attributes:
        name: Identifier of the tool to execute (must match a registered tool)
        tool_call_id: Unique identifier for this specific call, used to link
                     the call with its result in conversation history
        arguments: Dictionary of argument names to values, validated against
                  the tool's schema before execution

    Example:
        ToolCall(
            name="read_file",
            tool_call_id="call_abc123",
            arguments={"file_path": "example.txt"}
        )
    """
    name: str
    tool_call_id: Optional[str] = None
    arguments: Dict[str, Any]

class LLMResponse(BaseModel):
    """Structured response from an LLM with text and/or tool calls.

    This flexible model can represent different types of LLM responses:
    1. Text only: Direct answer without tools
    2. Tool calls only: Action requests without explanatory text
    3. Both: Reasoning text plus tool execution requests

    The model supports single or multiple tool calls, enabling the LLM to
    request parallel tool executions in a single turn.

    Attributes:
        text: Optional text content from the LLM (reasoning, explanation, or answer)
        tool_calls: Optional single ToolCall or list of ToolCalls for execution

    Examples:
        # Text only response
        LLMResponse(
            text="Hello! How can I help you today?",
            tool_calls=None
        )

        # Tool call only
        LLMResponse(
            text=None,
            tool_calls=ToolCall(
                name="write_file",
                arguments={"file_path": "test.txt", "content": "Hello"}
            )
        )

        # Text + tool call
        LLMResponse(
            text="I'll write that file for you.",
            tool_calls=ToolCall(
                name="write_file",
                arguments={"file_path": "test.txt", "content": "Hello"}
            )
        )

        # Multiple tool calls
        LLMResponse(
            text="Reading both files now.",
            tool_calls=[
                ToolCall(name="read_file", arguments={"file_path": "a.txt"}),
                ToolCall(name="read_file", arguments={"file_path": "b.txt"})
            ]
        )
    """
    text: Optional[str] = None
    tool_calls: Optional[Union[ToolCall, List[ToolCall]]] = None

    def has_tool_call(self) -> bool:
        """Check if the response contains any tool calls.

        Returns:
            True if tool_calls is not None, False otherwise
        """
        return self.tool_calls is not None

    def is_text_only(self) -> bool:
        """Check if the response contains only text without tool calls.

        Returns:
            True if response has text but no tool calls, False otherwise
        """
        return self.text is not None and not self.has_tool_call()

    def validate_agent_tools(self, available_tools: List[str]) -> bool:
        """Validate that all requested tools are available to the agent.

        Checks each tool call against the list of available tool names to ensure
        the LLM only requested tools it has access to.

        Args:
            available_tools: List of tool names that the agent can execute

        Returns:
            True if all tool calls reference available tools, False if any
            tool is not available
        """
        for item in self.tool_calls:
            if item.name not in available_tools:
                return False
        return True

#-----------------------------------------------------------
class ToolResult(BaseModel):
    """Result of a tool execution.

    Captures the outcome of executing a tool, including success/failure status
    and either the result data or error information. Used to provide feedback
    to the LLM about tool execution outcomes.

    Attributes:
        success: True if tool executed successfully, False if it failed
        result: The tool's output data (only present if successful)
        error: Error message explaining what went wrong (only present if failed)

    Examples:
        # Successful execution
        ToolResult(
            success=True,
            result="File contents: Hello World",
            error=None
        )

        # Failed execution
        ToolResult(
            success=False,
            result=None,
            error="File not found: missing.txt"
        )
    """
    success: bool = False
    result: Optional[str] = None
    error: Optional[str] = None

class UserMessage(BaseModel):
    """Message from the human user.

    Represents input from the user to the AI assistant. These messages typically
    contain questions, requests, or instructions that trigger agent processing.

    Attributes:
        role: Always "user" (from MessageRole.USER)
        content: The user's message text

    Example:
        UserMessage(content="Please read the file example.txt")
    """
    role: str = MessageRole.USER.value
    content: str

class AIMessage(BaseModel):
    """Message from the AI assistant.

    Represents responses generated by the LLM. The content can be either plain
    text or a structured LLMResponse object containing text and/or tool calls.

    Attributes:
        role: Always "assistant" (from MessageRole.ASSISTANT)
        content: Either a string response or an LLMResponse object with
                text and/or tool calls

    Examples:
        # Simple text response
        AIMessage(content="I can help you with that!")

        # Response with tool call
        AIMessage(content=LLMResponse(
            text="I'll read that file now.",
            tool_calls=ToolCall(name="read_file", arguments={...})
        ))
    """
    role: str = MessageRole.ASSISTANT.value
    content: Union[str, LLMResponse]

class SystemMessage(BaseModel):
    """System-level instruction message.

    Contains instructions that define the AI's behavior, role, constraints, and
    context. System messages are typically added once at the start of a conversation
    and guide all subsequent AI responses.

    Attributes:
        role: Always "system" (from MessageRole.SYSTEM)
        content: The system instruction text

    Example:
        SystemMessage(content="You are a helpful coding assistant. Always provide clear, well-commented code.")
    """
    role: str = MessageRole.SYSTEM.value
    content: str

class ToolMessage(BaseModel):
    """Message containing tool execution results.

    Represents the outcome of a tool execution, providing feedback to the LLM
    about whether the tool succeeded and what data it returned. The tool_call_id
    links this result to the original ToolCall that requested it.

    Attributes:
        role: Always "tool" (from MessageRole.TOOL)
        content: ToolResult object with success status and result/error data
        tool_call_id: Identifier linking this result to the original tool call

    Example:
        ToolMessage(
            content=ToolResult(success=True, result="Hello World"),
            tool_call_id="call_abc123"
        )
    """
    role: str = MessageRole.TOOL.value
    content: ToolResult
    tool_call_id: Optional[str] = None

# Union type for any message
Message = Union[UserMessage, AIMessage, SystemMessage, ToolMessage]
