from pydantic import BaseModel, Field
from typing import Literal, Union, Dict, Any, List, TypedDict, Optional
from enum import Enum


class LLMEnums(Enum):
    OPENAI = "OPENAI"
    GEMINI = "GEMINI"

class MessageRole(Enum):
    """Enum for message roles"""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"

class OpenAIMessageRole(Enum):
    SYSTEM = "system"
    USER =  "user"
    ASSISTANT = "assistant"
    TOOL = "tool"

#-----------------------------------------------------------
class ToolCall(BaseModel):
    """Represents a function/tool call"""
    name: str
    tool_call_id: Optional[str] = None
    arguments: Dict[str, Any]

class LLMResponse(BaseModel):
    """
    Flexible response structure from LLM.

    Can contain:
    - text: Direct text response from the LLM
    - function_call: A tool/function call with name and arguments
    - Both: Text explanation + function call

    Examples:

    1. Text only:
    {
        "text": "Hello! How can I help you today?",
        "function_call": None
    }

    2. Function call only:
    {
        "text": None,
        "function_call": {
            "name": "write_file",
            "arguments": {"file_path": "test.txt", "content": "Hello"}
        }
    }

    3. Both:
    {
        "text": "I'll write that file for you.",
        "function_call": {
            "name": "write_file",
            "arguments": {"file_path": "test.txt", "content": "Hello"}
        }
    }
    """
    text: Optional[str] = None
    tool_calls: Optional[Union[ToolCall, List[ToolCall]]] = None

    def has_tool_call(self) -> bool:
        """Check if response has a tool call"""
        return self.tool_calls is not None

    def is_text_only(self) -> bool:
        """Check if response is text only"""
        return self.text is not None and not self.has_tool_call()

    def validate_agent_tools(self, available_tools: List[str]) -> bool:
        """Validate if the tool call is for an available tool"""
        for item in self.tool_calls:
            if item.name not in available_tools:
                return False
        return True

#-----------------------------------------------------------
class ToolResult(BaseModel):
    """Represents the result of a tool execution"""
    success: bool = False
    result: Optional[str] = None
    error: Optional[str] = None

class UserMessage(BaseModel):
    """Represents a human/user message"""
    role: str = MessageRole.USER.value
    content: str

class AIMessage(BaseModel):
    """Represents an AI/assistant message"""
    role: str = MessageRole.ASSISTANT.value
    content: Union[str, LLMResponse]

class SystemMessage(BaseModel):
    """Represents a system message"""
    role: str = MessageRole.SYSTEM.value
    content: str

class ToolMessage(BaseModel):
    """Represents a tool message"""
    role: str = MessageRole.TOOL.value
    content: ToolResult
    tool_call_id: Optional[str] = None

# Union type for any message
Message = Union[UserMessage, AIMessage, SystemMessage, ToolMessage]
