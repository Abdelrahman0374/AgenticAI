from typing import List, Optional
from sdk.models.models import LLMResponse
from ..models import Message, SystemMessage, AIMessage, UserMessage, ToolMessage, ToolResult, MessageRole

class Memory():
    def __init__(self, system_message: Optional[str] = None):
        self.messages: List[Message] = []
        if system_message is not None:
            self.add_system_message(system_message)

    def add_system_message(self, content: str):
        """Add a system message."""
        self.messages.append(SystemMessage(content= content))

    def add_user_message(self, content: str):
        """Add a user message."""
        self.messages.append(UserMessage(content= content))

    def add_assistant_message(self, content: str):
        """Add an assistant message."""
        self.messages.append(AIMessage(content= content))

    def add_tool_message(self, content: ToolResult, tool_call_id: str = None):
        """Add a tool message.

        Args:
            content: The ToolResult from tool execution.
        """
        self.messages.append(ToolMessage(content= content, tool_call_id= tool_call_id))

    def get_messages(self) -> List[dict]:
        """Get all messages as dictionaries."""
        return self.messages

    def clear(self):
        """Clear all messages."""
        self.messages = []
