from typing import List, Optional
from sdk.models.models import LLMResponse
from ..models import Message, SystemMessage, AIMessage, UserMessage, ToolMessage, ToolResult, MessageRole

class Memory():
    """Manages conversation history and message storage for agent interactions.

    The Memory class maintains a chronological list of messages exchanged between
    the user, AI assistant, system, and tools. It provides a clean interface for
    adding different message types and retrieving the full conversation history.

    Message types supported:
    - SystemMessage: Initial instructions/context for the AI
    - UserMessage: Messages from the user
    - AIMessage: Responses from the AI assistant
    - ToolMessage: Results from tool executions

    Attributes:
        messages: List of Message objects in chronological order

    Example:
        memory = Memory(system_message="You are a helpful coding assistant")
        memory.add_user_message("Hello!")
        memory.add_assistant_message("Hi! How can I help you?")

        # Get all messages for LLM processing
        history = memory.get_messages()
    """
    def __init__(self, system_message: Optional[str] = None):
        """Initialize Memory with optional system message.

        Args:
            system_message: Optional system prompt to set agent behavior and context.
                          If provided, added as the first message in history.
        """
        self.messages: List[Message] = []
        if system_message is not None:
            self.add_system_message(system_message)

    def add_system_message(self, content: str):
        """Add a system message to set context and instructions for the AI.

        System messages typically define the AI's role, behavior, constraints,
        and any special instructions. Usually added once at initialization.

        Args:
            content: The system message text (e.g., "You are a helpful assistant")
        """
        self.messages.append(SystemMessage(content= content))

    def add_user_message(self, content: str):
        """Add a user message to the conversation history.

        Represents input from the human user. This typically triggers the agent
        to process and generate a response.

        Args:
            content: The user's message text
        """
        self.messages.append(UserMessage(content= content))

    def add_assistant_message(self, content: str):
        """Add an AI assistant message to the conversation history.

        Stores the AI's response, which can be plain text or include tool calls.
        The content can be either a string or an LLMResponse object.

        Args:
            content: The assistant's response text or LLMResponse object
        """
        self.messages.append(AIMessage(content= content))

    def add_tool_message(self, content: ToolResult, tool_call_id: str = None):
        """Add a tool execution result to the conversation history.

        Records the outcome of a tool execution, including success/failure status
        and any returned data or error messages. These messages provide feedback
        to the AI about tool execution outcomes.

        Args:
            content: ToolResult object containing execution results:
                    - success: bool indicating if tool executed successfully
                    - result: str with the tool's output (if successful)
                    - error: str with error message (if failed)
            tool_call_id: Unique identifier linking this result to the original
                         tool call request from the AI
        """
        self.messages.append(ToolMessage(content= content, tool_call_id= tool_call_id))

    def get_messages(self) -> List[dict]:
        """Retrieve all messages in chronological order.

        Returns the complete conversation history for processing by the LLM or
        for serialization/storage.

        Returns:
            List of Message objects (SystemMessage, UserMessage, AIMessage,
            ToolMessage) in the order they were added
        """
        return self.messages

    def clear(self):
        """Remove all messages from memory.

        Resets the conversation history to an empty state. Useful for starting
        a new conversation or clearing context when memory grows too large.

        Note:
            This permanently removes all messages including any system messages.
            You may need to re-add the system message after clearing.
        """
        self.messages = []
