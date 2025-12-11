from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from ..models.models import Message, LLMResponse


class LLMInterface(ABC):
    """Abstract interface for Large Language Model providers.

    Defines the contract that all LLM provider implementations must follow.
    This interface enables swapping between different LLM providers (OpenAI,
    Anthropic, etc.) without changing agent code.

    The primary method, generate_text(), handles:
    - Processing conversation history (messages)
    - Function/tool calling via schema definitions
    - Generating structured responses with text and/or tool calls

    Implementations should:
    - Convert internal Message objects to provider-specific format
    - Handle tool schemas in provider's function calling format
    - Parse provider responses into standardized LLMResponse objects
    - Manage API authentication and rate limiting

    Example Implementation:
        class MyLLMProvider(LLMInterface):
            def __init__(self, api_key: str):
                self.client = MyLLMClient(api_key)

            def generate_text(self, messages, tools=None):
                # Convert messages to provider format
                # Call provider API
                # Parse and return LLMResponse
                pass
    """

    @abstractmethod
    def generate_text(self,
                      messages: List[Message],
                      tools: Optional[List[Any]]
                     ) -> LLMResponse:
        """Generate a text response from the LLM with optional tool calling.

        Args:
            messages: List of Message objects (UserMessage, AIMessage, SystemMessage,
                     ToolMessage) representing the conversation history. Messages
                     should be in chronological order.
            tools: Optional list of tool schema dictionaries for function calling.
                  Each tool schema should follow this format:
                  {
                      "type": "function",
                      "function": {
                          "name": str,  # Unique tool identifier
                          "description": str,  # What the tool does
                          "parameters": dict  # JSON schema for arguments
                      }
                  }

        Returns:
            LLMResponse: Structured response containing:
                - text: Optional reasoning or direct response text
                - tool_calls: Optional list of ToolCall objects with:
                    - name: Tool to execute
                    - tool_call_id: Unique identifier for this call
                    - arguments: Dict of validated arguments

        Raises:
            Exception: Provider-specific errors (authentication, rate limits, etc.)

        Example:
            messages = [SystemMessage("You are helpful"),
                       UserMessage("What's 2+2?")]
            response = llm.generate_text(messages, tools=None)
            print(response.text)  # "2+2 equals 4"
        """
        pass
