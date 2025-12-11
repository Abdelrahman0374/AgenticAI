from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from ..models.models import Message, LLMResponse


class LLMInterface(ABC):
    """
    Interface for LLM providers.

    The generate_text method should return a structured response with:
    - text: reasoning about the action to take
    - action: the action to perform with name and parameters
    """

    @abstractmethod
    def generate_text(self,
                      messages: List[Message],
                      tools: Optional[List[Any]]
                     ) -> LLMResponse:
        """
        Generate text response from the LLM.

        Args:
            messages: List of Message objects (HumanMessage, AIMessage, etc.) representing the conversation history
            tools: List of tool dictionaries from get_schema() in the format:
                [
                    {
                        "type": "function",
                        "function": {
                            "name": str,
                            "description": str,
                        "parameters": dict  # JSON schema from args_schema.model_json_schema()
                    }
                }

        Returns:
            LLMResponse with text and action fields
        """
        pass
