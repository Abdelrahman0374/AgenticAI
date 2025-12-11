from abc import ABC, abstractmethod
from typing import Type, Any, Dict
from unicodedata import name
from pydantic import BaseModel
from ..models import ToolResult

class BaseTool(ABC):
    """Base class for all tools.

    Attributes:
        name: Unique identifier for the tool.
        description: Human-readable description of what the tool does.
        args_schema: Pydantic model class for validating tool arguments.
        schema: JSON schema dict for the tool (follows function calling format).
    """

    def __init__(self, name: str, description: str, args_schema: Type[BaseModel]):
        """Initialize the base tool.

        Args:
            name: Unique identifier for the tool.
            description: Description of what the tool does.
            args_schema: Pydantic model class for argument validation.
        """
        self.name = name
        self.description = description
        self.args_schema = args_schema
        self.schema = {
            "type": "function",
            "name": self.name,
            "description": self.description,
            "parameters": self.args_schema.model_json_schema()
        }

    @abstractmethod
    def run(self, **kwargs) -> ToolResult:
        """Execute the tool with the provided arguments.

        Args:
            **kwargs: Validated arguments matching the args_schema.

        Returns:
            A ToolResult instance with the execution result.
        """
        pass

    def get_schema(self) -> Dict[str, Any]:
        """Get the JSON schema for this tool.

        Returns:
            A dictionary containing the tool's schema in function calling format.
        """
        return self.schema

    def validate_args(self, args: Dict[str, Any]) -> BaseModel:
        """Validate and parse arguments using the tool's schema.

        Args:
            args: Dictionary of arguments to validate.

        Returns:
            A Pydantic model instance with validated arguments.

        Raises:
            ValidationError: If arguments don't match the schema.
        """
        return self.args_schema(**args)
