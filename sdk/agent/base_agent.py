from abc import ABC, abstractmethod

class BaseAgent(ABC):
    """Abstract base class for all agent implementations.

    This class defines the core interface that all agents must implement.
    Agents are responsible for processing user input, making decisions,
    executing tools, and generating responses through an iterative loop.

    Subclasses must implement:
        run(): Main execution loop for processing requests

    Example:
        class MyAgent(BaseAgent):
            def run(self, user_input: str) -> str:
                # Implementation here
                pass
    """
    @abstractmethod
    def run(self):
        """Agent Main Execution Loop"""
        pass
