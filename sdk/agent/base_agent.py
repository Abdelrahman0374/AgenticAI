from abc import ABC, abstractmethod

class BaseAgent(ABC):
    @abstractmethod
    def run(self):
        """Run the agent loop."""
        pass
