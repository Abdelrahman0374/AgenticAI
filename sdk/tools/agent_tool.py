from .base_tool import BaseTool
from ..models import ToolResult
from pydantic import BaseModel, Field

class Args(BaseModel):
    query: str = Field(..., description="Message to forward to the agent")

class AgentTool(BaseTool):
    """
    Generic adapter that exposes ANY Agent instance as a callable tool.
    """
    def __init__(self, agent, name: str, description: str = None):
        """
        agent: the Agent instance
        name: unique tool name (LLM will call this)
        """
        super().__init__(
            name=name,
            description=description or f"Calls agent: {name}",
            args_schema=Args
        )
        self.agent = agent

    def run(self, query: str) -> ToolResult:
        """Forward query to underlying agent

        Args:
            query: The query to forward to the agent.

        Returns:
            A ToolResult with the agent's response.
        """
        try:
            result = self.agent.run(query)
            return ToolResult(success=True, result=result)
        except Exception as e:
            return ToolResult(success=False, error=f"Agent execution failed: {str(e)}")
