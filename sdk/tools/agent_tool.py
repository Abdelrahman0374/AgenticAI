from .base_tool import BaseTool
from ..models import ToolResult
from pydantic import BaseModel, Field

class Args(BaseModel):
    """Argument schema for AgentTool.

    Attributes:
        query: Message or question to forward to the delegated agent
    """
    query: str = Field(..., description="Message to forward to the agent")

class AgentTool(BaseTool):
    """Tool adapter that exposes an Agent instance as a callable tool.

    Enables agent composition by allowing one agent to delegate tasks to another
    specialized agent. This supports hierarchical agent architectures where a
    coordinator agent can invoke multiple specialized agents as tools.

    The delegated agent receives a query string, processes it through its own
    loop (with its own tools and LLM), and returns the final result.

    Attributes:
        name: Unique identifier for this agent tool
        description: Description for the LLM explaining when to use this agent
        agent: The Agent instance to delegate to
        args_schema: Args schema for query validation

    Example:
        # Create a specialized agent
        file_agent = Agent(
            tools=[ReadFileTool(), WriteFileTool()],
            name="FileAgent",
            system_message="You manage files."
        )

        # Wrap it as a tool
        file_tool = AgentTool(
            agent=file_agent,
            name="file_agent",
            description="Delegate file operations to the file agent"
        )

        # Use in a coordinator agent
        coordinator = Agent(
            tools=[file_tool],
            name="Coordinator"
        )
    """
    def __init__(self, agent, name: str, description: str = None):
        """Initialize the agent tool with a wrapped agent.

        Args:
            agent: The Agent instance to delegate queries to
            name: Unique tool name (what the LLM will call to invoke this agent)
            description: Explanation of what this agent does and when to use it.
                        If not provided, defaults to "Calls agent: {name}"
        """
        super().__init__(
            name=name,
            description=description or f"Calls agent: {name}",
            args_schema=Args
        )
        self.agent = agent

    def run(self, query: str) -> ToolResult:
        """Forward a query to the underlying agent and return its result.

        Executes the delegated agent's run() method with the provided query
        and wraps the response in a ToolResult.

        Args:
            query: The query or task to delegate to the agent

        Returns:
            ToolResult with:
                - success=True and result=agent_response if execution succeeds
                - success=False and error=error_message if execution fails
        """
        try:
            result = self.agent.run(query)
            return ToolResult(success=True, result=result)
        except Exception as e:
            return ToolResult(success=False, error=f"Agent execution failed: {str(e)}")
