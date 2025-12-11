from sdk.llm.llm_interface import LLMInterface
from sdk.tools import BaseTool
from sdk.memory import Memory
from sdk.models.models import LLMEnums, LLMResponse, ToolResult, ToolCall
from sdk.llm import Factory
from sdk.agent.base_agent import BaseAgent
from typing import List, Dict, Any, Optional

class Agent(BaseAgent):
    """Primary agent implementation with LLM integration and tool execution.

    This class provides a complete agent that can:
    - Process user queries through an LLM
    - Execute tools based on LLM decisions
    - Maintain conversation history via Memory
    - Support multi-turn interactions with tool calling

    The agent follows a think-act-observe loop:
    1. Think: LLM generates a response (text and/or tool calls)
    2. Act: Execute any requested tool calls
    3. Observe: Add tool results to memory and repeat

    Attributes:
        name: Optional name identifier for the agent
        system_message: System prompt to guide agent behavior
        tools: Dictionary mapping tool names to BaseTool instances
        history: Memory instance tracking conversation messages
        llm: LLMInterface implementation for text generation
        tools_schema: List of tool schemas for LLM function calling

    Example:
        from sdk import Agent, Memory, Factory
        from sdk.tools import ReadFileTool, WriteFileTool

        agent = Agent(
            tools=[ReadFileTool(), WriteFileTool()],
            name="FileAssistant",
            system_message="You help users manage files.",
            history=Memory(),
            llm=Factory().create()
        )
        result = agent.run("Read example.txt")
    """
    def __init__(self,
                 tools: List[BaseTool],
                 name: str = None,
                 system_message: str = None,
                 history: Memory = None,
                 llm: LLMInterface = None):
        """Initialize the agent with tools, memory, and LLM configuration.

        Args:
            tools: List of BaseTool instances the agent can execute
            name: Optional identifier for the agent
            system_message: System prompt to guide agent behavior and personality
            history: Memory instance for conversation tracking (creates new if None)
            llm: LLMInterface implementation (uses Factory default if None)

        Raises:
            ValueError: If tools list is empty or contains invalid tool instances
        """
        self.name = name
        self.system_message = system_message
        self.tools = {tool.name: tool for tool in tools}
        self.history = history
        self.llm = llm
        self.tools_schema = [tool.get_schema() for tool in tools]

        # LLm validation
        if self.llm is None:
            self.llm = Factory().create()

        # TODO: Handling: system message, history -> Validation

    def _think(self):
        """Generate LLM response based on conversation history and available tools.

        Retrieves messages from history, passes them to the LLM along with tool schemas,
        and returns the LLM's decision (text response and/or tool calls).

        Returns:
            LLMResponse: Contains text content and/or tool calls from the LLM
        """
        messages = self.history.get_messages()
        tool_schema = self.tools_schema if self.tools_schema else {}
        response = self.llm.generate_text(messages=messages, tools=tool_schema)
        return response


    def _execute(self, tool_calls: List[ToolCall]) -> List[ToolResult]:
        """Execute a list of tool calls and return their results.

        For each tool call:
        1. Validates the tool exists in the agent's tool list
        2. Validates the arguments against the tool's schema
        3. Executes the tool with validated arguments
        4. Captures results or errors

        Args:
            tool_calls: List of ToolCall objects specifying which tools to execute and with what arguments

        Returns:
            List of ToolResult objects, one for each tool call, containing
            success status and either result data or error messages
        """
        results = []

        for tc in tool_calls:
            # Validate tool exists
            if not self._validate_tool_calls(tc.name):
                results.append(ToolResult(
                    success=False,
                    error=f"Tool name of the '{tc.name}' is not available in the agent's tool list."
                ))
                continue

            try:
                # Get the tool instance
                tool = self.tools[tc.name]
                # Validate arguments
                validated_args = tool.validate_args(tc.arguments)
                # Execute the tool
                result = tool.run(**validated_args.model_dump())
                results.append(result)
            except Exception as e:
                results.append(ToolResult(
                    success=False,
                    error=f"Error executing tool '{tc.name}': {str(e)}"
                ))

        return results

    def _validate_tool_calls(self, tool_name: str) -> bool:
        """Check if a tool name exists in the agent's available tools.

        Args:
            tool_name: Name of the tool to validate

        Returns:
            True if the tool is available, False otherwise
        """
        return tool_name in self.tools

    def run(self, user_input: Optional[str] = None) -> str:
        """Execute the agent's main processing loop.

        The agent follows this cycle:
        1. Add user input to memory (if provided)
        2. Think: Get LLM response based on history and available tools
        3. Add LLM response to memory
        4. If text-only response, return it (exit condition)
        5. If tool calls present, execute them
        6. Add tool results to memory
        7. Repeat from step 2

        This loop continues until the LLM returns a text-only response without
        any tool calls, indicating the task is complete.

        Args:
            user_input: Optional user message to process. If None, continues
                       from previous conversation state

        Returns:
            The final text response from the LLM after all tool executions
            are complete

        Example:
            agent = Agent(tools=[...], history=Memory())
            response = agent.run("What's in example.txt?")
            # Agent will call read_file tool and respond with contents
        """
        if user_input:
            self.history.add_user_message(user_input)

        while True:
            response = self._think()

            # Update memory with assistant message
            self.history.add_assistant_message(response)

            # Decide next action
            if response.is_text_only():
                # Exit Condition: when LLM returns text only
                return response.text

            elif response.has_tool_call():
                # Ensure tool_calls is a list
                tool_calls = response.tool_calls if isinstance(response.tool_calls, list) else [response.tool_calls]

                # Execute all tool calls
                results = self._execute(tool_calls)

                # Update memory with tool results
                for i, (tc, result) in enumerate(zip(tool_calls, results)):
                    tool_call_id = tc.tool_call_id if tc.tool_call_id else tc.name
                    self.history.add_tool_message(result, tool_call_id=tool_call_id)
