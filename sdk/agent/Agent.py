from sdk.llm.llm_interface import LLMInterface
from sdk.tools import BaseTool
from sdk.memory import Memory
from sdk.models.models import LLMEnums, LLMResponse, ToolResult, ToolCall
from sdk.llm import Factory
from sdk.agent.base_agent import BaseAgent
from typing import List, Dict, Any, Optional

class Agent(BaseAgent):
    """Agent that inherits from BaseAgent and uses registry"""
    def __init__(self,
                 tools: List[BaseTool],
                 name: str = None,
                 system_message: str = None,
                 history: Memory = None,
                 llm: LLMInterface = None):
        """
        Initialize agent with registry and tool names

        Args:
            registry: ToolRegistry instance containing all available tools
            tools: List of tool names this agent can use
            provider_name: Name of LLM provider (default: LLMEnums.GEMINI.value)
            api_key: API key for the provider
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
        """LLM thinking process - get tool schemas and generate response"""
        messages = self.history.get_messages()
        tool_schema = self.tools_schema if self.tools_schema else {}
        response = self.llm.generate_text(messages=messages, tools=tool_schema)
        return response


    def _execute(self, tool_calls: List[ToolCall]) -> List[ToolResult]:
        """Execute tool calls and return results"""
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
        """Validate if the tool call is available in the agent's tool list"""
        return tool_name in self.tools

    def run(self, user_input: Optional[str] = None) -> str:
        """Run the agent loop"""
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
