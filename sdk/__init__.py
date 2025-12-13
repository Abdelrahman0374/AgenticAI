"""SDK package for building AI agents with LLM integration.

This package provides a comprehensive framework for creating AI agents with:
- Agent orchestration and tool execution
- Memory management for conversation history
- LLM provider integration (OpenAI, etc.)
- Tools
- Structured message and response models

Typical usage:
    from sdk import Agent, Memory, Factory
    from sdk.tools import ReadFileTool, WriteFileTool

    llm = Factory().create()
    memory = Memory(system_message="You are a helpful assistant")
    agent = Agent(tools=[ReadFileTool(), WriteFileTool()], history=memory, llm=llm)
    response = agent.run("Read the file example.txt")
"""

from sdk.agent import Agent
from sdk.memory import Memory
from sdk.tools import ReadFileTool, WriteFileTool, AgentTool, AskUserTool
from sdk.llm import Factory
from sdk.models import LLMResponse, ToolResult, ToolCall, Message, UserMessage, AIMessage, ToolMessage, SystemMessage


__all__ = [
    "Agent",
    "Memory",
    "ReadFileTool",
    "WriteFileTool",
    "AgentTool",
    "AskUserTool",
    "Factory",
    "LLMResponse",
    "ToolResult",
    "ToolCall",
    "Message",
    "UserMessage",
    "AIMessage",
    "ToolMessage",
    "SystemMessage"
]
