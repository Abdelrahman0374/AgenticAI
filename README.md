# Agent SDK

A Python SDK for building AI agents with LLM integration, tool execution, and memory management.

## Features

- **Agent System**: Complete agent implementation with think-act-observe loop
- **LLM Integration**: Support for multiple LLM providers (OpenAI)
- **Tool Execution**: Extensible tool system for agent capabilities
- **Memory Management**: Conversation history tracking
- **Type Safety**: Pydantic models for data validation

## Installation

```bash
pip install openai  # Required for LLM functionality
```

Set up your OpenAI API key:

```bash
export OPENAI_API_KEY='your-api-key-here'  # Linux/Mac
set OPENAI_API_KEY=your-api-key-here      # Windows
```

## Quick Start

```python
from sdk import Agent, Memory, Factory
from sdk.tools import ReadFileTool, WriteFileTool

# Initialize the agent
agent = Agent(
    tools=[ReadFileTool(), WriteFileTool()],
    name="FileAssistant",
    system_message="You help users manage files.",
    history=Memory(),
    llm=Factory().create()
)

# Run the agent
result = agent.run("Read example.txt and summarize it")
print(result)
```

## Architecture

### Agent Module

- **Agent**: Main agent class with LLM integration and tool execution
- **BaseAgent**: Abstract base class for custom agent implementations

### LLM Module

- **Factory**: Creates LLM provider instances
- **LLMInterface**: Abstract interface for LLM providers
- **Providers**: Currently supports OpenAI (GPT-4o-mini default)

### Tools Module

- **BaseTool**: Abstract base class for creating custom tools
- **ReadFileTool**: Read file contents
- **WriteFileTool**: Write content to files
- **AgentTool**: Execute sub-agents as tools

### Memory Module

- **Memory**: Manages conversation history and context

### Models Module

- **LLMEnums**: Enumerations for LLM-related constants
- **LLMResponse**: Structured LLM response format
- **ToolResult**: Tool execution results
- **ToolCall**: Tool invocation data

## Creating Custom Tools

```python
from sdk.tools import BaseTool
from pydantic import BaseModel, Field

class MyToolArgs(BaseModel):
    param: str = Field(description="Parameter description")

class MyTool(BaseTool):
    def __init__(self):
        super().__init__(
            name="my_tool",
            description="What this tool does",
            args_schema=MyToolArgs
        )

    def execute(self, **kwargs):
        # Your tool logic here
        return {"result": "success"}
```

## Agent Workflow

The agent follows a think-act-observe loop:

1. **Think**: LLM processes the query and decides on actions (text response and/or tool calls)
2. **Act**: Execute any requested tool calls
3. **Observe**: Add tool results to memory and continue if needed
4. **Respond**: Return final result to user

## Examples

### Simple Query

```python
agent = Agent(tools=[], llm=Factory().create())
response = agent.run("What is 2+2?")
```

### With Tools

```python
from sdk.tools import ReadFileTool, WriteFileTool

agent = Agent(
    tools=[ReadFileTool(), WriteFileTool()],
    llm=Factory().create()
)

response = agent.run("Read data.txt and write a summary to summary.txt")
```

### Custom System Message

```python
agent = Agent(
    tools=[],
    system_message="You are a helpful coding assistant specializing in Python.",
    llm=Factory().create()
)
```

## Project Structure

```
sdk/
├── __init__.py
├── agent/
│   ├── Agent.py          # Main agent implementation
│   └── base_agent.py     # Agent base class
├── llm/
│   ├── factory.py        # LLM factory
│   ├── llm_interface.py  # LLM interface
│   └── providers/
│       └── openai.py     # OpenAI provider
├── memory/
│   └── memory.py         # Memory management
├── models/
│   └── models.py         # Data models
└── tools/
    ├── base_tool.py      # Tool base class
    ├── read_file.py      # File reading tool
    ├── write_file.py     # File writing tool
    └── agent_tool.py     # Sub-agent tool
```

## Requirements

- Python 3.8+
- openai
- pydantic

## License
