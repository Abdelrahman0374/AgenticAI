from .base_tool import BaseTool
from ..models import ToolResult
from pydantic import BaseModel, Field


class AskUserArgs(BaseModel):
    """Argument schema for the AskUserTool.

    Attributes:
        question: The question to ask the user
    """
    question: str = Field(..., description="The question to ask the user")


class AskUserTool(BaseTool):
    """Tool for asking the user a question and getting their input.

    This tool enables the agent to interact with the user by asking questions
    and receiving responses during execution. Useful when the agent needs
    clarification, additional information, or user decisions.

    The tool pauses execution, prompts the user with a question, waits for
    their response, and returns the answer to the agent.

    Attributes:
        name: "ask_user"
        description: Tool description for LLM
        args_schema: AskUserArgs for argument validation
        input_function: Callable to get user input (defaults to built-in input())

    Example:
        tool = AskUserTool()
        result = tool.run(question="What color do you prefer?")
        if result.success:
            print(result.result)  # User's answer

        # Custom input function (e.g., for GUI)
        def gui_input(prompt):
            return my_gui.show_dialog(prompt)

        tool = AskUserTool(input_function=gui_input)
    """

    def __init__(self, input_function=None):
        """Initialize the ask user tool.

        Args:
            input_function: Optional callable that takes a prompt string and
                          returns user input. Defaults to built-in input().
                          Useful for custom UIs or testing.
        """
        super().__init__(
            name="ask_user",
            description="Ask the user a question and get their response. Use this when you need clarification, additional information, or user decisions.",
            args_schema=AskUserArgs
        )
        self.input_function = input_function or input

    def run(self, question: str) -> ToolResult:
        """Ask the user a question and return their response.

        Displays the question to the user, waits for their input, and returns
        the response as a ToolResult.

        Args:
            question: The question to ask the user

        Returns:
            ToolResult with:
                - success=True and result=user_response if successful
                - success=False and error=message if failed (e.g., EOF, interrupt)

        Error conditions:
            - User interrupts with Ctrl+C (KeyboardInterrupt)
            - Input stream closed (EOFError)
            - Other I/O errors
        """
        try:
            # Ask the question and get user input
            user_response = self.input_function(f"\n{question}\nYour answer: ")

            # Return the response
            return ToolResult(
                success=True,
                result=user_response
            )
        except Exception as e:
            return ToolResult(
                success=False,
                error=f"Error getting user input: {str(e)}"
            )
