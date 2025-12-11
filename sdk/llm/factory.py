from sdk.llm.providers.openai import OpenAIProvider
from .llm_interface import LLMInterface
import os

class Factory():
    """Factory for creating LLM provider instances.

    Implements the Factory Method pattern to create and configure LLM provider
    instances. Currently supports OpenAI's GPT models with configurable API keys
    and model selection.

    The factory handles:
    - Environment variable configuration (OPENAI_API_KEY)
    - Default model selection
    - Provider initialization

    Example:
        # Uses environment variable OPENAI_API_KEY
        llm = Factory().create()

        # Custom configuration
        import os
        os.environ['OPENAI_API_KEY'] = 'your-key-here'
        llm = Factory().create()
    """
    def create(self) -> LLMInterface | None:
        """Create and configure an LLM provider instance.

        Retrieves the OpenAI API key from the OPENAI_API_KEY environment variable
        and creates an OpenAI provider instance with the default model (gpt-4o-mini).

        Returns:
            LLMInterface: Configured OpenAI provider instance ready for use

        Raises:
            ValueError: If OPENAI_API_KEY environment variable is not set

        Note:
            Future versions may support additional providers and configuration options.
        """
        # If no config provided, use environment variables
        api_key = os.getenv("OPENAI_API_KEY")
        model =  "gpt-4o-mini"
        return OpenAIProvider(
                api_key=api_key,
                model=model
        )
