from sdk.llm.providers.openai import OpenAIProvider
from .llm_interface import LLMInterface
import os

class Factory():
    """
    The Creator class in the Factory Method Pattern.
    It declares the factory method that returns an ILLMProvider object.
    """
    def create(self) -> LLMInterface | None:
        # If no config provided, use environment variables
        api_key = os.getenv("OPENAI_API_KEY")
        model =  "gpt-4o-mini"
        return OpenAIProvider(
                api_key=api_key,
                model=model
        )
