"""LLM integration module for Large Language Model providers.

This module provides:
- Factory: Factory class for creating LLM provider instances
- LLMInterface: Abstract interface for LLM providers
- OpenAI provider implementation for GPT models
"""

from .factory import Factory

__all__ = [
    "Factory",
]
