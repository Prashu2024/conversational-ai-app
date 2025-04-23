import logging
from typing import List, Dict, AsyncGenerator

from .config import settings
from .llm_clients.base import BaseLLMClient
from .llm_clients.gemini import GeminiClient
from .llm_clients.openai import OpenAIClient
from .llm_clients.anthropic import AnthropicClient

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ConversationalAgent:
    """
    Manages the conversation flow, history, and interaction with the selected LLM.
    """
    def __init__(self):
        """Initializes the agent and the appropriate LLM client based on settings."""
        self.llm_client: BaseLLMClient | None = None
        self._initialize_llm_client()

    def _initialize_llm_client(self):
        """Creates an instance of the LLM client based on the configuration."""
        provider = settings.LLM_PROVIDER
        logger.info(f"Initializing LLM client for provider: {provider}")

        try:
            if provider == "GEMINI":
                if not settings.GEMINI_API_KEY:
                    raise ValueError("GEMINI_API_KEY is not configured.")
                self.llm_client = GeminiClient(
                    api_key=settings.GEMINI_API_KEY,
                    model_name=settings.GEMINI_MODEL_NAME
                )
            elif provider == "OPENAI":
                if not settings.OPENAI_API_KEY:
                    raise ValueError("OPENAI_API_KEY is not configured.")
                self.llm_client = OpenAIClient(
                    api_key=settings.OPENAI_API_KEY,
                    model_name=settings.OPENAI_MODEL_NAME
                )
            elif provider == "ANTHROPIC":
                if not settings.ANTHROPIC_API_KEY:
                    raise ValueError("ANTHROPIC_API_KEY is not configured.")
                self.llm_client = AnthropicClient(
                    api_key=settings.ANTHROPIC_API_KEY,
                    model_name=settings.ANTHROPIC_MODEL_NAME
                )
            else:
                # This case should ideally be caught by config validation, but added for safety
                raise ValueError(f"Unsupported LLM provider configured: {provider}")

            logger.info(f"Successfully initialized LLM client: {self.llm_client.__class__.__name__}")

        except ValueError as ve:
             logger.error(f"Configuration error: {ve}")
             # Decide how to handle this - raise, or set client to None and handle in chat methods
             self.llm_client = None
             # Optionally raise a more specific configuration error
             # raise ConfigurationError(f"Failed to initialize LLM client: {ve}") from ve
        except Exception as e:
            logger.error(f"Unexpected error initializing LLM client for {provider}: {e}", exc_info=True)
            self.llm_client = None
            # raise InitializationError(f"Unexpected error initializing LLM client: {e}") from e


    def _check_client(self):
        """Checks if the LLM client is initialized."""
        if self.llm_client is None:
            logger.error("LLM client is not initialized. Check configuration and API keys.")
            raise RuntimeError("Conversational agent is not properly configured. LLM client is missing.")


    async def get_response(self, history: List[Dict[str, str]]) -> str:
        """
        Gets a standard, non-streaming response from the configured LLM.

        Args:
            history (List[Dict[str, str]]): The conversation history including the latest user message.

        Returns:
            str: The LLM's response.

        Raises:
            RuntimeError: If the LLM client is not initialized or if the API call fails.
        """
        self._check_client()
        logger.info(f"Getting standard response using {self.llm_client.__class__.__name__}")
        try:
            # The history passed here should already include the latest user message
            response = await self.llm_client.generate_response(history)
            return response
        except Exception as e:
            logger.error(f"Error getting response from LLM: {e}", exc_info=True)
            # Return a user-friendly error message or re-raise
            return f"[Error communicating with LLM: {e}]"
            # Or re-raise specific errors if needed: raise e

    async def get_streaming_response(self, history: List[Dict[str, str]]) -> AsyncGenerator[str, None]:
        """
        Gets a streaming response from the configured LLM.

        Args:
            history (List[Dict[str, str]]): The conversation history including the latest user message.

        Yields:
            str: Chunks of the LLM's response.

        Raises:
            RuntimeError: If the LLM client is not initialized.
        """
        self._check_client()
        logger.info(f"Getting streaming response using {self.llm_client.__class__.__name__}")
        try:
            # The history passed here includes the latest user message
            async for chunk in self.llm_client.generate_streaming_response(history):
                yield chunk
        except Exception as e:
            logger.error(f"Error getting streaming response from LLM: {e}", exc_info=True)
            # Yield an error message within the stream
            yield f"[Error during streaming: {e}]"
            # Avoid re-raising here as it would break the generator flow for the caller


# Global agent instance (can be managed differently, e.g., with dependency injection in FastAPI)
agent = ConversationalAgent()
