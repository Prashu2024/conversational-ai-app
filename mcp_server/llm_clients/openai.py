from openai import AsyncOpenAI # Use async client
from typing import List, Dict, Any, AsyncGenerator
import os
import logging

from .base import BaseLLMClient

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OpenAIClient(BaseLLMClient):
    """
    LLM client implementation for OpenAI models (GPT-4, GPT-3.5-turbo).
    """
    def __init__(self, api_key: str, model_name: str | None = None, **kwargs):
        """
        Initializes the OpenAI client.

        Args:
            api_key (str): The OpenAI API key.
            model_name (str | None): The specific OpenAI model name (e.g., "gpt-4").
                                     Defaults to "gpt-3.5-turbo" if not provided.
            **kwargs: Additional arguments for the OpenAI client constructor (rarely needed).
        """
        resolved_model_name = model_name or os.getenv("OPENAI_MODEL_NAME", "gpt-3.5-turbo")
        super().__init__(api_key=api_key, model_name=resolved_model_name)

        if not api_key:
            raise ValueError("OpenAI API key is required.")

        try:
            # Use the async client
            self.client = AsyncOpenAI(api_key=self.api_key, **kwargs)
            logger.info(f"OpenAI client initialized successfully for model: {self.model_name}")
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {e}", exc_info=True)
            raise RuntimeError(f"OpenAI initialization failed: {e}") from e

    def _validate_history(self, history: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Ensure history roles are valid for OpenAI ('system', 'user', 'assistant')."""
        validated_history = []
        valid_roles = {"system", "user", "assistant"}
        for message in history:
            role = message.get("role")
            content = message.get("content")
            if role in valid_roles and content is not None:
                validated_history.append({"role": role, "content": str(content)}) # Ensure content is string
            else:
                logger.warning(f"Invalid role '{role}' or missing content in OpenAI history, skipping message.")
        return validated_history

    async def generate_response(
        self,
        history: List[Dict[str, str]],
        **kwargs # Allows passing temperature, max_tokens etc.
    ) -> str:
        """
        Generate a non-streaming response using the OpenAI model.
        """
        if not history:
            return "Hello! How can I assist you today?"

        validated_history = self._validate_history(history)
        if not validated_history:
             logger.error("No valid messages found in history for OpenAI.")
             return "[Error: No valid conversation history provided]"

        logger.debug(f"Sending history to OpenAI: {validated_history}")

        try:
            response = await self.client.chat.completions.create(
                model=self.model_name,
                messages=validated_history,
                stream=False,
                **kwargs # Pass additional arguments like temperature, max_tokens
            )
            # Accessing the response content
            if response.choices and response.choices[0].message:
                content = response.choices[0].message.content
                logger.debug(f"Received OpenAI response: {content[:100] if content else 'None'}...")
                return content or "[No Content Received]"
            else:
                logger.warning("OpenAI response did not contain expected choices or message.")
                return "[No Content Received]"

        except Exception as e:
            logger.error(f"Error during OpenAI API call: {e}", exc_info=True)
            # Improve error reporting (e.g., check for specific OpenAI error types)
            raise RuntimeError(f"OpenAI API call failed: {e}") from e

    async def generate_streaming_response(
        self,
        history: List[Dict[str, str]],
        **kwargs # Allows passing temperature, max_tokens etc.
    ) -> AsyncGenerator[str, None]:
        """
        Generate a streaming response using the OpenAI model.
        """
        if not history:
            yield "Hello! How can I assist you today?"
            return

        validated_history = self._validate_history(history)
        if not validated_history:
             logger.error("No valid messages found in history for OpenAI streaming.")
             yield "[Error: No valid conversation history provided]"
             return

        logger.debug(f"Sending history to OpenAI (stream): {validated_history}")

        try:
            stream = await self.client.chat.completions.create(
                model=self.model_name,
                messages=validated_history,
                stream=True,
                **kwargs # Pass additional arguments
            )
            async for chunk in stream:
                # Accessing streaming content
                if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content is not None:
                    content_chunk = chunk.choices[0].delta.content
                    # logger.debug(f"Received OpenAI stream chunk: {content_chunk}") # Verbose
                    yield content_chunk
        except Exception as e:
            logger.error(f"Error during OpenAI streaming API call: {e}", exc_info=True)
            yield f"[Error: Failed to get streaming response from OpenAI - {e}]"
            # raise RuntimeError(f"OpenAI streaming API call failed: {e}") from e

