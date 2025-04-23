from anthropic import AsyncAnthropic # Use async client
from typing import List, Dict, Any, AsyncGenerator
import os
import logging

from .base import BaseLLMClient

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AnthropicClient(BaseLLMClient):
    """
    LLM client implementation for Anthropic Claude models.
    """
    def __init__(self, api_key: str, model_name: str | None = None, **kwargs):
        """
        Initializes the Anthropic client.

        Args:
            api_key (str): The Anthropic API key.
            model_name (str | None): The specific Claude model name (e.g., "claude-3-opus-20240229").
                                     Defaults to "claude-3-sonnet-20240229" if not provided.
            **kwargs: Additional arguments for the Anthropic client constructor.
        """
        # Default to a generally available and capable model like Sonnet if none specified
        resolved_model_name = model_name or os.getenv("ANTHROPIC_MODEL_NAME", "claude-3-sonnet-20240229")
        super().__init__(api_key=api_key, model_name=resolved_model_name)

        if not api_key:
            raise ValueError("Anthropic API key is required.")

        try:
            # Use the async client
            self.client = AsyncAnthropic(api_key=self.api_key, **kwargs)
            logger.info(f"Anthropic client initialized successfully for model: {self.model_name}")
        except Exception as e:
            logger.error(f"Failed to initialize Anthropic client: {e}", exc_info=True)
            raise RuntimeError(f"Anthropic initialization failed: {e}") from e

    def _validate_and_format_history(self, history: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """
        Validates history and formats it for Anthropic API.
        Ensures alternating user/assistant roles and handles potential 'system' prompt.
        """
        formatted_history = []
        system_prompt = None
        valid_roles = {"system", "user", "assistant"}
        last_role = None

        for i, message in enumerate(history):
            role = message.get("role")
            content = message.get("content")

            if not role or content is None:
                logger.warning(f"Skipping message with missing role or content: {message}")
                continue

            if role == "system":
                if i == 0: # System prompt should ideally be first
                    system_prompt = str(content)
                    logger.debug(f"Using system prompt: {system_prompt[:100]}...")
                else:
                    logger.warning("System prompt found later in history, ignoring.")
                continue # Don't add system prompt to messages list here

            if role not in valid_roles:
                 logger.warning(f"Unsupported role '{role}' for Anthropic, skipping.")
                 continue

            # Ensure alternating user/assistant roles (Anthropic requirement)
            if last_role and role == last_role:
                logger.warning(f"Consecutive '{role}' messages detected. Anthropic requires alternating roles. Merging or skipping might be needed, but currently skipping previous.")
                if formatted_history: formatted_history.pop() # Remove previous to allow current
                # Or, potentially merge content if feasible (more complex)

            # Anthropic expects string content
            formatted_history.append({"role": role, "content": str(content)})
            last_role = role


        # Anthropic API requires the conversation to start with a 'user' message if no system prompt
        if formatted_history and formatted_history[0]["role"] == "assistant":
             logger.warning("History starts with 'assistant'. Prepending a placeholder user message.")
             # This is a workaround. Ideally, the calling logic ensures valid sequences.
             formatted_history.insert(0, {"role": "user", "content": "(Context starts)"})


        # Return value includes messages and optional system prompt for the API call
        return {"messages": formatted_history, "system": system_prompt}


    async def generate_response(
        self,
        history: List[Dict[str, str]],
        max_tokens: int = 1024, # Default max tokens for Anthropic
        **kwargs # Allows passing temperature etc.
    ) -> str:
        """
        Generate a non-streaming response using the Anthropic model.
        """
        if not history:
            return "Hello! How may I help you today?"

        formatted_input = self._validate_and_format_history(history)
        messages = formatted_input["messages"]
        system_prompt = formatted_input["system"]

        if not messages:
            logger.error("No valid messages found after formatting history for Anthropic.")
            return "[Error: No valid conversation history provided]"

        logger.debug(f"Sending messages to Anthropic: {messages}")
        if system_prompt: logger.debug(f"Using system prompt: {system_prompt}")

        try:
            # Note: Anthropic uses 'max_tokens' directly in the main call, not within kwargs usually
            api_kwargs = {
                "model": self.model_name,
                "messages": messages,
                "max_tokens": max_tokens,
                **kwargs # Pass other args like temperature
            }
            if system_prompt:
                api_kwargs["system"] = system_prompt

            response = await self.client.messages.create(**api_kwargs)

            # Accessing the response content
            if response.content and isinstance(response.content, list) and len(response.content) > 0:
                 # Assuming the first block is the primary text response
                 if hasattr(response.content[0], 'text'):
                     full_response = response.content[0].text
                     logger.debug(f"Received Anthropic response: {full_response[:100]}...")
                     return full_response
                 else:
                    logger.warning("Anthropic response content block does not have 'text' attribute.")
                    return "[No Text Content Received]"
            else:
                logger.warning("Anthropic response did not contain expected content blocks.")
                # Check stop reason
                if response.stop_reason:
                    logger.info(f"Anthropic generation stopped due to: {response.stop_reason}")
                    if response.stop_reason == "max_tokens":
                        return "[Response Truncated: Max tokens reached]"
                return "[No Content Received]"

        except Exception as e:
            logger.error(f"Error during Anthropic API call: {e}", exc_info=True)
            # Add more specific Anthropic error handling if needed
            raise RuntimeError(f"Anthropic API call failed: {e}") from e

    async def generate_streaming_response(
        self,
        history: List[Dict[str, str]],
        max_tokens: int = 1024, # Default max tokens for Anthropic
        **kwargs # Allows passing temperature etc.
    ) -> AsyncGenerator[str, None]:
        """
        Generate a streaming response using the Anthropic model.
        """
        if not history:
            yield "Hello! How may I help you today?"
            return

        formatted_input = self._validate_and_format_history(history)
        messages = formatted_input["messages"]
        system_prompt = formatted_input["system"]

        if not messages:
            logger.error("No valid messages found after formatting history for Anthropic streaming.")
            yield "[Error: No valid conversation history provided]"
            return

        logger.debug(f"Sending messages to Anthropic (stream): {messages}")
        if system_prompt: logger.debug(f"Using system prompt: {system_prompt}")

        try:
            # Use context manager for streaming
            api_kwargs = {
                "model": self.model_name,
                "messages": messages,
                "max_tokens": max_tokens,
                **kwargs # Pass other args like temperature
            }
            if system_prompt:
                api_kwargs["system"] = system_prompt

            async with await self.client.messages.stream(**api_kwargs) as stream:
                 async for text_chunk in stream.text_stream:
                    # logger.debug(f"Received Anthropic stream chunk: {text_chunk}") # Verbose
                    yield text_chunk

                 # After streaming, get final message details if needed
                 final_message = await stream.get_final_message()
                 if final_message.stop_reason:
                     logger.info(f"Anthropic stream finished. Stop reason: {final_message.stop_reason}")
                     if final_message.stop_reason == "max_tokens":
                         yield "\n[Response Truncated: Max tokens reached]"


        except Exception as e:
            logger.error(f"Error during Anthropic streaming API call: {e}", exc_info=True)
            yield f"[Error: Failed to get streaming response from Anthropic - {e}]"
            # raise RuntimeError(f"Anthropic streaming API call failed: {e}") from e

