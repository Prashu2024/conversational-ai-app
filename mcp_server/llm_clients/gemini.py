import google.generativeai as genai
from google.generativeai.types import GenerationConfig, ContentDict, PartDict, HarmCategory, HarmBlockThreshold
from typing import List, Dict, Any, AsyncGenerator
import os
import logging

from .base import BaseLLMClient

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Safety settings configuration - adjust as needed
DEFAULT_SAFETY_SETTINGS = {
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
}

# Generation configuration - adjust as needed
DEFAULT_GENERATION_CONFIG = GenerationConfig(
    # temperature=0.7, # Example: Control randomness
    # top_p=1.0,       # Example: Nucleus sampling
    # top_k=40,        # Example: Top-k sampling
    # max_output_tokens=2048 # Example: Limit response length
)

class GeminiClient(BaseLLMClient):
    """
    LLM client implementation for Google Gemini models.
    """
    def __init__(self, api_key: str, model_name: str | None = None, **kwargs):
        """
        Initializes the Gemini client.

        Args:
            api_key (str): The Google AI API key.
            model_name (str | None): The specific Gemini model name (e.g., "gemini-2.0-flash").
                                     Defaults to "gemini-pro" if not provided.
            **kwargs: Additional arguments (e.g., safety_settings, generation_config).
        """
        resolved_model_name = model_name or os.getenv("GEMINI_MODEL_NAME", "gemini-2.0-flash")
        super().__init__(api_key=api_key, model_name=resolved_model_name)

        if not api_key:
            raise ValueError("Gemini API key is required.")

        try:
            genai.configure(api_key=api_key)
            self.safety_settings = kwargs.get("safety_settings", DEFAULT_SAFETY_SETTINGS)
            self.generation_config = kwargs.get("generation_config", DEFAULT_GENERATION_CONFIG)
            self.model = genai.GenerativeModel(
                model_name=self.model_name,
                safety_settings=self.safety_settings,
                generation_config=self.generation_config
            )
            logger.info(f"Gemini client initialized successfully for model: {self.model_name}")
        except Exception as e:
            logger.error(f"Failed to initialize Gemini client: {e}", exc_info=True)
            raise RuntimeError(f"Gemini initialization failed: {e}") from e

    def _convert_history_to_gemini_format(self, history: List[Dict[str, str]]) -> List[ContentDict]:
        """
        Converts the generic history format to Gemini's ContentDict format.
        Handles potential role issues and ensures 'parts' is always a list of PartDict.
        """
        gemini_history: List[ContentDict] = []
        for message in history:
            role = message.get("role")
            content = message.get("content", "")

            # Gemini uses 'user' and 'model' roles. Map 'assistant' to 'model'.
            if role == "assistant":
                gemini_role = "model"
            elif role == "user":
                gemini_role = "user"
            else:
                logger.warning(f"Unsupported role '{role}' in history, skipping message.")
                continue # Skip messages with unsupported roles

            # Ensure content is wrapped in PartDict structure
            part: PartDict = {"text": content}
            gemini_history.append({"role": gemini_role, "parts": [part]})

        return gemini_history

    async def generate_response(
        self,
        history: List[Dict[str, str]],
        **kwargs # Allows passing extra args like temperature, top_p etc. if needed later
    ) -> str:
        """
        Generate a non-streaming response using the Gemini model.
        """
        if not history:
            return "Hello! How can I help you today?" # Or some default greeting

        gemini_history = self._convert_history_to_gemini_format(history)
        # The last message is the user's prompt, separate it if needed by the API structure
        # For gemini-pro's generate_content, the whole history is passed.
        # The last message in the formatted history implicitly acts as the prompt.

        logger.debug(f"Sending history to Gemini: {gemini_history}")

        try:
            # Use generate_content for conversational history
            response = await self.model.generate_content_async(
                gemini_history,
                stream=False,
                # You could potentially override generation_config or safety_settings here via kwargs
                # generation_config=GenerationConfig(**kwargs) if kwargs else self.generation_config
            )
            # Accessing the text content safely
            if response.parts:
                 full_response = "".join(part.text for part in response.parts if hasattr(part, 'text'))
                 logger.debug(f"Received Gemini response: {full_response[:100]}...") # Log snippet
                 return full_response
            elif response.candidates and response.candidates[0].content.parts:
                 # Handle cases where response might be nested differently
                 full_response = "".join(part.text for part in response.candidates[0].content.parts if hasattr(part, 'text'))
                 logger.debug(f"Received Gemini response (candidate): {full_response[:100]}...") # Log snippet
                 return full_response
            else:
                 logger.warning("Gemini response did not contain expected text parts.")
                 # Check for blocked prompts
                 if response.prompt_feedback.block_reason:
                     block_reason = response.prompt_feedback.block_reason.name
                     logger.error(f"Gemini request blocked due to: {block_reason}")
                     return f"[Blocked by Safety Filter: {block_reason}]"
                 return "[No Content Received]"

        except Exception as e:
            logger.error(f"Error during Gemini API call: {e}", exc_info=True)
            # Consider more specific error handling based on google.api_core.exceptions
            raise RuntimeError(f"Gemini API call failed: {e}") from e


    async def generate_streaming_response(
        self,
        history: List[Dict[str, str]],
        **kwargs # Allows passing extra args like temperature, top_p etc. if needed later
    ) -> AsyncGenerator[str, None]:
        """
        Generate a streaming response using the Gemini model.
        """
        if not history:
            yield "Hello! How can I help you today?"
            return

        gemini_history = self._convert_history_to_gemini_format(history)
        logger.debug(f"Sending history to Gemini (stream): {gemini_history}")

        try:
            # Use generate_content with stream=True
            async for chunk in await self.model.generate_content_async(
                gemini_history,
                stream=True,
                # generation_config=GenerationConfig(**kwargs) if kwargs else self.generation_config
            ):
                # Safely access text part of the chunk
                if chunk.parts:
                    text_chunk = "".join(part.text for part in chunk.parts if hasattr(part, 'text'))
                    if text_chunk:
                        # logger.debug(f"Received Gemini stream chunk: {text_chunk}") # Can be very verbose
                        yield text_chunk
                elif chunk.prompt_feedback.block_reason:
                     block_reason = chunk.prompt_feedback.block_reason.name
                     logger.error(f"Gemini stream blocked during generation due to: {block_reason}")
                     yield f"[Blocked by Safety Filter: {block_reason}]"
                     break # Stop streaming if blocked

        except Exception as e:
            logger.error(f"Error during Gemini streaming API call: {e}", exc_info=True)
            # Yield an error message to the stream
            yield f"[Error: Failed to get streaming response from Gemini - {e}]"
            # Optionally re-raise or handle differently
            # raise RuntimeError(f"Gemini streaming API call failed: {e}") from e

