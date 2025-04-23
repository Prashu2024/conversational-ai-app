import abc
from typing import List, Dict, Any, AsyncGenerator

class BaseLLMClient(abc.ABC):
    """
    Abstract base class for Large Language Model clients.
    Defines the common interface for interacting with different LLM providers.
    """

    @abc.abstractmethod
    def __init__(self, api_key: str, model_name: str | None = None, **kwargs):
        """
        Initialize the LLM client.

        Args:
            api_key (str): The API key for the LLM provider.
            model_name (str | None): The specific model to use (optional).
            **kwargs: Additional provider-specific arguments.
        """
        self.api_key = api_key
        self.model_name = model_name
        print(f"Initialized {self.__class__.__name__} with model: {self.model_name or 'default'}") # Basic logging

    @abc.abstractmethod
    async def generate_response(
        self,
        history: List[Dict[str, str]],
        **kwargs
    ) -> str:
        """
        Generate a non-streaming response based on the conversation history.

        Args:
            history (List[Dict[str, str]]): A list of message dictionaries,
                                             e.g., [{"role": "user", "content": "Hello"},
                                                    {"role": "assistant", "content": "Hi there!"}]
            **kwargs: Additional provider-specific arguments for generation.

        Returns:
            str: The generated text response from the LLM.

        Raises:
            Exception: If the API call fails.
        """
        pass

    @abc.abstractmethod
    async def generate_streaming_response(
        self,
        history: List[Dict[str, str]],
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """
        Generate a streaming response based on the conversation history.

        Args:
            history (List[Dict[str, str]]): The conversation history.
            **kwargs: Additional provider-specific arguments for generation.

        Yields:
            str: Chunks of the generated text response.

        Raises:
            Exception: If the API call fails.
        """
        # This is crucial for the generator type hint
        # Even if the implementation overrides it completely,
        # this satisfies the abstract method requirement for an async generator.
        if False: # Never executed, but satisfies type checker
            yield
