import os
from dotenv import load_dotenv
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables from a .env file if it exists
# Useful for local development
dotenv_path = os.path.join(os.path.dirname(__file__), '..', '.env') # Look for .env in parent dir
if os.path.exists(dotenv_path):
    logger.info(f"Loading environment variables from: {dotenv_path}")
    load_dotenv(dotenv_path=dotenv_path)
else:
    logger.info(".env file not found, relying on system environment variables.")

class Settings:
    """
    Holds the application configuration settings, loaded from environment variables.
    """
    # LLM Provider Choice
    LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "GEMINI").upper()

    # API Keys
    GEMINI_API_KEY: str | None = os.getenv("GEMINI_API_KEY")
    OPENAI_API_KEY: str | None = os.getenv("OPENAI_API_KEY")
    ANTHROPIC_API_KEY: str | None = os.getenv("ANTHROPIC_API_KEY")

    # Optional Model Names (allow overrides via env vars)
    GEMINI_MODEL_NAME: str | None = os.getenv("GEMINI_MODEL_NAME", "gemini-2.0-flash") # Default gemini model
    OPENAI_MODEL_NAME: str | None = os.getenv("OPENAI_MODEL_NAME", "gpt-3.5-turbo") # Default openai model
    ANTHROPIC_MODEL_NAME: str | None = os.getenv("ANTHROPIC_MODEL_NAME", "claude-3-sonnet-20240229") # Default anthropic model

    # Server Configuration
    MCP_SERVER_URL: str = os.getenv("MCP_SERVER_URL", "http://127.0.0.1:8000") # For Gradio to connect

    def __init__(self):
        """Validate required settings."""
        logger.info(f"Selected LLM Provider: {self.LLM_PROVIDER}")
        if self.LLM_PROVIDER == "GEMINI" and not self.GEMINI_API_KEY:
            logger.warning("LLM_PROVIDER is GEMINI, but GEMINI_API_KEY is not set.")
            # raise ValueError("GEMINI_API_KEY must be set when LLM_PROVIDER is GEMINI")
        elif self.LLM_PROVIDER == "OPENAI" and not self.OPENAI_API_KEY:
            logger.warning("LLM_PROVIDER is OPENAI, but OPENAI_API_KEY is not set.")
            # raise ValueError("OPENAI_API_KEY must be set when LLM_PROVIDER is OPENAI")
        elif self.LLM_PROVIDER == "ANTHROPIC" and not self.ANTHROPIC_API_KEY:
            logger.warning("LLM_PROVIDER is ANTHROPIC, but ANTHROPIC_API_KEY is not set.")
            # raise ValueError("ANTHROPIC_API_KEY must be set when LLM_PROVIDER is ANTHROPIC")
        elif self.LLM_PROVIDER not in ["GEMINI", "OPENAI", "ANTHROPIC"]:
            raise ValueError(f"Invalid LLM_PROVIDER: {self.LLM_PROVIDER}. Choose from GEMINI, OPENAI, ANTHROPIC.")

        logger.info("Configuration loaded successfully.")
        logger.debug(f"Settings: {self.__dict__}") # Log all settings at debug level


# Create a single instance of settings to be imported elsewhere
settings = Settings()