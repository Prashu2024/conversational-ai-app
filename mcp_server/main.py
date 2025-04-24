import logging
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from contextlib import asynccontextmanager
from typing import List, Dict
import os

from .config import settings
from .models import ChatRequest, ChatResponse, HealthCheckResponse, ChatMessage
# Import the globally initialized agent instance
from .agent import agent, ConversationalAgent

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Lifespan Management ---
# Use lifespan events for initialization/cleanup if needed
# For this example, the agent is initialized globally when agent.py is imported.
# If initialization were more complex or needed async operations, lifespan would be better.
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Application startup...")
    # Initialize resources here if needed (e.g., database connections)
    # Re-check agent initialization status at startup
    if agent.llm_client is None:
         logger.error("LLM Client failed to initialize during startup. Check config/keys.")
         # Depending on requirements, you might want to prevent startup
         # raise RuntimeError("Critical component (LLM Client) failed to initialize.")
    else:
        logger.info(f"LLM Client '{agent.llm_client.__class__.__name__}' seems initialized.")
    yield
    # Clean up resources here if needed
    logger.info("Application shutdown...")

# --- FastAPI App Initialization ---
app = FastAPI(
    title="Conversational AI MCP Server",
    description="Master Control Plane for handling chat requests and orchestrating LLM responses.",
    version="1.0.0",
    lifespan=lifespan # Use the lifespan context manager
)

# --- CORS Middleware ---
# Allow requests from your Gradio UI origin (and potentially others)
# Be more specific in production environments
origins = [
    "http://localhost",       # Common local development origin
    "http://localhost:7860",  # Default Gradio local port
    "http://127.0.0.1",
    "http://127.0.0.1:7860", # Default Gradio local IP
    # Add the origin where your Gradio app is hosted if deployed
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins, # Allows specified origins
    allow_credentials=True,
    allow_methods=["*"],   # Allows all methods (GET, POST, etc.)
    allow_headers=["*"],   # Allows all headers
)

# --- API Endpoints ---

@app.get("/", include_in_schema=False)
async def root():
    """Redirects root path to docs."""
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url='/docs')

@app.get("/health", response_model=HealthCheckResponse, tags=["Management"])
async def health_check():
    """Performs a health check of the server and LLM client status."""
    client_initialized = agent.llm_client is not None
    if not client_initialized:
        logger.warning("Health Check: LLM Client is not initialized.")
    return HealthCheckResponse(
        status="OK" if client_initialized else "WARNING",
        llm_provider=settings.LLM_PROVIDER,
        llm_client_initialized=client_initialized
    )

@app.post("/chat", response_model=ChatResponse, tags=["Chat"])
async def handle_chat(request: ChatRequest):
    """
    Handles a standard (non-streaming) chat request.
    Receives conversation history and returns a single complete response.
    """
    logger.info(f"Received standard chat request. History length: {len(request.history)}")
    if not request.history:
        raise HTTPException(status_code=400, detail="Chat history cannot be empty.")

    # Convert Pydantic models back to simple dicts for the agent
    history_dicts = [msg.model_dump() for msg in request.history]

    try:
        response_content = await agent.get_response(history_dicts)
        logger.info("Successfully generated standard response.")
        return ChatResponse(content=response_content)
    except RuntimeError as e:
        logger.error(f"Runtime error during chat processing: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error during chat processing: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="An internal server error occurred.")


async def stream_generator(history: List[Dict[str, str]]):
    """Helper async generator function to yield chunks from the agent."""
    try:
        async for chunk in agent.get_streaming_response(history):
            yield chunk
    except RuntimeError as e:
        logger.error(f"Runtime error during streaming: {e}", exc_info=True)
        yield f"[STREAM_ERROR: {e}]" # Send error message within the stream
    except Exception as e:
        logger.error(f"Unexpected error during streaming: {e}", exc_info=True)
        yield f"[STREAM_ERROR: An internal server error occurred during streaming.]"


@app.post("/chat_stream", tags=["Chat"])
async def handle_chat_stream(request: ChatRequest):
    """
    Handles a streaming chat request using Server-Sent Events (SSE).
    Receives conversation history and streams back the response chunk by chunk.
    """
    logger.info(f"Received streaming chat request. History length: {len(request.history)}")
    if not request.history:
        # StreamingResponse doesn't handle HTTPExceptions well before starting the stream
        # Return an error message directly in the stream if possible, or log and return empty stream
        logger.error("Chat history cannot be empty for streaming request.")
        async def empty_stream():
            yield "[ERROR: Chat history cannot be empty]"
            return
        return StreamingResponse(empty_stream(), media_type="text/event-stream")


    history_dicts = [msg.model_dump() for msg in request.history]

    # Return a StreamingResponse that uses the async generator
    # Use text/event-stream for Server-Sent Events (SSE) which Gradio can handle
    # Or use text/plain if the client expects raw text chunks
    return StreamingResponse(stream_generator(history_dicts), media_type="text/plain")
    # If using SSE format explicitly:
    # async def sse_formatted_stream(history: List[Dict[str, str]]):
    #     try:
    #         async for chunk in agent.get_streaming_response(history):
    #             if chunk: # Avoid sending empty data lines
    #                 yield f"data: {chunk}\n\n" # SSE format: "data: <content>\n\n"
    #         yield "event: end\ndata: {}\n\n" # Optional: signal end of stream
    #     except Exception as e:
    #         logger.error(f"Error during SSE streaming: {e}", exc_info=True)
    #         yield f"event: error\ndata: {{\"message\": \"{str(e)}\"}}\n\n"
    # return StreamingResponse(sse_formatted_stream(history_dicts), media_type="text/event-stream")


# --- Main Execution ---
# This block allows running the server directly using `python -m mcp_server.main`
if __name__ == "__main__":
    logger.info("Starting MCP Server...")
    # Get host and port from environment variables or use defaults
    host = os.getenv("MCP_HOST", "127.0.0.1")
    port = int(os.getenv("MCP_PORT", "8000"))
    # Use reload=True for development only
    reload = os.getenv("MCP_RELOAD", "false").lower() == "true"

    uvicorn.run(
        "mcp_server.main:app", # Path to the app instance
        host=host,
        port=port,
        reload=reload, # Enable auto-reload for development
        log_level="info" # Control uvicorn's logging level
    )

