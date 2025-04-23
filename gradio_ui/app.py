import gradio as gr
import requests
import json
import os
import logging
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Configuration Loading ---
# Load .env file from the parent directory relative to this script's location
dotenv_path = os.path.join(os.path.dirname(__file__), '..', '.env')
if os.path.exists(dotenv_path):
    logger.info(f"Loading environment variables from: {dotenv_path}")
    load_dotenv(dotenv_path=dotenv_path)
else:
    # Try loading from the current directory as a fallback
    dotenv_path_current = os.path.join(os.path.dirname(__file__), '.env')
    if os.path.exists(dotenv_path_current):
        logger.info(f"Loading environment variables from: {dotenv_path_current}")
        load_dotenv(dotenv_path=dotenv_path_current)
    else:
        logger.warning(".env file not found in parent or current directory. Relying on system environment variables.")


# Get MCP Server URL from environment variable or use default
MCP_SERVER_URL = os.getenv("MCP_SERVER_URL", "http://127.0.0.1:8000")
STREAMING_ENDPOINT = f"{MCP_SERVER_URL}/chat_stream"
STANDARD_ENDPOINT = f"{MCP_SERVER_URL}/chat" # If needed for non-streaming fallback
HEALTH_ENDPOINT = f"{MCP_SERVER_URL}/health"

logger.info(f"MCP Server URL configured: {MCP_SERVER_URL}")

# --- Health Check ---
def check_server_health():
    """Checks if the backend MCP server is reachable and configured."""
    try:
        response = requests.get(HEALTH_ENDPOINT, timeout=5) # 5 second timeout
        response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
        health_data = response.json()
        logger.info(f"Health check successful: {health_data}")
        if not health_data.get("llm_client_initialized"):
             return f"⚠️ Server Status: OK | LLM Provider: {health_data.get('llm_provider', 'N/A')} (Client NOT Initialized - Check API Key)"
        return f"✅ Server Status: OK | LLM Provider: {health_data.get('llm_provider', 'N/A')}"
    except requests.exceptions.ConnectionError:
        logger.error(f"Health check failed: Connection error to {HEALTH_ENDPOINT}")
        return f"❌ Server Status: Connection Error - Cannot reach MCP at {MCP_SERVER_URL}"
    except requests.exceptions.Timeout:
        logger.error(f"Health check failed: Request timed out to {HEALTH_ENDPOINT}")
        return "❌ Server Status: Timeout - MCP server is not responding quickly."
    except requests.exceptions.RequestException as e:
        logger.error(f"Health check failed: {e}", exc_info=True)
        status_code = e.response.status_code if e.response is not None else "N/A"
        return f"❌ Server Status: Error (Code: {status_code}) - Check MCP server logs."
    except Exception as e:
        logger.error(f"Health check failed with unexpected error: {e}", exc_info=True)
        return f"❌ Server Status: Unexpected Error - {e}"

# --- Gradio Chat Logic ---

# Function to handle user messages and stream responses
# Gradio's Chatbot component expects a function that takes (message, history)
# and yields updates to the history.
async def predict_streaming(message: str, history: list[list[str | None]]):
    """
    Sends the user message and history to the FastAPI streaming endpoint
    and yields the response chunks back to the Gradio Chatbot.
    """
    logger.info(f"User message: {message}")
    logger.debug(f"Current history: {history}")

    # Convert Gradio history (list of pairs) to the format expected by the MCP API (list of dicts)
    api_history = []
    for user_msg, assistant_msg in history:
        if user_msg:
            api_history.append({"role": "user", "content": user_msg})
        if assistant_msg:
            api_history.append({"role": "assistant", "content": assistant_msg})

    # Add the latest user message
    api_history.append({"role": "user", "content": message})

    logger.debug(f"Sending history to API: {api_history}")

    # Prepare the request payload
    payload = {"history": api_history}

    # Initialize assistant's response in the history for streaming
    history.append([message, ""]) # Add user message and placeholder for assistant response

    full_response = ""
    try:
        # Use requests with stream=True for streaming endpoint
        # Note: requests itself doesn't handle async generators well for SSE.
        # A better approach for robust SSE handling might involve httpx or aiohttp,
        # but for basic text streaming, iterating over lines can work.
        # We will use `requests` here for simplicity, assuming text/plain streaming.
        with requests.post(STREAMING_ENDPOINT, json=payload, stream=True, timeout=120) as response: # Increased timeout for long streams
            response.raise_for_status() # Check for HTTP errors

            # Stream the response chunks
            for chunk_bytes in response.iter_content(chunk_size=None): # Read chunks as they arrive
                 if chunk_bytes:
                    chunk = chunk_bytes.decode('utf-8') # Decode bytes to string
                    # Check for potential error messages embedded in the stream
                    if chunk.startswith("[STREAM_ERROR:") or chunk.startswith("[Error:") or chunk.startswith("[Blocked by Safety Filter:"):
                         logger.error(f"Received error chunk from stream: {chunk}")
                         full_response += f"\n**{chunk}**" # Append error prominently
                         # Update the last message in history with the error
                         history[-1][1] = full_response
                         yield history # Update Gradio UI with error
                         break # Stop processing further chunks on error

                    # Append chunk to the assistant's message in the last history pair
                    full_response += chunk
                    history[-1][1] = full_response # Update the assistant's response
                    yield history # Yield the updated history to Gradio for real-time display

    except requests.exceptions.ConnectionError:
        error_msg = f"**[Connection Error]** Could not connect to the backend server at {STREAMING_ENDPOINT}. Please ensure it's running."
        logger.error(error_msg)
        history[-1][1] = error_msg # Update placeholder with error
        yield history
    except requests.exceptions.Timeout:
        error_msg = "**[Timeout Error]** The request to the backend server timed out. The server might be busy or unavailable."
        logger.error(error_msg)
        history[-1][1] = error_msg
        yield history
    except requests.exceptions.RequestException as e:
        status_code = e.response.status_code if e.response is not None else "N/A"
        error_detail = str(e)
        try: # Try to get more detail from response body if possible
            error_detail = e.response.json().get("detail", str(e))
        except: pass
        error_msg = f"**[API Error]** Failed to get response from server (Status: {status_code}). Detail: {error_detail}"
        logger.error(error_msg, exc_info=True)
        history[-1][1] = error_msg
        yield history
    except Exception as e:
        error_msg = f"**[Unexpected Error]** An error occurred in the Gradio UI: {e}"
        logger.error(error_msg, exc_info=True)
        history[-1][1] = error_msg
        yield history

    logger.info("Streaming finished.")


# --- Gradio Interface Definition ---
with gr.Blocks(theme=gr.themes.Soft(primary_hue="blue", secondary_hue="sky")) as demo:
    gr.Markdown(
        """
        # Conversational AI Chat Agent
        Chat with an AI assistant powered by a configurable backend (Gemini, OpenAI, or Anthropic).
        Type your message below and press Enter.
        """
    )

    # Health check display
    health_status = gr.Textbox(label="Backend Server Status", value="Checking health...", interactive=False, max_lines=2)

    # Chat interface
    chatbot = gr.Chatbot(
        label="Conversation",
        bubble_full_width=False,
        height=500, # Adjust height as needed
        # Example avatar images (replace with actual URLs or remove if not desired)
        # avatar_images=(("https://placehold.co/40x40/007bff/ffffff?text=U"), # User Avatar (Blue)
        #                ("https://placehold.co/40x40/28a745/ffffff?text=AI")) # AI Avatar (Green)
    )

    # Input textbox
    msg_input = gr.Textbox(
        label="Your Message",
        placeholder="Type your message here and press Enter...",
        lines=2 # Start with 2 lines, can expand
    )

    # Buttons
    with gr.Row():
        # Submit button (alternative to pressing Enter)
        submit_btn = gr.Button("➡️ Send", variant="primary")
        # Clear button
        clear_btn = gr.ClearButton([msg_input, chatbot], value="🗑️ Clear Chat")

    # --- Event Handling ---
    # Link Enter key press in textbox to the prediction function
    msg_input.submit(predict_streaming, [msg_input, chatbot], chatbot)

    # Link Submit button click to the prediction function
    submit_btn.click(predict_streaming, [msg_input, chatbot], chatbot)

    # Update health status when the interface loads
    demo.load(check_server_health, None, health_status, every=5) # Check health every 5s


# --- Launch the Gradio App ---
if __name__ == "__main__":
    logger.info("Launching Gradio UI...")
    # Set share=True to create a public link (useful for testing/demos)
    # Set server_name="0.0.0.0" to allow access from other devices on the network
    demo.launch(
        server_name="0.0.0.0", # Listen on all network interfaces
        server_port=7860,      # Default Gradio port
        # share=True,          # Set to True to generate a public link (requires internet)
        debug=True             # Enable Gradio debug mode for more detailed logs
    )
    logger.info("Gradio UI has stopped.")
