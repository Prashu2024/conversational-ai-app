# End-to-End Conversational AI Application

This project implements a complete conversational AI system featuring a user interface (Gradio), a central control plane server (FastAPI), and a configurable backend supporting multiple Large Language Models (LLMs) like Google Gemini, OpenAI GPT models, and Anthropic Claude.

## Features

* **Web-Based Chat UI:** Interactive chat interface built with Gradio for real-time conversations.
* **Master Control Plane (MCP):** A robust FastAPI server that:
    * Handles user requests.
    * Orchestrates the conversation flow.
    * Manages interaction with the selected LLM.
    * Supports streaming responses (Server-Sent Events).
* **Configurable LLM Backend:** Easily switch between different LLM providers (Gemini, OpenAI, Anthropic) via environment variables.
* **Context Management:** The agent maintains conversation history to provide coherent responses.
* **Streaming Support:** Real-time display of AI responses as they are generated.
* **Modular Design:** Code is organized into distinct components (UI, Server, LLM Clients, Agent).

## Prerequisites

* Python 3.8+
* `pip` (Python package installer)
* Access keys for the LLM APIs you intend to use (Google AI Studio, OpenAI Platform, Anthropic Console).

## Setup Instructions

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/Prashu2024/conversational-ai-app
    cd conversational-ai-app
    ```

2.  **Set Up Environment Variables:**
    * Copy the example environment file:
        ```bash
        cp .env.example .env
        ```
    * Edit the `.env` file with a text editor and fill in your details:
        * `LLM_PROVIDER`: Set to `"GEMINI"`, `"OPENAI"`, or `"ANTHROPIC"` to choose the active LLM.
        * `GEMINI_API_KEY`: Your API key for Google Gemini.
        * `OPENAI_API_KEY`: Your API key for OpenAI.
        * `ANTHROPIC_API_KEY`: Your API key for Anthropic Claude.
        * (Optional) `*_MODEL_NAME`: Specify particular model versions if needed (e.g., `OPENAI_MODEL_NAME="gpt-4"`). Defaults are provided in `mcp_server/config.py`.
        * `MCP_SERVER_URL`: Keep the default (`http://127.0.0.1:8000`) if running locally.

3.  **Install Dependencies:**
    * **Server Dependencies:**
        ```bash
        cd mcp_server
        pip install -r requirements.txt
        cd ..
        ```
    * **UI Dependencies:**
        ```bash
        cd gradio_ui
        pip install -r requirements.txt
        cd ..
        ```
    *Note: You might want to use a Python virtual environment (`venv`) to keep dependencies isolated.*
    ```bash
    python -m venv venv
    source venv/bin/activate # On Windows use `venv\Scripts\activate`
    # Now run the pip install commands
    ```

## Running the Application

You need to run two components separately: the FastAPI server and the Gradio UI.

1.  **Start the FastAPI MCP Server:**
    * Open a terminal in the `conversational-ai-app` directory.
    * Make sure your virtual environment is activated if you created one.
    * Run the server using Uvicorn:
        ```bash
        python -m mcp_server.main
        ```
    * The server should start, typically on `http://127.0.0.1:8000`. You'll see log messages indicating the selected LLM provider and initialization status. Check for any errors related to API keys or configuration.

2.  **Start the Gradio UI:**
    * Open a *second* terminal in the `conversational-ai-app` directory.
    * Make sure your virtual environment is activated if you created one.
    * Run the Gradio app:
        ```bash
        python -m gradio_ui.app
        ```
    * Gradio will start the UI, usually on `http://127.0.0.1:7860`. It will provide a local URL in the terminal. Open this URL in your web browser.
    * The Gradio interface should load, and the "Backend Server Status" box should indicate whether it successfully connected to the FastAPI server and the LLM provider configured.

3.  **Chat!**
    * Type messages into the input box in the Gradio UI and press Enter or click "Send".
    * The UI will send the message to the FastAPI server, which will process it using the configured LLM and stream the response back.

## Switching LLM Providers (currently tested on Google Gemini API only)

1.  **Stop** both the FastAPI server (Ctrl+C in its terminal) and the Gradio UI (Ctrl+C in its terminal).
2.  **Edit** the `.env` file and change the `LLM_PROVIDER` variable to the desired provider (`"GEMINI"`, `"OPENAI"`, or `"ANTHROPIC"`).
3.  Ensure the corresponding API key (`GEMINI_API_KEY`, `OPENAI_API_KEY`, or `ANTHROPIC_API_KEY`) is correctly set in the `.env` file for the chosen provider.
4.  **Restart** the FastAPI server first, then restart the Gradio UI as described in the "Running the Application" section. The server logs and the Gradio health check should reflect the newly selected provider.

## Error Handling & Logging

* Both the FastAPI server and the Gradio UI include logging. Check the terminal output for errors or informational messages.
* FastAPI logs provide details about incoming requests, LLM interactions, and potential issues during processing.
* Gradio logs show UI events and errors encountered when communicating with the backend.
* Common errors include:
    * Incorrect or missing API keys in `.env`.
    * Network issues preventing Gradio from reaching the FastAPI server.
    * Rate limits or other errors from the LLM provider APIs.
    * Invalid conversation history formats (e.g., incorrect roles).

## Code Quality & Practices

* **Modularity:** Code is separated into logical components.
* **Typing:** Python type hints are used for better code clarity and maintainability.
* **Asynchronous Code:** `async`/`await` is used in the FastAPI server and LLM clients for efficient I/O operations.
* **Configuration:** Settings are externalized to environment variables.
* **Error Handling:** Basic `try...except` blocks are included, and errors are logged. More specific error handling could be added.
* **Comments:** Code includes comments explaining key parts.
