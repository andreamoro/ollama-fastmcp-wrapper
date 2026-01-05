# Ollama-FastMCP Wrapper

A proxy service that bridges [Ollama](https://ollama.ai) with [FastMCP](https://gofastmcp.com) thus allowing models to be used locally in conjunction with MCP servers and their tools directly on the local machine.

---

## ✨ Features

- Connect/disconnect to multiple **MCP servers** at runtime (using FastMCP).
- Expose FastMCP tools as callable functions to Ollama models.
- Use Ollama locally with tool-augmented reasoning.
- Historical conversation with the LLM model persistable on disk with **async I/O**.
- Automatically summarise historical conversation with the model.
- Run as:
  - **API Server** (via FastAPI + Uvicorn)
  - **Interactive CLI**

---

## :question: What is MCP?

The Model Context Protocol (MCP) is a protocol that lets you build servers to expose data and functionality to LLM applications in a secure, standardized way. 
Limited to this wrapper, MCP use is limited to the Tools part.

## ⚡ Installation

1. Clone this repository and install dependencies:

   ```bash
   git clone https://github.com/andreamoro/ollama-fastmcp-wrapper.git
   ```

    Install dependencies according to the package manager you are using. If it's uv:

    ```bash
    cd ollama-fastmcp-wrapper
    uv sync
    ```
    Otherwise go with the traditional (obsolete) pip:
    
    ```bash
    cd ollama-fastmcp-wrapper
    pip install -r requirements.txt
    ```

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Python requirements include:<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; - aiofiles (async file I/O)<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; - fastapi<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; - fastmcp<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; - uvicorn<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; - ollama<br>

2. Make sure you have:
- [Ollama Client](https://ollama.com/download) up and running: 
  ```bash
  ollama serve
  ```
- An MCP server to use the Tools capability
  - Without an MCP server, you can use this wrapper as a conversation interface
  - Two example MCP servers are included in the `mcp_servers/` directory:
    - **math_server.py** - Basic arithmetic operations (add, subtract, multiply, divide)
    - **ipinfo_server.py** - IP geolocation lookup with 20 preset organizations

## &#9973; Usage

Run the wrapper:

```bash
python ollama_wrapper.py
```

You'll be asked which mode to start in:

- **API mode** → starts a REST API on `http://127.0.0.1:8000`
- **CLI mode** → starts a terminal-based chat loop

### 📚 Demo Scripts

The `demos/` directory contains comprehensive usage examples in both shell and Python formats:

- **basic_chat** - Simple chat without tools
- **math_operations** - Using the math MCP server
- **ipinfo_lookup** - IP geolocation queries
- **server_management** - Connect/disconnect/list servers
- **history_management** - Conversation persistence
- **temperature_test_multi_model.py** - Advanced temperature testing across multiple models

See [`demos/README.md`](demos/README.md) for detailed usage instructions and examples.

---

### 🖥️ API Mode

Start the API:

```bash
python ollama_wrapper.py
# choose "api"
```

**📖 For comprehensive API documentation, usage patterns, and examples, see [API_USAGE.md](API_USAGE.md)**

#### API Endpoints

**Root:**
- `GET /` → API documentation and endpoint listing

**Chat:**
- `POST /chat` → Send a chat request (with optional MCP tools)

**Model Management:**
- `GET /model` → Get current session model
- `GET /model/list` → List all available Ollama models
- `POST /model/switch/{model_name}` → Switch session model and reset context
- `GET /ollama/status` → Quick health check for Ollama (5s timeout)

**History:**
- `GET /history` → Get current conversation history
- `GET /history/clear` → Clear the current conversation history
- `GET /history/load/{file_name}` → Load conversation history from disk
- `GET /history/overwrite/{file_name}` → Overwrite an existing conversation file
- `GET /history/save/{file_name}` → Save conversation history to disk

**Servers:**
- `GET /servers` → List available FastMCP servers from config
- `POST /servers/{server_name}/connect` → Connect to an MCP server
- `POST /servers/{server_name}/disconnect` → Disconnect from an MCP server
- `GET /servers/{server_name}/tools` → List available tools for a specific MCP server

#### Quick Start Example

```bash
# Start API server
python ollama_wrapper.py api

# Simple chat request (uses session model from config)
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello! Tell me about Python"}'

# Connect to MCP server and use tools
curl -X POST http://localhost:8000/servers/math/connect
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "What is 25 * 4?",
    "mcp_server": "math"
  }'
```

For detailed usage patterns including model switching, multi-model testing, temperature testing, and more, see [API_USAGE.md](API_USAGE.md).

---

### 💬 CLI Mode

```bash
python ollama_wrapper.py
# choose "cli"
```

**📖 For comprehensive CLI documentation, commands, and usage patterns, see [CLI_USAGE.md](CLI_USAGE.md)**

#### Quick Start

```bash
# Start CLI
python ollama_wrapper.py cli

# Chat naturally
You: Hello! How are you?
Bot: I'm doing well, thank you for asking!

# Use commands
You: /help       # Show available commands
You: /model      # Change model interactively
You: /clear      # Clear conversation context
You: /exit       # Exit CLI
```

For detailed usage patterns including model switching, persistent conversations, and temperature adjustment, see [CLI_USAGE.md](CLI_USAGE.md).

---

## ⚙️ Configuration

### Configuration Files

Version 0.4.0 introduces a separated configuration structure:

1. **Wrapper Configuration** (`wrapper_config.toml` - root directory):
   ```toml
   [wrapper]
   transport = "HTTP"              # Transport method: "HTTP" or "STDIO" (default: HTTP)
   host = "0.0.0.0"                # Server host address (default: 0.0.0.0)
   port = 8000                     # Server port (default: 8000)
   history_file = ""               # Path to conversation history file (default: none)
   overwrite_history = false       # Overwrite history file on exit (default: false)
   max_history_messages = 20       # Maximum messages before summarization kicks in (default: 20)

   [ollama]
   host = "localhost"              # Ollama instance host (default: localhost)
   port = 11434                    # Ollama instance port (default: 11434)
   timeout = 300                   # Request timeout in seconds (default: 300)
   label = ""                      # Optional label to identify this Ollama instance
   model = { default = "llama3.2:3b", temperature = 0.2 }  # Model settings
   ```

   **Wrapper Settings:**
   - `max_history_messages`: Maximum number of messages before automatic summarization (default: 20)
     - When the message count exceeds this limit, older messages are summarized to save context
     - Helps maintain conversation continuity while keeping token usage manageable
     - Adjust based on your model's context window and use case

   **Ollama Settings:**
   - `host`: Ollama instance host (use with port for remote/tunneled instances)
   - `port`: Ollama instance port (default: 11434)
   - `timeout`: Request timeout in seconds (default: 300). Prevents the wrapper from hanging indefinitely when SSH tunnels drop or remote Ollama becomes unresponsive.
   - `label`: Human-readable label to identify this Ollama instance (prompted at startup if not set)

   **Model Settings:**
   - `default`: Default model name if not specified in requests
   - `temperature`: Controls response randomness (0.0-2.0)
     - Low (0.0-0.3): Consistent, deterministic responses (recommended for factual tasks)
     - Medium (0.7-1.0): Balanced creativity
     - High (1.5-2.0): Very creative, less predictable
     - Temperature can be overridden per request via API

2. **MCP Servers Configuration** (`mcp_servers/mcp_servers_config.toml`):
   ```toml
   [[servers]]
   name = "math"
   command = "uv"
   args = ["run", "--with", "fastmcp", "mcp_servers/math_server.py"]
   host = "http://localhost:5000/mcp"
   port = 5000
   enabled = true

   [[servers]]
   name = "ipinfo"
   command = "uv"
   args = ["run", "--with", "fastmcp", "mcp_servers/ipinfo_server.py"]
   host = "http://localhost:5001/mcp"
   port = 5001
   enabled = true
   token_file = "mcp_tokens.toml"  # Optional: specify token file (default: mcp_tokens.toml)
   ```

3. **API Tokens** (`mcp_servers/mcp_tokens.toml` - gitignored):
   ```toml
   # Copy from mcp_tokens.toml.example and add your tokens
   [ipinfo]
   token = "your_ipinfo_token_here"
   ```

### Configuration Priority

Command-line arguments take precedence over config file settings:
- If you specify `--host` or `--port` on the command line, those values will be used
- If not specified on command line, values from `wrapper_config.toml` will be used
- If not in config file, default values will be used

### Transport Methods

- **STDIO transport** → spawn server locally
- **HTTP transport** → connect to remote MCP server

### Command-Line Arguments

**Positional Arguments:**
- `mode` - Operation mode: `api` or `cli` (default: `api`)
- `model` - Ollama model to use (e.g., `llama3.2:3b`, `gemma3:1b`)

**Optional Arguments:**
- `-c, --wrapper-config <file>` - Path to wrapper configuration file (default: `wrapper_config.toml`)
- `--mcp-config <filename>` - MCP servers config filename in mcp_servers/ directory (default: `mcp_servers_config.toml`)
- `--history-file <file>` - Path to conversation history file to load/save
- `-o, --overwrite-history` - Allow overwriting existing history file
- `-t, --transport <method>` - Transport method: `HTTP` or `STDIO` (default: from config or `HTTP`)
- `--host <address>` - API server host address (default: from config or `0.0.0.0`)
- `--port <number>` - API server port number (default: from config or `8000`)

**Ollama Connection Arguments:**
- `--ollama-host <address>` - Ollama instance host (default: from config or `localhost`)
- `--ollama-port <number>` - Ollama instance port (default: from config or `11434`)
- `--ollama-timeout <seconds>` - Request timeout in seconds (default: from config or `300`)
- `--ollama-label <label>` - Label to identify the Ollama instance (prompted if not provided)

#### Examples

```bash
# Use custom wrapper config file
python ollama_wrapper.py api -c my_wrapper_config.toml

# Override config file settings
python ollama_wrapper.py api --host 127.0.0.1 --port 9000

# Specify transport method
python ollama_wrapper.py api -t STDIO

# Load conversation history
python ollama_wrapper.py cli --history-file my_conversation.json

# Start with specific model and auto-save history
python ollama_wrapper.py api llama3.2:3b --history-file conversation.json -o

# Use alternate MCP servers config
python ollama_wrapper.py api --mcp-config alternate_servers.toml

# Connect to remote Ollama via SSH tunnel (with custom timeout)
python ollama_wrapper.py api --ollama-host localhost --ollama-port 11435 \
  --ollama-timeout 600 --ollama-label "remote-vps-via-tunnel"
```

---

## 📊 Architecture Diagram

```mermaid
flowchart TD
    O[Ollama] --> W[Ollama-FastMCP Wrapper]
    W -->|API mode| A[API]
    W -->|CLI mode| C[CLI]
    A --> M{ MCP Server Available?} 
    C --> Z
    M -->|Yes| F[FastMCP Tools]
    F --> Z[Chat]
    M -->|No| Z
```

---

## :bulb: Release History / Roadmap

Please check the [Changelog](CHANGELOG.md) file for more information.

---

## License

MIT License © 2025 Andrea MORO

