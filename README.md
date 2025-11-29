# Ollama-FastMCP Wrapper

A proxy service that bridges [Ollama](https://ollama.ai) with [FastMCP](https://gofastmcp.com) thus allowing models to be used locally in conjunction with MCP servers and their tools directly on the local machine.

---

## ✨ Features

- Connect/disconnect to multiple **MCP servers** at runtime (using FastMCP).
- Expose FastMCP tools as callable functions to Ollama models.
- Use Ollama locally with tool-augmented reasoning.
- Historical conversation with the LLM model persistable on disk.
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

You’ll be asked which mode to start in:

- **API mode** → starts a REST API on `http://127.0.0.1:8000`
- **CLI mode** → starts a terminal-based chat loop

---

### 🖥️ API Mode

Start the API:

```bash
python ollama_wrapper.py
# choose "api"
```

Endpoints:

- `GET /servers` → List connected servers with their tools and available API endpoints
- `POST /connect/{server_name}` → Connect a FastMCP server
- `POST /disconnect/{server_name}` → Disconnect a server
- `POST /chat` → Send a chat request
- `GET /history` → Get current conversation history
- `POST /load_history/{file_name}` → Load conversation history from disk
- `POST /save_history/{file_name}` → Persists the conversation history on disk
- `POST /overwrite_history/{file_name}` → Overwrite an existing conversation file with the ongoing conversation

#### Usage Scenarios

**Scenario 1: Using Tools with Explicit Connection**
```bash
# Step 1: Connect to MCP server
curl -X POST http://localhost:8000/connect/math

# Step 2: Chat with tools
curl http://localhost:8000/chat -H "Content-Type: application/json" -d '{
  "message": "Add 5 and 10, then multiply the result by 20.",
  "model": "llama3.2:3b",
  "mcp_server": "math"
}'
```

**Scenario 2: Auto-Connect (specifying server in chat request)**
```bash
# Server auto-connects when specified in mcp_server parameter
curl http://localhost:8000/chat -H "Content-Type: application/json" -d '{
  "message": "What is 15 multiplied by 3?",
  "model": "llama3.2:3b",
  "mcp_server": "math"
}'
# Server "math" is automatically connected if not already
```

**Scenario 3: Pure Chat (no tools)**
```bash
# Use empty mcp_server for pure Ollama chat without tools
curl http://localhost:8000/chat -H "Content-Type: application/json" -d '{
  "message": "Hello, how are you?",
  "model": "llama3.2:3b",
  "mcp_server": ""
}'
# No tools are sent to Ollama, faster response
```

**Scenario 4: Managing Server Connections**
```bash
# List available servers
curl http://localhost:8000/servers

# Disconnect a server
curl -X POST http://localhost:8000/disconnect/math

# Get conversation history
curl http://localhost:8000/history
```

---

### 💬 CLI Mode

```bash
python ollama_wrapper.py
# choose "cli"
```

Then type messages:

```
You: Hello!
Bot: Hi there
```

Exit with `/exit` or `/quit`.

---

## ⚙️ Configuration

### Configuration Files

Version 0.4.0 introduces a separated configuration structure:

1. **Wrapper Configuration** (`wrapper_config.toml` - root directory):
   ```toml
   [wrapper]
   transport = "HTTP"              # Transport method: "HTTP" or "STDIO" (default: HTTP)
   host = "0.0.0.0"               # Server host address (default: 0.0.0.0)
   port = 8000                     # Server port (default: 8000)
   history_file = ""               # Path to conversation history file (default: none)
   overwrite_history = false       # Overwrite history file on exit (default: false)
   ```

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
- If not specified on command line, values from `server_config.toml` will be used
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

