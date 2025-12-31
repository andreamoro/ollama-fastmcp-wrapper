# CLI Usage Guide

Comprehensive guide for using the Ollama-FastMCP Wrapper in CLI (Command Line Interface) mode.

---

## Table of Contents

- [Getting Started](#getting-started)
- [CLI Commands](#cli-commands)
  - [Chat Commands](#chat-commands)
  - [Model Management](#model-management)
  - [History Management](#history-management)
- [Configuration](#configuration)
- [Usage Patterns](#usage-patterns)
  - [Pattern 1: Simple Conversation](#pattern-1-simple-conversation)
  - [Pattern 2: Model Switching](#pattern-2-model-switching)
  - [Pattern 3: Persistent Conversations](#pattern-3-persistent-conversations)
  - [Pattern 4: Temperature Adjustment](#pattern-4-temperature-adjustment)
- [Model Selection](#model-selection)
- [Troubleshooting](#troubleshooting)
- [Related Documentation](#related-documentation)

---

## Getting Started

Start the CLI mode:

```bash
python ollama_wrapper.py cli
```

Or simply select CLI when prompted:

```bash
python ollama_wrapper.py
# Choose: cli
```

**[⬆ Back to top](#table-of-contents)**

---

## CLI Commands

### Chat Commands

**Basic chat:**
```
You: Hello! How are you?
Bot: I'm doing well, thank you for asking! How can I help you today?
```

**Exit:**
```
You: /exit
```
or
```
You: /quit
```

**Clear conversation context:**
```
You: /clear
🧹 Conversation context cleared
```

**Show help:**
```
You: /help
📖 Available CLI commands:
  /help                         - Show this help message
  /exit or /quit                - Exit the CLI
  /clear                        - Clear conversation context
  /model                        - Change the current model interactively
  /load <file_name>             - Load conversation history from a file
  /save <file_name>             - Save conversation history to a file
  /overwrite <file_name>        - Overwrite existing conversation file
```

### Model Management

**Change model interactively:**
```
You: /model

🔄 Current model: llama3.2:3b

📋 Available Ollama models:
   1. llama3.2:3b
   2. gemma2:2b
   3. gemma2:9b

Select model (1-3, model name, or 'c' to cancel): 2
✅ Model changed: llama3.2:3b → gemma2:2b
🔄 Conversation context reset
   Family: gemma2
   Parameters: 2.6B
   Quantization: Q4_0
```

**Model selection options:**
- Type a number (e.g., `2`)
- Type exact model name (e.g., `gemma2:2b`)
- Type partial match (e.g., `gemma` matches all gemma models)
- Type `c` to cancel

### History Management

**Save conversation:**
```
You: /save my_conversation.json
💾 Conversation history saved to my_conversation.json
```

**Load conversation:**
```
You: /load my_conversation.json
📂 Conversation history loaded from my_conversation.json (5 messages)
```

**Overwrite existing file:**
```
You: /overwrite my_conversation.json
💾 Conversation history saved to my_conversation.json (overwritten)
```

**[⬆ Back to top](#table-of-contents)**

---

## Configuration

CLI mode uses settings from `wrapper_config.toml`:

```toml
[wrapper]
host = "0.0.0.0"
port = 8000
history_file = ""               # Auto-load history file on startup
overwrite_history = false       # Allow overwriting history on exit

[model]
default = "llama3.2:3b"        # Default model for CLI
temperature = 0.2               # Default temperature (0.0-2.0)
max_history_messages = 20       # Max messages before summarization
```

**Temperature settings:**
- **Low (0.0-0.3)**: Consistent, deterministic responses (recommended for factual tasks)
- **Medium (0.7-1.0)**: Balanced creativity
- **High (1.5-2.0)**: Very creative, less predictable

**[⬆ Back to top](#table-of-contents)**

---

## Usage Patterns

### Pattern 1: Simple Conversation

Basic conversational interaction with context preservation.

```bash
# Start CLI
python ollama_wrapper.py cli

# Chat naturally - context is preserved
You: What is Python?
Bot: Python is a high-level, interpreted programming language...

You: What are its main features?
Bot: Python's main features include...

You: Can you show me an example?
Bot: Here's a simple Python example...

# Exit
You: /exit
```

**Key points:**
- All messages are kept in conversation history
- Model remembers previous context
- Temperature set in config file

---

### Pattern 2: Model Switching

Switch between models during a conversation.

```bash
# Start with default model
python ollama_wrapper.py cli

You: Explain quantum computing
Bot: [Response from llama3.2:3b]

# Switch to a different model
You: /model
Select model: gemma2:9b
✅ Model changed: llama3.2:3b → gemma2:9b
🔄 Conversation context reset

# Start fresh conversation with new model
You: Explain quantum computing
Bot: [Response from gemma2:9b - fresh context]
```

**Important:**
- Switching models **resets conversation context**
- Prevents mixing responses from different models
- Previous conversation is lost unless saved

---

### Pattern 3: Persistent Conversations

Save and resume conversations across sessions.

**Session 1: Save conversation**
```bash
python ollama_wrapper.py cli

You: I'm working on a Python project about data analysis
Bot: That sounds interesting! What kind of data are you analyzing?

You: Financial time series data
Bot: Great! For financial time series analysis, you'll want to use...

# Save before exiting
You: /save finance_project.json
💾 Conversation history saved to finance_project.json

You: /exit
```

**Session 2: Resume conversation**
```bash
python ollama_wrapper.py cli

You: /load finance_project.json
📂 Conversation history loaded from finance_project.json (4 messages)

# Continue where you left off
You: What libraries should I use for this?
Bot: [Continues with context from previous session]
```

**Auto-load on startup:**
```toml
# wrapper_config.toml
[wrapper]
history_file = "finance_project.json"
overwrite_history = true  # Auto-save on exit
```

Then simply:
```bash
python ollama_wrapper.py cli
# Automatically loads finance_project.json
# Automatically saves on exit
```

---

### Pattern 4: Temperature Adjustment

Adjust creativity/randomness for different tasks.

**Configuration approach** (wrapper_config.toml):
```toml
[model]
default = "llama3.2:3b"
temperature = 0.2  # For factual, consistent responses
```

```bash
python ollama_wrapper.py cli

You: What is the capital of France?
Bot: The capital of France is Paris.
# Consistent, factual answer
```

**For creative tasks:**
```toml
[model]
default = "llama3.2:3b"
temperature = 1.5  # For creative responses
```

```bash
python ollama_wrapper.py cli

You: Write a poem about programming
Bot: [Creative, varied poetry]
# More creative and diverse responses
```

**Note:** Unlike API mode, CLI mode doesn't support per-request temperature override. You must change the config file and restart.

**[⬆ Back to top](#table-of-contents)**

---

## Model Selection

When the configured model doesn't exist or when using `/model`:

### Numbered Selection
```
Select model (1-3, model name, or 'c' to cancel): 2
```

### Exact Name
```
Select model (1-3, model name, or 'c' to cancel): gemma2:9b
```

### Partial Match
```
Select model (1-3, model name, or 'c' to cancel): gemma
❌ Multiple models match 'gemma':
   1. gemma2:2b
   2. gemma2:9b
   3. gemma3:1b

Select model (1-3, model name, or 'c' to cancel): 2
✅ Model changed: llama3.2:3b → gemma2:9b
```

### Fuzzy Matching at Startup

When config has invalid model `gemma2` (without size), CLI shows similar models:

```
❌ Error: Model 'gemma2' not found

💡 Found 2 similar model(s):
   1. gemma2:2b
   2. gemma2:9b

Select model (1-2, model name, or 'c' to cancel):
```

**[⬆ Back to top](#table-of-contents)**

---

## Troubleshooting

### Model Not Found

**Problem:**
```
❌ Error: Model 'llama3.2:3b' not found
```

**Solution:**
Download the model first:
```bash
ollama pull llama3.2:3b
```

Or select a different model from the list presented.

---

### Cannot Connect to Ollama

**Problem:**
```
❌ Cannot connect to Ollama server
```

**Solution:**
Start Ollama server:
```bash
ollama serve
```

---

### History File Not Found

**Problem:**
```
❌ History file not found: my_conversation.json
```

**Solution:**
Check the file path. Files are relative to current directory unless absolute path is provided:
```
You: /load /full/path/to/my_conversation.json
```

---

### Context Too Long

**Problem:** Model responses slow down after long conversation.

**Solution:**
Clear context or adjust `max_history_messages` in config:
```
You: /clear
```

Or in `wrapper_config.toml`:
```toml
[model]
max_history_messages = 10  # Reduce for shorter context
```

**[⬆ Back to top](#table-of-contents)**

---

## Related Documentation

- **[README.md](README.md)** - Project overview and installation
- **[API_USAGE.md](API_USAGE.md)** - API mode usage (supports stateless mode, MCP tools)
- **[CHANGELOG.md](CHANGELOG.md)** - Version history and changes
- **[demos/README.md](demos/README.md)** - Example scripts (API-focused)

**Key Differences: CLI vs API Mode**

| Feature | CLI Mode | API Mode |
|---------|----------|----------|
| **MCP Tools** | ❌ Not supported | ✅ Full support |
| **Stateless Mode** | ❌ Not available | ✅ Available |
| **Temperature Override** | ❌ Config only | ✅ Per-request |
| **Model Override** | ❌ Must switch | ✅ Per-request (stateless) |
| **Use Case** | Interactive chat | Programmatic access, testing |

**[⬆ Back to top](#table-of-contents)**

---

**Note:** CLI mode is designed for simple, interactive conversations. For advanced features like MCP tools, stateless requests, or multi-model testing, use [API mode](API_USAGE.md).
