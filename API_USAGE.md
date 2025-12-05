# API Usage Guide

Comprehensive guide for interacting with the Ollama-FastMCP Wrapper API.

---

## Table of Contents

- [Model Management](#model-management)
- [Conversation Modes](#conversation-modes)
  - [Stateful Mode](#stateful-mode-default)
  - [Stateless Mode](#stateless-mode)
- [Usage Patterns](#usage-patterns)
  - [Pattern 1: Simple Conversation](#pattern-1-simple-conversation)
  - [Pattern 2: Model Switching During Conversation](#pattern-2-model-switching-during-conversation)
  - [Pattern 3: Multi-Model Testing](#pattern-3-multi-model-testing)
  - [Pattern 4: Private/Secret Queries](#pattern-4-privatesecret-queries)
  - [Pattern 5: Temperature Testing](#pattern-5-temperature-testing)
  - [Pattern 6: MCP Tool Usage](#pattern-6-mcp-tool-usage)
- [API Endpoints Reference](#api-endpoints-reference)
  - [Chat Endpoints](#chat)
  - [Model Management Endpoints](#model-management-1)
  - [History Management Endpoints](#history-management)
  - [Server Management Endpoints](#server-management)
- [Best Practices](#best-practices)
- [Migration Notes](#migration-notes)
- [Troubleshooting](#troubleshooting)
- [Related Documentation](#related-documentation)

---

## Model Management

The wrapper supports flexible model management with clear separation between session-level and request-level model selection.

### Session Model

The **session model** is the default model used for conversational (stateful) requests:
- Initialized from `wrapper_config.toml` (`model.default`)
- Persists across stateful chat requests
- Can be changed using the `/model/switch` endpoint

### Model Selection Rules

| Mode | Model Parameter Behavior |
|------|-------------------------|
| **Stateful** (`stateless=false`) | Must match session model, or use `/model/switch` |
| **Stateless** (`stateless=true`) | Can be any valid model, no restrictions |

**[⬆ Back to top](#table-of-contents)**

---

## Conversation Modes

### Stateful Mode (Default)

**Conversational mode** where context accumulates across requests.

- Messages are added to conversation history
- Full context sent to model with each request
- Model must match session model (enforced)
- Suitable for: Interactive conversations, context-dependent queries

**Example:**
```bash
# First message (uses session model from config)
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Hello! What is the capital of France?",
    "stateless": false
  }'

# Follow-up (remembers previous context)
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "What is its population?",
    "stateless": false
  }'
```

### Stateless Mode

**One-shot mode** where each request is independent.

- No messages added to history
- Only system prompt + current message sent
- Any model can be used
- No context contamination
- Suitable for: Testing, experimentation, "secret" queries

**Example:**
```bash
# Try different models without affecting session
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "What is 2+2?",
    "model": "gemma2:2b",
    "stateless": true
  }'

curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "What is 2+2?",
    "model": "llama3.2:3b",
    "stateless": true
  }'
# Session model unchanged, no context pollution
```

**[⬆ Back to top](#table-of-contents)**

---

## Usage Patterns

### Pattern 1: Simple Conversation

Single model, conversational context.

```bash
# Start conversation (uses session model)
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Tell me about Python"}'

# Continue conversation
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What are its main features?"}'

# Clear context if needed
curl -X GET http://localhost:8000/history/clear
```

### Pattern 2: Model Switching During Conversation

Change models mid-conversation with context reset.

```bash
# Start with default model
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello"}'

# Check current model
curl -X GET http://localhost:8000/model

# Switch to different model (resets context)
curl -X POST http://localhost:8000/model/switch/llama3.2:3b

# Continue with new model (fresh context)
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello again"}'
```

### Pattern 3: Multi-Model Testing

Test multiple models without affecting session.

```bash
# Test model A (stateless)
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Explain quantum computing",
    "model": "gemma2:2b",
    "stateless": true
  }'

# Test model B (stateless)
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Explain quantum computing",
    "model": "llama3.2:3b",
    "stateless": true
  }'

# Session model and context unchanged
# Continue normal conversation
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Back to our conversation"}'
```

### Pattern 4: Private/Secret Queries

Send messages without persisting to history.

```bash
# Normal conversation
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What is machine learning?"}'

# Private query (not saved to history)
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Sensitive question here",
    "stateless": true
  }'

# Continue normal conversation (secret query not in context)
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Can you elaborate on ML?"}'
```

### Pattern 5: Temperature Testing

Test different temperature settings across models.

```bash
# Get available models
curl -X GET http://localhost:8000/model/list

# Test with different temperatures (stateless)
for temp in 0.0 0.5 1.0 1.5 2.0; do
  curl -X POST http://localhost:8000/chat \
    -H "Content-Type: application/json" \
    -d "{
      \"message\": \"Write a creative story\",
      \"model\": \"gemma2:2b\",
      \"temperature\": $temp,
      \"stateless\": true
    }"
done
```

### Pattern 6: MCP Tool Usage

Using FastMCP tools with model selection.

```bash
# Connect to MCP server
curl -X POST http://localhost:8000/connect/math_server

# Use tools in conversation
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "What is 123 + 456?",
    "mcp_server": "math_server"
  }'

# Use tools with different model (stateless)
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "What is 789 * 12?",
    "model": "llama3.2:3b",
    "mcp_server": "math_server",
    "stateless": true
  }'
```

**[⬆ Back to top](#table-of-contents)**

---

## API Endpoints Reference

### Chat

**POST /chat**

Main chat endpoint.

**Request Body:**
```json
{
  "message": "Your message here",
  "model": "optional-model-name",
  "mcp_server": "optional-server-name",
  "temperature": 0.7,
  "stateless": false
}
```

**Parameters:**
- `message` (required): The user's message
- `model` (optional): Model name. For stateful requests, must match session model or use `/model/switch`. For stateless requests, can be any valid model.
- `mcp_server` (optional): MCP server to use for tools
- `temperature` (optional): Override temperature (0.0-2.0)
- `stateless` (optional): If true, one-shot mode without context

**Response:**
```json
{
  "response": "AI response here",
  "tools_used": ["tool1", "tool2"],
  "metrics": {
    "total_duration": 1234567890,
    "load_duration": 123456,
    "prompt_eval_count": 10,
    "eval_count": 20
  }
}
```

**Behavior:**
- **Stateful** (`stateless=false`): Adds to history, uses session model
- **Stateless** (`stateless=true`): Independent query, any model

**Error Cases:**
```json
{
  "detail": "Model mismatch: session model is 'gemma2:3b', but request specified 'llama3.2:3b'. Use POST /model/switch/llama3.2:3b to change session model, or set stateless=true for one-off queries."
}
```

### Model Management

**GET /model**

Get current session model.

**Response:**
```json
{
  "model": "gemma2:3b",
  "source": "config"
}
```

---

**POST /model/switch/{model_name}**

Change session model and reset conversation context.

**Parameters:**
- `model_name` (path): Name of the model to switch to

**Response:**
```json
{
  "status": "success",
  "old_model": "gemma2:3b",
  "new_model": "llama3.2:3b",
  "message": "Model switched and conversation context reset"
}
```

**Behavior:**
- Changes `self.model` to specified model
- Resets conversation history (equivalent to CLI `/model`)
- Validates model exists in Ollama

**Error Cases:**
```json
{
  "detail": "Model 'invalid:model' not found in Ollama. Use GET /model/list to see available models."
}
```

---

**GET /model/list**

List all available models from Ollama.

**Response:**
```json
{
  "models": [
    {
      "name": "gemma2:2b",
      "size": 1234567890,
      "modified": "2025-01-15T10:30:00Z"
    },
    {
      "name": "llama3.2:3b",
      "size": 2345678901,
      "modified": "2025-01-14T09:20:00Z"
    }
  ],
  "count": 2
}
```

### History Management

**GET /history/clear**

Clear conversation history without changing model.

**Response:**
```json
{
  "status": "success",
  "message": "Conversation history cleared"
}
```

---

**GET /history/load/{file_name}**

Load conversation history from file.

**Response:**
```json
{
  "status": "success",
  "message": "History loaded from conversation.json",
  "messages_count": 10
}
```

---

**GET /history/save/{file_name}**

Save conversation history to file (fails if exists).

**Response:**
```json
{
  "status": "success",
  "message": "History saved to conversation.json",
  "messages_count": 10
}
```

---

**GET /history/overwrite/{file_name}**

Save conversation history, overwriting existing file.

**Response:**
```json
{
  "status": "success",
  "message": "History saved to conversation.json",
  "messages_count": 10
}
```

### Server Management

**POST /connect/{server_name}**

Connect to an MCP server.

**Response:**
```json
{
  "status": "connected",
  "server": "math_server",
  "tools_count": 4
}
```

---

**POST /disconnect/{server_name}**

Disconnect from an MCP server.

**Response:**
```json
{
  "status": "disconnected",
  "server": "math_server"
}
```

---

**GET /servers**

List connected and available MCP servers.

**Response:**
```json
{
  "connected": {
    "math_server": ["add", "subtract", "multiply", "divide"]
  },
  "available": ["math_server", "ipinfo_server"]
}
```

**[⬆ Back to top](#table-of-contents)**

---

## Best Practices

### For Interactive Conversations
- Use stateful mode (default)
- Stick to one model per conversation
- Use `/model/switch` to change models explicitly
- Clear history with `/history/clear` when starting new topics

### For Testing & Experimentation
- Always use `stateless=true`
- Specify model explicitly in each request
- No need to worry about context contamination
- Perfect for comparing model responses

### For Multi-Model Pipelines
- Use stateless mode for each step
- Pass results between models via message content
- No shared context between models
- Clean separation of concerns

### Error Handling
- Check for model mismatch errors in stateful mode
- Use `/model/list` to validate model names
- Handle MCP server connection errors
- Validate temperature range (0.0-2.0)

**[⬆ Back to top](#table-of-contents)**

---

## Migration Notes

### From Version 0.7.x

**Breaking Changes:**
- Stateful requests now enforce model matching
- Model parameter must match session model or error is returned

**Migration Path:**
- **If using stateless=true**: No changes needed
- **If mixing models in stateful mode**:
  - Use `/model/switch` before changing models, OR
  - Add `stateless=true` to requests with different models

**Example Migration:**

**Before (0.7.x):**
```bash
# This would silently use different models with mixed context
curl -X POST /chat -d '{"message": "Hello", "model": "gemma2:3b"}'
curl -X POST /chat -d '{"message": "Hi", "model": "llama3.2:3b"}'
```

**After (0.8.0):**
```bash
# Option 1: Switch model explicitly
curl -X POST /model/switch/gemma2:3b
curl -X POST /chat -d '{"message": "Hello"}'
curl -X POST /model/switch/llama3.2:3b
curl -X POST /chat -d '{"message": "Hi"}'

# Option 2: Use stateless for multi-model queries
curl -X POST /chat -d '{"message": "Hello", "model": "gemma2:3b", "stateless": true}'
curl -X POST /chat -d '{"message": "Hi", "model": "llama3.2:3b", "stateless": true}'
```

**[⬆ Back to top](#table-of-contents)**

---

## Troubleshooting

### "Model mismatch" Error

**Problem:** Getting error about model mismatch in stateful requests.

**Solution:**
```bash
# Check current session model
curl -X GET http://localhost:8000/model

# Option A: Switch session model
curl -X POST http://localhost:8000/model/switch/desired-model

# Option B: Use stateless mode
curl -X POST /chat -d '{"message": "...", "model": "desired-model", "stateless": true}'
```

### Model Not Found

**Problem:** "Model not found in Ollama" error.

**Solution:**
```bash
# List available models
curl -X GET http://localhost:8000/model/list

# Pull model if needed (via Ollama CLI)
ollama pull llama3.2:3b

# Try again
curl -X POST /model/switch/llama3.2:3b
```

### Context Contamination

**Problem:** Responses seem influenced by previous queries with different models.

**Solution:**
```bash
# Always use stateless mode for multi-model testing
curl -X POST /chat -d '{"message": "...", "model": "model1", "stateless": true}'
curl -X POST /chat -d '{"message": "...", "model": "model2", "stateless": true}'

# Or clear history when switching models in stateful mode
curl -X POST /model/switch/new-model  # This auto-clears history
```

**[⬆ Back to top](#table-of-contents)**

---

## Related Documentation

- [README.md](README.md) - Main project documentation and installation guide
- [demos/README.md](demos/README.md) - Working examples and demo scripts for all API patterns
- [CHANGELOG.md](CHANGELOG.md) - Version history and changes

**[⬆ Back to top](#table-of-contents)**
