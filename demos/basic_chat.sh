#!/bin/bash
# Basic Chat Demo - Chat without MCP tools
# This demonstrates pure Ollama chat without any tool usage

# Source the wrapper URL configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/get_wrapper_url.sh"
HOST="$WRAPPER_URL"

echo "=== Basic Chat Demo ==="
echo "Sending a simple chat message without tools..."
echo

curl -X POST "$HOST/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Hello! How are you doing?",
    "model": "gemma3:1b",
    "mcp_server": ""
  }' | jq '.'

# "model": "llama3.2:3b",

echo
echo "Demo complete!"
