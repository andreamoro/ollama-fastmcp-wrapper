#!/bin/bash
# Basic Chat Demo - Chat without MCP tools
# This demonstrates pure Ollama chat without any tool usage

HOST="http://localhost:8000"

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
