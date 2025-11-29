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
    "message": "Hello! Can you tell me a fun fact about programming?",
    "model": "llama3.2:3b",
    "mcp_server": ""
  }' | jq '.'

echo
echo "Demo complete!"
