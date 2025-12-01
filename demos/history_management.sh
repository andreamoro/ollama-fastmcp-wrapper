#!/bin/bash
# History Management Demo - Load and save conversation history
# This demonstrates conversation persistence

HOST="http://localhost:8000"
HISTORY_FILE="demo_conversation.json"

echo "=== History Management Demo ==="
echo

echo "Step 1: Starting a conversation..."
curl -X POST "$HOST/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Hello! My name is Demo User.",
    "model": "llama3.2:3b",
    "mcp_server": ""
  }' | jq '.'
echo

echo "Step 2: Continuing the conversation..."
curl -X POST "$HOST/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "What is the capital of France?",
    "model": "llama3.2:3b",
    "mcp_server": ""
  }' | jq '.'
echo

echo "Step 3: Getting current history..."
curl -X GET "$HOST/history" | jq '.'
echo

echo "Step 4: Saving conversation history..."
curl "$HOST/history/save/$HISTORY_FILE" | jq '.'
echo

echo "Saved conversation to: $HISTORY_FILE"
echo "You can load this history later by restarting the wrapper with:"
echo "  python ollama_wrapper.py api --history-file $HISTORY_FILE"
echo

echo "Demo complete!"
