#!/bin/bash
# Math Operations Demo - Using the math MCP server
# This demonstrates connecting to a server and using its tools

# Source the wrapper URL configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/get_wrapper_url.sh"
HOST="$WRAPPER_URL"

echo "=== Math Operations Demo ==="
echo

echo "Step 1: Connecting to math server..."
curl -X POST "$HOST/servers/math" | jq '.'/connect
echo

echo "Step 2: Performing math operations..."
curl -X POST "$HOST/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Calculate: (15 + 25) * 3, then divide the result by 10",
    "model": "llama3.2:3b",
    "mcp_server": "math"
  }' | jq '.'
echo

echo "Step 3: Another calculation..."
curl -X POST "$HOST/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "What is 100 divided by 4, then multiply by 7?",
    "model": "llama3.2:3b",
    "mcp_server": "math"
  }' | jq '.'
echo

echo "Demo complete!"
