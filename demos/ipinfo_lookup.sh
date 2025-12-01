#!/bin/bash
# IPInfo Lookup Demo - Using the IPInfo MCP server
# This demonstrates IP geolocation and organization lookup

HOST="http://localhost:8000"

echo "=== IPInfo Lookup Demo ==="
echo

echo "Step 1: Connecting to ipinfo server..."
curl -X POST "$HOST/servers/ipinfo" | jq '.'/connect
echo

echo "Step 2: Looking up a specific IP address..."
curl -X POST "$HOST/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Look up information for IP address 8.8.8.8",
    "model": "llama3.2:3b",
    "mcp_server": "ipinfo"
  }' | jq '.'
echo

echo "Step 3: Looking up organization info..."
curl -X POST "$HOST/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Show me IP information for GitHub",
    "model": "llama3.2:3b",
    "mcp_server": "ipinfo"
  }' | jq '.'
echo

echo "Step 4: Listing available organizations..."
curl -X POST "$HOST/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "List all available preset organizations",
    "model": "llama3.2:3b",
    "mcp_server": "ipinfo"
  }' | jq '.'
echo

echo "Demo complete!"
