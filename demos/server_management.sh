#!/bin/bash
# Server Management Demo - Connect, list, and disconnect servers
# This demonstrates server lifecycle management

HOST="http://localhost:8000"

echo "=== Server Management Demo ==="
echo

echo "Step 1: List available servers (initially empty)..."
curl -X GET "$HOST/servers" | jq '.'
echo

echo "Step 2: Connect to math server..."
curl -X POST "$HOST/servers/math" | jq '.'/connect
echo

echo "Step 3: Connect to ipinfo server..."
curl -X POST "$HOST/servers/ipinfo" | jq '.'/connect
echo

echo "Step 4: List connected servers with their tools..."
curl -X GET "$HOST/servers" | jq '.'
echo

echo "Step 5: Disconnect from math server..."
curl -X POST "$HOST/servers/math" | jq '.'/connect
echo

echo "Step 6: List servers again (only ipinfo should remain)..."
curl -X GET "$HOST/servers" | jq '.'
echo

echo "Step 7: Disconnect from ipinfo server..."
curl -X POST "$HOST/servers/ipinfo" | jq '.'/connect
echo

echo "Step 8: List servers one final time (should be empty)..."
curl -X GET "$HOST/servers" | jq '.'
echo

echo "Demo complete!"
