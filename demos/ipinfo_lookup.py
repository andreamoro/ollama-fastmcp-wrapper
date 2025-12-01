#!/usr/bin/env python3
"""
IPInfo Lookup Demo - Using the IPInfo MCP server
This demonstrates IP geolocation and organization lookup
"""

import requests
import json
from demo_config import API_URL

HOST = API_URL

def main():
    print("=== IPInfo Lookup Demo ===")
    print()
    
    print("Step 1: Connecting to ipinfo server...")
    response = requests.post(f"{HOST}/servers/ipinfo/connect")
    print(json.dumps(response.json(), indent=2))
    print()
    
    print("Step 2: Looking up a specific IP address...")
    response = requests.post(
        f"{HOST}/chat",
        json={
            "message": "Look up information for IP address 8.8.8.8",
            "model": "llama3.2:3b",
            "mcp_server": "ipinfo"
        }
    )
    print(json.dumps(response.json(), indent=2))
    print()
    
    print("Step 3: Looking up organization info...")
    response = requests.post(
        f"{HOST}/chat",
        json={
            "message": "Show me IP information for GitHub",
            "model": "llama3.2:3b",
            "mcp_server": "ipinfo"
        }
    )
    print(json.dumps(response.json(), indent=2))
    print()
    
    print("Step 4: Listing available organizations...")
    response = requests.post(
        f"{HOST}/chat",
        json={
            "message": "List all available preset organizations",
            "model": "llama3.2:3b",
            "mcp_server": "ipinfo"
        }
    )
    print(json.dumps(response.json(), indent=2))
    print()
    
    print("Demo complete!")

if __name__ == "__main__":
    main()
