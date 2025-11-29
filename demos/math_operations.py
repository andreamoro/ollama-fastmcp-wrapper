#!/usr/bin/env python3
"""
Math Operations Demo - Using the math MCP server
This demonstrates connecting to a server and using its tools
"""

import requests
import json

HOST = "http://localhost:8000"

def main():
    print("=== Math Operations Demo ===")
    print()
    
    print("Step 1: Connecting to math server...")
    response = requests.post(f"{HOST}/connect/math")
    print(json.dumps(response.json(), indent=2))
    print()
    
    print("Step 2: Performing math operations...")
    response = requests.post(
        f"{HOST}/chat",
        json={
            "message": "Calculate: (15 + 25) * 3, then divide the result by 10",
            "model": "llama3.2:3b",
            "mcp_server": "math"
        }
    )
    print(json.dumps(response.json(), indent=2))
    print()
    
    print("Step 3: Another calculation...")
    response = requests.post(
        f"{HOST}/chat",
        json={
            "message": "What is 100 divided by 4, then multiply by 7?",
            "model": "llama3.2:3b",
            "mcp_server": "math"
        }
    )
    print(json.dumps(response.json(), indent=2))
    print()
    
    print("Demo complete!")

if __name__ == "__main__":
    main()
