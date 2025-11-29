#!/usr/bin/env python3
"""
Basic Chat Demo - Chat without MCP tools
This demonstrates pure Ollama chat without any tool usage
"""

import requests
import json

HOST = "http://localhost:8000"

def main():
    print("=== Basic Chat Demo ===")
    print("Sending a simple chat message without tools...")
    print()
    
    response = requests.post(
        f"{HOST}/chat",
        json={
            "message": "Hello! Can you tell me a fun fact about programming?",
            "model": "llama3.2:3b",
            "mcp_server": ""
        }
    )
    
    print(json.dumps(response.json(), indent=2))
    print()
    print("Demo complete!")

if __name__ == "__main__":
    main()
