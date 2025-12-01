#!/usr/bin/env python3
"""
History Management Demo - Load and save conversation history
This demonstrates conversation persistence
"""

import requests
import json
from demo_config import API_URL

HOST = API_URL
HISTORY_FILE = "demo_conversation.json"

def main():
    print("=== History Management Demo ===")
    print()
    
    print("Step 1: Starting a conversation...")
    response = requests.post(
        f"{HOST}/chat",
        json={
            "message": "Hello! My name is Demo User.",
            "model": "llama3.2:3b",
            "mcp_server": ""
        }
    )
    print(json.dumps(response.json(), indent=2))
    print()
    
    print("Step 2: Continuing the conversation...")
    response = requests.post(
        f"{HOST}/chat",
        json={
            "message": "What is the capital of France?",
            "model": "llama3.2:3b",
            "mcp_server": ""
        }
    )
    print(json.dumps(response.json(), indent=2))
    print()
    
    print("Step 3: Getting current history...")
    response = requests.get(f"{HOST}/history")
    print(json.dumps(response.json(), indent=2))
    print()
    
    print("Step 4: Saving conversation history...")
    response = requests.post(f"{HOST}/save_history/{HISTORY_FILE}")
    print(json.dumps(response.json(), indent=2))
    print()
    
    print(f"Saved conversation to: {HISTORY_FILE}")
    print("You can load this history later by restarting the wrapper with:")
    print(f"  python ollama_wrapper.py api --history-file {HISTORY_FILE}")
    print()
    
    print("Demo complete!")

if __name__ == "__main__":
    main()
