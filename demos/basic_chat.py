#!/usr/bin/env python3
"""
Basic Chat Demo - Chat without MCP tools
This demonstrates pure Ollama chat without any tool usage
"""

import requests
import json
import timeit

HOST = "http://localhost:8000"

def main():
    print("=== Basic Chat Demo ===")
    print("Sending a simple chat message without tools...")
    print()

    response = requests.post(
        f"{HOST}/chat",
        json={
            "message": "Hello! Can you tell me a fun fact about programming?",
            "model": "aliafshar/gemma3-it-qat-tools:4b",
            "mcp_server": ""
        }
    )

    print(json.dumps(response.json(), indent=2))
    print()
    print("Demo complete!")

def seconds_to_units(seconds):
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60
    return f"Execution time: {hours} hours, {minutes} minutes, {secs} seconds"

if __name__ == "__main__":
    execution_time =  seconds_to_units(timeit.timeit(main(), number=1))
    print(execution_time)
    main()
