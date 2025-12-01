#!/usr/bin/env python3
"""
Server Management Demo - Connect, list, and disconnect servers
This demonstrates server lifecycle management
"""

import requests
import json
from demo_config import API_URL

HOST = API_URL

def main():
    print("=== Server Management Demo ===")
    print()
    
    print("Step 1: List available servers (initially empty)...")
    response = requests.get(f"{HOST}/servers")
    print(json.dumps(response.json(), indent=2))
    print()
    
    print("Step 2: Connect to math server...")
    response = requests.post(f"{HOST}/connect/math")
    print(json.dumps(response.json(), indent=2))
    print()
    
    print("Step 3: Connect to ipinfo server...")
    response = requests.post(f"{HOST}/connect/ipinfo")
    print(json.dumps(response.json(), indent=2))
    print()
    
    print("Step 4: List connected servers with their tools...")
    response = requests.get(f"{HOST}/servers")
    print(json.dumps(response.json(), indent=2))
    print()
    
    print("Step 5: Disconnect from math server...")
    response = requests.post(f"{HOST}/disconnect/math")
    print(json.dumps(response.json(), indent=2))
    print()
    
    print("Step 6: List servers again (only ipinfo should remain)...")
    response = requests.get(f"{HOST}/servers")
    print(json.dumps(response.json(), indent=2))
    print()
    
    print("Step 7: Disconnect from ipinfo server...")
    response = requests.post(f"{HOST}/disconnect/ipinfo")
    print(json.dumps(response.json(), indent=2))
    print()
    
    print("Step 8: List servers one final time (should be empty)...")
    response = requests.get(f"{HOST}/servers")
    print(json.dumps(response.json(), indent=2))
    print()
    
    print("Demo complete!")

if __name__ == "__main__":
    main()
