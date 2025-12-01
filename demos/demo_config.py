"""
Shared configuration utilities for demo scripts
Reads settings from wrapper_config.toml
"""

import os
import re

def get_wrapper_url():
    """Read wrapper host and port from wrapper_config.toml"""
    try:
        # Get the project root directory
        demos_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(demos_dir)
        config_file = os.path.join(project_root, "wrapper_config.toml")

        host = "localhost"  # Default
        port = 8000  # Default

        pattern = re.compile(
            r'^\s*(host|port)\s*=\s*["\']?([^"\']+)["\']?',
            re.IGNORECASE
        )

        with open(config_file, 'r', encoding='utf-8-sig') as f:
            for line in f:
                clean = line.split('#', 1)[0].strip()  # remove comments
                match = pattern.match(clean)
                if not match:
                    continue

                key, value = match.groups()
                key = key.lower()
                value = value.strip()

                if key == "host":
                    host = "localhost" if value == "0.0.0.0" else value

                elif key == "port":
                    # Ensure it's numeric
                    if not value.isdigit():
                        raise ValueError(f"Invalid port number: {value}")
                    port = int(value)

        if host is None or port is None:
            raise ValueError("Missing host or port in config")

        return f"http://{host}:{port}"
    except Exception as e:
        # Fallback to default
        print(f"Warning: Could not read wrapper config ({e}), using default http://localhost:8000")
        return "http://localhost:8000"

# Read wrapper URL from config once on import
API_URL = get_wrapper_url()
