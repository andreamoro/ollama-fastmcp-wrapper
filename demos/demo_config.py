"""
Shared configuration utilities for demo scripts
Reads settings from wrapper_config.toml
"""

import os

def get_wrapper_url():
    """Read wrapper host and port from wrapper_config.toml"""
    try:
        # Get the project root directory
        demos_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(demos_dir)
        config_file = os.path.join(project_root, "wrapper_config.toml")

        host = "localhost"  # Default
        port = 8000  # Default

        with open(config_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith('host ='):
                    # Extract host value: host = "0.0.0.0" -> 0.0.0.0
                    host_value = line.split('=', 1)[1].strip().strip('"').strip("'")
                    # If listening on 0.0.0.0, connect to localhost
                    if host_value == "0.0.0.0":
                        host = "localhost"
                    else:
                        host = host_value
                elif line.startswith('port ='):
                    # Extract port value: port = 8000 -> 8000
                    port = int(line.split('=', 1)[1].strip())

        return f"http://{host}:{port}"
    except Exception as e:
        # Fallback to default
        print(f"Warning: Could not read wrapper config ({e}), using default http://localhost:8000")
        return "http://localhost:8000"

# Read wrapper URL from config once on import
API_URL = get_wrapper_url()
