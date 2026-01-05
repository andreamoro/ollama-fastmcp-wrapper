"""
Shared configuration utilities for demo scripts
Reads settings from wrapper_config.toml
"""

import re
from pathlib import Path

def get_wrapper_url():
    """
    Read wrapper host and port from wrapper_config.toml

    Returns:
        str: The wrapper URL in format http://host:port

    Note:
        - Falls back to http://localhost:8000 if config cannot be read
        - Converts 0.0.0.0 to localhost for client connections
        - Only reads from [wrapper] section, ignoring [ollama] section
    """
    try:
        # Get the project root directory using Path
        demos_dir = Path(__file__).resolve().parent
        project_root = demos_dir.parent
        config_file = project_root / "wrapper_config.toml"

        host = "localhost"  # Default
        port = 8000  # Default

        section_pattern = re.compile(r'^\s*\[([^\]]+)\]')
        config_pattern = re.compile(
            r'^\s*(host|port)\s*=\s*["\']?([^"\']+)["\']?',
            re.IGNORECASE
        )

        current_section = None

        with open(config_file, 'r', encoding='utf-8-sig') as f:
            for line in f:
                clean = line.split('#', 1)[0].strip()  # remove comments

                # Check for section header
                section_match = section_pattern.match(clean)
                if section_match:
                    current_section = section_match.group(1).strip()
                    continue

                # Only process config lines in [wrapper] section
                if current_section != "wrapper":
                    continue

                config_match = config_pattern.match(clean)
                if not config_match:
                    continue

                key, value = config_match.groups()
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
