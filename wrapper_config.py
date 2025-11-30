"""
Wrapper Configuration Module

This module handles the configuration for the Ollama-FastMCP wrapper itself.
It reads wrapper settings from the [wrapper] section of a TOML file.

Configuration Structure in TOML:
    [wrapper]                    # Wrapper configuration section
    transport = "HTTP"           # Transport method: "HTTP" or "STDIO"
    host = "0.0.0.0"            # Host address for the wrapper's API server
    port = 8000                  # Port for the wrapper's API server
    history_file = ""            # Path to conversation history file
    overwrite_history = false    # Whether to overwrite existing history file
    model = { default = "llama3.2:3b", temperature = 0.2 }  # Model settings

Transport Modes:
    - HTTP: Wrapper connects to independently-running MCP servers via HTTP.
            Servers must be started manually before connecting.
    - STDIO: Wrapper spawns MCP servers as subprocesses and communicates
             via standard input/output. Servers are started automatically.
"""

from dataclasses import dataclass
import tomllib


@dataclass
class WrapperConfig:
    """
    Configuration for the Ollama-FastMCP Wrapper.

    This controls how the wrapper itself operates, separate from the
    MCP server configurations.

    Attributes:
        transport: Transport method for MCP communication ("HTTP" or "STDIO")
                  - HTTP: Connect to running servers via network
                  - STDIO: Spawn servers as subprocesses
        host: IP address the wrapper's API server binds to
              - "0.0.0.0" = all interfaces (accessible from network)
              - "127.0.0.1" = localhost only (local access only)
        port: TCP port the wrapper's API server listens on
        history_file: Optional path to load/save conversation history
        overwrite_history: If true, overwrite history file on exit instead of error
        model: Dictionary with model settings (default model name, temperature, etc.)
    """
    transport: str = "HTTP"
    host: str = "0.0.0.0"
    port: int = 8000
    history_file: str = ""
    overwrite_history: bool = False
    model: dict = None

    def __post_init__(self):
        """Set default model configuration if not provided."""
        if self.model is None:
            self.model = {"default": "llama3.2:3b", "temperature": 0.2}

    @classmethod
    def from_toml(cls, config_path: str):
        """
        Load wrapper configuration from TOML file.

        Reads the [wrapper] section from the TOML file. If the section doesn't
        exist, returns a WrapperConfig with default values.

        Args:
            config_path: Path to the TOML configuration file

        Returns:
            WrapperConfig instance with loaded or default settings

        Raises:
            FileNotFoundError: If config file doesn't exist
            tomllib.TOMLDecodeError: If config file is invalid TOML

        Example:
            config = WrapperConfig.from_toml("wrapper_config.toml")
            print(f"Wrapper will run on {config.host}:{config.port}")
        """
        with open(config_path, "rb") as f:
            data = tomllib.load(f)

        # Get [wrapper] section from TOML, or empty dict if not present
        wrapper_data = data.get('wrapper', {})

        # Create config from TOML data, or use defaults if section missing
        return cls(**wrapper_data) if wrapper_data else cls()
