"""
Wrapper Configuration Module

This module handles the configuration for the Ollama-FastMCP wrapper itself.
It reads wrapper settings from the [wrapper] section and Ollama connection
settings from the [ollama] section of a TOML file.

Configuration Structure in TOML:
    [wrapper]                    # Wrapper configuration section
    transport = "HTTP"           # Transport method: "HTTP" or "STDIO"
    host = "0.0.0.0"            # Host address for the wrapper's API server
    port = 8000                  # Port for the wrapper's API server
    history_file = ""            # Path to conversation history file
    overwrite_history = false    # Whether to overwrite existing history file
    max_history_messages = 20    # Maximum messages before summarization kicks in

    [ollama]                     # Ollama instance configuration section
    host = "localhost"           # Ollama instance host
    port = 11434                 # Ollama instance port
    timeout = 300                # Request timeout in seconds (prevents hang on tunnel drops)
    label = ""                   # Optional label to identify this instance
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
        max_history_messages: Maximum number of messages before summarization kicks in
    """
    transport: str = "HTTP"
    host: str = "0.0.0.0"
    port: int = 8000
    history_file: str = ""
    overwrite_history: bool = False
    max_history_messages: int = 20

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


@dataclass
class OllamaConfig:
    """
    Configuration for the Ollama instance connection.

    This controls how the wrapper connects to the Ollama instance.

    Attributes:
        host: Ollama instance host address (e.g., "localhost", "192.168.1.100")
        port: Ollama instance port (default: 11434)
        timeout: Request timeout in seconds (default: 300). Prevents hang on tunnel drops.
        label: Optional human-readable label to identify this instance (e.g., "remote-vps-via-tunnel")
        model: Dictionary with model settings (default model name, temperature, etc.)
    """
    host: str = "localhost"
    port: int = 11434
    timeout: int = 300
    label: str = ""
    model: dict = None

    def __post_init__(self):
        """Set default model configuration if not provided."""
        if self.model is None:
            self.model = {"default": "llama3.2:3b", "temperature": 0.2}

    @property
    def url(self) -> str:
        """Get the full Ollama URL (e.g., 'http://localhost:11434')."""
        return f"http://{self.host}:{self.port}"

    @classmethod
    def from_toml(cls, config_path: str):
        """
        Load Ollama configuration from TOML file.

        Reads the [ollama] section from the TOML file. If the section doesn't
        exist, returns an OllamaConfig with default values.

        Args:
            config_path: Path to the TOML configuration file

        Returns:
            OllamaConfig instance with loaded or default settings

        Raises:
            FileNotFoundError: If config file doesn't exist
            tomllib.TOMLDecodeError: If config file is invalid TOML

        Example:
            config = OllamaConfig.from_toml("wrapper_config.toml")
            print(f"Connecting to Ollama at {config.url}")
        """
        with open(config_path, "rb") as f:
            data = tomllib.load(f)

        # Get [ollama] section from TOML, or empty dict if not present
        ollama_data = data.get('ollama', {})

        # Create config from TOML data, or use defaults if section missing
        return cls(**ollama_data) if ollama_data else cls()
