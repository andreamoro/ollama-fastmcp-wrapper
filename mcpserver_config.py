"""
MCP Server Configuration Module

This module handles the configuration for FastMCP servers that can be used with
the Ollama-FastMCP wrapper. It reads server configurations from a TOML file.

Configuration Structure in TOML:
    [[servers]]              # Array of server configurations
    name = "math"            # Unique identifier for the server
    command = "uv"           # Command to spawn the server (STDIO mode only)
    args = ["run", ...]      # Arguments for the command (STDIO mode only)
    host = "http://..."      # URL to connect to the server (HTTP mode only)
    port = 5000              # Port number the server runs on
    enabled = true           # Whether this server should be available
"""

from dataclasses import dataclass, field
from typing import Dict, List

import tomllib

@dataclass
class MCPServerConfig:
    """
    Configuration for a single FastMCP server.

    Attributes:
        name: Unique identifier for this server (used to reference it in API calls)
        command: Shell command to execute the server (used with STDIO transport)
        args: List of command-line arguments (used with STDIO transport)
        host: URL to connect to the server (used with HTTP transport)
              Example: "http://localhost:5000/mcp"
        port: Port number the server runs on (informational for HTTP, used for STDIO)
        enabled: Whether this server is available for connections
    """
    name: str = ""
    command: str = ""
    args: List[str] = field(default_factory=list)
    host: str = ""
    port: int = 0
    enabled: bool = False

    def __post_init__(self):
        """
        Post-initialization processing.

        Ensures 'args' is always a list, even if a single string was provided
        in the TOML configuration.
        """
        if isinstance(self.args, str):
            self.args = [self.args]

@dataclass
class Config:
    """
    Container for all MCP server configurations.

    This class manages multiple server configurations and provides convenient
    access methods.

    Usage:
        config = Config.from_toml("server_config.toml")

        # Check if server exists
        if "math" in config:
            # Access by name
            math_server = config["math"]

        # Get all servers as dict
        all_servers = config.servers
    """
    _servers: List[MCPServerConfig] = field(default_factory=list, init=False)

    @classmethod
    def from_toml(cls, config_path: str):
        """
        Create Config object from TOML file.

        Reads the TOML configuration file and parses all [[servers]] sections
        into MCPServerConfig objects.

        Args:
            config_path: Path to the TOML configuration file

        Returns:
            Config instance with all servers loaded

        Raises:
            FileNotFoundError: If config file doesn't exist
            tomllib.TOMLDecodeError: If config file is invalid TOML
        """
        with open(config_path, "rb") as f:
            data = tomllib.load(f)

        instance = cls()

        # Parse servers - handle array of tables ([[servers]])
        # Each [[servers]] entry in TOML becomes a separate server config
        for server_data in data.get('servers', []):
            server = MCPServerConfig(**server_data)
            instance._servers.append(server)

        return instance

    @property
    def servers(self) -> Dict[str, MCPServerConfig]:
        """
        Return servers as a dictionary for easy lookup.

        Returns:
            Dict mapping server names to their configurations
        """
        return {server.name: server for server in self._servers}

    def __getitem__(self, name: str) -> MCPServerConfig:
        """
        Get server configuration by name using indexer syntax.

        Args:
            name: The server name to look up

        Returns:
            MCPServerConfig for the requested server

        Raises:
            KeyError: If server with given name doesn't exist

        Example:
            config = Config.from_toml("config.toml")
            math_config = config["math"]
        """
        for server in self._servers:
            if server.name == name:
                return server
        raise KeyError(f"Server '{name}' not found")

    def __contains__(self, name: str) -> bool:
        """
        Check if a server exists by name.

        Args:
            name: The server name to check

        Returns:
            True if server exists, False otherwise

        Example:
            if "math" in config:
                print("Math server is configured")
        """
        return any(server.name == name for server in self._servers)
