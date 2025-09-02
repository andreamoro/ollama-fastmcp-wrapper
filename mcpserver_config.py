from dataclasses import dataclass, field
from typing import Dict, List

import tomllib

@dataclass
class MCPServerConfig:
    name: str = ""
    command: str = ""
    args: List[str] = field(default_factory=list)
    host: str = ""
    port: int = 0
    enabled: bool = False

    def __post_init__(self):
        # Ensure args is always a list
        if isinstance(self.args, str):
            self.args = [self.args]

@dataclass
class Config:
    _servers: List[MCPServerConfig] = field(default_factory=list, init=False)

    @classmethod
    def from_toml(cls, config_path: str):
        """Create MCPConfig object from TOML file"""
        with open(config_path, "rb") as f:
            data = tomllib.load(f)

        instance = cls()

        # Parse servers - handle array of tables
        # servers = []
        for server_data in data.get('servers', []):
            # Apply defaults where server config is None
            server = MCPServerConfig(**server_data)
            instance._servers.append(server)

        return instance

    @property
    def servers(self) -> Dict[str, MCPServerConfig]:
        """Return servers as a dict for easy lookup"""
        return {server.name: server for server in self._servers}

    def __getitem__(self, name: str) -> MCPServerConfig:
        """Get server by name using indexer syntax"""
        for server in self._servers:
            if server.name == name:
                return server
        raise KeyError(f"Server '{name}' not found")

    def __contains__(self, name: str) -> bool:
        """Check if server exists by name"""
        return any(server.name == name for server in self._servers)
