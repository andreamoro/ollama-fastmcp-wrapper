# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.4.0] - 2025-11-29

### Breaking Changes
- **Complete configuration separation**: MCP server configuration moved to `mcp_servers/` directory
- **Configuration file reorganization**:
  - Wrapper config: `wrapper_config.toml` (root directory)
  - MCP servers config: `mcp_servers/mcp_servers_config.toml` (mcp_servers/ directory)
  - MCP tokens: `mcp_servers/mcp_tokens.toml` (mcp_servers/ directory, gitignored)
- **Command-line argument changes**:
  - `-c/--config` renamed to `-c/--wrapper-config`
  - New `--mcp-config` argument (expects filename only, resolved to mcp_servers/ directory)
- **Removed python-dotenv dependency**: All configuration now uses TOML format

### Added
- **IPInfo MCP Server**: New server for IP geolocation lookup via ipinfo.io API
  - 20 preset organizations for demos (Google, Facebook, GitHub, etc.)
  - Three tools: `lookup_ip()`, `lookup_organization()`, `list_organizations()`
  - Token management via `mcp_servers/mcp_tokens.toml`
- **MCP servers directory structure**: All MCP server code now in `mcp_servers/` directory
- **Enhanced `/servers` endpoint**: Now dynamically lists all connected servers with their available tools
- **Token management**: TOML-based secure token storage with example file

### Changed
- **Configuration module architecture**: `mcpserver_config.py` now resolves paths relative to `mcp_servers/` directory
- **`/servers` endpoint response**: Returns runtime state (connected servers with their tools) instead of config-based listing
- **Tool listing**: Completely dynamic from FastMCP `client.list_tools()` - no manual maintenance needed
- **Error handling**: More descriptive error messages for connection failures

### Fixed
- **Global config initialization**: Fixed `mcp_config` NoneType error on server connection
- **Config file location**: Proper separation between wrapper and MCP server configurations
- **`/servers` endpoint**: Removed redundant status when servers are connected

### Removed
- **python-dotenv dependency**: Replaced with native Python `tomllib` for token management

## [0.3.0] - 2025-11-29

### Added
- Wrapper configuration section in `server_config.toml` with `[wrapper]` settings
- Configurable server host and port via command-line arguments (`--host`, `--port`)
- Configuration priority system: command-line > config file > defaults
- Separate `wrapper_config.py` module for wrapper-specific configuration
- Comprehensive inline documentation for all configuration modules
- `math_server.py` now reads configuration from `server_config.toml`
- Detailed docstrings following Python documentation standards
- Automatic conversation history saving after each chat interaction
- `history_file` and `overwrite_history` parameters to `OllamaWrapper.__init__`
- Automatic loading of conversation history on startup if `history_file` is specified
- Auto-save method that respects the `overwrite_history` flag
- `GET /history` endpoint to retrieve current conversation history

### Changed
- Renamed internal `config` variable to `mcp_config` for clarity
- Separated MCP server config from wrapper config into distinct modules
- Updated README.md with comprehensive configuration documentation
- Improved validation logic for host and port parameters

### Fixed
- Configuration alignment between `math_server.py` and wrapper
- Math server no longer uses hardcoded host/port values
- Tools no longer persist globally across requests - tools are now scoped per request
- When `mcp_server` is empty (""), no tools are sent to Ollama (prevents unintended tool usage)
- Conversation history now automatically saves to configured file after each message

## [0.2.0] - 2025-09-02

### Added
- External MCP config file support via `server_config.toml`
- Command-line argument parser for mode and model selection
- `-c/--config` argument to specify custom configuration file
- `--history-file` argument to load conversation history
- `-o/--overwrite-history` flag for history file management
- `-t/--transport` argument to specify transport method (HTTP/STDIO)

### Changed
- MCP server configuration moved from inline Python dict to TOML file
- Server configurations now support both STDIO and HTTP transport modes

## [0.1.0] - 2025-08-19

### Added
- Initial release of Ollama-FastMCP Wrapper
- API mode with FastAPI server
- CLI mode for interactive chat
- Support for multiple FastMCP servers
- Connect/disconnect servers at runtime via API
- Message history with automatic summarization
- Conversation persistence with save/load functionality
- Tool calling support for Ollama models
- STDIO and HTTP transport methods for MCP servers
- Basic math MCP server example

### API Endpoints
- `GET /servers` - List available FastMCP servers
- `POST /connect/{server_name}` - Connect to an MCP server
- `POST /disconnect/{server_name}` - Disconnect from an MCP server
- `GET /list_tools?server_name={name}` - List tools for a server
- `POST /chat` - Send chat requests with optional tool usage
- `POST /save_history/{file_name}` - Save conversation history
- `POST /overwrite_history/{file_name}` - Overwrite existing conversation history

[unreleased]: https://github.com/andreamoro/ollama-fastmcp-wrapper/compare/v0.4.0...HEAD
[0.4.0]: https://github.com/andreamoro/ollama-fastmcp-wrapper/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/andreamoro/ollama-fastmcp-wrapper/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/andreamoro/ollama-fastmcp-wrapper/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/andreamoro/ollama-fastmcp-wrapper/releases/tag/v0.1.0
