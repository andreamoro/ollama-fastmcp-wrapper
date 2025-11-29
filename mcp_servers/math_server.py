"""
Math MCP Server

This is a FastMCP server that provides basic mathematical operations as tools.
It demonstrates how to create an MCP server that can be used by the Ollama-FastMCP wrapper.

The server reads its configuration from mcp_servers_config.toml to ensure alignment with
the wrapper's expectations.
"""

from fastmcp import FastMCP
import mcpserver_config
import sys

# Initialize FastMCP with a server name
# This name should match the 'name' field in server_config.toml
mcp = FastMCP("math")

@mcp.tool()
def add(a: int, b: int) -> int:
    """Adds or sum two integer numbers."""
    return a + b

@mcp.tool()
def multiply(a: int, b: int) -> int:
    """Multiplies two integer numbers."""
    return a * b

if __name__ == "__main__":
    """
    Start the MCP server using configuration from mcp_servers_config.toml

    The server can run in two transport modes:
    1. STDIO: For local testing where the wrapper spawns the server as a subprocess
              and communicates via standard input/output
    2. HTTP: For network communication where the server runs independently and the
             wrapper connects to it via HTTP

    Configuration is read from mcp_servers_config.toml [[servers]] section where name="math"
    """

    # Load configuration from TOML file
    # Default to 'mcp_servers_config.toml' in the current directory
    config_file = sys.argv[1] if len(sys.argv) > 1 else "mcp_servers_config.toml"

    try:
        config = mcpserver_config.Config.from_toml(config_file)

        # Find this server's configuration by name
        server_config = config["math"]

        # Extract host and port from the configuration
        # For HTTP transport, the 'host' field contains the full URL
        # We need to extract just the hostname and use the 'port' field
        if server_config.host:
            # Parse the host URL to extract the hostname
            # Example: "http://localhost:5000/mcp" -> "localhost"
            import urllib.parse
            parsed = urllib.parse.urlparse(server_config.host)
            hostname = parsed.hostname or "localhost"
        else:
            hostname = "localhost"

        port = server_config.port if server_config.port else 5000

        print(f"🚀 Starting Math MCP Server...")
        print(f"📡 Server: {mcp.name}")
        print(f"🔗 Running on: http://{hostname}:{port}")
        print(f"📝 Config file: {config_file}")

        # Start the server with configuration from file
        # Note: This always uses HTTP transport. For STDIO transport, the wrapper
        # spawns this script as a subprocess when transport='STDIO' is configured
        mcp.run(transport='http', host=hostname, port=port)

    except KeyError:
        print(f"❌ Error: Server 'math' not found in {config_file}")
        print("   Please ensure [[servers]] section with name='math' exists in the config file.")
        sys.exit(1)
    except FileNotFoundError:
        print(f"❌ Error: Config file '{config_file}' not found")
        print("   Usage: python math_server.py [config_file]")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error starting server: {e}")
        sys.exit(1)
