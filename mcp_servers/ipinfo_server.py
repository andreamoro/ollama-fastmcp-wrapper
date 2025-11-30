"""
IP Info MCP Server

This FastMCP server provides IP geolocation lookup using the ipinfo.io API.
It includes preset IPs for well-known organizations for demo purposes.

The server reads its configuration from mcp_servers_config.toml and API token from mcp_tokens.toml.
"""

from fastmcp import FastMCP
import mcpserver_config
import sys
import urllib.request
import json
import tomllib
from pathlib import Path

# Initialize FastMCP server
mcp = FastMCP("ipinfo")

# Preset IPs for well-known organizations (for demo purposes)
KNOWN_IPS = {
    "google_dns": "8.8.8.8",
    "google": "142.250.185.46",
    "facebook": "157.240.241.35",
    "cloudflare_dns": "1.1.1.1",
    "amazon": "205.251.242.103",
    "microsoft": "20.112.52.29",
    "apple": "17.253.144.10",
    "netflix": "52.84.24.18",
    "twitter": "104.244.42.193",
    "github": "140.82.121.4",
    "italian_gov": "194.242.232.104",  # governo.it
    "european_parliament": "212.68.215.195",
    "bbc": "151.101.192.81",
    "cnn": "151.101.1.67",
    "nyt": "151.101.1.164",  # New York Times
    "wikipedia": "185.15.58.224",
    "stackoverflow": "151.101.1.69",
    "reddit": "151.101.65.140",
    "linkedin": "108.174.10.10",
    "youtube": "142.250.185.78"
}

def get_ipinfo_token(token_file_path: str = None) -> str:
    """
    Get IPInfo API token from specified token file.

    Args:
        token_file_path: Optional path to token file. If None, defaults to mcp_tokens.toml
                        in the mcp_servers directory.

    Returns:
        str: The IPInfo API token
    """
    # Default to mcp_tokens.toml in mcp_servers directory if not specified
    if token_file_path is None:
        token_file_path = str(Path(__file__).parent / "mcp_tokens.toml")

    token_file = Path(token_file_path)

    if not token_file.exists():
        raise ValueError(
            f"Token file not found: {token_file}\n"
            "Please create the token file from mcp_tokens.toml.example"
        )

    try:
        with open(token_file, "rb") as f:
            tokens = tomllib.load(f)

        if "ipinfo" not in tokens or "token" not in tokens["ipinfo"]:
            raise ValueError(
                f"IPInfo token not found in {token_file}\n"
                "Please add: [ipinfo]\\ntoken = \"your_token_here\""
            )

        return tokens["ipinfo"]["token"]

    except tomllib.TOMLDecodeError as e:
        raise ValueError(f"Invalid TOML in {token_file}: {e}")

def query_ipinfo(ip_address: str) -> dict:
    """
    Query the ipinfo.io API for information about an IP address.

    Args:
        ip_address: The IP address to lookup

    Returns:
        dict: IP information including location, org, etc.
    """
    token = get_ipinfo_token()
    url = f"https://ipinfo.io/{ip_address}?token={token}"

    try:
        with urllib.request.urlopen(url) as response:
            data = response.read()
            return json.loads(data)
    except Exception as e:
        return {"error": f"Failed to query IP: {str(e)}"}

@mcp.tool()
def lookup_ip(ip_address: str) -> str:
    """
    Lookup geolocation information for a specific IP address.

    Args:
        ip_address: The IP address to lookup (e.g., "8.8.8.8")

    Returns:
        Formatted string with IP information including city, region, country, organization
    """
    result = query_ipinfo(ip_address)

    if "error" in result:
        return result["error"]

    # Format the response
    info_parts = [
        f"IP: {result.get('ip', 'N/A')}",
        f"Location: {result.get('city', 'N/A')}, {result.get('region', 'N/A')}, {result.get('country', 'N/A')}",
        f"Coordinates: {result.get('loc', 'N/A')}",
        f"Organization: {result.get('org', 'N/A')}",
        f"Timezone: {result.get('timezone', 'N/A')}"
    ]

    return "\n".join(info_parts)

@mcp.tool()
def lookup_organization(org_name: str) -> str:
    """
    Lookup IP information for a well-known organization from preset list.

    Available organizations:
    - google_dns, google, facebook, cloudflare_dns, amazon
    - microsoft, apple, netflix, twitter, github
    - italian_gov, european_parliament, bbc, cnn, nyt
    - wikipedia, stackoverflow, reddit, linkedin, youtube

    Args:
        org_name: Name of the organization (e.g., "google", "facebook")

    Returns:
        Formatted string with IP information for the organization
    """
    org_name_lower = org_name.lower().replace(" ", "_")

    if org_name_lower not in KNOWN_IPS:
        available = ", ".join(sorted(KNOWN_IPS.keys()))
        return f"Organization '{org_name}' not found in preset list.\nAvailable: {available}"

    ip_address = KNOWN_IPS[org_name_lower]
    return f"Organization: {org_name}\n" + lookup_ip(ip_address)

@mcp.tool()
def list_organizations() -> str:
    """
    List all available preset organizations with their IP addresses.

    Returns:
        Formatted list of organizations and their IPs
    """
    lines = ["Available organizations:"]
    for org, ip in sorted(KNOWN_IPS.items()):
        lines.append(f"  • {org}: {ip}")
    return "\n".join(lines)

if __name__ == "__main__":
    """
    Start the IPInfo MCP server using configuration from mcp_servers_config.toml

    Requires token in mcp_servers/mcp_tokens.toml file.
    """

    # Load configuration from TOML file
    config_file = sys.argv[1] if len(sys.argv) > 1 else "mcp_servers_config.toml"

    try:
        # Load server configuration
        config = mcpserver_config.Config.from_toml(config_file)
        server_config = config["ipinfo"]

        # Get API token from configured token file
        token_file_path = server_config.get_token_file_path()
        token = get_ipinfo_token(token_file_path)

        # Extract host and port from configuration
        if server_config.host:
            import urllib.parse
            parsed = urllib.parse.urlparse(server_config.host)
            hostname = parsed.hostname or "localhost"
        else:
            hostname = "localhost"

        port = server_config.port if server_config.port else 5001

        print(f"🚀 Starting IPInfo MCP Server...")
        print(f"📡 Server: {mcp.name}")
        print(f"🔗 Running on: http://{hostname}:{port}")
        print(f"📝 Config file: {config_file}")
        print(f"🔑 API Token: {'*' * (len(token) - 4) + token[-4:]}")

        # Start the server
        mcp.run(transport='http', host=hostname, port=port)

    except ValueError as e:
        print(f"❌ Error: {e}")
        sys.exit(1)
    except KeyError:
        print(f"❌ Error: Server 'ipinfo' not found in {config_file}")
        print("   Please ensure [[servers]] section with name='ipinfo' exists in the config file.")
        sys.exit(1)
    except FileNotFoundError:
        print(f"❌ Error: Config file '{config_file}' not found")
        print("   Usage: python ipinfo_server.py [config_file]")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error starting server: {e}")
        sys.exit(1)
