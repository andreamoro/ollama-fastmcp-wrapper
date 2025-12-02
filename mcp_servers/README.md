# MCP Servers

This directory contains FastMCP servers that can be used with the Ollama-FastMCP wrapper.

## Available Servers

### 1. Math Server (`math_server.py`)

Provides basic mathematical operations.

**Tools:**
- `add(a, b)` - Add two integers
- `multiply(a, b)` - Multiply two integers

**Usage:**
```bash
python mcp_servers/math_server.py
```

**Configuration:** Port 5000 (default)

---

### 2. IPInfo Server (`ipinfo_server.py`)

Provides IP geolocation lookup using the ipinfo.io API.

**Tools:**
- `lookup_ip(ip_address)` - Get location info for any IP address
- `lookup_organization(org_name)` - Lookup preset organization IPs
- `list_organizations()` - List all available preset organizations

**Preset Organizations (20 total):**
- google_dns, google, facebook, cloudflare_dns, amazon
- microsoft, apple, netflix, twitter, github
- italian_gov, european_parliament, bbc, cnn, nyt
- wikipedia, stackoverflow, reddit, linkedin, youtube

**Setup:**

1. Get a free API token from [ipinfo.io](https://ipinfo.io/signup)
2. Create a `.env` file in the project root:
   ```bash
   cp .env.example .env
   ```
3. Edit `.env` and add your token:
   ```
   IPINFO_TOKEN=your_actual_token_here
   ```

**Usage:**
```bash
python mcp_servers/ipinfo_server.py
```

**Configuration:** Port 5001 (default)

---

## Running Servers

Both servers read configuration from `wrapper_config.toml` in the project root.

**Start individual server:**
```bash
# Math server
python mcp_servers/math_server.py

# IPInfo server (requires .env with IPINFO_TOKEN)
python mcp_servers/ipinfo_server.py
```

**Start with custom config:**
```bash
python mcp_servers/math_server.py path/to/config.toml
```

## Creating New MCP Servers

To create a new MCP server:

1. Create a new `.py` file in this directory
2. Follow the pattern from `math_server.py` or `ipinfo_server.py`:
   - Import `FastMCP` and `mcpserver_config`
   - Initialize with `mcp = FastMCP("server_name")`
   - Define tools using `@mcp.tool()` decorator
   - Load config and start server in `if __name__ == "__main__":`
3. Add server configuration to `wrapper_config.toml`
4. Test the server independently before using with wrapper

## Configuration Format

Add your server to `wrapper_config.toml`:

```toml
[[servers]]
name = "your_server"
command = "uv"
args = ["run", "--with", "fastmcp", "/path/to/your_server.py"]
host = "http://localhost:PORT/mcp"
port = PORT
enabled = true
token_file = "mcp_tokens.toml" # Optional
```
