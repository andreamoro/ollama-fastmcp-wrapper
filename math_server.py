from fastmcp import FastMCP

# Initialize FastMCP with a name
mcp = FastMCP("math")

@mcp.tool()
def add(a: int, b: int) -> int:
    """Adds or sum two integer numbers."""
    return a + b

@mcp.tool()
def multiply(a: int, b: int) -> int:
    """Multiplies two integer numbers."""
    return a * b

# Start the MCP server
if __name__ == "__main__":
    # Initialize and run the server

    # use this for local testing
    # it requires the client to know the local file path where the server is running
    # mcp.run(transport='stdio')

    # use this for running the server over HTTP
    # it requires the client to connect to the server using the HTTP URL
    mcp.run(transport='http', host='localhost', port=5000)
