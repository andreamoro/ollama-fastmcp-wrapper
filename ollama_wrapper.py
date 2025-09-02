#!/usr/bin/env python3
"""
A proxy service that exposes FastMCP tools to Ollama via API endpoint
Usage:
- uv run ollama_wrapper.py api gemma3:1b
or
- python ollama_wrapper.py api llama3.2:3b

Then use:
curl http://localhost:8000/chat
    -H "Content-Type: application/json"
    -d '{"message": "Hello", "model": "llama3.2:3b", "mcp_server": ""}'
"""

import argparse
from pathlib import Path
import mcpserver_config
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Tuple, Optional
import ollama
import json
import uvicorn
from fastmcp import Client
from fastmcp.client.transports import StdioTransport
from contextlib import asynccontextmanager
from enum import Enum

config = None
# Global FastMCP clients storage
fastmcp_clients = {} # client that have been already initialised
fastmcp_tools = {}
fastmcp_tools_expanded = []

class TransportMethod(Enum):
    HTTP = 1
    STDIO = 2

class ChatRequest(BaseModel):
    message: str
    model: str = "gemma3:1b"
    mcp_server: Optional[str] = ""

class ChatResponse(BaseModel):
    response: str
    tools_used: List[str] = []

class MessageHistory:
    def __init__(self,
                system_prompt="You are a helpful assistant.",
                max_messages=20,
                summarise_model="gemma3:1b"
            ):
        self.system_prompt = system_prompt
        self.max_messages = max_messages
        self.summarise_model = summarise_model
        self.messages = [{"role": "system", "content": system_prompt}]
        self.summary = None

    def add(self, role, content):
        self.messages.append({"role": role, "content": content})
        self._trim()

    def _summarise(self):
        history_text = "\n".join(
            f"{m['role'].capitalize()}: {m['content']}"
            for m in self.messages if m['role'] in ["user", "assistant"]
        )
        response = ollama.chat(
            model=self.summarise_model,
            messages=[
                {"role": "system", "content": "Summarise the following conversation briefly."},
                {"role": "user", "content": history_text}
            ]
        )
        return response["message"]["content"]

    def _trim(self):
        if len(self.messages) > self.max_messages:
            self.summary = self._summarise()
            self.messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "system", "content": f"Conversation so far: {self.summary}"}
            ] + self.messages[-(self.max_messages // 2):]

    def get(self):
        return self.messages

    def reset(self):
        self.messages = [{"role": "system", "content": self.system_prompt}]
        self.summary = None

    def save(self, path):
        """Save message history to a JSON file for restart the conversation."""
        with open(path, "w") as f:
            json.dump({"messages": self.messages, "summary": self.summary}, f, indent=2)

    def load(self, path):
        """Load previous conversation history from a JSON file."""
        with open(path, "r") as f:
            data = json.load(f)
        self.messages = data.get("messages", [])
        self.summary = data.get("summary", None)

class OllamaWrapper:
    """A wrapper to connect Ollama with FastMCP servers and expose an API."""

    def __init__(self,
                model="llama3.2:3b",
                history=None,
                transport:TransportMethod=TransportMethod.HTTP
            ):
        self.model = model
        self.message_history = history or MessageHistory()
        self.app = FastAPI(title="Ollama-FastMCP Wrapper", version="1.0.0", lifespan=OllamaWrapper._lifespan)
        self.transport = transport

        @self.app.get("/list_tools")
        async def list_tools(server_name: str):
            """List available tools for a given FastMCP server"""
            return await self._list_tools(server_name=server_name)

        @self.app.get("/servers")
        async def list_servers() -> dict:
            """List available FastMCP servers"""
            return await self._list_servers()

        @self.app.post("/connect/{server_name}")
        async def connect_server(server_name: str) -> dict:
            """Connect a FastMCP server."""
            return await self._connect_server(server_name)

        @self.app.post("/disconnect/{server_name}")
        async def disconnect_server(server_name: str) -> dict:
            """Disconnect from a FastMCP server."""
            return await self._disconnect_server(server_name)

        @self.app.post("/chat", response_model=ChatResponse)
        async def chat(request: ChatRequest):
            return await self.chat(request)

        @self.app.post("/load_history/{file_name}")
        async def load_history(file_name: str):
            """Load previously saved message history from the specified file."""
            return await self._load_history(file_name)

        @self.app.post("/save_history/{file_name}")
        async def save_history(file_name: str):
            """Save current message history to the specified file."""
            return await self._save_history(file_name, overwrite=False)

        @self.app.post("/overwrite_history/{file_name}")
        async def overwrite_history(file_name: str):
            """Save current message history overwriting the specified file."""
            return await self._save_history(file_name, overwrite=True)

    @staticmethod
    @asynccontextmanager
    async def _lifespan(app: FastAPI):
        """Lifespan event to initialise FastMCP clients"""
        # This is occurring at startup and exit
        # It can be used for preloading or initializing resources
        print("🔗 Initialising FastMCP clients...")

        yield
        # Clean up FastMCP connections on shutdown
        for server_name, client in fastmcp_clients.items():
            try:
                await client.disconnect() # TO-DO: check for possible bug
                print(f"✅ Disconnected from FastMCP server: {server_name}")
            except Exception as e:
                print(f"⚠️  Error disconnecting from {server_name}: {e}")

    async def initialise_mcp_client(
            self,
            server_name: str
            ) -> Tuple[Client, List[Dict[str, Any]]]:
        """Initialise FastMCP client connection or return existing if any."""
        if server_name is None or len(server_name) == 0 or server_name not in config:
            raise ValueError(f"Unknown FastMCP server: {server_name}")

        # Return existing client if already initialised
        if server_name in fastmcp_clients:
            return fastmcp_clients[server_name], fastmcp_tools[server_name]

        # Initialize new client if configuration exists
        mcp_server = config[server_name]

        if not mcp_server.enabled:
            raise HTTPException(status_code=500, detail=f"Failed to connect. FastMCP server {server_name} is not enabled.")

        try:
            match self.transport:
                case TransportMethod.STDIO:
                    # Use StdioTransport to spawn a server for local testing
                    # This allows the client to communicate with the FastMCP server
                    # via standard input/output.
                    #
                    # Local MCP Server Python file must have a mcp.run(transport='stdio') in the init
                    if not mcp_server.command or not mcp_server.args:
                        raise ValueError(f"Invalid command or args for StdioTransport in the {server_name} server config.")

                    transport = StdioTransport(command=mcp_server.command, args=mcp_server.args)
                    client = Client(transport)
                case TransportMethod.HTTP:
                    # To simulate what would happen in a global network.
                    # When using the HTTP transport, a valid Client along with protocol and port
                    # should have been specified in the config.
                    # The Server is not spawn at run-time (hence you have to initiate manually).
                    if not mcp_server.host:
                        raise ValueError(f"Invalid host for HTTP transport in the {server_name} server config.")

                    client = Client(mcp_server.host)
                case _:
                    raise ValueError(f"Unsupported transport method: {self.transport}")

            # async with client:
            #     # Ping to ensure connection works
            #     await client.ping()
            #     print(f"{server_name} ping done.")

            tools_list = []
            async with client:
                # Get available tools
                tools_list = await client.list_tools()

            # Convert FastMCP tools to Ollama format
            ollama_tools = []
            for tool in tools_list:
                ollama_tool = {
                    "type": "function",
                    "function": {
                        "name": f"{server_name}::{tool.name}", # Include server name as a namespace to disambiguate
                        "description": tool.description,
                        "parameters": tool.inputSchema if hasattr(tool, 'inputSchema') else {},
                    },
                }
                ollama_tools.append(ollama_tool)

            fastmcp_clients[server_name] = client
            fastmcp_tools[server_name] = ollama_tools

            # Update a global variable with all tools from all MCP servers
            global fastmcp_tools_expanded
            fastmcp_tools_expanded = await self.__get_full_list_mcp_tools()

            return client, ollama_tools

        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to connect to FastMCP server {server_name}: {e}")

    async def __call_fastmcp_tool(
            self,
            server_name: str,
            tool_name: str,
            arguments: Dict[str, Any]
            ) -> Any:
        """
        This function handles calling a tool on a specified FastMCP server
        and returns the result."""

        if server_name not in fastmcp_clients:
            raise HTTPException(
                status_code=400,
                detail=f"FastMCP server {server_name} not initialised"
                )

        client = fastmcp_clients[server_name]
        try:
            # Use FastMCP's simplified tool calling
            async with client:
                result = await client.call_tool(tool_name, arguments)

            # FastMCP returns results in a more direct format
            if hasattr(result, 'content'):
                return result.content
            elif isinstance(result, dict) and 'content' in result:
                return result['content']
            else:
                return str(result)

        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"FastMCP tool call failed: {e}"
            )

    async def __coerce_parameters(
            self,
            server_name: str,
            tool_name: str,
            parameters: Dict[str, Any]
        ) -> Dict[str, Any]:
        """
        Coerce parameters to their expected types based on
        the schema defined by the MCP Server."""

        # Get the tool specification, raising errors if the server or tool is not found
        if server_name not in fastmcp_tools:
            raise HTTPException(
                status_code=400,
                detail=f"FastMCP server {server_name} not initialised"
                )

        tools = fastmcp_tools[server_name]
        tool = next((t for t in tools if t['function']['name'] == f"{server_name}::{tool_name}"), None)
        if not tool:
            raise HTTPException(
                status_code=404,
                detail=f"Tool {tool_name} not found in server {server_name}"
                )

        spec = tool.get('function').get('parameters').get('properties')
        coerced_params = {}
        for key, value in parameters.items():
            expected_type = spec.get(key, {}).get("type")
            if expected_type == "integer":
                try:
                    coerced_params[key] = int(value)
                except Exception:
                    pass
            elif expected_type == "float":
                try:
                    coerced_params[key] = float(value)
                except Exception:
                    pass
            elif expected_type == "string":
                coerced_params[key] = str(value)
            elif expected_type == "boolean":
                if isinstance(value, str):
                    coerced_params[key] = value.lower() in ("true", "1", "yes")
                else:
                    coerced_params[key] = bool(value)
            else:
                coerced_params[key] = value
        # Return coerced parameters
        return coerced_params

    async def __get_full_list_mcp_tools(self) -> List[Dict[str, Any]]:
        """Get a flatten tools list from the available MCP server."""
        try:
            if len(fastmcp_tools) > 0:
                return [tools for mcp in fastmcp_tools for tools in fastmcp_tools[mcp]]
            return []
        except Exception as e:
            print(f"Error fetching tools: {e}")
            return []

    async def chat(self, request: ChatRequest) -> ChatResponse:
        """
        Chat endpoint that uses FastMCP tools with Ollama

        This function:
        1. Connects to specified FastMCP server if any
        2. Determine if any tools are attached and send the message to Ollama to process
        3. If the model determine the need of the tools, call them via FastMCP and get results
        3. Sends user message + tools to Ollama model if available, or just the message if not
        4. If model requests tools, executes them via FastMCP
        5. Returns final response
        """

        # STEP 1: Initialise an MCP server if specified
        # This is optional, if no server is specified, it will just use Ollama
        if request.mcp_server is not None and len(request.mcp_server) != 0:
            try:
                await self.initialise_mcp_client(request.mcp_server)
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

        tools_used = []
        self.message_history.add("user", request.message)

        # messages = [{"role": "user", "content": request.message}]

        try:
            # STEP 2: Call to Ollama with available tools if any
            if len(fastmcp_tools_expanded) > 0:
                response = ollama.chat(
                    model=request.model,
                    messages=self.message_history.get(),
                    tools=fastmcp_tools_expanded,
                    options={'temperature': 0}
                )

                # STEP 3: Handle tool calls if requested
                if 'tool_calls' in response.get('message', {}):
                    tool_messages = []
                    tool_messages.append(response['message'])

                    for tool_call in response['message']['tool_calls']:
                        tool_name = tool_call['function']['name']
                        tool_args = tool_call['function']['arguments']

                        print(f"🔧 FastMCP calling tool: {tool_name} with args: {tool_args}")
                        tools_used.append(tool_name)

                        # Resolve the server name and tool name by checking for the namespace
                        if "::" in tool_name:
                            server_name, tool_name = tool_name.split("::", 1)

                            # I've noticed that often LLMs don't respect parameters signature
                            # Coercing parameters to expected types.
                            # The coercion function might require additional checks and improvements
                            tool_args = await self.__coerce_parameters(server_name, tool_name, tool_args)

                            # Execute tool via FastMCP
                            tool_result = await self.__call_fastmcp_tool(server_name, tool_name, tool_args)

                            # Add tool result to conversation
                            tool_messages.append({
                                "role": "tool",
                                "content": str(tool_result),
                                "tool_call_id": tool_call.get('id', '')
                            })

                    # Get final response from Ollama
                    response_w_tools = ollama.chat(
                        model=request.model,
                        messages=tool_messages
                    )

                    # Response with tools
                    self.message_history.add("assistant", response_w_tools['message']['content'])
                    return ChatResponse(
                        response=response_w_tools['message']['content'],
                        tools_used=tools_used
                    )
                else:
                    # Direct response, no tools used
                    self.message_history.add("assistant", response['message']['content'])
                    return ChatResponse(
                        response=response['message']['content'],
                        tools_used=[]
                    )
            else:
                # No FastMCP tools available, just return Ollama response
                response = ollama.chat(
                    model=request.model,
                    messages=self.message_history.get()
                )
                self.message_history.add("assistant", response['message']['content'])
                return ChatResponse(
                    response=response['message']['content'],
                    tools_used=[]
                )

        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Chat failed: {e}")

    async def _list_tools(
            self,
            server_name: str
            ) -> dict:
        """List available tools for a given FastMCP server"""
        try:
            client, tools = await self.initialise_mcp_client(server_name)
            return {
                "server": server_name,
                "tools": [tool['function']['name'] for tool in tools],
                "detailed_tools": tools
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    async def _list_servers(self) -> dict:
        """List available FastMCP servers"""
        result = {}
        for name, _ in config.servers.items():
            result[name] = {"enabled": True if fastmcp_clients.get(name, False) else False}
        return {"servers": result}

    async def _connect_server(self, server_name: str) -> dict:
        """Connect a FastMCP server.

        Example usage:
        ```bash
        curl -X POST http://0.0.0.0:8000/connect/math
        """
        try:
            await self.initialise_mcp_client(server_name)
            # # update the global tools list
            # global fastmcp_tools_expanded
            # fastmcp_tools_expanded = await __get_full_list_mcp_tools()
            print(f"✅ FastMCP {server_name} server initialised")
            return {"status_code": 200, "detail": f"{server_name} server successfully connected."}

        except Exception as e:
            print(f"⚠️  Warning: Could not initialise {server_name} FastMCP server: {e}")
            raise HTTPException(status_code=400, detail=f"Could not initialise {server_name}.")

    async def _disconnect_server(self, server_name: str) -> dict:
        """Disconnect from a FastMCP server.

        Example usage:
        ```bash
        curl -X POST http://0.0.0.0:8000/disconnect/math
        """
        if server_name in fastmcp_clients:
            try:
                await fastmcp_clients[server_name].close()
                del fastmcp_clients[server_name]
                del fastmcp_tools[server_name]
                print(f"Disconnected from {server_name}")
                return {"status_code": 200, "detail": f"{server_name} server successfully disconnected."}
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Disconnect failed: {e}")
        else:
            raise HTTPException(status_code=404, detail=f"Server {server_name} not connected")

    async def _load_history(self, file_name: str) -> None:
        """Load message history from a file."""
        dir = Path('messages_history')

        if not dir.exists():
            raise HTTPException(status_code=500, detail="Load history failed: there are no conversation histories available.")

        file_path = dir.joinpath(file_name)
        if file_path.suffix != ".json":
            file_name = str(file_path.with_suffix(".json"))
            file_path = Path(file_name)

        try:
            self.message_history.load(file_name)
            return {"status_code": 200, "detail": f"History loaded from {file_name}"}
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Load history failed: {e}")

    async def _save_history(self, file_name: str, overwrite: bool=False) -> None:
        """Save current message history to a file."""
        # Parse the file path and change the extension to json if necessary
        dir = Path('messages_history')
        if not dir.exists():
            dir.mkdir()

        file_path = dir.joinpath(file_name)
        if file_path.suffix != ".json":
            file_name = str(file_path.with_suffix(".json"))
            file_path = Path(file_name)

        if file_path.exists() and not overwrite:
            raise HTTPException(status_code=500, detail=f"Save history failed: file {file_name} already exists. Use overwrite option to save using the same file name.")

        try:
            self.message_history.save(file_name)
            return {"status_code": 200, "detail": f"History saved to {file_name}"}
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Save history failed: {e}")

    def run_api(self, host:str="0.0.0.0", port:int=8000):
        """Run as an API server.
        In this mode, the wrapper allow the possibility to connect/disconnect
        FastMCP servers at runtime via API calls.

        Arguments:
            message (str): The user message to send to the model.
            model (str): The Ollama model to use (default: "llama3.2:3b").
            mcp_server (str, optional): The FastMCP server to connect to (default: None).

        Example usage:
            ```bash
            curl http://localhost:8000/chat -H "Content-Type: application/json"
            -d '{"message": "I need to sum these two numbers: 5 and 10.
            I want the sum output to be multiplied by 20 and get the final result.
            Use multiple tools if necessary.", "model": "llama3.2:3b", "mcp_server": ""}'
            ```
        """
        if host == "" or host == "localhost" or host == "0.0.0.0":
            host = "127.0.0.1"
        if port <= 0 or port > 65535:
            port = 8000

        print("🚀 Starting Ollama-FastMCP Wrapper...")
        print("📡 This service bridges Ollama models with FastMCP servers")
        print(f"🔗 Available at: http://{host}:{port}")
        print(f"📚 Docs at: http://{host}:{port}/docs")

        uvicorn.run(self.app, host=host, port=port)

    def run_cli(self):
        """Run as a CLI chat interface.
        In this mode, the wrapper works as a simple chat interface.

        TO-DO: implement MCP connection logic.

        Example usage:
            ```bash
            python ollama_wrapper.py
            ```
        """
        import asyncio
        import re

        async def chat_session():
            print("💬 Starting Ollama-FastMCP Wrapper CLI...")
            print("Type:")
            print("- '/exit' or '/quit' to close the CLI.")
            print("- '/load <file_name>' to load a previous conversation from a file.")
            print("- '/save or /overwrite <file_name>' to save the current conversation to a file.")
            while True:
                user_input = input("You: ")
                if user_input.lower() in {"/exit", "/quit"}:
                    print("👋 Goodbye!")
                    break

                file_match = re.search(r"/(load|save|overwrite)\s+([^\s]+)", user_input)
                if file_match is None or not file_match.group(2).strip():
                    if "/load" in user_input.lower() or "/save" in user_input.lower():
                        print("⚠️  Error: a file name was not specified.")
                        continue
                else:
                    file_name = file_match.group(2)
                    if file_name == "":
                        print("⚠️  Error: a file name was not specified.")
                        continue
                    else:
                        if user_input.lower().find("load") > 0:
                            ret = await self._load_history(file_name=file_name)
                        elif user_input.lower().find("save") > 0:
                            ret = await self._save_history(file_name=file_name, overwrite=False)
                        elif user_input.lower().find("overwrite") > 0:
                            ret = await self._save_history(file_name=file_name, overwrite=True)
                        print(ret)
                        continue

                if user_input != "":
                    try:
                        response = await self.chat(
                                ChatRequest(message=user_input, model=self.model)
                                )
                        print(f"Bot: {response.response}")
                    except Exception as e:
                        print(f"⚠️  Error: {e}")

        asyncio.run(chat_session())

if __name__ == "__main__":
    """Start the Ollama-FastMCP Wrapper.

    TO-DO: Add argument parser for configuring the API host.
    TO-DO: Add logging support.

    Example:
        uv python run ollama_wrapper.py [mode] [model]
        or
        uv python run ollama_wrapper.py cli gemma3:1b
    """
    # Create the argument parser
    parser = argparse.ArgumentParser(
        description='A proxy service that bridges Ollama with FastMCP thus allowing MCP servers to be plugged and used locally.',
        epilog='Example usage: python ollama_wrapper.py cli'
    )

    # Positional argument for mode
    # Without a double parameter and without a dash at the beginning
    # the parser make it positional.
    parser.add_argument('mode',
                        choices=['api', 'cli'],
                        nargs='?',
                        default='api',
                        help='Operation mode'
                    )

    parser.add_argument('model',
                        type=str,
                        nargs='?',
                        help='An Ollama model previously downloaded.')

    # Optional arguments
    parser.add_argument('-c', '--config',
                        type=str,
                        nargs='?',
                        default='server_config.toml',
                        help='Specify the configuration file path.'
                    )

    parser.add_argument('--history-file',
                        type=str,
                        nargs='?',
                        default='',
                        help='Specify an existing history file path to load conversation history.'
                    )

    parser.add_argument('-o', '--overwrite-history',
                        action='store_true',
                        help='Overwrite existing history file on exit (if --history-file is set).'
                    )

    parser.add_argument('-t', '--transport',
                        choices=['HTTP', 'STDIO'],
                        nargs='?',
                        default='HTTP',
                        help='Transport method to connect to FastMCP servers (default: http).')

    # Parse the arguments
    args = parser.parse_args()

    config = mcpserver_config.Config.from_toml(args.config)
    wrapper = OllamaWrapper(
        model=args.model,
        transport=TransportMethod[args.transport], # TransportMethod.HTTP
        history=args.history_file
        )

    # while True:
    #     mode = input("Start in [api/cli]? ").strip().lower()
    #     if mode in {"api", "cli"}:
    #         break
    #     print("❌ Invalid choice. Please type 'api' or 'cli'.")

    if args.mode == "api":
        wrapper.run_api()
    elif args.mode == "cli":
        wrapper.run_cli()
    else:
        raise ValueError('Invalid mode type specified.')
