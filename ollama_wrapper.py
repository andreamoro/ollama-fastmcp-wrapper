#!/usr/bin/env python3
"""
A proxy service that exposes FastMCP tools to Ollama via API endpoint
Usage:
- uv run ollama_wrapper.py api llama3.2:3b
or
- python ollama_wrapper.py api llama3.2:3b --host 0.0.0.0 --port 8080

Then use:
curl http://localhost:8000/chat
    -H "Content-Type: application/json"
    -d '{"message": "Hello", "model": "llama3.2:3b", "mcp_server": ""}'
"""

import argparse
from pathlib import Path
from mcp_servers import mcpserver_config
import wrapper_config
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
import aiofiles

mcp_config = None
# Global FastMCP clients storage
fastmcp_clients = {} # client that have been already initialised
fastmcp_tools = {}
fastmcp_tools_expanded = []

class TransportMethod(Enum):
    HTTP = 1
    STDIO = 2

class ChatRequest(BaseModel):
    message: str
    model: str = "llama3.2:3b"
    mcp_server: Optional[str] = ""
    temperature: Optional[float] = None  # Optional temperature override (0.0-2.0)
    stateless: bool = False  # If True, don't persist message to history (one-shot mode)

class ChatResponse(BaseModel):
    response: str
    tools_used: List[str] = []
    metrics: Optional[dict] = None  # Ollama metrics: tokens, timing, etc.

class MessageHistory:
    def __init__(self,
                system_prompt="You are a helpful assistant.",
                max_messages=20,
                summarise_model="llama3.2:3b"
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

    async def save(self, path):
        """Save message history to a JSON file for restart the conversation (async)."""
        data = json.dumps({"messages": self.messages, "summary": self.summary}, indent=2)
        async with aiofiles.open(path, "w") as f:
            await f.write(data)

    async def load(self, path):
        """Load previous conversation history from a JSON file (async)."""
        async with aiofiles.open(path, "r") as f:
            content = await f.read()
        data = json.loads(content)
        self.messages = data.get("messages", [])
        self.summary = data.get("summary", None)

class OllamaWrapper:
    """A wrapper to connect Ollama with FastMCP servers and expose an API."""

    def __init__(self,
                model="llama3.2:3b",
                history=None,
                transport:TransportMethod=TransportMethod.HTTP,
                history_file:str="",
                overwrite_history:bool=False,
                config_temperature:float=0.2,
                max_history_messages:int=20
            ):
        self.model = model
        self.message_history = history or MessageHistory(max_messages=max_history_messages)
        self.history_file = history_file
        self.overwrite_history = overwrite_history
        self.config_temperature = config_temperature  # Default temperature from config
        self.transport = transport

        # Create lifespan with access to self
        @asynccontextmanager
        async def lifespan_with_history(app: FastAPI):
            """Lifespan event to initialise FastMCP clients and load history"""
            print("🔗 Initialising FastMCP clients...")

            # Load history from file if specified
            if self.history_file:
                try:
                    await self.message_history.load(self.history_file)
                    print(f"📖 Loaded conversation history from {self.history_file}")
                except FileNotFoundError:
                    print(f"📝 History file {self.history_file} not found, starting fresh")
                except Exception as e:
                    print(f"⚠️  Warning: Could not load history file: {e}")

            yield

            # Clean up FastMCP connections on shutdown
            for server_name, client in fastmcp_clients.items():
                try:
                    await client.disconnect()
                    print(f"✅ Disconnected from FastMCP server: {server_name}")
                except Exception as e:
                    print(f"⚠️  Error disconnecting from {server_name}: {e}")

        self.app = FastAPI(title="Ollama-FastMCP Wrapper", version="0.7.1", lifespan=lifespan_with_history)

        @self.app.get("/")
        async def root():
            """Root endpoint - lists all available API endpoints"""
            return {
                "name": "Ollama-FastMCP Wrapper",
                "version": "0.7.1",
                "description": "A proxy service that bridges Ollama with FastMCP",
                "endpoints": {
                    "GET /": "This endpoint - lists all available endpoints",

                    "# Chat": "",
                    "POST /chat": "Send a chat message (with optional MCP tools)",

                    "# History": "",
                    "GET /history": "Get current conversation history",
                    "GET /history/clear": "Clear the current conversation history",
                    "GET /history/load/{file_name}": "Load conversation history from file",
                    "GET /history/overwrite/{file_name}": "Overwrite existing conversation file",
                    "GET /history/save/{file_name}": "Save conversation history to file",

                    "# Models": "",
                    "GET /models": "List installed Ollama models with details",

                    "# Servers": "",
                    "GET /servers": "List connected servers and available servers from config",
                    "POST /servers/{server_name}/connect": "Connect to an MCP server",
                    "POST /servers/{server_name}/disconnect": "Disconnect from an MCP server",
                    "GET /servers/{server_name}/tools": "List available tools for a specific MCP server"
                },
                "chat_parameters": {
                    "message": "string (required) - The message to send",
                    "model": "string (default: 'llama3.2:3b') - Ollama model to use",
                    "mcp_server": "string (optional) - MCP server name to use tools from",
                    "temperature": "float (optional, 0.0-2.0) - Response randomness/creativity",
                    "stateless": "bool (default: false) - Don't persist message to history"
                },
                "documentation": "https://github.com/your-repo/ollama-fastmcp-wrapper"
            }

        @self.app.get("/servers")
        async def list_servers() -> dict:
            """List available FastMCP servers"""
            return await self._list_servers()

        @self.app.get("/servers/{server_name}/tools")
        async def list_server_tools(server_name: str):
            """List available tools for a given FastMCP server"""
            return await self._list_tools(server_name=server_name)

        @self.app.get("/models")
        async def list_models() -> dict:
            """List available Ollama models"""
            return await self._list_models()

        @self.app.get("/history")
        async def get_history() -> dict:
            """Get current conversation history"""
            return await self._get_history()

        @self.app.post("/servers/{server_name}/connect")
        async def connect_server(server_name: str) -> dict:
            """Connect a FastMCP server."""
            return await self._connect_server(server_name)

        @self.app.post("/servers/{server_name}/disconnect")
        async def disconnect_server(server_name: str) -> dict:
            """Disconnect from a FastMCP server."""
            return await self._disconnect_server(server_name)

        @self.app.post("/chat", response_model=ChatResponse)
        async def chat(request: ChatRequest):
            return await self.chat(request)

        @self.app.get("/history/load/{file_name}")
        async def load_history(file_name: str):
            """Load previously saved message history from the specified file."""
            return await self._load_history(file_name)

        @self.app.get("/history/save/{file_name}")
        async def save_history(file_name: str):
            """Save current message history to the specified file."""
            return await self._save_history(file_name, overwrite=False)

        @self.app.get("/history/overwrite/{file_name}")
        async def overwrite_history(file_name: str):
            """Save current message history overwriting the specified file."""
            return await self._save_history(file_name, overwrite=True)

        @self.app.get("/history/clear")
        async def clear_history():
            """Clear the current conversation history."""
            self.message_history.reset()
            return {"status": "success", "message": "Conversation history cleared"}

    async def initialise_mcp_client(
            self,
            server_name: str
            ) -> Tuple[Client, List[Dict[str, Any]]]:
        """Initialise FastMCP client connection or return existing if any."""
        if server_name is None or len(server_name) == 0 or server_name not in mcp_config:
            raise ValueError(f"Unknown FastMCP server: {server_name}")

        # Return existing client if already initialised
        if server_name in fastmcp_clients:
            return fastmcp_clients[server_name], fastmcp_tools[server_name]

        # Initialize new client if configuration exists
        mcp_server = mcp_config[server_name]

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

    async def __auto_save_history(self):
        """
        Automatically save conversation history to file if configured (async).

        Saves to the file specified in self.history_file if set.
        Respects the overwrite_history flag for whether to overwrite existing files.
        """
        if not self.history_file:
            return  # No history file configured, skip saving

        try:
            file_path = Path(self.history_file)

            # Create parent directory if it doesn't exist
            if file_path.parent and not file_path.parent.exists():
                file_path.parent.mkdir(parents=True, exist_ok=True)

            # Check if file exists and overwrite flag
            if file_path.exists() and not self.overwrite_history:
                # File exists but we're not allowed to overwrite - skip
                # This prevents overwriting on the first save
                pass

            # Save the history (async)
            await self.message_history.save(str(file_path))

        except Exception as e:
            # Don't fail the chat if history saving fails, just log it
            print(f"⚠️  Warning: Could not auto-save history: {e}")

    async def chat(self, request: ChatRequest) -> ChatResponse:
        """
        Chat endpoint that uses FastMCP tools with Ollama

        This function:
        1. Uses tools from already-connected FastMCP servers
        2. Sends user message + tools to Ollama model if available, or just the message if not
        3. If model requests tools, executes them via FastMCP
        4. Returns final response

        Note (v0.5.0+): Servers must be explicitly connected via /connect/{server_name}
        before they can be used in chat. Auto-loading has been removed.
        """

        # STEP 1: Get tools from already-connected server (if specified)
        # No auto-loading - server must be explicitly connected first
        request_tools = []
        if request.mcp_server is not None and len(request.mcp_server) != 0:
            # Check if server is already connected
            if request.mcp_server not in fastmcp_clients:
                raise HTTPException(
                    status_code=400,
                    detail=f"Server '{request.mcp_server}' is not connected. Please connect first using POST /connect/{request.mcp_server}"
                )

            # Use tools only for the requested server
            request_tools = fastmcp_tools.get(request.mcp_server, [])

        tools_used = []

        # Build message list based on stateless mode
        if request.stateless:
            # One-shot mode: only system prompt + current message
            messages = [
                {"role": "system", "content": self.message_history.messages[0]["content"]},
                {"role": "user", "content": request.message}
            ]
        else:
            # Conversational mode: use full history
            self.message_history.add("user", request.message)
            messages = self.message_history.get()

        # Determine temperature: request parameter > config > default
        temperature = request.temperature if request.temperature is not None else self.config_temperature

        try:
            # STEP 2: Call to Ollama with tools only if an MCP server was specified
            # This ensures tools don't persist from previous requests
            if len(request_tools) > 0:
                response = ollama.chat(
                    model=request.model,
                    messages=messages,
                    tools=request_tools,
                    options={'temperature': temperature}
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
                        messages=tool_messages,
                        options={'temperature': temperature}
                    )

                    # Response with tools
                    if not request.stateless:
                        self.message_history.add("assistant", response_w_tools['message']['content'])
                        # TODO: Enable auto-save in future (async implementation ready)
                        # await self.__auto_save_history()

                    # Extract metrics from response
                    metrics = self._extract_metrics(response_w_tools)

                    return ChatResponse(
                        response=response_w_tools['message']['content'],
                        tools_used=tools_used,
                        metrics=metrics
                    )
                else:
                    # Direct response, no tools used
                    if not request.stateless:
                        self.message_history.add("assistant", response['message']['content'])
                        # TODO: Enable auto-save in future (async implementation ready)
                        # await self.__auto_save_history()

                    # Extract metrics from response
                    metrics = self._extract_metrics(response)

                    return ChatResponse(
                        response=response['message']['content'],
                        tools_used=[],
                        metrics=metrics
                    )
            else:
                # No FastMCP tools available, just return Ollama response
                response = ollama.chat(
                    model=request.model,
                    messages=messages,
                    options={'temperature': temperature}
                )
                if not request.stateless:
                    self.message_history.add("assistant", response['message']['content'])
                    # TODO: Enable auto-save in future (async implementation ready)
                    # await self.__auto_save_history()

                # Extract metrics from response
                metrics = self._extract_metrics(response)

                return ChatResponse(
                    response=response['message']['content'],
                    tools_used=[],
                    metrics=metrics
                )

        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Chat failed: {e}")

    def _extract_metrics(self, response: dict) -> dict:
        """Extract timing and token metrics from Ollama response"""
        metrics = {}

        # Token counts
        if 'prompt_eval_count' in response:
            metrics['prompt_tokens'] = response['prompt_eval_count']
        if 'eval_count' in response:
            metrics['completion_tokens'] = response['eval_count']

        # Timing (nanoseconds to seconds)
        if 'prompt_eval_duration' in response:
            metrics['prompt_eval_duration_s'] = response['prompt_eval_duration'] / 1e9
        if 'eval_duration' in response:
            metrics['eval_duration_s'] = response['eval_duration'] / 1e9
        if 'total_duration' in response:
            metrics['total_duration_s'] = response['total_duration'] / 1e9

        # Calculate tokens per second
        if 'eval_count' in response and 'eval_duration' in response and response['eval_duration'] > 0:
            metrics['tokens_per_second'] = response['eval_count'] / (response['eval_duration'] / 1e9)

        return metrics

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
        """List both connected and available FastMCP servers"""
        # Build connected servers list
        connected_servers = {}
        for name in fastmcp_clients.keys():
            tools = fastmcp_tools.get(name, [])
            tool_list = []
            for tool in tools:
                tool_info = {
                    "name": tool["function"]["name"],
                    "description": tool["function"]["description"]
                }
                tool_list.append(tool_info)

            connected_servers[name] = {
                "status": "connected",
                "tools_count": len(tool_list),
                "tools": tool_list
            }

        # Build available servers list (only enabled ones from config)
        available_servers = {}
        if mcp_config:
            for server_name, server_config in mcp_config.servers.items():
                if server_config.enabled:
                    available_servers[server_name] = {
                        "enabled": True,
                        "host": server_config.host if server_config.host else None,
                        "port": server_config.port if server_config.port else None
                    }

        return {
            "connected": connected_servers,
            "available": available_servers
        }

    async def _list_models(self) -> dict:
        """List available Ollama models"""
        try:
            models_response = ollama.list()
            models_list = []

            for model in models_response['models']:
                model_info = {
                    "name": model['model'],
                    "size": model['size'],
                    "size_gb": round(model['size'] / 1e9, 2),
                    "modified_at": str(model['modified_at']),
                    "family": model['details']['family'],
                    "parameter_size": model['details']['parameter_size'],
                    "quantization": model['details']['quantization_level']
                }
                models_list.append(model_info)

            return {
                "models": models_list,
                "count": len(models_list)
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to list models: {e}")

    async def _get_history(self) -> dict:
        """
        Get current conversation history.

        Returns the in-memory conversation state including all messages and summary.

        Returns:
            dict: Contains 'messages' (list of conversation messages) and 'summary' (if any)

        Example usage:
            ```bash
            curl http://localhost:8000/history
            ```
        """
        return {
            "messages": self.message_history.get(),
            "summary": self.message_history.summary
        }

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
        """Load message history from a file (async)."""
        dir = Path('messages_history')

        if not dir.exists():
            raise HTTPException(status_code=500, detail="Load history failed: there are no conversation histories available.")

        file_path = dir.joinpath(file_name)
        if file_path.suffix != ".json":
            file_name = str(file_path.with_suffix(".json"))
            file_path = Path(file_name)

        try:
            await self.message_history.load(file_name)
            return {"status_code": 200, "detail": f"History loaded from {file_name}"}
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Load history failed: {e}")

    async def _save_history(self, file_name: str, overwrite: bool=False) -> None:
        """Save current message history to a file (async)."""
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
            await self.message_history.save(file_name)
            return {"status_code": 200, "detail": f"History saved to {file_name}"}
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Save history failed: {e}")

    def run_api(self, host:str="0.0.0.0", port:int=8000):
        """Run as an API server.
        In this mode, the wrapper allow the possibility to connect/disconnect
        FastMCP servers at runtime via API calls.

        Arguments:
            host (str): The host address to bind the server to (default: "0.0.0.0").
            port (int): The port number to bind the server to (default: 8000).

        Example usage:
            ```bash
            # Start server on default address (0.0.0.0:8000)
            python ollama_wrapper.py api

            # Start server on custom address
            python ollama_wrapper.py api --host 127.0.0.1 --port 8080

            # First, connect to an MCP server
            curl -X POST http://localhost:8000/connect/math

            # Then make a chat request with tools
            curl http://localhost:8000/chat -H "Content-Type: application/json" \\
            -d '{"message": "Add 5 and 10, then multiply the result by 20.", \\
                 "model": "llama3.2:3b", "mcp_server": "math"}'

            # Or chat without tools (pure Ollama)
            curl http://localhost:8000/chat -H "Content-Type: application/json" \\
            -d '{"message": "Hello, how are you?", \\
                 "model": "llama3.2:3b", "mcp_server": ""}'
            ```
        """
        # Validate port range
        if port <= 0 or port > 65535:
            print(f"⚠️  Invalid port {port}. Using default port 8000.")
            port = 8000

        # Display host for informational purposes
        display_host = host if host not in ("", "0.0.0.0") else "localhost"

        print("🚀 Starting Ollama-FastMCP Wrapper...")
        print("📡 This service bridges Ollama models with FastMCP servers")
        print(f"🔗 Available at: http://{display_host}:{port}")
        print(f"📚 Docs at: http://{display_host}:{port}/docs")

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

            # Validate model is specified
            if not self.model:
                print("\n❌ Error: No model specified in configuration")
                print("   Please set 'model.default' in wrapper_config.toml")
                print("\n   Example:")
                print("   model = { default = \"llama3.2:3b\", temperature = 0.2 }")
                import sys
                sys.exit(1)

            # Validate model exists in Ollama and display capabilities
            try:
                model_info = ollama.show(self.model)
                print(f"\n🤖 Model: {self.model}")
                if 'details' in model_info:
                    details = model_info['details']
                    print(f"   Family: {details.get('family', 'N/A')}")
                    print(f"   Parameters: {details.get('parameter_size', 'N/A')}")
                    print(f"   Quantization: {details.get('quantization_level', 'N/A')}")
                if 'model_info' in model_info:
                    arch = model_info['model_info'].get('general.architecture', None)
                    if arch:
                        print(f"   Architecture: {arch}")
            except Exception as e:
                print(f"\n❌ Error: Model '{self.model}' not found")
                print(f"   {e}")

                # Try to show available models and let user select
                try:
                    models = ollama.list()
                    if models['models']:
                        all_models = [m['model'] for m in models['models']]

                        # Find similar models (fuzzy match)
                        similar_models = []
                        if self.model:
                            # Extract base name (before colon) for fuzzy matching
                            model_base = self.model.split(':')[0].lower()
                            similar_models = [m for m in all_models if model_base in m.lower()]

                        # Decide which list to show for selection
                        if similar_models:
                            # Show only fuzzy matches
                            print(f"\n💡 Found {len(similar_models)} similar model(s):")
                            models_to_select = similar_models
                        else:
                            # Show all models (up to 10)
                            if len(all_models) <= 10:
                                print(f"\n💡 Available models in Ollama:")
                                models_to_select = all_models
                            else:
                                print(f"\n💡 Available models in Ollama ({len(all_models)} total, showing first 10):")
                                models_to_select = all_models[:10]

                        # Display numbered list for selection
                        for idx, model_name in enumerate(models_to_select, 1):
                            print(f"   {idx}. {model_name}")

                        # Interactive selection
                        while True:
                            try:
                                choice = input(f"\nSelect model (1-{len(models_to_select)}, or 'c' to cancel): ").strip()

                                if choice.lower() == 'c':
                                    print("\n👋 Selection cancelled")
                                    import sys
                                    sys.exit(0)

                                choice_idx = int(choice) - 1

                                if 0 <= choice_idx < len(models_to_select):
                                    self.model = models_to_select[choice_idx]
                                    print(f"✅ Selected: {self.model}")

                                    # Show model capabilities
                                    try:
                                        model_info = ollama.show(self.model)
                                        if 'details' in model_info:
                                            details = model_info['details']
                                            print(f"   Family: {details.get('family', 'N/A')}")
                                            print(f"   Parameters: {details.get('parameter_size', 'N/A')}")
                                            print(f"   Quantization: {details.get('quantization_level', 'N/A')}")
                                    except:
                                        pass  # Don't fail if we can't get details
                                    break
                                else:
                                    print(f"❌ Please enter a number between 1 and {len(models_to_select)}")
                            except ValueError:
                                print("❌ Please enter a valid number or 'c' to cancel")
                            except KeyboardInterrupt:
                                print("\n\n👋 Selection cancelled")
                                import sys
                                sys.exit(0)
                    else:
                        print("\n💡 No models installed in Ollama.")
                        print("   To download a model, run:")
                        print("   ollama pull llama3.2:3b")
                        import sys
                        sys.exit(1)
                except Exception as list_error:
                    # Can't reach Ollama server
                    print(f"\n❌ Cannot connect to Ollama server: {list_error}")
                    print("   Please ensure Ollama is running:")
                    print("   ollama serve")
                    import sys
                    sys.exit(1)

            print("\nType '/help' for available commands")
            while True:
                user_input = input("You: ")
                if user_input.lower() in {"/exit", "/quit"}:
                    print("👋 Goodbye!")
                    break

                # Handle /help command
                if user_input.lower().strip() == "/help":
                    print("\n📖 Available CLI commands:")
                    print("  /help                         - Show this help message")
                    print("  /exit or /quit                - Exit the CLI")
                    print("  /model                        - Change the current model interactively")
                    print("  /load <file_name>             - Load conversation history from a file")
                    print("  /save <file_name>             - Save conversation history to a file")
                    print("  /overwrite <file_name>        - Overwrite existing conversation file")
                    print("\n💡 Tips:")
                    print("  - Just type your message to chat with the AI")
                    print("  - Use arrow keys to navigate command history")
                    print("  - Press Ctrl+C to cancel model selection or input")
                    continue

                # Handle /model command for model selection
                if user_input.lower().strip() == "/model":
                    try:
                        models_response = ollama.list()
                        available_models = [m['model'] for m in models_response['models']]

                        if not available_models:
                            print("❌ No models installed in Ollama.")
                            continue

                        # Show current model
                        print(f"\n🔄 Current model: {self.model}")

                        # Find similar models (fuzzy match based on current model)
                        similar_models = []
                        if self.model:
                            # Extract base name (before colon) for fuzzy matching
                            model_base = self.model.split(':')[0].lower()
                            similar_models = [m for m in available_models if model_base in m.lower()]

                        # Decide which list to show for selection
                        if similar_models:
                            # Show only fuzzy matches
                            print(f"\n📋 Found {len(similar_models)} similar model(s):")
                            models_to_select = similar_models
                        else:
                            # Show all models (up to 10)
                            if len(available_models) <= 10:
                                print(f"\n📋 Available Ollama models:")
                                models_to_select = available_models
                            else:
                                print(f"\n📋 Available Ollama models ({len(available_models)} total, showing first 10):")
                                models_to_select = available_models[:10]

                        for idx, model_name in enumerate(models_to_select, 1):
                            print(f"   {idx}. {model_name}")

                        while True:
                            try:
                                choice = input(f"\nSelect model (1-{len(models_to_select)}, or 'c' to cancel): ").strip()

                                if choice.lower() == 'c':
                                    print("Model selection cancelled")
                                    break

                                choice_idx = int(choice) - 1

                                if 0 <= choice_idx < len(models_to_select):
                                    old_model = self.model
                                    self.model = models_to_select[choice_idx]

                                    # Reset conversation context when model changes
                                    self.message_history.reset()

                                    print(f"✅ Model changed: {old_model} → {self.model}")
                                    print("🔄 Conversation context reset")

                                    # Show model capabilities
                                    try:
                                        model_info = ollama.show(self.model)
                                        if 'details' in model_info:
                                            details = model_info['details']
                                            print(f"   Family: {details.get('family', 'N/A')}")
                                            print(f"   Parameters: {details.get('parameter_size', 'N/A')}")
                                            print(f"   Quantization: {details.get('quantization_level', 'N/A')}")
                                    except:
                                        pass  # Don't fail if we can't get details
                                    break
                                else:
                                    print(f"❌ Please enter a number between 1 and {len(models_to_select)}")
                            except ValueError:
                                print("❌ Please enter a valid number or 'c' to cancel")
                    except Exception as e:
                        print(f"❌ Error fetching models: {e}")
                    continue

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

    TO-DO: Add logging support.

    Example:
        uv python run ollama_wrapper.py [mode] [model]
        or
        uv python run ollama_wrapper.py cli llama3.2:3b
        or
        uv python run ollama_wrapper.py api --host 127.0.0.1 --port 8080
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
    parser.add_argument('-c', '--wrapper-config',
                        type=str,
                        nargs='?',
                        default='wrapper_config.toml',
                        help='Specify the wrapper configuration file path.'
                    )

    parser.add_argument('--mcp-config',
                        type=str,
                        nargs='?',
                        default='mcp_servers_config.toml',
                        help='Specify the MCP servers configuration file name.'
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
                        default=None,
                        help='Transport method to connect to FastMCP servers (default: from config or HTTP).')

    parser.add_argument('--host',
                        type=str,
                        nargs='?',
                        default=None,
                        help='Host address for the API server (default: from config or 0.0.0.0).')

    parser.add_argument('--port',
                        type=int,
                        nargs='?',
                        default=None,
                        help='Port number for the API server (default: from config or 8000).')

    # Parse the arguments
    args = parser.parse_args()

    # Load configurations from separate TOML files
    mcp_config = mcpserver_config.Config.from_toml(args.mcp_config)
    wrap_config = wrapper_config.WrapperConfig.from_toml(args.wrapper_config)

    # Command-line arguments take precedence over config file
    # Use config values as defaults if command-line args are not explicitly provided
    transport = args.transport if args.transport is not None else wrap_config.transport
    host = args.host if args.host is not None else wrap_config.host
    port = args.port if args.port is not None else wrap_config.port
    history_file = args.history_file if args.history_file else wrap_config.history_file
    overwrite_history = args.overwrite_history if args.overwrite_history else wrap_config.overwrite_history

    # Ensure we have valid defaults if not set in config
    transport = transport or "HTTP"
    host = host or "0.0.0.0"
    port = port or 8000

    # Get model from args or config (with fallback)
    model = args.model if args.model else (wrap_config.model.get('default') if wrap_config.model else None)

    # Get temperature from config (with default fallback)
    config_temperature = wrap_config.model.get('temperature', 0.2) if wrap_config.model else 0.2

    # Get max_history_messages from config (with default fallback)
    max_history_messages = wrap_config.max_history_messages

    wrapper = OllamaWrapper(
        model=model,
        transport=TransportMethod[transport],
        history_file=history_file,
        overwrite_history=overwrite_history,
        config_temperature=config_temperature,
        max_history_messages=max_history_messages
        )

    # while True:
    #     mode = input("Start in [api/cli]? ").strip().lower()
    #     if mode in {"api", "cli"}:
    #         break
    #     print("❌ Invalid choice. Please type 'api' or 'cli'.")

    if args.mode == "api":
        wrapper.run_api(host=host, port=port)
    elif args.mode == "cli":
        wrapper.run_cli()
    else:
        raise ValueError('Invalid mode type specified.')
