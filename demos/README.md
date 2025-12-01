# Demo Scripts

This directory contains demonstration scripts showing how to use the Ollama-FastMCP Wrapper. Each demo is available in both shell script (`.sh`) and Python (`.py`) formats.

## Prerequisites

1. **Ensure Ollama is running:**
   ```bash
   ollama serve
   ```

2. **Start the wrapper in API mode:**
   ```bash
   python ollama_wrapper.py api
   ```

## Available Demos

### 1. Basic Chat (`basic_chat.*`)
Demonstrates simple chat interaction without MCP tools.

**Shell:**
```bash
./demos/basic_chat.sh
```

**Python:**
```bash
python demos/basic_chat.py
```

**What it shows:**
- Sending chat messages without tool usage
- Pure Ollama conversation

---

### 2. Math Operations (`math_operations.*`)
Demonstrates using the math MCP server for calculations.

**Shell:**
```bash
./demos/math_operations.sh
```

**Python:**
```bash
python demos/math_operations.py
```

**What it shows:**
- Connecting to an MCP server
- Using math tools (add, multiply)
- Tool-augmented LLM reasoning

---

### 3. IPInfo Lookup (`ipinfo_lookup.*`)
Demonstrates IP geolocation using the IPInfo MCP server.

**Shell:**
```bash
./demos/ipinfo_lookup.sh
```

**Python:**
```bash
python demos/ipinfo_lookup.py
```

**What it shows:**
- Connecting to IPInfo server
- Looking up IP addresses
- Querying preset organizations
- Listing available organizations

---

### 4. Server Management (`server_management.*`)
Demonstrates server lifecycle: connect, list, disconnect.

**Shell:**
```bash
./demos/server_management.sh
```

**Python:**
```bash
python demos/server_management.py
```

**What it shows:**
- Listing connected servers
- Connecting to multiple servers
- Viewing server tools dynamically
- Disconnecting servers
- Server state management

---

### 5. History Management (`history_management.*`)
Demonstrates conversation persistence.

**Shell:**
```bash
./demos/history_management.sh
```

**Python:**
```bash
python demos/history_management.py
```

**What it shows:**
- Maintaining conversation context
- Retrieving conversation history
- Saving conversations to disk
- Loading saved conversations

---

### 6. Temperature Test (`temperature_test.*`)
Compares response quality with different temperature settings and displays performance metrics.

**Shell:**
```bash
./demos/temperature_test.sh
```

**Python:**
```bash
python demos/temperature_test.py
```

**What it shows:**
- How temperature affects response consistency
- Low temperature (0.1): Factual, deterministic
- Default temperature (0.2): Consistent but natural
- Medium temperature (0.8): Varied, conversational
- High temperature (1.5): Creative, less predictable
- Performance metrics in tabular format:
  - **TPS (Tokens Per Second)**: Generation speed
  - **Tokens**: Number of completion tokens
  - **Time**: Total request time including network overhead
  - **Total Duration**: Ollama processing time
- Best practices for different use cases

---

### 7. Enhanced Temperature Test - Multi-Model (`temperature_test_multi_model.*`)
Compare multiple Ollama models across different temperature settings with interactive configuration and persistent results.

**Shell:**
```bash
./demos/temperature_test_multi_model.sh
```

**Python:**
```bash
python demos/temperature_test_multi_model.py [prompt_file_or_text]

# Examples:
python demos/temperature_test_multi_model.py                         # Interactive prompts
python demos/temperature_test_multi_model.py "Custom prompt"         # Direct prompt
python demos/temperature_test_multi_model.py coreference_resolution.txt  # Just filename (auto-searches demos/prompts/)
python demos/temperature_test_multi_model.py demos/prompts/myfile.txt    # Relative path
```

**Interactive Configuration:**
1. **Prompt Selection**: Use default, enter custom text, or load from file
2. **Model Selection**: Choose specific models or test all installed models
3. **Temperature Selection**: Select from 8 predefined temperatures or choose custom values
4. **Output File**: Auto-generated timestamp filename or custom name

**What it shows:**
- **Interactive model selection**: Choose which models to test from your installed Ollama models
- **Interactive temperature selection**: Choose from 8 temperature options (0.0 to 2.0) or use defaults
- **Custom prompts**: Load prompts from files or enter directly
- **Multi-model comparison**: See how different models perform at the same temperature
- **Comprehensive metrics**: TPS, tokens, timing for each model/temperature combination
- **Persistent results**:
  - JSON output with complete test metadata and results
  - Auto-generated Markdown report with tables and detailed responses
  - Timestamps (start/end) and total duration tracking
- **Three result views**:
  1. Results grouped by model (see each model across all temperatures)
  2. Cross-model comparison (compare all models at each temperature)
  3. Full responses for detailed analysis
- **Summary statistics**: Fastest model, average TPS, response length variance
- **Model insights**: Compare small vs. large models, different families (llama, qwen, phi3, etc.)
- Uses the `/models` API endpoint to dynamically discover available models

**Output Files:**
- `YYYYMMDD_HHMMSS_multi_test_comparison.json` - Complete test data in JSON format
- `YYYYMMDD_HHMMSS_multi_test_comparison.md` - Human-readable Markdown report

**Example use cases:**
- Find the fastest model for your hardware
- Compare accuracy vs. speed tradeoffs
- Test which model works best for your specific use case
- Evaluate different quantization levels (Q4 vs. Q5 vs. Q8)
- Build a database of model performance benchmarks
- Document model selection decisions for projects

**Running long tests without hangups:**

For extensive multi-model tests that may take hours, use `nohup` to prevent terminal disconnection from interrupting the test:

```bash
# Run in background, immune to hangups
nohup python demos/temperature_test_multi_model.py &

# Or with uv
nohup uv run python demos/temperature_test_multi_model.py &

# Monitor progress in real-time
tail -f nohup.out

# Check if still running
ps aux | grep temperature_test_multi_model
```

**Benefits:**
- Test continues even if SSH connection drops
- Safe to close terminal or log out
- Results automatically saved incrementally (survives interruptions)
- Output captured in `nohup.out`

**Note:** Results are saved after each individual test completes, so even if the process is interrupted, partial results are preserved in the JSON output file.

---

## Python Dependencies

Python demos require the `requests` library:

```bash
pip install requests
```

Or if using `uv`:

```bash
uv pip install requests
```

## Shell Dependencies

Shell demos require `jq` for JSON formatting and `bc` for calculations:

**Ubuntu/Debian:**
```bash
sudo apt-get install jq bc
```

**macOS:**
```bash
brew install jq
# bc is typically pre-installed on macOS
```

## Notes

- **Python demos automatically read configuration** from `wrapper_config.toml` (host and port)
- Shell demos assume the wrapper is running on `http://localhost:8000`
  - Modify the `HOST` variable in shell scripts if using a different address
- The demos will create a `demo_conversation.json` file in the current directory
- Each demo is self-contained and can be run independently
- Check the wrapper logs for detailed server connection information

## Troubleshooting

**Connection refused:**
- Ensure the wrapper is running: `python ollama_wrapper.py api`

**Server connection fails:**
- Check that MCP servers are configured in `mcp_servers/mcp_servers_config.toml`
- Verify the servers are enabled (`enabled = true`)

**IPInfo errors:**
- Ensure you've created `mcp_servers/mcp_tokens.toml` with your API token
- Verify the token is valid at https://ipinfo.io/account

**Tool calls not working:**
- Make sure you've connected to the server first
- Verify the `mcp_server` field in chat requests matches the connected server name
