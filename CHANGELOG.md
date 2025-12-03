# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.6.4] - 2025-12-03

### Added
- **Temperature test utilities (`demos/temperature_test_utils.py`):**
  - Added `clean_llm_response_data()` function to handle duplicate JSON keys from malformed LLM outputs (e.g., gemma2:2b)
  - Added `append_result_to_markdown()` for progressive markdown export during tests
  - Added `convert_json_to_markdown()` standalone utility for regenerating markdown from JSON
  - Centralized format functions with format tokens (DRY principle):
    - `format_metadata_section()`
    - `format_summary_table_header()`
    - `format_summary_table_row()`
    - `format_detailed_response()`
  - Specular output structure across console and markdown formats
- **Temperature test multi-model script (`demos/temperature_test_multi_model.py`):**
  - Progressive markdown export alongside JSON during test execution
- **Architecture improvements:**
  - LLM responses cleaned at source (immediately after generation) for efficiency
  - Progressive export to both JSON and Markdown formats
  - Text formatters use tokens (h2, bullet, bold) for consistent, maintainable code

### Changed
- **Temperature test utilities (`demos/temperature_test_utils.py`):**
  - Renamed `save_results_to_json()` → `export_results_to_json()` for naming consistency
  - Refactored `format_summary_display()` to use format tokens instead of duplicated code blocks
  - Both JSON and Markdown now receive clean data (no duplicate processing)

### Fixed
- **LLM response handling**: Duplicate JSON keys in LLM outputs are now cleaned before export
  - Some models (e.g., gemma2:2b) generate malformed JSON with duplicate keys
  - Cleaning happens once at source for efficiency

## [0.6.3] - 2025-12-02

### Fixed
- **Configuration**: Removed deprecated `server_config.toml` file
  - All configuration now uses `wrapper_config.toml` exclusively
  - Updated documentation to reference the correct config file

## [0.6.2] - 2025-12-01

### Added
- **Temperature test multi-model script (`demos/temperature_test_multi_model.py`):**
  - Added `test_number` field to each test result in JSON output
  - Added `elapsed_time_readable` field to each test result for human-readable timing
  - Enhanced summary statistics with detailed timing for the fastest model:
    - Elapsed time (both seconds and readable format)
    - Total duration (both seconds and readable format)
    - Completion tokens count
    - TPS (tokens per second)
- **Temperature test utilities (`demos/temperature_test_utils.py`):**
  - Added centralized `format_summary_display()` function to avoid code duplication
  - Enhanced Markdown export with `test_number` column in summary table
  - Improved summary statistics section with organized subsections:
    - Fastest Model (with all timing details)
    - Averages (TPS and tokens)
    - Response Lengths (min/max/avg)
  - Summary formatting is now consistent between console output and Markdown export
- **Demo configuration (`demos/demo_config.py`):**
  - Added robust config parsing with regex support
  - UTF-8 BOM encoding support
  - Comment handling in config files
- **History endpoint (`ollama_wrapper.py`):**
  - Added `GET /history/clear` endpoint to reset conversation history

### Changed
- **History endpoints changed from POST to GET** (idempotent operations):
  - `POST /load_history/{file_name}` → `GET /history/load/{file_name}`
  - `POST /save_history/{file_name}` → `GET /history/save/{file_name}`
  - `POST /overwrite_history/{file_name}` → `GET /history/overwrite/{file_name}`

### Fixed
- **Temperature test multi-model script (`demos/temperature_test_multi_model.py`):**
  - Fixed output directory path to save results to `demos/test_results/` instead of root `test_results/`
  - Fixed summary statistics to include the temperature of the fastest model, not just the model name

## [0.6.1] - 2025-12-01

### ⚠️ BREAKING CHANGES

**RESTful API Endpoint Restructuring - Complete**

All server-related and history-related endpoints have been restructured to follow RESTful resource hierarchy.

**Migration Guide:**
```bash
# SERVER ENDPOINTS
# OLD (v0.5.x and earlier)
curl http://localhost:8000/list_tools?server_name=math
curl -X POST http://localhost:8000/connect/math
curl -X POST http://localhost:8000/disconnect/math

# NEW (v0.6.1+)
curl http://localhost:8000/servers/math/tools
curl -X POST http://localhost:8000/servers/math/connect
curl -X POST http://localhost:8000/servers/math/disconnect

# HISTORY ENDPOINTS
# OLD (v0.5.x and earlier)
curl -X POST http://localhost:8000/load_history/conversation.json
curl -X POST http://localhost:8000/save_history/conversation.json
curl -X POST http://localhost:8000/overwrite_history/conversation.json

# NEW (v0.6.1+)
curl http://localhost:8000/history/load/conversation.json
curl http://localhost:8000/history/save/conversation.json
curl http://localhost:8000/history/overwrite/conversation.json
curl http://localhost:8000/history/clear
```

### Changed
- **Server-related endpoints restructured** to follow RESTful resource hierarchy:
  - `GET /list_tools?server_name={name}` → `GET /servers/{server_name}/tools`
  - `POST /connect/{server_name}` → `POST /servers/{server_name}/connect`
  - `POST /disconnect/{server_name}` → `POST /servers/{server_name}/disconnect`
  - All server operations are under `/servers/{server_name}/...`
- **History-related endpoints restructured** to follow RESTful resource hierarchy:
  - `POST /load_history/{file_name}` → `GET /history/load/{file_name}`
  - `POST /save_history/{file_name}` → `GET /history/save/{file_name}`
  - `POST /overwrite_history/{file_name}` → `GET /history/overwrite/{file_name}`
  - All history operations are under `/history/...`
  - Changed from POST to GET (these are idempotent operations)

### Added
- **Root endpoint**: `GET /` returns comprehensive API documentation
  - Lists all available endpoints with descriptions
  - Documents chat parameters
  - Self-documenting API for easier discovery
  - Organized by resource groups for better clarity
- **History clear endpoint**: `GET /history/clear` to reset conversation history
  - Clears the current conversation without deleting files
  - Useful for starting fresh conversations

## [0.6.0] - 2025-12-01

### Deprecated
This version was immediately superseded by 0.6.1 which completed the RESTful restructuring.

## [0.5.1] - 2025-12-01

### Added
- **Stateless mode for one-shot requests**: New `stateless` parameter in `ChatRequest`
  - When `stateless: true`, messages are not added to conversation history
  - Useful for independent, single-turn interactions without context accumulation
  - Ideal for batch processing, API integrations, or stateless microservices
  - Default: `false` (preserves conversational behavior)
  - Example: `{"message": "Analyze this text", "model": "llama3.2:3b", "stateless": true}`

### Changed
- **Message history behavior**: Chat endpoint now supports both conversational and stateless modes
  - Stateless mode: Only system prompt + current message sent to Ollama (no history persistence)
  - Conversational mode: Full message history preserved (default)
  - Auto-save only triggered in conversational mode

## [0.5.0] - 2025-11-30

### ⚠️ BREAKING CHANGES

**Chat Endpoint No Longer Auto-Connects to MCP Servers**

Previously, the `/chat` endpoint would automatically connect to an MCP server if `mcp_server` was specified in the request. This has been removed for cleaner separation of concerns.

**Migration Guide:**
```bash
# OLD (v0.4.x) - Auto-connected on first chat request
curl -X POST http://localhost:8000/chat -d '{
  "message": "Calculate 5 + 3",
  "mcp_server": "math"  # Automatically connected if not already
}'

# NEW (v0.5.0+) - Must explicitly connect first
curl -X POST http://localhost:8000/connect/math  # Connect first!
curl -X POST http://localhost:8000/chat -d '{
  "message": "Calculate 5 + 3",
  "mcp_server": "math"
}'
```

### Changed
- **Chat endpoint no longer auto-connects** to MCP servers
  - Explicit server connection required via `/connect/{server_name}` endpoint
  - Cleaner separation of connection management from chat operations
  - More predictable behavior and better error handling
  - Prevents accidental server connections

### Added
- **Explicit connection management**: Use `/connect/{server_name}` and `/disconnect/{server_name}`
  - Better control over server lifecycle
  - Clear separation between connection and chat operations

## [0.4.0] - 2025-11-28

Initial public release with basic functionality.
