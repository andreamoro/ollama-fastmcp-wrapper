# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.10.0] - 2026-01-17

### Added
- **Per-request timeout for Ollama calls**:
  - ChatRequest now accepts optional `timeout` parameter (in seconds)
  - Allows adaptive timeout based on prompt length for long-running requests
  - Creates temporary Ollama client with custom timeout when specified
  - Useful for batch processing with variable-length prompts
- **Model selection 'list all' option**:
  - When fuzzy search shows filtered results, press 'l' to list all available models
  - Option only appears when viewing a subset of models
- **Coreference text file analysis tools** (`demos/coreference/`):
  - `coreference_test_from_textfile.py`: Analyze arbitrary text files for coreference resolution
  - `compare_textfile_results.py`: Compare results grouped by source → model → prompt → date
  - `compare_pre_post.py`: Compare pre/post fix results side by side
  - `compare_specular.py`: Mirror comparison of result pairs
  - Supports `--wrapper-host` and `--wrapper-port` CLI arguments
  - Saves full prompt in JSON output for reproducibility
  - Adaptive HTTP timeout (1.5x Ollama timeout) for clean error propagation

### Changed
- **CLI argument naming**:
  - Renamed `--host` to `--wrapper-host` for clarity (distinguishes from `--ollama-host`)
  - Renamed `--port` to `--wrapper-port` for clarity (distinguishes from `--ollama-port`)
- **Ollama label default**:
  - No longer prompts interactively for label if not provided
  - Defaults to `local-server` when not specified in config or CLI
- **Coreference utilities**:
  - `send_to_model()` now passes timeout to wrapper for per-request Ollama timeout
  - HTTP timeout set to 1.5x the Ollama timeout to allow clean error handling
  - `compare_results.py` now excludes comparison reports from file listing

## [0.9.0] - 2026-01-06

### Added
- **Ollama request timeout configuration**:
  - Added `timeout` parameter to `[ollama]` config section (default: 300 seconds)
  - Added `--ollama-timeout` CLI argument to override config value
  - Prevents wrapper from hanging indefinitely when SSH tunnels drop or remote Ollama becomes unresponsive
  - Displays timeout value at startup: `⏱ Request timeout: 300s`
- **Quick Ollama health check endpoint**:
  - Added `GET /ollama/status` endpoint with 5-second timeout for fast availability check
  - Returns immediately with status: `available`, `unavailable`, or `error`
  - `/model/list` now performs quick health check before attempting full request
  - API responds immediately instead of blocking when Ollama is unreachable
- **Ollama server-side configuration documentation** (`OLLAMA_SERVER.md`):
  - Environment variables: `OLLAMA_NUM_PARALLEL`, `OLLAMA_MAX_LOADED_MODELS`, `OLLAMA_MAX_QUEUE`, `OLLAMA_HOST`
  - Systemd configuration examples for persistent settings
  - SSH tunneling guide for remote Ollama instances
  - VRAM considerations and parallelism tuning
- **Coreference resolution testing demo** (`demos/coreference/`):
  - Comprehensive LLM evaluation framework for pronoun resolution tasks
  - Multi-language support: English, Spanish, Russian, Chinese, Italian, French, German
  - Winograd Schema Challenge-style tests requiring world knowledge and reasoning
  - Multiple prompt strategies: zero-shot, chain-of-thought reasoning, compact CoT
  - **Accurate timing metrics**:
    - Separates Ollama internal processing time from wall-clock time
    - Tracks queue wait time when running parallel tests
    - Enables accurate benchmarking even with `OLLAMA_NUM_PARALLEL=1`
  - **Comparison tool** (`compare_results.py`):
    - Side-by-side comparison of multiple test runs
    - Per-test timing breakdown with accuracy indicators
    - Section headers show accuracy percentage and prompt file used
  - JSON and Markdown export with detailed metrics per test case
  - Watcher mode for real-time test monitoring

## [0.8.0] - 2025-12-05

### Added
- **API model management**:
  - Added `GET /model` endpoint to get current session model
  - Added `GET /model/list` endpoint to list all available Ollama models
  - Added `POST /model/switch/{model_name}` endpoint to change session model and reset context
  - Model switch endpoint returns detailed response with old model, new model, and model capabilities
  - Comprehensive API documentation in `API_USAGE.md` with usage patterns and examples
- **Stateless mode enhancements**:
  - Stateless requests can now use any model without restrictions
  - Perfect for multi-model testing and experimentation
  - No context contamination between different models
- **Model validation**:
  - Stateful requests now enforce model matching to prevent context contamination
  - Clear error messages guide users to use `/model/switch` or `stateless=true`
  - Validates model exists in Ollama before switching

### Changed
- **ChatRequest model parameter**:
  - Changed from hardcoded default `"llama3.2:3b"` to optional parameter
  - Now uses session model from `wrapper_config.toml` when not specified
  - Aligns with CLI mode behavior for consistency
- **API root endpoint**:
  - Updated to include new model management endpoints
  - Improved chat parameters documentation
  - Clarified stateful vs stateless behavior

### Breaking Changes
- **Stateful requests with model parameter**:
  - Previously: Could specify any model in stateful requests (caused context contamination)
  - Now: Model parameter must match session model, or use `/model/switch` endpoint
  - **Migration**: Use `stateless=true` for multi-model queries, or `/model/switch` to change session model
- **Temperature testing scripts**: No changes needed - already use `stateless=true`

### Removed
- **Legacy endpoint cleanup**:
  - Removed duplicate `GET /models` endpoint in favor of `GET /model/list`
  - All model listing now consolidated under Model Management section

### Fixed
- **Default model configuration**:
  - Fixed bug where API mode used hardcoded default instead of config file
  - Session model now properly initialized from `wrapper_config.toml`

## [0.7.1] - 2025-12-04

### Added
- **CLI enhancements for model management**:
  - Added `/help` command to display all available CLI commands and tips
  - Added `/model` command for interactive model switching during conversations
  - Model validation at startup with helpful error messages
  - Interactive model selection when invalid model detected at startup
  - Cancel option ('c') for all model selection prompts
  - Fuzzy matching for similar model names (showing first 10 results)
  - Model capabilities display (family, parameters, quantization) after selection

### Changed
- **Conversation context management**:
  - Conversation history now resets automatically when switching models via `/model` command
  - Prevents context leakage between different models
- **Startup messages**:
  - Changed from listing all commands to simple "Type '/help' for available commands"
  - Cleaner, less cluttered startup experience

### Fixed
- **Model configuration handling**:
  - Fixed bug where CLI mode didn't fall back to `wrapper_config.toml` model when `--model` arg not provided
  - Added proper validation that model is specified before starting CLI session

## [0.7.0] - 2025-12-03

### Added
- **Async I/O for conversation history**: Major refactoring for non-blocking file operations
  - Added `aiofiles` dependency for async file I/O
  - `MessageHistory.save()` and `.load()` are now async methods
  - History loading integrated into FastAPI lifespan for proper async support
  - All history endpoints (`/history/load`, `/history/save`, `/history/overwrite`) use async I/O
  - Comprehensive test suite with 16 tests covering async operations, concurrency, and error handling
- **Test infrastructure**:
  - Added `tests/test_async_history.py` with comprehensive async history tests
  - Added `tests/conftest.py` with pytest fixtures and test result reporting
  - Added `tests/README.md` documenting test coverage and usage
  - Automated test result reporting to `tests/test_results/` directory
  - Test results exported in both JSON and Markdown formats
  - Added pytest markers for slow tests (tests requiring Ollama server)

### Changed
- **MessageHistory**: Save and load operations now use async/await pattern
- **OllamaWrapper**: Lifespan event now includes async history loading on startup
- **Breaking change**: All save/load operations now require `await` when called directly

### Fixed
- **Configuration**: Removed deprecated `server_config.toml` file
  - All configuration now uses `wrapper_config.toml` exclusively
  - Updated documentation to reference the correct config file

## [0.6.8] - 2025-12-03

### Fixed
- **Temperature test wrapper availability checking:**
  - Added `check_wrapper_running()` function to test wrapper availability before execution
  - `get_available_models()` now exits with clear error message if wrapper is not running
  - Added early wrapper check in `temperature_test_multi_model.py` to fail fast
  - Prevents scripts from hanging indefinitely with connection refused errors
  - Provides actionable error messages guiding users to start the wrapper
- **Missing variable definition in temperature test:**
  - Fixed `NameError` where `is_non_interactive` variable was used but not defined
  - Variable now properly set from `args.default` after argument parsing

### Changed
- **Refactored `demo_config.py`:**
  - Replaced `os.path` with `pathlib.Path` for modern path handling
  - Improved documentation with detailed docstring
  - More Pythonic and maintainable code structure

## [0.6.7] - 2025-12-03

### Fixed
- **STDIO transport support in MCP servers:**
  - Fixed long-standing bug where `math_server.py` and `ipinfo_server.py` were hardcoded to HTTP transport
  - Servers now properly detect and use transport mode (stdio or http)
  - Auto-detection: if stdin is not a terminal, assumes STDIO mode
  - Manual override via command-line argument: `python server.py [config] [transport]`
  - Resolves transport mode mismatch between wrapper and MCP servers

## [0.6.6] - 2025-12-03

### Fixed
- **`/servers` endpoint now shows both connected and available servers:**
  - Returns `connected` object with currently mounted servers and their tools
  - Returns `available` object with enabled servers from config file
  - Previously only showed connected servers or empty response
  - Provides better visibility into the MCP server ecosystem

### Changed
- **API documentation updated:**
  - `GET /servers` description now reflects it lists both connected and available servers

## [0.6.5] - 2025-12-03

### Added
- **Temperature test multi-model script (`demos/temperature_test_multi_model.py`):**
  - Added `argparse` support with `--prompt` argument to specify prompt file or text via command line
  - Added `--default` argument for non-interactive mode
  - Added comprehensive help text with usage examples
  - Added detailed code documentation in main() and __main__ entry point

### Changed
- **Temperature test multi-model script (`demos/temperature_test_multi_model.py`):**
  - **BREAKING**: Prompt must now be specified with `--prompt` flag instead of positional argument
  - Old: `python script.py prompt_file.txt --default`
  - New: `python script.py --prompt prompt_file.txt --default`

## [0.6.4] - 2025-12-03

### Added
- **Temperature test utilities (`demos/temperature_test_utils.py`):**
  - Added `clean_llm_response_data()` function to handle duplicate JSON keys from malformed LLM outputs (e.g., gemma2:2b)
  - Added `get_recent_prompts()` to list top 5 most recent prompt files alphabetically ordered
  - Interactive prompt selection now supports:
    - Selecting recent prompts by number (1-5)
    - Direct filename entry
    - Custom text input
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
- **Wrapper configuration (`wrapper_config.toml`, `wrapper_config.py`):**
  - Added `max_history_messages` config parameter (default: 20) to control conversation summarization threshold
  - Configurable when `MessageHistory` triggers conversation summarization
- **Architecture improvements:**
  - LLM responses cleaned at source (immediately after generation) for efficiency
  - Progressive export to both JSON and Markdown formats
  - Text formatters use tokens (h2, bullet, bold) for consistent, maintainable code

### Changed
- **Temperature test utilities (`demos/temperature_test_utils.py`):**
  - Renamed `save_results_to_json()` → `export_results_to_json()` for naming consistency
  - Refactored `format_summary_display()` to use format tokens instead of duplicated code blocks
  - Both JSON and Markdown now receive clean data (no duplicate processing)
- **Ollama wrapper (`ollama_wrapper.py`):**
  - `OllamaWrapper.__init__()` now accepts `max_history_messages` parameter
  - `MessageHistory` max_messages threshold is now configurable via config file

### Fixed
- **LLM response handling**: Duplicate JSON keys in LLM outputs are now cleaned before export
  - Some models (e.g., gemma2:2b) generate malformed JSON with duplicate keys
  - Cleaning happens once at source for efficiency

## [0.6.3] - 2025-12-02

### Fixed
- **Configuration**: Removed deprecated `server_config.toml` file (superseded by async feature in v0.7.0)
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
