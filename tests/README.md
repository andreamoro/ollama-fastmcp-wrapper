# Tests for Ollama-FastMCP-Wrapper

## Async Conversation History Tests

The `test_async_history.py` file contains test cases for the async implementation of conversation history.

### Running the Tests

**Prerequisites:**
```bash
# Install test dependencies
uv pip install -e ".[test]"
# or with pip:
pip install -e ".[test]"
```

**Run all tests:**
```bash
pytest tests/
```

**Run specific test file:**
```bash
pytest tests/test_async_history.py
```

**Run with verbose output:**
```bash
pytest tests/test_async_history.py -v
```

**Run specific test:**
```bash
pytest tests/test_async_history.py::TestMessageHistoryAsync::test_save_creates_file -v
```

### Test Coverage Areas

Before merging `feature/async-conversation-history` branch, ensure all tests pass:

1. **MessageHistory async operations**
   - `save()` creates files correctly
   - `load()` restores conversations
   - Round-trip save/load preserves data
   - Concurrent operations don't block
   - Error handling for missing files

2. **FastAPI history endpoints**
   - `/history/save/{file_name}` works with async I/O
   - `/history/load/{file_name}` works with async I/O
   - `/history/overwrite/{file_name}` respects overwrite flag
   - Error responses for invalid operations

3. **Lifespan integration**
   - History loads during startup
   - Missing files handled gracefully
   - No blocking during app initialization

## Test Results

Test results are automatically saved to `tests/test_results/` after each test run:

- **`latest.md`** - Most recent test report (human-readable)
- **`test_report_TIMESTAMP.md`** - Markdown reports with timing analysis
- **`test_report_TIMESTAMP.json`** - Detailed JSON reports for CI/CD
- **`history_sample_TIMESTAMP.json`** - Sample conversation history files from tests

The `test_results/` directory is gitignored to avoid committing test artifacts.

### Example Report

After running tests, check `tests/test_results/latest.md` for:
- Test summary (passed/failed/skipped)
- Individual test results with timing
- Slowest tests analysis
- Sample conversation files with actual Ollama summaries

## TODO

- [ ] Add performance benchmarks (async vs sync)
- [ ] Set up CI/CD to run tests automatically
