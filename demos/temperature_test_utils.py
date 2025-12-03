"""
Shared utilities for temperature testing demos
"""

import requests
import time
import json
from pathlib import Path
from demo_config import API_URL

DEFAULT_PROMPT = "Explain what a binary search algorithm does in one sentence."

# TEMPERATURE CONFIGURATION
# Central source of all possible temperature/description configurations.
# Keys (1-8) correspond to the displayed user selection number.
ALL_TEMPERATURE_CONFIGS = {
    1: (None, "Default from Config"),  # Uses config_temp, requires get_config_temperature
    2: (0.0, "Zero (Maximum Determinism)"),
    3: (0.1, "Very Low (Deterministic)"),
    4: (0.5, "Low-Medium"),
    5: (0.8, "Medium (Balanced)"),
    6: (1.0, "Medium-High"),
    7: (1.5, "High (Creative)"),
    8: (2.0, "Maximum (Very Creative)")
}

# The keys corresponding to the standard tests
DEFAULT_TEMPERATURE_TESTS = [2, 4, 6]
_CONFIG_CACHE = None

# Use API_URL from shared config
HOST = API_URL

def _load_config_data():
    """Reads wrapper_config.toml once and caches the default model and temperature."""
    global _CONFIG_CACHE

    # Check if cache is already populated
    if _CONFIG_CACHE is not None:
        return

    # Initialize cache with fallback values (0.2 embedded here)
    cache = {
        'model': None,
        'temperature': 0.2 # <-- FALLBACK VALUE IS NOW HARDCODED
    }

    try:
        # Get the path to wrapper_config.toml using pathlib
        demos_dir = Path(__file__).resolve().parent
        config_path = demos_dir.parent / 'wrapper_config.toml'

        if not config_path.exists():
            _CONFIG_CACHE = cache
            return

        with open(config_path, 'r') as f:
            for line in f:
                # Look for model line containing temperature and default
                if 'model' in line and 'temperature' in line and 'default' in line:

                    # 1. Extract Temperature Value
                    try:
                        temp_part = line.split('temperature')[1]
                        temp_str = temp_part.split('=')[1].strip().rstrip('}').strip()
                        cache['temperature'] = float(temp_str)
                    except Exception:
                        pass

                    # 2. Extract Default Model Name
                    try:
                        parts = line.split('default =')
                        if len(parts) > 1:
                            value_part = parts[1].strip()
                            start_quote = value_part.find('"')
                            end_quote = value_part.find('"', start_quote + 1)
                            if start_quote != -1 and end_quote != -1:
                                cache['model'] = value_part[start_quote + 1:end_quote]
                    except Exception:
                        pass

                    # Assume we only need one line for both values
                    break

    except Exception:
        # On any error, return the initialized cache with fallbacks
        pass

    _CONFIG_CACHE = cache

def get_config_temperature():
    """Read default temperature from wrapper_config.toml using the cache."""
    _load_config_data()
    # If the cache load failed entirely, return the module fallback
    return _CONFIG_CACHE.get('temperature', 0.2)

def get_config_model():
    """Read default model name from wrapper_config.toml using the cache."""
    _load_config_data()
    return _CONFIG_CACHE.get('model', None)

def get_resolved_temperature(temp):
    """
    Returns the numerical temperature value, substituting the config default
    for None. Used primarily for sorting and numerical comparison.
    """
    if temp is None:
        return get_config_temperature()
    return temp

def test_temperature_model(model, temp, description, prompt=None):
    """Test a specific model and temperature combination"""
    if prompt is None:
        prompt = DEFAULT_PROMPT

    payload = {
        "message": prompt,
        "model": model,
        "mcp_server": "",
        "stateless": True  # Use stateless mode for independent testing
    }

    if temp is not None:
        payload["temperature"] = temp

    start_time = time.time()
    try:
        response = requests.post(f"{HOST}/chat", json=payload)
        elapsed_time = time.time() - start_time

        if response.status_code != 200:
            return None

        result = response.json()

        # Get actual config temperature if using default
        actual_temp = temp if temp is not None else f"default ({get_config_temperature()})"

        return {
            "model": model,
            "temperature": actual_temp,
            "description": description,
            "response": result['response'],
            "elapsed_time": elapsed_time,
            "metrics": result.get('metrics', {})
        }
    except Exception as e:
        print(f"Error testing {model} at temp={temp}: {e}")
        return None

def format_duration(seconds):
    """Format duration in hours:minutes:seconds or shorter format"""
    if seconds < 1:
        # Less than 1 second: show milliseconds
        return f"{seconds*1000:.0f}ms"
    elif seconds < 60:
        # Less than 1 minute: show seconds with 2 decimals
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        # Less than 1 hour: show minutes:seconds
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes}m {secs}s"
    else:
        # 1 hour or more: show hours:minutes:seconds
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours}h {minutes}m {secs}s"

def check_wrapper_running():
    """
    Check if the Ollama-FastMCP wrapper is running and accessible.

    Returns:
        bool: True if wrapper is running, False otherwise
    """
    try:
        response = requests.get(f"{HOST}/", timeout=3)
        return response.status_code == 200
    except requests.exceptions.ConnectionError:
        return False
    except requests.exceptions.Timeout:
        return False
    except Exception:
        return False

def get_available_models():
    """
    Fetch available models from the API.

    Returns:
        list: List of model names, or None on error

    Raises:
        SystemExit: If wrapper is not running
    """
    try:
        response = requests.get(f"{HOST}/models", timeout=5)
        if response.status_code == 200:
            data = response.json()
            return [m['name'] for m in data['models']]
        else:
            print(f"Error fetching models: {response.status_code}")
            return None
    except requests.exceptions.ConnectionError as e:
        print(f"\n❌ Error: Cannot connect to Ollama-FastMCP wrapper at {HOST}")
        print(f"   Connection refused: {e}")
        print(f"\n💡 Please ensure the wrapper is running:")
        print(f"   uv run python ollama_wrapper.py api")
        print(f"\n   Or check wrapper_config.toml for the correct host/port settings.")
        import sys
        sys.exit(1)
    except requests.exceptions.Timeout:
        print(f"\n❌ Error: Connection to {HOST} timed out")
        print(f"\n💡 Please check if the wrapper is running and accessible.")
        import sys
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error connecting to API: {e}")
        import sys
        sys.exit(1)

def print_summary():
    """Print standard temperature testing summary"""
    print("\nTemperature Guidelines:")
    print("  • Lower temperature (0.1-0.3): Best for factual tasks, coding, math")
    print("  • Medium temperature (0.7-1.0): Good for natural conversation")
    print("  • Higher temperature (1.5+): Best for creative writing, brainstorming")
    print("\nMetrics Explanation:")
    print("  • TPS = Tokens Per Second (generation speed)")
    print("  • Time = Total request time including network overhead")
    print("  • Total Duration = Ollama processing time")

def select_temperatures():
    """
    Allow user to select which temperatures to test, sorting the menu
    numerically based on resolved temperature values.
    """
    config_temp = get_config_temperature()

    # 1. Prepare a sortable list from ALL_TEMPERATURE_CONFIGS
    # Format: [(resolved_temp, original_temp, description), ...]
    sortable_configs = []
    for key, (temp, desc) in ALL_TEMPERATURE_CONFIGS.items():
        resolved_temp = get_resolved_temperature(temp)
        sortable_configs.append((resolved_temp, temp, desc))

    # 2. Sort the list numerically
    sortable_configs.sort(key=lambda x: x[0])

    print("\nAvailable temperature settings:")

    # 3. Print the sorted menu using ENUMERATE for 1-based indexing
    for i, (resolved_temp, temp, desc) in enumerate(sortable_configs, 1):
        temp_display = temp if temp is not None else f"default ({config_temp})"
        # Print using the sorted index 'i'
        print(f"  {i}. {temp_display:<20} - {desc}")

    print("\nEnter temperature numbers to test (comma-separated, or 'all' for all, or 'default' for standard 4):")
    print("Example: 1,3,5  or  all  or  default")

    user_input = input("> ").strip().lower()

    if user_input == 'all':
        # Select all original (temp, desc) tuples from the sorted list
        selected = [(t, d) for r, t, d in sortable_configs]
    elif user_input == 'default' or user_input == '':
        # This is a cleaner way: filter the fully sorted list based on the numerical default values
        default_resolved_values = [get_resolved_temperature(ALL_TEMPERATURE_CONFIGS[k][0]) for k in DEFAULT_TEMPERATURE_TESTS]

        selected = []
        for r_temp, temp, desc in sortable_configs:
            # Check if the resolved temperature matches one of the default numerical values
            if any(abs(r_temp - dv) < 1e-6 for dv in default_resolved_values):
                selected.append((temp, desc))

    else:
        # Handle manual selection based on the SORTED MENU index 'i'
        try:
            indices = [int(x.strip()) for x in user_input.split(',')]
            selected = []
            for i in indices:
                if 1 <= i <= len(sortable_configs):
                    # Index i maps to sortable_configs[i-1]. We extract (temp, desc) (index 1 and 2)
                    selected.append( (sortable_configs[i-1][1], sortable_configs[i-1][2]) )

            if not selected:
                print("Invalid selection. Using default temperatures.")
                # Fallback logic to get default, already sorted
                default_resolved_values = [get_resolved_temperature(ALL_TEMPERATURE_CONFIGS[k][0]) for k in DEFAULT_TEMPERATURE_TESTS]
                for r_temp, temp, desc in sortable_configs:
                    if any(abs(r_temp - dv) < 1e-6 for dv in default_resolved_values):
                        selected.append((temp, desc))
        except Exception:
            print("Invalid selection. Using default temperatures.")
            # Fallback logic
            default_resolved_values = [get_resolved_temperature(ALL_TEMPERATURE_CONFIGS[k][0]) for k in DEFAULT_TEMPERATURE_TESTS]
            selected = []
            for r_temp, temp, desc in sortable_configs:
                if any(abs(r_temp - dv) < 1e-6 for dv in default_resolved_values):
                    selected.append((temp, desc))

    # The list is already sorted by the display logic, so no final sort is needed.
    return selected

def resolve_prompt_path(path):
    """Resolve prompt file path - tries current dir, then demos/prompts/, then project root"""
    if not path:
        return None

    path_obj = Path(path)

    # 1. If absolute path, use as-is
    # .is_absolute() replaces os.path.isabs()
    if path_obj.is_absolute():
        # .exists() replaces os.path.exists()
        return path_obj if path_obj.exists() else None

    # 2. Try current directory (relative to CWD)
    if path_obj.exists():
        return path_obj

    # 3. Try relative to demos/prompts/ directory
    # Path(__file__).resolve().parent gets the absolute path to the directory containing this script.
    demos_dir = Path(__file__).resolve().parent
    # The / operator replaces os.path.join()
    prompts_path = demos_dir / 'prompts' / path
    if prompts_path.exists():
        # Note: We return the Path object, not a string
        return prompts_path

    # 4. Try project root + path
    project_root = demos_dir.parent
    root_path = project_root / path
    if root_path.exists():
        return root_path

    return None

def get_recent_prompts(max_count=5):
    """
    Get list of recent prompt files from demos/prompts/ directory.

    Files are selected by most recent modification time, but returned in alphabetical order.

    Args:
        max_count: Maximum number of recent files to return

    Returns:
        List of filenames (alphabetically sorted)
    """
    demos_dir = Path(__file__).resolve().parent
    prompts_dir = demos_dir / 'prompts'

    if not prompts_dir.exists():
        return []

    # Get all text files in prompts directory (exclude .gitkeep, README.md)
    prompt_files = []
    for file_path in prompts_dir.iterdir():
        if file_path.is_file() and file_path.suffix in ['.txt', '.md'] and file_path.name not in ['.gitkeep', 'README.md']:
            mtime = file_path.stat().st_mtime
            prompt_files.append((file_path.name, mtime))

    # Sort by modification time to get most recent files
    prompt_files.sort(key=lambda x: x[1], reverse=True)

    # Take top max_count most recent files
    recent_files = [name for name, _ in prompt_files[:max_count]]

    # Sort alphabetically for display
    recent_files.sort()

    return recent_files

def get_prompt_from_file_or_input(prompt_arg=None):
    """Get prompt from file, command line arg, or user input

    File path resolution order:
    1. Absolute paths (e.g., /home/user/prompt.txt)
    2. Relative to current directory (e.g., ./prompt.txt)
    3. Relative to demos/prompts/ (e.g., coreference_resolution.txt)
    4. Relative to project root (e.g., demos/prompts/coreference_resolution.txt)
    """
    # If command line argument provided
    if prompt_arg and prompt_arg != '':
        # Try to resolve as file path
        resolved_path = resolve_prompt_path(prompt_arg)

        if resolved_path:
            try:
                with open(resolved_path, 'r') as f:
                    prompt = f.read().strip()
                    print(f"✓ Loaded prompt from: {resolved_path}")
                    return prompt
            except Exception as e:
                print(f"Error reading prompt file: {e}")
                return DEFAULT_PROMPT
        else:
            # Not a file, treat as direct prompt text
            return prompt_arg

    # Ask user interactively
    print("\nPrompt configuration:")
    print("  - Press Enter to use default prompt")

    # Show recent prompts if available
    recent_prompts = get_recent_prompts(max_count=5)
    if recent_prompts:
        print(f"\n  Top {len(recent_prompts)} most recent prompt files by date (alphabetically sorted):")
        for idx, filename in enumerate(recent_prompts, 1):
            print(f"    {idx}. {filename}")
        print("\n  You can:")
        print(f"    • Enter a number (1-{len(recent_prompts)}) to use a recent prompt")
        print("    • Enter just the filename (e.g., coreference_resolution.txt)")

    print("  - Enter a file path to load from file (supports multiple formats):")
    print("    • Just filename: searches demos/prompts/")
    print("    • Relative path: demos/prompts/myfile.txt")
    print("    • Absolute path: /full/path/to/file.txt")
    print("  - Enter text directly to use as prompt")
    print(f"\nDefault prompt: '{DEFAULT_PROMPT}'")

    user_input = input("\nPrompt (number/file/text/Enter for default): ").strip()

    # Check if user entered a number (selecting from recent prompts)
    if user_input.isdigit():
        idx = int(user_input) - 1
        if 0 <= idx < len(recent_prompts):
            user_input = recent_prompts[idx]
            print(f"Selected recent prompt: {user_input}")
        else:
            print(f"Invalid selection. Using default prompt.")
            return DEFAULT_PROMPT

    if user_input == '':
        return DEFAULT_PROMPT

    # Try to resolve as file path
    resolved_path = resolve_prompt_path(user_input)

    if resolved_path:
        try:
            with open(resolved_path, 'r') as f:
                prompt = f.read().strip()
                print(f"✓ Loaded prompt from: {resolved_path}")
                return prompt
        except Exception:
            # If file read fails, treat as direct text
            pass

    # Treat as direct text
    return user_input

def format_summary_display(summary, format_type='console'):
    """
    Format summary statistics for display in console or markdown.

    Args:
        summary: Dictionary containing summary statistics
        format_type: 'console' or 'markdown'

    Returns:
        String formatted for the specified output type
    """
    if not summary:
        return ""

    # Define format tokens based on output type
    if format_type == 'markdown':
        h2 = '##'
        h3 = '###'
        bullet = '-'
        bold_start = '**'
        bold_end = '**'
        section_sep = '\n'
    elif format_type == 'console':
        h2 = '\n'
        h3 = '  •'
        bullet = '    -'
        bold_start = ''
        bold_end = ''
        section_sep = '\n'
    else:
        raise ValueError(f"Unsupported format_type: {format_type}")

    # Build content using format tokens - specular structure for both formats
    lines = []
    lines.append(f"{h2} Performance Statistics{section_sep}")

    # Fastest model section - identical structure for both formats
    lines.append(f"{h3} Fastest Model (Highest TPS){section_sep}")
    lines.append(f"{bullet} {bold_start}Model{bold_end}: {summary.get('fastest_model', 'N/A')}")
    lines.append(f"{bullet} {bold_start}Temperature{bold_end}: {summary.get('fastest_temperature', 'N/A')}")
    lines.append(f"{bullet} {bold_start}TPS{bold_end}: {summary.get('highest_tps', 'N/A')}")
    lines.append(f"{bullet} {bold_start}Tokens{bold_end}: {summary.get('fastest_completion_tokens', 'N/A')}")
    lines.append(f"{bullet} {bold_start}Elapsed Time{bold_end}: {summary.get('fastest_elapsed_time_readable', 'N/A')} ({summary.get('fastest_elapsed_time_s', 'N/A')}s)")
    lines.append(f"{bullet} {bold_start}Total Duration{bold_end}: {summary.get('fastest_total_duration_readable', 'N/A')} ({summary.get('fastest_total_duration_s', 'N/A')}s){section_sep}")

    # Averages section - identical structure for both formats
    lines.append(f"{h3} Averages{section_sep}")
    lines.append(f"{bullet} {bold_start}Average TPS{bold_end}: {summary.get('average_tps', 'N/A')}")
    lines.append(f"{bullet} {bold_start}Average Tokens{bold_end}: {summary.get('average_tokens', 'N/A')}{section_sep}")

    # Response lengths section - identical structure for both formats
    lines.append(f"{h3} Response Lengths{section_sep}")
    lines.append(f"{bullet} {bold_start}Min{bold_end}: {summary.get('min_response_length', 'N/A')} chars")
    lines.append(f"{bullet} {bold_start}Max{bold_end}: {summary.get('max_response_length', 'N/A')} chars")
    lines.append(f"{bullet} {bold_start}Avg{bold_end}: {summary.get('avg_response_length', 'N/A')} chars")

    return '\n'.join(lines)


def clean_llm_response_data(data):
    """
    Clean LLM response data by removing duplicate keys in nested JSON.

    Some LLMs (e.g., gemma2:2b) generate malformed JSON with duplicate keys
    in their responses, such as:
        {"context": "first", "context": "second"}

    This function recursively cleans such duplicates from the data structure.
    When duplicate keys exist, Python's json.loads() keeps the last occurrence.

    Args:
        data: The data structure to clean (dict, list, str, or primitive)

    Returns:
        Cleaned data structure with duplicate keys removed

    Note:
        This should be applied to individual test results immediately after
        generation to ensure all_results contains only clean data.
    """
    if isinstance(data, dict):
        # Process dict - recursively clean nested structures
        return {key: clean_llm_response_data(value) for key, value in data.items()}
    elif isinstance(data, list):
        # Recursively clean list items
        return [clean_llm_response_data(item) for item in data]
    elif isinstance(data, str):
        # Check if string contains JSON (common in LLM responses)
        stripped = data.strip()
        if stripped.startswith('{') or stripped.startswith('['):
            try:
                parsed = json.loads(stripped)
                cleaned = clean_llm_response_data(parsed)
                return json.dumps(cleaned)
            except (json.JSONDecodeError, ValueError):
                # Not valid JSON, return as-is
                pass
        return data
    else:
        # Primitive types (int, float, bool, None) - return as-is
        return data


def format_metadata_section(meta, format_type='markdown'):
    """
    Format metadata section for text export (markdown, plain text, etc.).

    Note: JSON export doesn't use this - it uses json.dump() directly.

    Args:
        meta: test_metadata dict from results_data
        format_type: 'markdown' for now (extensible to 'html', 'text', etc.)

    Returns:
        Formatted string for metadata section
    """
    if format_type == 'markdown':
        h2 = '##'
        bullet = '-'
        bold_start = '**'
        bold_end = '**'
    else:
        raise ValueError(f"Unsupported format_type: {format_type}")

    lines = [
        f"{h2} Test Information\n",
        f"{bullet} {bold_start}Start Time{bold_end}: {meta['start_timestamp']}",
        f"{bullet} {bold_start}End Time{bold_end}: {meta['end_timestamp']}",
        f"{bullet} {bold_start}Total Duration{bold_end}: {meta['total_duration_readable']}",
        f"{bullet} {bold_start}Prompt{bold_end}: {meta['prompt']}",
        f"{bullet} {bold_start}Models Tested{bold_end}: {', '.join(meta['models_tested'])}",
        f"{bullet} {bold_start}Temperatures Tested{bold_end}: {', '.join(map(str, meta['temperatures_tested']))}",
        f"{bullet} {bold_start}Total Tests{bold_end}: {meta['total_tests']}\n"
    ]
    return '\n'.join(lines)


def format_summary_table_header(format_type='markdown'):
    """
    Format summary table header for text export.

    Note: JSON export doesn't use this - it uses json.dump() directly.
    """
    if format_type == 'markdown':
        return (
            "## Summary Table\n\n"
            "| Test # | Model | Temperature | TPS | Tokens | Elapsed Time | Total Duration | Description |\n"
            "|--------|-------|-------------|-----|--------|--------------|----------------|-------------|\n"
        )
    else:
        raise ValueError(f"Unsupported format_type: {format_type}")


def format_summary_table_row(result, format_type='markdown'):
    """
    Format a single summary table row for text export.

    Note: JSON export doesn't use this - it uses json.dump() directly.

    Args:
        result: Single test result dict
        format_type: 'markdown' for now (extensible to other text formats)

    Returns:
        Formatted string for one table row
    """
    test_num = result.get('test_number', 'N/A')
    model = result['model']
    temp = result['temperature']
    tps = result['metrics'].get('tokens_per_second', 0)
    tokens = result['metrics'].get('completion_tokens', 0)
    elapsed = result.get('elapsed_time_readable', format_duration(result['elapsed_time']))
    total_dur = format_duration(result['metrics'].get('total_duration_s', 0))
    desc = result['description']

    if format_type == 'markdown':
        return f"| {test_num} | {model} | {temp} | {tps:.2f} | {tokens} | {elapsed} | {total_dur} | {desc} |\n"
    else:
        raise ValueError(f"Unsupported format_type: {format_type}")


def format_detailed_response(result, format_type='markdown'):
    """
    Format detailed response section for a single test result (text export only).

    Note: JSON export doesn't use this - it uses json.dump() directly.

    Args:
        result: Single test result dict
        format_type: 'markdown' for now (extensible to other text formats)

    Returns:
        Formatted string for detailed response
    """
    test_num = result.get('test_number', 'N/A')
    tps = result['metrics'].get('tokens_per_second', 0)
    tokens = result['metrics'].get('completion_tokens', 0)
    elapsed = result.get('elapsed_time_readable', format_duration(result['elapsed_time']))
    total_dur = format_duration(result['metrics'].get('total_duration_s', 0))

    if format_type == 'markdown':
        h3 = '###'
        bullet = '-'
        bold_start = '**'
        bold_end = '**'

        lines = [
            f"{h3} Test #{test_num}: {result['model']} - Temperature {result['temperature']} ({result['description']})\n",
            f"{bold_start}Response:{bold_end}\n> {result['response']}\n",
            f"{bold_start}Metrics:{bold_end}",
            f"{bullet} TPS: {tps:.2f}",
            f"{bullet} Tokens: {tokens}",
            f"{bullet} Elapsed Time: {elapsed}",
            f"{bullet} Total Duration: {total_dur}\n"
        ]
        return '\n'.join(lines)
    else:
        raise ValueError(f"Unsupported format_type: {format_type}")


def export_results_to_json(results_data, filename):
    """
    Export test results to JSON file, ensuring the output directory exists.

    Args:
        results_data: Clean test results data (should be pre-cleaned at source)
        filename: Path to the output JSON file
    """

    filepath = Path(filename)
    # Use Path.parent and Path.mkdir() for robust directory creation
    filepath.parent.mkdir(parents=True, exist_ok=True)

    with open(filename, 'w') as f:
        json.dump(results_data, f, indent=2)


def append_result_to_markdown(result, md_filename, is_first_result=False):
    """
    Append a single test result to markdown file for progressive updates.

    Args:
        result: Single test result dict (should be pre-cleaned)
        md_filename: Path to the markdown file
        is_first_result: If True, creates file with headers; if False, appends to existing
    """
    md_path = Path(md_filename)
    md_path.parent.mkdir(parents=True, exist_ok=True)

    mode = 'w' if is_first_result else 'a'

    with open(md_filename, mode) as f:
        if is_first_result:
            # Initialize file with header and table header
            f.write("# Temperature Test Results\n\n")
            f.write(format_summary_table_header())

        # Append summary table row using centralized formatter
        f.write(format_summary_table_row(result))


def convert_json_to_markdown(json_filename):
    """
    Standalone utility to convert a saved JSON results file to markdown format.

    This function reads an existing JSON file and generates a complete markdown
    export. It's designed for maintenance tasks such as:
    - Regenerating markdown from edited/cleaned JSON files
    - Batch conversion of old JSON test results
    - Recovery when markdown files are corrupted or lost

    Args:
        json_filename: Path to the JSON results file

    Returns:
        Path to the generated markdown file

    Note:
        This function is not used in the normal progressive export workflow.
        During normal test execution, markdown is generated progressively
        using append_result_to_markdown().
    """
    json_path = Path(json_filename)

    # Load JSON data
    with open(json_filename, 'r') as f:
        results_data = json.load(f)

    md_filename = json_path.with_suffix('.md')
    md_filename.parent.mkdir(parents=True, exist_ok=True)

    with open(md_filename, 'w') as f:
        # Header
        f.write("# Temperature Test Results\n\n")

        # Metadata using centralized formatter
        f.write(format_metadata_section(results_data['test_metadata']))
        f.write("\n")

        # Summary table using centralized formatters
        f.write(format_summary_table_header())
        for result in results_data['results']:
            f.write(format_summary_table_row(result))

        # Detailed responses using centralized formatter
        f.write("\n## Detailed Responses\n\n")
        for result in results_data['results']:
            f.write(format_detailed_response(result))
            f.write("\n")

        # Summary insights
        if 'summary' in results_data and results_data['summary']:
            summary_text = format_summary_display(results_data['summary'], format_type='markdown')
            f.write(summary_text)
            f.write('\n')

    print(f"✓ Markdown converted from JSON: {md_filename}")
    return md_filename


def export_results_to_markdown(results_data, json_filename):
    """
    Finalize the progressive markdown export by adding metadata and detailed sections.

    This function completes the progressive markdown file that was built by
    append_result_to_markdown(). It prepends metadata and appends detailed
    responses and summary sections.

    Args:
        results_data: Complete test results data (from centralized clean data bag)
        json_filename: Path to the JSON file (used to determine markdown filename)

    Returns:
        Path to the finalized markdown file
    """
    json_path = Path(json_filename)
    md_filename = json_path.with_suffix('.md')

    # Read existing progressive content (summary table)
    existing_content = ""
    if md_filename.exists():
        with open(md_filename, 'r') as f:
            existing_content = f.read()

    # Now rewrite with complete structure
    with open(md_filename, 'w') as f:
        # Header
        f.write("# Temperature Test Results\n\n")

        # Metadata using centralized formatter
        f.write(format_metadata_section(results_data['test_metadata']))
        f.write("\n")

        # Write existing summary table (skip the old header if present)
        if "## Summary Table" in existing_content:
            # Extract from "## Summary Table" onwards
            table_start = existing_content.find("## Summary Table")
            existing_content = existing_content[table_start:]
            f.write(existing_content)
            f.write("\n")
        else:
            # Fallback: regenerate summary table using centralized formatters
            f.write(format_summary_table_header())
            for result in results_data['results']:
                f.write(format_summary_table_row(result))
            f.write("\n")

        # Detailed responses using centralized formatter
        f.write("## Detailed Responses\n\n")
        for result in results_data['results']:
            f.write(format_detailed_response(result))
            f.write("\n")

        # Summary insights
        if 'summary' in results_data and results_data['summary']:
            summary_text = format_summary_display(results_data['summary'], format_type='markdown')
            f.write(summary_text)
            f.write('\n')

    print(f"✓ Markdown export finalized: {md_filename}")
    return md_filename
