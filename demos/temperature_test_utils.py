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

def get_available_models():
    """Fetch available models from the API"""
    try:
        response = requests.get(f"{HOST}/models")
        if response.status_code == 200:
            data = response.json()
            return [m['name'] for m in data['models']]
        else:
            print(f"Error fetching models: {response.status_code}")
            return None
    except Exception as e:
        print(f"Error connecting to API: {e}")
        return None

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
    print("  - Enter a file path to load from file (supports multiple formats):")
    print("    • Just filename: coreference_resolution.txt (searches demos/prompts/)")
    print("    • Relative path: demos/prompts/myfile.txt")
    print("    • Absolute path: /full/path/to/file.txt")
    print("  - Enter text directly to use as prompt")
    print(f"\nDefault prompt: '{DEFAULT_PROMPT}'")

    user_input = input("Prompt (file/text/Enter for default): ").strip()

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

    if format_type == 'console':
        # Console format with bullet points
        lines = []
        lines.append("\nPerformance:")
        lines.append(f"  • Fastest Model: {summary.get('fastest_model', 'N/A')} at temperature {summary.get('fastest_temperature', 'N/A')}")
        lines.append(f"    - TPS: {summary.get('highest_tps', 'N/A')}")
        lines.append(f"    - Tokens: {summary.get('fastest_completion_tokens', 'N/A')}")
        lines.append(f"    - Elapsed Time: {summary.get('fastest_elapsed_time_readable', 'N/A')} ({summary.get('fastest_elapsed_time_s', 'N/A')}s)")
        lines.append(f"    - Total Duration: {summary.get('fastest_total_duration_readable', 'N/A')} ({summary.get('fastest_total_duration_s', 'N/A')}s)")
        lines.append(f"  • Average TPS: {summary.get('average_tps', 'N/A')}")
        lines.append(f"  • Average Tokens: {summary.get('average_tokens', 'N/A')}")

        lines.append("\nResponse Lengths:")
        lines.append(f"  • Min: {summary.get('min_response_length', 'N/A')} chars")
        lines.append(f"  • Max: {summary.get('max_response_length', 'N/A')} chars")
        lines.append(f"  • Avg: {summary.get('avg_response_length', 'N/A')} chars")

        return '\n'.join(lines)

    elif format_type == 'markdown':
        # Markdown format with headers and lists
        lines = []
        lines.append("## Summary Statistics\n")

        # Fastest model section
        lines.append("### Fastest Model (Highest TPS)\n")
        lines.append(f"- **Model**: {summary.get('fastest_model', 'N/A')}")
        lines.append(f"- **Temperature**: {summary.get('fastest_temperature', 'N/A')}")
        lines.append(f"- **TPS**: {summary.get('highest_tps', 'N/A')}")
        lines.append(f"- **Tokens**: {summary.get('fastest_completion_tokens', 'N/A')}")
        lines.append(f"- **Elapsed Time**: {summary.get('fastest_elapsed_time_readable', 'N/A')} ({summary.get('fastest_elapsed_time_s', 'N/A')}s)")
        lines.append(f"- **Total Duration**: {summary.get('fastest_total_duration_readable', 'N/A')} ({summary.get('fastest_total_duration_s', 'N/A')}s)\n")

        # Averages section
        lines.append("### Averages\n")
        lines.append(f"- **Average TPS**: {summary.get('average_tps', 'N/A')}")
        lines.append(f"- **Average Tokens**: {summary.get('average_tokens', 'N/A')}\n")

        # Response lengths section
        lines.append("### Response Lengths\n")
        lines.append(f"- **Min**: {summary.get('min_response_length', 'N/A')} chars")
        lines.append(f"- **Max**: {summary.get('max_response_length', 'N/A')} chars")
        lines.append(f"- **Avg**: {summary.get('avg_response_length', 'N/A')} chars")

        return '\n'.join(lines)

    else:
        raise ValueError(f"Unsupported format_type: {format_type}")

def save_results_to_json(results_data, filename):
    """Save test results to JSON file, ensuring the output directory exists."""

    filepath = Path(filename)
    # Use Path.parent and Path.mkdir() for robust directory creation
    filepath.parent.mkdir(parents=True, exist_ok=True)

    with open(filename, 'w') as f:
        json.dump(results_data, f, indent=2)

def export_results_to_markdown(results_data, json_filename):
    """Export results to markdown format"""

    json_path = Path(json_filename)
    md_filename = json_path.with_suffix('.md')
    md_filename.parent.mkdir(parents=True, exist_ok=True)

    with open(md_filename, 'w') as f:
        # Header
        f.write("# Temperature Test Results\n\n")

        # Metadata
        meta = results_data['test_metadata']
        f.write("## Test Information\n\n")
        f.write(f"- **Start Time**: {meta['start_timestamp']}\n")
        f.write(f"- **End Time**: {meta['end_timestamp']}\n")
        f.write(f"- **Total Duration**: {meta['total_duration_readable']}\n")
        f.write(f"- **Prompt**: {meta['prompt']}\n")
        f.write(f"- **Models Tested**: {', '.join(meta['models_tested'])}\n")
        f.write(f"- **Temperatures Tested**: {', '.join(map(str, meta['temperatures_tested']))}\n")
        f.write(f"- **Total Tests**: {meta['total_tests']}\n\n")

        # Summary table
        f.write("## Summary Table\n\n")
        f.write("| Test # | Model | Temperature | TPS | Tokens | Elapsed Time | Total Duration | Description |\n")
        f.write("|--------|-------|-------------|-----|--------|--------------|----------------|-------------|\n")

        for result in results_data['results']:
            test_num = result.get('test_number', 'N/A')
            model = result['model']
            temp = result['temperature']
            tps = result['metrics'].get('tokens_per_second', 0)
            tokens = result['metrics'].get('completion_tokens', 0)
            elapsed = result.get('elapsed_time_readable', format_duration(result['elapsed_time']))
            total_dur = format_duration(result['metrics'].get('total_duration_s', 0))
            desc = result['description']

            f.write(f"| {test_num} | {model} | {temp} | {tps:.2f} | {tokens} | {elapsed} | {total_dur} | {desc} |\n")

        # Full responses
        f.write("\n## Detailed Responses\n\n")

        for result in results_data['results']:
            test_num = result.get('test_number', 'N/A')
            f.write(f"### Test #{test_num}: {result['model']} - Temperature {result['temperature']} ({result['description']})\n\n")
            f.write(f"**Response:**\n> {result['response']}\n\n")
            f.write(f"**Metrics:**\n")
            f.write(f"- TPS: {result['metrics'].get('tokens_per_second', 0):.2f}\n")
            f.write(f"- Tokens: {result['metrics'].get('completion_tokens', 0)}\n")
            f.write(f"- Elapsed Time: {result.get('elapsed_time_readable', format_duration(result['elapsed_time']))}\n")
            f.write(f"- Total Duration: {format_duration(result['metrics'].get('total_duration_s', 0))}\n\n")

        # Summary insights - use centralized formatting function
        if 'summary' in results_data and results_data['summary']:
            summary_text = format_summary_display(results_data['summary'], format_type='markdown')
            f.write(summary_text)
            f.write('\n')

    print(f"✓ Markdown export saved to: {md_filename}")
    return md_filename
