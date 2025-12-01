"""
Shared utilities for temperature testing demos
"""

import requests
import time
import json
import sys
import os
from datetime import datetime

HOST = "http://localhost:8000"
DEFAULT_PROMPT = "Explain what a binary search algorithm does in one sentence."

def get_config_temperature():
    """Read default temperature from wrapper_config.toml"""
    try:
        # Get the parent directory (project root)
        demos_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(demos_dir)
        config_path = os.path.join(project_root, 'wrapper_config.toml')

        if not os.path.exists(config_path):
            return 0.2  # Fallback default

        with open(config_path, 'r') as f:
            for line in f:
                # Look for model line with temperature
                if 'model' in line and 'temperature' in line:
                    # Parse inline table: model = { default = "...", temperature = 0.2 }
                    try:
                        # Extract temperature value
                        temp_part = line.split('temperature')[1]
                        temp_str = temp_part.split('=')[1].strip().rstrip('}').strip()
                        return float(temp_str)
                    except:
                        pass
        return 0.2  # Fallback if not found
    except:
        return 0.2  # Fallback on any error

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

# Standard temperature test configurations
TEMPERATURE_TESTS = [
    (0.1, "Very Low (Deterministic)"),
    (None, "Default from Config"),
    (0.8, "Medium (Balanced)"),
    (1.5, "High (Creative)")
]

def select_temperatures():
    """Allow user to select which temperatures to test"""
    # Read actual default from config
    config_temp = get_config_temperature()

    available_temps = [
        (0.0, "Zero (Maximum Determinism)"),
        (0.1, "Very Low (Deterministic)"),
        (None, f"Default from Config ({config_temp})"),
        (0.5, "Low-Medium"),
        (0.8, "Medium (Balanced)"),
        (1.0, "Medium-High"),
        (1.5, "High (Creative)"),
        (2.0, "Maximum (Very Creative)")
    ]

    print("\nAvailable temperature settings:")
    for i, (temp, desc) in enumerate(available_temps, 1):
        temp_display = temp if temp is not None else "default"
        print(f"  {i}. {temp_display:8} - {desc}")

    print("\nEnter temperature numbers to test (comma-separated, or 'all' for all, or 'default' for standard 4):")
    print("Example: 1,3,5  or  all  or  default")

    user_input = input("> ").strip().lower()

    if user_input == 'all':
        return available_temps
    elif user_input == 'default' or user_input == '':
        return TEMPERATURE_TESTS

    try:
        indices = [int(x.strip()) - 1 for x in user_input.split(',')]
        selected = [available_temps[i] for i in indices if 0 <= i < len(available_temps)]
        return selected if selected else TEMPERATURE_TESTS
    except:
        print("Invalid selection. Using default temperatures.")
        return TEMPERATURE_TESTS

def resolve_prompt_path(path):
    """Resolve prompt file path - tries current dir, then demos/prompts/, then project root"""
    if not path:
        return None

    # If absolute path, use as-is
    if os.path.isabs(path):
        return path if os.path.exists(path) else None

    # Try current directory
    if os.path.exists(path):
        return path

    # Try relative to demos/prompts/ directory
    demos_dir = os.path.dirname(os.path.abspath(__file__))
    prompts_path = os.path.join(demos_dir, 'prompts', path)
    if os.path.exists(prompts_path):
        return prompts_path

    # Try project root + path
    project_root = os.path.dirname(demos_dir)
    root_path = os.path.join(project_root, path)
    if os.path.exists(root_path):
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
        except:
            # If file read fails, treat as direct text
            pass

    # Treat as direct text
    return user_input

def save_results_to_json(results_data, filename):
    """Save test results to JSON file"""
    with open(filename, 'w') as f:
        json.dump(results_data, f, indent=2)
    print(f"\n✓ Results saved to: {filename}")

def export_results_to_markdown(results_data, json_filename):
    """Export results to markdown format"""
    md_filename = json_filename.replace('.json', '.md')

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
        f.write("| Model | Temperature | TPS | Tokens | Time | Total Duration | Description |\n")
        f.write("|-------|-------------|-----|--------|------|----------------|-------------|\n")

        for result in results_data['results']:
            model = result['model']
            temp = result['temperature']
            tps = result['metrics'].get('tokens_per_second', 0)
            tokens = result['metrics'].get('completion_tokens', 0)
            elapsed = format_duration(result['elapsed_time'])
            total_dur = format_duration(result['metrics'].get('total_duration_s', 0))
            desc = result['description']

            f.write(f"| {model} | {temp} | {tps:.2f} | {tokens} | {elapsed} | {total_dur} | {desc} |\n")

        # Full responses
        f.write("\n## Detailed Responses\n\n")

        for result in results_data['results']:
            f.write(f"### {result['model']} - Temperature {result['temperature']} ({result['description']})\n\n")
            f.write(f"**Response:**\n> {result['response']}\n\n")
            f.write(f"**Metrics:**\n")
            f.write(f"- TPS: {result['metrics'].get('tokens_per_second', 0):.2f}\n")
            f.write(f"- Tokens: {result['metrics'].get('completion_tokens', 0)}\n")
            f.write(f"- Time: {format_duration(result['elapsed_time'])}\n")
            f.write(f"- Total Duration: {format_duration(result['metrics'].get('total_duration_s', 0))}\n\n")

        # Summary insights
        if 'summary' in results_data:
            f.write("## Summary Statistics\n\n")
            summary = results_data['summary']
            for key, value in summary.items():
                f.write(f"- **{key.replace('_', ' ').title()}**: {value}\n")

    print(f"✓ Markdown export saved to: {md_filename}")
    return md_filename
