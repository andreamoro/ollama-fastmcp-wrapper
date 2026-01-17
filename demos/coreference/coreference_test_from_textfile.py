#!/usr/bin/env python3
"""
Coreference Resolution Test from Text Files

Analyzes .txt files for coreference resolution using selected prompts and models.
Unlike coreference_test.py which uses JSON test cases with gold truth,
this script analyzes full text files without correctness scoring.

Usage:
    First load a local or a remote server
    uv run ollama_wrapper.py api --ollama-model xyz --ollama-host localhost --ollama-port 11435 --ollama-label "Contabo remote server"

    Then choose in another terminal session run one of the following:

    # Interactive mode
    uv run python coreference_test_from_textfile.py

    # Non-interactive with defaults
    uv run python coreference_test_from_textfile.py --default

    # Specify model
    uv run python coreference_test_from_textfile.py --model llama3.2

    # Specify prompt file(s)
    uv run python coreference_test_from_textfile.py --prompt coref_3_ita.txt
    uv run python coreference_test_from_textfile.py --prompts coref_1_ita.txt coref_2_ita.txt coref_3_ita.txt

    # Run all prompts against a text file
    uv run python coreference_test_from_textfile.py --all-prompts --texts sample.txt

    # Specify text file(s)
    uv run python coreference_test_from_textfile.py --texts sample1.txt sample2.txt

    # Analyze all .txt files
    uv run python coreference_test_from_textfile.py --all-texts

    # Connect to a different host/port
    uv run python coreference_test_from_textfile.py --wrapper-host 192.168.1.10 --wrapper-port 9000
"""

import sys
import re
import argparse
from datetime import datetime
from pathlib import Path

from coreference_utils import (
    load_prompt_template,
    send_to_model,
    format_duration,
    check_wrapper_running,
    get_available_models,
    get_ollama_config,
    export_results_to_json,
    export_results_to_markdown,
    calculate_adaptive_timeout,
    set_host,
    HOST
)

# Directories
SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "results"
PROMPTS_DIR = SCRIPT_DIR / "prompts"
DATASETS_DIR = SCRIPT_DIR / "datasets"


def get_available_texts():
    """List available .txt files from the datasets directory."""
    if not DATASETS_DIR.exists():
        return []

    texts = []
    for f in sorted(DATASETS_DIR.iterdir()):
        if f.is_file() and f.suffix == '.txt':
            texts.append(f.name)
    return texts


def get_available_prompts():
    """List available prompt files from the prompts directory."""
    if not PROMPTS_DIR.exists():
        return []

    prompts = []
    for f in sorted(PROMPTS_DIR.iterdir()):
        if f.is_file() and f.suffix == '.txt':
            prompts.append(f.name)
    return prompts


def select_texts(available_texts):
    """Interactive text file selection (multi-select)."""
    print("\nAvailable text files:")
    for i, text in enumerate(available_texts, 1):
        print(f"  {i}. {text}")
    print("  a. [Select all]")

    print("\nEnter text number(s) separated by comma, or 'a' for all:")
    user_input = input("> ").strip()

    if user_input.lower() == 'a':
        return available_texts

    selected = []
    try:
        indices = [int(x.strip()) - 1 for x in user_input.split(',')]
        for idx in indices:
            if 0 <= idx < len(available_texts):
                selected.append(available_texts[idx])
    except ValueError:
        pass

    if not selected:
        print(f"Invalid selection. Using first file: {available_texts[0]}")
        return [available_texts[0]]

    return selected


def select_prompt(available_prompts):
    """Interactive prompt selection (multi-select)."""
    print("\nAvailable prompts:")
    for i, prompt in enumerate(available_prompts, 1):
        print(f"  {i}. {prompt}")
    print("  a. [Select all]")

    print("\nEnter prompt number(s) separated by comma, 'a' for all, or press Enter for 'coref_3.txt':")
    user_input = input("> ").strip()

    if user_input == '':
        default = 'coref_3.txt'
        if default in available_prompts:
            return [default]
        return [available_prompts[0]] if available_prompts else []

    if user_input.lower() == 'a':
        return available_prompts

    selected = []
    try:
        indices = [int(x.strip()) - 1 for x in user_input.split(',')]
        for idx in indices:
            if 0 <= idx < len(available_prompts):
                selected.append(available_prompts[idx])
    except ValueError:
        pass

    if not selected:
        print(f"Invalid selection. Using {available_prompts[0]}")
        return [available_prompts[0]]

    return selected


def select_model(available_models):
    """Interactive model selection."""
    print("\nAvailable models:")
    for i, model in enumerate(available_models, 1):
        print(f"  {i}. {model}")

    print("\nEnter model number (or press Enter for first model):")
    user_input = input("> ").strip()

    if user_input == '':
        return available_models[0]

    try:
        idx = int(user_input) - 1
        if 0 <= idx < len(available_models):
            return available_models[idx]
    except ValueError:
        pass

    print(f"Invalid selection. Using {available_models[0]}")
    return available_models[0]


def load_text_file(filename):
    """Load text content from a file in the datasets directory."""
    filepath = DATASETS_DIR / filename
    with open(filepath, 'r', encoding='utf-8') as f:
        return f.read()


def get_output_filename(model_name, prompt_name=None, source_file=None, timestamp=None):
    """Generate output filename with datetime, model name, prompt name, and source file."""
    if timestamp is None:
        timestamp = datetime.now()
    datetime_str = timestamp.strftime("%Y%m%d_%H%M%S")
    safe_model = model_name.replace(':', '-').replace('/', '-')

    parts = [datetime_str, safe_model]

    if prompt_name:
        safe_prompt = Path(prompt_name).stem.replace(' ', '_')
        parts.append(safe_prompt)

    if source_file:
        safe_source = Path(source_file).stem.replace(' ', '_')
        parts.append(safe_source)

    parts.append("textfile")

    return OUTPUT_DIR / f"{'-'.join(parts)}.json"


def run_text_analysis(text_content, prompt_template, model, temperature=None, timeout=None):
    """
    Run coreference analysis on a text file.

    Args:
        text_content: The full text content to analyze
        prompt_template: The prompt template string (should have {text} placeholder)
        model: Model name
        temperature: Optional temperature
        timeout: Optional timeout in seconds

    Returns:
        dict: Result with model response and timing
    """
    # Format the prompt with the text
    # Use simple string replacement to avoid issues with JSON braces in prompts
    prompt = prompt_template.replace('{text}', text_content)

    # Send to model
    response, elapsed_time, metrics = send_to_model(prompt, model, temperature, timeout)

    return {
        'model_response': response,
        'elapsed_time': elapsed_time,
        'metrics': metrics,
        'error': None if response else 'No response from model'
    }


def export_textfile_results_to_markdown(results_data, filename):
    """Export text file analysis results to Markdown."""
    filepath = Path(filename)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    with open(filepath, 'w', encoding='utf-8') as f:
        f.write("# Coreference Analysis from Text Files\n\n")

        # Metadata
        meta = results_data.get('metadata', {})
        f.write("## Analysis Information\n\n")
        f.write(f"- **Model**: {meta.get('model', 'N/A')}\n")
        f.write(f"- **Prompt**: {meta.get('prompt', 'N/A')}\n")
        f.write(f"- **Temperature**: {meta.get('temperature', 'N/A')}\n")
        f.write(f"- **Timestamp**: {meta.get('timestamp', 'N/A')}\n")
        f.write(f"- **Total Duration**: {meta.get('total_duration_readable', 'N/A')}\n")
        f.write(f"- **Files Analyzed**: {len(results_data.get('results', []))}\n\n")

        # Results for each file
        f.write("## Results\n\n")
        for i, r in enumerate(results_data.get('results', []), 1):
            f.write(f"### {i}. {r.get('source_file', 'Unknown')}\n\n")
            f.write(f"**Text length**: {r.get('text_length', 0)} characters\n")
            f.write(f"**Processing time**: {r.get('elapsed_time', 0):.2f}s\n\n")

            f.write("**Model Response**:\n\n")
            response = r.get('model_response', 'No response')
            if response:
                # Check if response already has markdown code blocks
                if response.strip().startswith('```'):
                    # Already formatted, don't double-wrap
                    f.write(f"{response}\n\n")
                else:
                    # Plain text, wrap it
                    f.write(f"```\n{response}\n```\n\n")
            else:
                f.write("*No response received*\n\n")

            f.write("---\n\n")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Analyze text files for coreference resolution',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode
  uv run python coreference_test_from_textfile.py

  # Non-interactive with defaults
  uv run python coreference_test_from_textfile.py --default

  # Specify model
  uv run python coreference_test_from_textfile.py --model llama3.2

  # Specify prompt file
  uv run python coreference_test_from_textfile.py --prompt coref_3.txt

  # Specify text file(s)
  uv run python coreference_test_from_textfile.py --texts sample1.txt sample2.txt

  # Analyze all .txt files
  uv run python coreference_test_from_textfile.py --all-texts
        """
    )

    parser.add_argument(
        '--default',
        action='store_true',
        help='Non-interactive mode with default settings'
    )
    parser.add_argument(
        '--model',
        type=str,
        default=None,
        help='Model name to use'
    )
    parser.add_argument(
        '--prompt',
        type=str,
        default=None,
        help='Single prompt template file (from prompts/ folder or full path)'
    )
    parser.add_argument(
        '--prompts',
        type=str,
        nargs='+',
        default=None,
        help='Multiple prompt template files (from prompts/ folder)'
    )
    parser.add_argument(
        '--all-prompts',
        action='store_true',
        help='Use all prompt files in prompts/ folder'
    )
    parser.add_argument(
        '--texts',
        type=str,
        nargs='+',
        default=None,
        help='Text file(s) to analyze (from datasets/ folder)'
    )
    parser.add_argument(
        '--all-texts',
        action='store_true',
        help='Analyze all .txt files in datasets/ folder'
    )
    parser.add_argument(
        '--temperature',
        type=float,
        default=0.1,
        help='Temperature setting (default: 0.1)'
    )
    parser.add_argument(
        '--wrapper-host',
        type=str,
        default=None,
        help='Wrapper API host (e.g., localhost, 192.168.1.10). Default: read from wrapper_config.toml'
    )
    parser.add_argument(
        '--wrapper-port',
        type=int,
        default=None,
        help='Wrapper API port (e.g., 9000). Default: read from wrapper_config.toml'
    )
    parser.add_argument(
        '--timeout',
        type=int,
        default=None,
        help='Request timeout in seconds (default: adaptive)'
    )

    args = parser.parse_args()

    # Build API URL from host/port if specified
    if args.wrapper_host or args.wrapper_port:
        # Start with defaults from config
        base_host = "localhost"
        base_port = 8000
        # Parse existing HOST to get defaults
        match = re.match(r'https?://([^:]+):(\d+)', HOST)
        if match:
            base_host = match.group(1)
            base_port = int(match.group(2))
        # Override with args
        final_host = args.wrapper_host if args.wrapper_host else base_host
        final_port = args.wrapper_port if args.wrapper_port else base_port
        api_url = f"http://{final_host}:{final_port}"
        set_host(api_url)
        current_host = api_url
    else:
        current_host = HOST

    print("=" * 70)
    print("COREFERENCE ANALYSIS FROM TEXT FILES")
    print("=" * 70)

    # Check wrapper
    print(f"\nChecking API at {current_host}...")
    is_running, error_msg = check_wrapper_running()
    if not is_running:
        print(f"Error: {error_msg}")
        sys.exit(1)
    print("API is ready.")

    # Get available models
    available_models = get_available_models()
    if not available_models:
        print("Error: No models available.")
        sys.exit(1)

    # Get available text files
    available_texts = get_available_texts()
    if not available_texts:
        print(f"Error: No .txt files found in {DATASETS_DIR}")
        print("Please add text files to analyze.")
        sys.exit(1)

    # Get current active model from wrapper
    ollama_config = get_ollama_config()
    active_model = ollama_config.get('active_model')

    # Select model
    if args.model:
        if args.model in available_models:
            model = args.model
        else:
            print(f"Warning: Model '{args.model}' not found. Available: {available_models}")
            sys.exit(1)
    elif args.default:
        model = available_models[0]
        print(f"Using default model: {model}")
    else:
        if active_model and active_model in available_models:
            print(f"\nCurrent active model in wrapper: {active_model}")
            use_active = input("Use this model? (Y/n): ").strip().lower()
            if use_active in ('', 'y', 'yes'):
                model = active_model
                print(f"Using active model: {model}")
            else:
                model = select_model(available_models)
        else:
            model = select_model(available_models)

    print(f"Selected model: {model}")

    # Select text files
    if args.texts:
        text_files = []
        for t in args.texts:
            if t in available_texts:
                text_files.append(t)
            elif (DATASETS_DIR / t).exists():
                text_files.append(t)
            else:
                print(f"Warning: Text file '{t}' not found, skipping.")
        if not text_files:
            print("Error: No valid text files specified.")
            sys.exit(1)
    elif args.all_texts:
        text_files = available_texts
        print(f"Analyzing all {len(text_files)} text files.")
    elif args.default:
        text_files = [available_texts[0]]
        print(f"Using default text file: {text_files[0]}")
    else:
        text_files = select_texts(available_texts)

    print(f"Selected text files: {text_files}")

    # Select prompt template(s)
    available_prompts = get_available_prompts()
    if not available_prompts:
        print("Error: No prompt files found in prompts/ directory.")
        sys.exit(1)

    # Determine which prompts to use
    prompt_names = []
    if args.all_prompts:
        prompt_names = available_prompts
        print(f"Using all {len(prompt_names)} prompts.")
    elif args.prompts:
        for p in args.prompts:
            if p in available_prompts:
                prompt_names.append(p)
            elif (PROMPTS_DIR / p).exists():
                prompt_names.append(p)
            else:
                print(f"Warning: Prompt file '{p}' not found, skipping.")
        if not prompt_names:
            print("Error: No valid prompt files specified.")
            sys.exit(1)
    elif args.prompt:
        prompt_path = Path(args.prompt)
        if prompt_path.is_absolute() and prompt_path.exists():
            prompt_names = [prompt_path.name]
        elif args.prompt in available_prompts:
            prompt_names = [args.prompt]
        elif (PROMPTS_DIR / args.prompt).exists():
            prompt_names = [args.prompt]
        else:
            print(f"Error: Prompt file '{args.prompt}' not found.")
            print(f"Available prompts: {available_prompts}")
            sys.exit(1)
    elif args.default:
        default_prompt = 'coref_3.txt'
        if default_prompt in available_prompts:
            prompt_names = [default_prompt]
        else:
            prompt_names = [available_prompts[0]]
        print(f"Using default prompt: {prompt_names[0]}")
    else:
        prompt_names = select_prompt(available_prompts)

    print(f"Selected prompts: {prompt_names}")

    # Show configuration
    print(f"\nConfiguration:")
    print(f"  Model: {model}")
    print(f"  Prompts: {len(prompt_names)}")
    for pn in prompt_names:
        print(f"    - {pn}")
    print(f"  Temperature: {args.temperature}")
    print(f"  Timeout: {args.timeout if args.timeout else 'adaptive'}")
    print(f"  Text files: {len(text_files)}")
    print(f"  Total runs: {len(prompt_names) * len(text_files)}")

    if not args.default:
        print("\nPress Enter to start analysis (or Ctrl+C to cancel)...")
        try:
            input()
        except KeyboardInterrupt:
            print("\n\nAnalysis cancelled by user.")
            sys.exit(0)

    # Get Ollama configuration for metadata
    ollama_host = ollama_config.get('host', 'unknown')
    ollama_label = ollama_config.get('label', '')
    wrapper_api_host = current_host

    # Run analysis for each (prompt, text_file) combination
    # Each text file gets its own output file
    all_output_files = []
    total_combinations = len(prompt_names) * len(text_files)
    combination_idx = 0

    for prompt_idx, prompt_name in enumerate(prompt_names, 1):
        print("\n" + "=" * 70)
        print(f"PROMPT [{prompt_idx}/{len(prompt_names)}]: {prompt_name}")
        print("=" * 70)

        prompt_file = PROMPTS_DIR / prompt_name
        prompt_template = load_prompt_template(prompt_file)

        for text_idx, text_filename in enumerate(text_files, 1):
            combination_idx += 1
            print(f"\n[{combination_idx}/{total_combinations}] Analyzing {text_filename} with {prompt_name}...")

            start_time = datetime.now()

            # Create output filename for this (prompt, text_file) combination
            output_file = get_output_filename(model, prompt_name, text_filename, start_time)
            print(f"  Output: {output_file.name}")

            # Load the text file
            text_content = load_text_file(text_filename)

            # Calculate the prompt length and timeout
            prompt = prompt_template.replace('{text}', text_content)
            effective_timeout = args.timeout if args.timeout else calculate_adaptive_timeout(len(prompt))
            print(f"  Timeout: {effective_timeout}s (prompt: {len(prompt)} chars)")

            # Run analysis
            result = run_text_analysis(text_content, prompt_template, model, args.temperature, args.timeout)
            result['source_file'] = text_filename
            result['text_length'] = len(text_content)
            result['full_prompt'] = prompt  # Preserve for reproducibility

            end_time = datetime.now()
            total_duration = (end_time - start_time).total_seconds()

            # Get Ollama metrics if available
            tps_info = ""
            if result.get('metrics', {}).get('tokens_per_second'):
                tps = result['metrics']['tokens_per_second']
                tps_info = f" | {tps:.1f}t/s"

            status = "OK" if result['model_response'] else "FAIL"
            elapsed = result['elapsed_time']
            if status == "FAIL":
                print(f"  Status: {status} (elapsed: {elapsed:.1f}s / timeout: {effective_timeout}s)")
            else:
                print(f"  Status: {status} ({elapsed:.1f}s){tps_info}")

            # Build results data for this single file
            results_data = {
                'metadata': {
                    'model': model,
                    'prompt': prompt_name,
                    'temperature': args.temperature,
                    'timestamp': start_time.isoformat(),
                    'end_timestamp': end_time.isoformat(),
                    'status': 'completed',
                    'total_duration_s': total_duration,
                    'total_duration_readable': format_duration(total_duration),
                    'source_files': [text_filename],
                    'files_count': 1,
                    'wrapper_api': wrapper_api_host,
                    'ollama_host': ollama_host,
                    'ollama_label': ollama_label
                },
                'results': [result]
            }

            # Save outputs
            export_results_to_json(results_data, output_file)
            export_textfile_results_to_markdown(results_data, output_file.with_suffix('.md'))
            all_output_files.append(output_file)

            print(f"  Saved: {output_file.name}")

    print("\n" + "=" * 70)
    print("ALL ANALYSES COMPLETE")
    print("=" * 70)
    print(f"\nGenerated {len(all_output_files)} result files:")
    for of in all_output_files:
        print(f"  - {of.name}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nAnalysis cancelled by user (Ctrl+C).")
        print("Exiting gracefully.")
        sys.exit(0)
