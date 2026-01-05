#!/usr/bin/env python3
"""
Coreference Resolution Test Runner

Tests pronoun resolution capabilities of LLM models across multiple languages.
Compares model responses against gold standard annotations and calculates
accuracy metrics.

Usage:
    # Interactive mode
    python coreference_test.py

    # Non-interactive with defaults
    python coreference_test.py --default

    # Specify model
    python coreference_test.py --model llama3.2

    # Specify prompt file
    python coreference_test.py --prompt gemma3b-prompt.txt

    # Specify dataset
    python coreference_test.py --dataset custom_testset.json
"""

import sys
import argparse
from datetime import datetime
from pathlib import Path

from coreference_utils import (
    load_prompt_template,
    load_test_dataset,
    send_to_model,
    extract_referent_from_response,
    extract_entity_type_from_response,
    match_referent,
    calculate_metrics,
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


def get_available_prompts():
    """List available prompt files from the prompts directory."""
    if not PROMPTS_DIR.exists():
        return []

    prompts = []
    for f in sorted(PROMPTS_DIR.iterdir()):
        if f.is_file() and f.suffix == '.txt':
            prompts.append(f.name)
    return prompts


def select_prompt(available_prompts):
    """Interactive prompt selection or custom prompt entry."""
    print("\nAvailable prompts:")
    for i, prompt in enumerate(available_prompts, 1):
        print(f"  {i}. {prompt}")
    print(f"  {len(available_prompts) + 1}. [Enter custom prompt]")

    print("\nEnter prompt number (or press Enter for default 'pronoun_resolution.txt'):")
    user_input = input("> ").strip()

    if user_input == '':
        default = 'pronoun_resolution.txt'
        if default in available_prompts:
            return default, None  # (filename, custom_text)
        return available_prompts[0] if available_prompts else None, None

    try:
        idx = int(user_input) - 1
        # Check if user selected "custom prompt" option
        if idx == len(available_prompts):
            print("\nEnter your custom prompt (use {text} and {pronoun} as placeholders):")
            print("(End with an empty line)")
            lines = []
            while True:
                line = input()
                if line == '':
                    break
                lines.append(line)
            custom_prompt = '\n'.join(lines)
            if '{text}' not in custom_prompt:
                print("Warning: Your prompt doesn't contain {text} placeholder. Adding default ending.")
                custom_prompt += "\n\nSENTENCE: {text}\nPRONOUN: {pronoun}\n"
            return None, custom_prompt  # (None filename, custom_text)

        if 0 <= idx < len(available_prompts):
            return available_prompts[idx], None
    except ValueError:
        pass

    print(f"Invalid selection. Using {available_prompts[0]}")
    return available_prompts[0], None


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


def get_output_filename(model_name, prompt_name=None, timestamp=None):
    """Generate output filename with datetime, model name, and prompt name."""
    if timestamp is None:
        timestamp = datetime.now()
    datetime_str = timestamp.strftime("%Y%m%d_%H%M%S")
    # Sanitize model name for filename (replace : and / with -)
    safe_model = model_name.replace(':', '-').replace('/', '-')
    # Include prompt name (without extension) if provided
    if prompt_name:
        safe_prompt = Path(prompt_name).stem.replace(' ', '_')
        return OUTPUT_DIR / f"{datetime_str}-{safe_model}-{safe_prompt}.json"
    return OUTPUT_DIR / f"{datetime_str}-{safe_model}.json"


def run_single_test(test_case, prompt_template, model, temperature=None, timeout=None):
    """
    Run a single test case.

    Args:
        test_case: Dictionary with text, pronoun, gold_truth
        prompt_template: The prompt template string
        model: Model name
        temperature: Optional temperature
        timeout: Optional timeout in seconds (None = adaptive)

    Returns:
        dict: Result with model answer, correctness, timing
    """
    # Format the prompt
    prompt = prompt_template.format(
        text=test_case['text'],
        pronoun=test_case['gold_truth']['pronoun']
    )

    # Send to model (timeout=None means adaptive timeout based on prompt length)
    response, elapsed_time, metrics = send_to_model(prompt, model, temperature, timeout)

    if response is None:
        return {
            'doc_key': test_case['doc_key'],
            'lang': test_case['lang'],
            'text': test_case['text'],
            'pronoun': test_case['gold_truth']['pronoun'],
            'gold_referent': test_case['gold_truth']['referent'],
            'model_answer': None,
            'model_raw_response': None,
            'correct': False,
            'elapsed_time': elapsed_time,
            'error': 'No response from model'
        }

    # Extract referent from response
    model_answer = extract_referent_from_response(response)

    # Extract entity type if provided (useful for NER integration)
    entity_type = extract_entity_type_from_response(response)

    # Check if correct (match_type: 'exact', 'partial', 'fuzzy (edit_dist=N)', or 'none')
    is_correct, norm_model, norm_gold, match_type = match_referent(
        model_answer,
        test_case['gold_truth']['referent'],
        test_case['lang']
    )

    return {
        'doc_key': test_case['doc_key'],
        'lang': test_case['lang'],
        'text': test_case['text'],
        'pronoun': test_case['gold_truth']['pronoun'],
        'gold_referent': test_case['gold_truth']['referent'],
        'gold_reason': test_case['gold_truth'].get('reason', ''),
        'model_answer': model_answer,
        'entity_type': entity_type,  # For NER integration (PERSON, ORG, LOC, etc.)
        'model_raw_response': response,
        'normalized_model': norm_model,
        'normalized_gold': norm_gold,
        'match_type': match_type,  # How the match was determined (exact/partial/fuzzy/none)
        'correct': is_correct,
        'elapsed_time': elapsed_time,
        'metrics': metrics
    }


def run_all_tests(test_cases, prompt_template, model, temperature=None, timeout=None, save_callback=None):
    """
    Run all test cases and collect results.

    Args:
        test_cases: List of test case dictionaries
        prompt_template: The prompt template string
        model: Model name
        temperature: Optional temperature
        timeout: Optional timeout in seconds (None = adaptive)
        save_callback: Optional function(results) to call after each test for progressive saving

    Returns:
        list: List of result dictionaries
    """
    results = []
    total = len(test_cases)
    times = []  # Track individual test times for running average

    for i, test_case in enumerate(test_cases, 1):
        # Print test header without newline - result will be appended
        print(f"[{i}/{total}] {test_case['doc_key']}: ", end="", flush=True)

        result = run_single_test(test_case, prompt_template, model, temperature, timeout)
        results.append(result)
        times.append(result.get('elapsed_time', 0))

        # Calculate progress stats
        completed = len(results)
        remaining = total - completed
        correct_count = sum(1 for r in results if r.get('correct', False))
        accuracy = (correct_count / completed * 100) if completed > 0 else 0
        avg_time = sum(times) / len(times) if times else 0
        eta_seconds = avg_time * remaining

        # Get Ollama metrics if available (tokens/sec)
        tps_info = ""
        if result.get('metrics', {}).get('tokens_per_second'):
            tps = result['metrics']['tokens_per_second']
            tps_info = f" {tps:.1f}t/s"

        # Format ETA compactly
        if eta_seconds < 60:
            eta_str = f"{eta_seconds:.0f}s"
        elif eta_seconds < 3600:
            eta_str = f"{int(eta_seconds // 60)}m"
        else:
            eta_str = f"{int(eta_seconds // 3600)}h{int((eta_seconds % 3600) // 60)}m"

        # Result on same line - very compact format
        status = "OK" if result['correct'] else "FAIL"
        model_ans = (result['model_answer'] or '-')[:20]

        # Single line per test: [1/30] test_name: OK 'answer' | 75% | ETA:30m | 1.5t/s
        print(f"{status} '{model_ans}' | {accuracy:.0f}% | ETA:{eta_str}{tps_info}")

        # Progressive save after each test
        if save_callback:
            save_callback(results)

    print()  # Final newline
    return results


def display_results(results_data):
    """Display results summary to console."""
    metrics = results_data.get('metrics', {})
    overall = metrics.get('overall', {})

    print()
    print("=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    print(f"\nOverall Accuracy: {overall.get('accuracy', 0)}%")
    print(f"Correct: {overall.get('correct', 0)} / {overall.get('total', 0)}")

    # Per-language breakdown
    by_lang = metrics.get('by_language', {})
    if by_lang:
        print("\nBy Language:")
        for lang, stats in sorted(by_lang.items()):
            print(f"  {lang}: {stats['accuracy']}% ({stats['correct']}/{stats['total']})")

    # Timing
    timing = metrics.get('timing', {})
    print(f"\nTiming:")
    print(f"  Average: {timing.get('average_s', 0):.3f}s")
    print(f"  Total: {format_duration(timing.get('total_s', 0))}")

    # Show incorrect cases
    print("\nIncorrect Cases:")
    incorrect = [r for r in results_data.get('results', []) if not r.get('correct')]
    if incorrect:
        for r in incorrect:
            print(f"  - {r['doc_key']}: Expected '{r['gold_referent']}', got '{r['model_answer']}'")
    else:
        print("  None! All correct.")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Test coreference/pronoun resolution across multiple languages',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode
  python coreference_test.py

  # Non-interactive with default model
  python coreference_test.py --default

  # Specify model
  python coreference_test.py --model llama3.2

  # Specify prompt file
  python coreference_test.py --prompt gemma3b-prompt.txt

  # Specify dataset file
  python coreference_test.py --dataset my_testset.json

  # Set temperature
  python coreference_test.py --temperature 0.1
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
        help='Prompt template file (from prompts/ folder or full path)'
    )
    parser.add_argument(
        '--dataset',
        type=str,
        default=None,
        help='Path to dataset JSON file'
    )
    parser.add_argument(
        '--temperature',
        type=float,
        default=0.1,
        help='Temperature setting (default: 0.1 for deterministic output)'
    )
    parser.add_argument(
        '--host',
        type=str,
        default=None,
        help='API host URL (default: read from wrapper_config.toml)'
    )
    parser.add_argument(
        '--timeout',
        type=int,
        default=None,
        help='Request timeout in seconds (default: adaptive based on text length, max 600s)'
    )

    args = parser.parse_args()

    # Override host if specified
    if args.host:
        set_host(args.host)
        # Re-import to get updated value
        from coreference_utils import HOST as current_host
    else:
        current_host = HOST

    print("=" * 70)
    print("COREFERENCE RESOLUTION TEST")
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

    # Get current active model from wrapper
    ollama_config = get_ollama_config()
    active_model = ollama_config.get('active_model')

    # Select model
    if args.model:
        # User specified a model via CLI
        if args.model in available_models:
            model = args.model
        else:
            print(f"Warning: Model '{args.model}' not found. Available: {available_models}")
            sys.exit(1)
    elif args.default:
        # User wants default (first available)
        model = available_models[0]
        print(f"Using default model: {model}")
    else:
        # Interactive mode - ask if user wants to use active model
        if active_model and active_model in available_models:
            print(f"\nCurrent active model in wrapper: {active_model}")
            use_active = input("Use this model? (Y/n): ").strip().lower()
            if use_active in ('', 'y', 'yes'):
                model = active_model
                print(f"Using active model: {model}")
            else:
                model = select_model(available_models)
        else:
            # No active model or it's not available, show selection
            model = select_model(available_models)

    print(f"Selected model: {model}")

    # Load dataset
    dataset_path = args.dataset
    if dataset_path:
        dataset_path = Path(dataset_path)
        if not dataset_path.is_absolute():
            # Try relative to datasets folder
            datasets_dir = SCRIPT_DIR / 'datasets'
            if (datasets_dir / dataset_path).exists():
                dataset_path = datasets_dir / dataset_path

    test_cases = load_test_dataset(dataset_path)
    print(f"Loaded {len(test_cases)} test cases.")

    # Select prompt template
    available_prompts = get_available_prompts()
    if not available_prompts:
        print("Error: No prompt files found in prompts/ directory.")
        sys.exit(1)

    custom_prompt = None
    if args.prompt:
        # Check if it's a full path or just filename
        prompt_path = Path(args.prompt)
        if prompt_path.is_absolute() and prompt_path.exists():
            prompt_file = prompt_path
        elif args.prompt in available_prompts:
            prompt_file = PROMPTS_DIR / args.prompt
        elif (PROMPTS_DIR / args.prompt).exists():
            prompt_file = PROMPTS_DIR / args.prompt
        else:
            print(f"Error: Prompt file '{args.prompt}' not found.")
            print(f"Available prompts: {available_prompts}")
            sys.exit(1)
        prompt_name = prompt_file.name
    elif args.default:
        prompt_name = 'pronoun_resolution.txt'
        prompt_file = PROMPTS_DIR / prompt_name
        if not prompt_file.exists():
            prompt_file = PROMPTS_DIR / available_prompts[0]
            prompt_name = available_prompts[0]
        print(f"Using default prompt: {prompt_name}")
    else:
        prompt_name, custom_prompt = select_prompt(available_prompts)
        if prompt_name:
            prompt_file = PROMPTS_DIR / prompt_name

    # Load prompt template (from file or custom)
    if custom_prompt:
        prompt_template = custom_prompt
        prompt_name = "custom"
        print("Using custom prompt")
    else:
        print(f"Selected prompt: {prompt_name}")
        prompt_template = load_prompt_template(prompt_file)

    # Show configuration
    print(f"\nConfiguration:")
    print(f"  Model: {model}")
    print(f"  Prompt: {prompt_name}")
    print(f"  Temperature: {args.temperature}")
    print(f"  Timeout: {args.timeout if args.timeout else 'adaptive (60-600s based on text length)'}")
    print(f"  Test cases: {len(test_cases)}")

    if not args.default:
        print("\nPress Enter to start testing (or Ctrl+C to cancel)...")
        try:
            input()
        except KeyboardInterrupt:
            print("\n\nTest cancelled by user.")
            sys.exit(0)

    # Run tests
    print("\n" + "=" * 70)
    print("RUNNING TESTS")
    print("=" * 70 + "\n")

    start_time = datetime.now()

    # Get Ollama configuration for metadata
    ollama_config = get_ollama_config()
    ollama_host = ollama_config.get('host', 'unknown')
    ollama_label = ollama_config.get('label', '')
    wrapper_api_host = current_host  # Record which wrapper API we're connecting to

    # Create output filename early (with timestamp from start)
    output_file = get_output_filename(model, prompt_name, start_time)
    print(f"Results will be saved to: {output_file.name}")
    print()

    # Progressive save callback - saves after each test
    def save_progress(current_results):
        current_time = datetime.now()
        current_duration = (current_time - start_time).total_seconds()
        current_metrics = calculate_metrics(current_results)

        progress_data = {
            'metadata': {
                'model': model,
                'prompt': prompt_name,
                'temperature': args.temperature,
                'timestamp': start_time.isoformat(),
                'last_update': current_time.isoformat(),
                'status': 'in_progress',
                'completed': len(current_results),
                'total': len(test_cases),
                'duration_s': current_duration,
                'duration_readable': format_duration(current_duration),
                'test_cases_count': len(test_cases),
                'wrapper_api': wrapper_api_host,
                'ollama_host': ollama_host,
                'ollama_label': ollama_label
            },
            'metrics': current_metrics,
            'results': current_results
        }
        export_results_to_json(progress_data, output_file)

    results = run_all_tests(test_cases, prompt_template, model, args.temperature, args.timeout, save_callback=save_progress)
    end_time = datetime.now()
    total_duration = (end_time - start_time).total_seconds()

    # Calculate final metrics
    metrics = calculate_metrics(results)

    # Build final results data
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
            'test_cases_count': len(test_cases),
            'wrapper_api': wrapper_api_host,
            'ollama_host': ollama_host,
            'ollama_label': ollama_label
        },
        'metrics': metrics,
        'results': results
    }

    # Final save
    export_results_to_json(results_data, output_file)
    export_results_to_markdown(results_data, output_file.with_suffix('.md'))

    print(f"\nResults saved to: {output_file}")
    print(f"Markdown report: {output_file.with_suffix('.md')}")

    # Display summary
    display_results(results_data)

    print("\n" + "=" * 70)
    print("TEST COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nTest cancelled by user (Ctrl+C).")
        print("Exiting gracefully.")
        sys.exit(0)
