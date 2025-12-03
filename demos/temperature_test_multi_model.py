#!/usr/bin/env python3
"""
Enhanced Temperature Test - Compare multiple models with different temperatures
Shows how temperature affects response quality across different model families
"""

import sys
from datetime import datetime
from pathlib import Path
from temperature_test_utils import (
    test_temperature_model,
    format_duration,
    get_available_models,
    get_config_temperature,
    get_config_model,
    get_prompt_from_file_or_input,
    select_temperatures,
    export_results_to_json,
    export_results_to_markdown,
    append_result_to_markdown,
    clean_llm_response_data,
    print_summary,
    format_summary_display,
    DEFAULT_PROMPT,
    ALL_TEMPERATURE_CONFIGS,
    DEFAULT_TEMPERATURE_TESTS
)

# Get the directory where this script is located
SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "test_results"

def select_models(available_models):
    """Allow user to select which models to test"""
    print("\nAvailable models:")
    for i, model in enumerate(available_models, 1):
        print(f"  {i}. {model}")

    print("\nEnter model numbers to test (comma-separated, or 'all' for all models):")
    print("Example: 1,3,5  or  all")

    user_input = input("> ").strip().lower()

    if user_input == 'all':
        return available_models

    try:
        indices = [int(x.strip()) - 1 for x in user_input.split(',')]
        selected = [available_models[i] for i in indices if 0 <= i < len(available_models)]
        return selected
    except Exception as e:
        print(f"Invalid selection ({e}). Using first model.")
        return [available_models[0]] if available_models else []

def get_output_filename():
    """Ask user for output filename, saving it to the predefined directory."""
    print("\nOutput file configuration:")
    print(f"  - Files will be saved in the '{OUTPUT_DIR}' directory.")
    print("  - Press Enter for auto-generated filename (YYYYMMDD_HHMMSS_multi_test_comparison.json)")
    print("  - Enter a filename (will add .json extension if needed)")

    user_input = input("Output filename (or Enter for auto): ").strip()

    if user_input == '':
        # Auto-generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_multi_test_comparison.json"
    else:
        # Add .json extension if not present
        filename = user_input
        if not filename.endswith('.json'):
            filename += '.json'

    # Use Path object and the / operator for clean path joining.
    # This function now returns a pathlib.Path object.
    return Path(OUTPUT_DIR) / filename

def get_output_filename_auto():
    """Auto-generate filename with timestamp, saving it to the predefined directory."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{timestamp}_multi_test_comparison.json"

    # Use Path operator / for clean path joining, returns a Path object
    return Path(OUTPUT_DIR) / filename

def calculate_summary_stats(results):
    """Calculate summary statistics from results"""
    if not results:
        return {}

    summary = {}

    # Find fastest model (highest TPS)
    fastest = max(results, key=lambda r: r['metrics'].get('tokens_per_second', 0))
    summary['fastest_model'] = fastest['model']
    summary['fastest_temperature'] = fastest['temperature']
    summary['highest_tps'] = round(fastest['metrics'].get('tokens_per_second', 0), 2)
    summary['fastest_elapsed_time_s'] = round(fastest['elapsed_time'], 2)
    summary['fastest_elapsed_time_readable'] = fastest['elapsed_time_readable']
    summary['fastest_total_duration_s'] = round(fastest['metrics'].get('total_duration_s', 0), 2)
    summary['fastest_total_duration_readable'] = format_duration(fastest['metrics'].get('total_duration_s', 0))
    summary['fastest_completion_tokens'] = fastest['metrics'].get('completion_tokens', 0)

    # Average statistics
    avg_tokens = sum(r['metrics'].get('completion_tokens', 0) for r in results) / len(results)
    avg_tps = sum(r['metrics'].get('tokens_per_second', 0) for r in results) / len(results)

    summary['average_tokens'] = round(avg_tokens, 2)
    summary['average_tps'] = round(avg_tps, 2)

    # Response length variance
    response_lengths = [len(r['response']) for r in results]
    summary['min_response_length'] = min(response_lengths)
    summary['max_response_length'] = max(response_lengths)
    summary['avg_response_length'] = round(sum(response_lengths) / len(response_lengths), 2)

    return summary

def build_results_data(all_results, selected_models, selected_temps, prompt, config_temp, start_time, status="initialized"):
    """Builds the comprehensive results data structure with metadata, results, and summary."""

    end_time = datetime.now()
    total_duration = (end_time - start_time).total_seconds()

    if status == "initialized":
        total_duration = 0
        end_time = None

    total_tests_count = len(selected_models) * len(selected_temps)

    return {
        "test_metadata": {
            "start_timestamp": start_time.isoformat(),
            "end_timestamp": end_time.isoformat() if end_time else None,
            "total_duration_s": total_duration,
            "total_duration_readable": format_duration(total_duration),
            "prompt": prompt,
            "models_tested": selected_models,
            "temperatures_tested": [t[0] if t[0] is not None else f"default ({config_temp})" for t in selected_temps],
            "total_tests": total_tests_count,
            "completed_tests": len(all_results),
            "status": status
        },
        "results": all_results,
        "summary": calculate_summary_stats(all_results) if all_results else {}
    }

def run_tests(selected_models, selected_temps, prompt, output_filename, config_temp, start_time):
    """Execute all temperature tests with incremental saving to prevent data loss"""

    start_timestamp = start_time.isoformat() # Use passed start_time

    all_results = []
    total_tests = len(selected_models) * len(selected_temps)
    current_test = 0

    # Determine markdown filename once
    md_filename = Path(output_filename).with_suffix('.md')

    for model in selected_models:
        for temp, desc in selected_temps:
            current_test += 1
            print(f"[{current_test}/{total_tests}] Testing {model} with {desc} (temp={temp})...")

            result = test_temperature_model(model, temp, desc, prompt)
            if result:
                # Clean the result immediately after generation (source cleaning)
                result = clean_llm_response_data(result)

                # Add test number and readable elapsed time
                result['test_number'] = current_test
                result['elapsed_time_readable'] = format_duration(result['elapsed_time'])
                all_results.append(result)

                # Progressive save to both JSON and Markdown after each test
                partial_results_data = build_results_data(
                    all_results,
                    selected_models,
                    selected_temps,
                    prompt,
                    config_temp,
                    start_time,
                    status="in_progress"
                )
                export_results_to_json(partial_results_data, output_filename)
                append_result_to_markdown(result, md_filename, is_first_result=(current_test == 1))

    end_time = datetime.now()
    total_duration = (end_time - start_time).total_seconds()

    return {
        'results': all_results,
        'start_timestamp': start_timestamp,
        'end_timestamp': end_time.isoformat(),
        'total_duration': total_duration
    }

def display_results_by_model(all_results, selected_models):
    """Display results organized by model"""
    print()
    print("=" * 80)
    print("RESULTS BY MODEL")
    print("=" * 80)

    for model in selected_models:
        model_results = [r for r in all_results if r['model'] == model]

        if not model_results:
            continue

        print(f"\nModel: {model}")
        print("-" * 80)

        # Table header
        header = f"{'Temperature':<20} {'TPS':<10} {'Tokens':<10} {'Time':<15} {'Total Duration':<15} {'Description':<25} {'Response Preview':<40}"
        print(header)
        print("-" * 80)

        # Table rows
        for r in model_results:
            temp_str = str(r['temperature'])
            tps = r['metrics'].get('tokens_per_second', 0)
            tokens = r['metrics'].get('completion_tokens', 0)
            elapsed = r.get('elapsed_time_readable', format_duration(r['elapsed_time']))
            total_dur = format_duration(r['metrics'].get('total_duration_s', 0))
            desc = r['description']
            response_preview = (r['response'][:37] + "...") if len(r['response']) > 40 else r['response']

            row = f"{temp_str:<20} {tps:<10.2f} {tokens:<10} {elapsed:<15} {total_dur:<15} {desc:<25} {response_preview:<40}"
            print(row)

def display_cross_model_comparison(all_results, selected_models, selected_temps):
    """Display comparison across models at each temperature"""
    if len(selected_models) <= 1:
        return

    config_temp = get_config_temperature()

    print()
    print("=" * 80)
    print("CROSS-MODEL COMPARISON (at each temperature)")
    print("=" * 80)

    for temp, desc in selected_temps:
        temp_display = temp if temp is not None else f"default ({config_temp})"
        temp_results = [r for r in all_results if r['temperature'] == temp_display]

        if not temp_results:
            continue

        print(f"\nTemperature: {temp_display} - {desc}")
        print("-" * 80)

        # Table header
        header = f"{'Model':<35} {'TPS':<10} {'Tokens':<10} {'Time':<15} {'Total Duration':<15} {'Response Preview':<50}"
        print(header)
        print("-" * 80)

        for r in temp_results:
            model_name = r['model']
            tps = r['metrics'].get('tokens_per_second', 0)
            tokens = r['metrics'].get('completion_tokens', 0)
            elapsed = r.get('elapsed_time_readable', format_duration(r['elapsed_time']))
            total_dur = format_duration(r['metrics'].get('total_duration_s', 0))
            response_preview = (r['response'][:47] + "...") if len(r['response']) > 50 else r['response']

            row = f"{model_name:<35} {tps:<10.2f} {tokens:<10} {elapsed:<15} {total_dur:<15} {response_preview:<50}"
            print(row)

def display_summary(results_data, selected_models):
    """Display summary statistics and insights"""
    print()
    print("=" * 80)
    print("SUMMARY & INSIGHTS")
    print("=" * 80)

    total_duration = results_data['test_metadata']['total_duration_s']
    all_results = results_data['results']
    summary = results_data['summary']

    print(f"\nTest Duration: {format_duration(total_duration)}")
    print(f"Total Tests: {len(all_results)}")

    # Use centralized formatting function
    summary_text = format_summary_display(summary, format_type='console')
    print(summary_text)

    print_summary()

    if len(selected_models) > 1:
        print("\nModel Comparison Tips:")
        print("  • Smaller models (1-3B) are faster but may be less accurate")
        print("  • Larger models (7B+) are slower but typically more capable")
        print("  • Quantization level (Q4, Q5, etc.) affects both speed and quality")

def display_full_responses(all_results, selected_models, selected_temps):
    """Display complete responses for detailed review"""
    config_temp = get_config_temperature()

    print()
    print("=" * 80)
    print("FULL RESPONSES")
    print("=" * 80)

    for model in selected_models:
        for temp, desc in selected_temps:
            temp_display = temp if temp is not None else f"default ({config_temp})"

            # Find matching result
            matching = [r for r in all_results if r['model'] == model and r['temperature'] == temp_display]

            if not matching:
                continue

            r = matching[0]

            print(f"\nModel: {model} | Temperature: {temp_display} ({desc})")
            print("-" * 80)
            print(f"Response ({len(r['response'])} chars):")
            print(r['response'])
            print()
            print(f"Metrics: {r['metrics'].get('completion_tokens', 0)} tokens @ {r['metrics'].get('tokens_per_second', 0):.2f} TPS")

def main():
    print("=" * 80)
    print("ENHANCED TEMPERATURE TEST - MULTI-MODEL COMPARISON")
    print("=" * 80)

    # Check for non-interactive flag (e.g., `--default`)
    is_non_interactive = '--default' in sys.argv

    # Get prompt (from command line arg, file, or interactive input)
    # Filter out the script name (sys.argv[0]) and the non-interactive flag ('--default')
    potential_prompts = [arg for arg in sys.argv[1:] if arg != '--default']

    # The prompt_arg is the first remaining argument, or None if no prompt was supplied.
    prompt_arg = potential_prompts[0] if potential_prompts else None

    # --- NON-INTERACTIVE PROMPT HANDLING ---
    if is_non_interactive and not prompt_arg:
        # In non-interactive mode with no explicit prompt argument, use the default.
        prompt = DEFAULT_PROMPT
        print(f"✓ Using default prompt for non-interactive mode: '{DEFAULT_PROMPT}'")
    else:
        # Use the utility function, which handles file resolution and interactive fallback.
        # If prompt_arg is None here, it means we are interactive and will ask the user.
        prompt = get_prompt_from_file_or_input(prompt_arg)

    # Get available models
    print("\nFetching available models from Ollama...")
    available_models = get_available_models()

    if not available_models:
        print("No models available. Please ensure Ollama is running and has models installed.")
        sys.exit(1)

    # Display test configuration
    config_temp = get_config_temperature()

    if is_non_interactive:
        print("\nRunning in non-interactive default mode.")

        # 1. Non-interactive Model Selection: Use config default with availability check
        default_model = get_config_model()

        if default_model and default_model in available_models:
            # Use the configured model
            selected_models = [default_model]
            print(f"✓ Using configured default model: {default_model}")

        elif default_model and default_model not in available_models:
            # RAISE EXCEPTION AS REQUESTED
            raise ValueError(
                f"Configured default model '{default_model}' is not available in the list of installed models. "
                "Please install the model or update 'wrapper_config.toml'."
            )

        else:
            # Fallback to the first available model if no default is configured/found
            fallback_model = available_models[0]
            selected_models = [fallback_model]
            print(f"⚠ Config default model not found. Falling back to first available model: {fallback_model}")

        # 2. Non-interactive Temperature Selection: Use the standard default set.
        selected_temps = [ALL_TEMPERATURE_CONFIGS[k] for k in DEFAULT_TEMPERATURE_TESTS]

        # 3. Non-interactive Output Filename: Auto-generate
        output_filename = get_output_filename_auto()

    else:
        # Existing interactive logic
        selected_models = select_models(available_models)
        if not selected_models:
            print("No models selected.")
            sys.exit(1)

        selected_temps = select_temperatures()
        output_filename = get_output_filename()

    print(f"\n{'=' * 80}")
    print("TEST CONFIGURATION")
    print(f"{'=' * 80}")

    # --- INITIALIZATION AND METADATA SAVE ---
    start_time = datetime.now()
    # Use the new helper function for the initial save
    initial_results_data = build_results_data(
        all_results=[],
        selected_models=selected_models,
        selected_temps=selected_temps,
        prompt=prompt,
        config_temp=config_temp,
        start_time=start_time,
        status="initialized"
    )

    # Save the file with the header/metadata immediately!
    export_results_to_json(initial_results_data, output_filename)
    print(f"\n✓ Initial metadata saved to: {output_filename}")
    print(f"Test in progress... Check {output_filename} for incremental results.")

    # Run all tests (with incremental saving)
    test_results = run_tests(
        selected_models,
        selected_temps,
        prompt,
        output_filename,
        config_temp,
        start_time # Pass the starting datetime object
    )

    if not test_results['results']:
        print("\nNo results collected. Please check if the API is running.")
        sys.exit(1)

    # Build FINAL results data structure using the helper function
    test_results = build_results_data(
        all_results=test_results['results'],
        selected_models=selected_models,
        selected_temps=selected_temps,
        prompt=prompt,
        config_temp=config_temp,
        start_time=start_time,
        status="completed"
    )

    print(f"Models: {', '.join(selected_models)}")
    print(f"Temperatures: {', '.join(str(t[0] if t[0] is not None else 'default') for t in selected_temps)}")
    print(f"Prompt: {prompt[:80]}..." if len(prompt) > 80 else f"Prompt: {prompt}")
    print(f"Output file: {output_filename}")
    print(f"{'=' * 80}")
    print("Note: Results are saved after each test to prevent data loss on interruption.")
    print(f"{'=' * 80}\n")

    # Final save with completed status and complete markdown export
    export_results_to_json(test_results, output_filename)
    export_results_to_markdown(test_results, output_filename)
    print(f"\n✓ Results saved to: {output_filename}")

    # Display results
    display_results_by_model(test_results['results'], selected_models)
    display_cross_model_comparison(test_results['results'], selected_models, selected_temps)
    display_summary(test_results, selected_models)

    # Footer
    print()
    print("=" * 80)
    print(f"Results saved to: {output_filename}")
    print(f"Markdown export: {output_filename.with_suffix('.md')}")
    print("=" * 80)
    print()
    print("Demo complete!")

if __name__ == "__main__":
    # Assuming you have a project with a virtual environment managed by uv
    # You can run this script in the background using:
    # nohup uv run python your_long_running_script.py &
    main()
