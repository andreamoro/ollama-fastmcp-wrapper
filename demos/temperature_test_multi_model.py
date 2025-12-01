#!/usr/bin/env python3
"""
Enhanced Temperature Test - Compare multiple models with different temperatures
Shows how temperature affects response quality across different model families
"""

import sys
from datetime import datetime
from temperature_test_utils import (
    test_temperature_model,
    format_duration,
    get_available_models,
    get_config_temperature,
    get_prompt_from_file_or_input,
    select_temperatures,
    save_results_to_json,
    export_results_to_markdown,
    print_summary,
    DEFAULT_PROMPT
)

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
    except:
        print("Invalid selection. Using first model.")
        return [available_models[0]] if available_models else []

def get_output_filename():
    """Ask user for output filename"""
    print("\nOutput file configuration:")
    print("  - Press Enter for auto-generated filename (YYYYMMDD_HHMMSS_multi_test_comparison.json)")
    print("  - Enter a filename (will add .json extension if needed)")

    user_input = input("Output filename (or Enter for auto): ").strip()

    if user_input == '':
        # Auto-generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{timestamp}_multi_test_comparison.json"

    # Add .json extension if not present
    if not user_input.endswith('.json'):
        user_input += '.json'

    return user_input

def calculate_summary_stats(results):
    """Calculate summary statistics from results"""
    if not results:
        return {}

    summary = {}

    # Find fastest model (highest TPS)
    fastest = max(results, key=lambda r: r['metrics'].get('tokens_per_second', 0))
    summary['fastest_model'] = fastest['model']
    summary['highest_tps'] = round(fastest['metrics'].get('tokens_per_second', 0), 2)

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

def run_tests(selected_models, selected_temps, prompt, output_filename, config_temp):
    """Execute all temperature tests with incremental saving to prevent data loss"""
    start_time = datetime.now()
    start_timestamp = start_time.isoformat()

    all_results = []
    total_tests = len(selected_models) * len(selected_temps)
    current_test = 0

    for model in selected_models:
        for temp, desc in selected_temps:
            current_test += 1
            print(f"[{current_test}/{total_tests}] Testing {model} with {desc} (temp={temp})...")

            result = test_temperature_model(model, temp, desc, prompt)
            if result:
                all_results.append(result)

                # Save after each test to prevent data loss
                partial_results_data = {
                    "test_metadata": {
                        "start_timestamp": start_timestamp,
                        "end_timestamp": datetime.now().isoformat(),
                        "total_duration_s": (datetime.now() - start_time).total_seconds(),
                        "total_duration_readable": format_duration((datetime.now() - start_time).total_seconds()),
                        "prompt": prompt,
                        "models_tested": selected_models,
                        "temperatures_tested": [t[0] if t[0] is not None else f"default ({config_temp})" for t in selected_temps],
                        "total_tests": len(all_results),
                        "status": "in_progress" if current_test < total_tests else "completed"
                    },
                    "results": all_results,
                    "summary": calculate_summary_stats(all_results)
                }
                save_results_to_json(partial_results_data, output_filename)

    end_time = datetime.now()
    end_timestamp = end_time.isoformat()
    total_duration = (end_time - start_time).total_seconds()

    return {
        'results': all_results,
        'start_timestamp': start_timestamp,
        'end_timestamp': end_timestamp,
        'total_duration': total_duration
    }

def display_results_by_model(all_results, selected_models):
    """Display results organized by model"""
    print()
    print("=" * 140)
    print("RESULTS BY MODEL")
    print("=" * 140)

    for model in selected_models:
        model_results = [r for r in all_results if r['model'] == model]

        if not model_results:
            continue

        print(f"\nModel: {model}")
        print("-" * 140)

        # Table header
        header = f"{'Temperature':<20} {'TPS':<10} {'Tokens':<10} {'Time':<15} {'Total Duration':<15} {'Description':<25} {'Response Preview':<40}"
        print(header)
        print("-" * 140)

        # Table rows
        for r in model_results:
            temp_str = str(r['temperature'])
            tps = r['metrics'].get('tokens_per_second', 0)
            tokens = r['metrics'].get('completion_tokens', 0)
            elapsed = format_duration(r['elapsed_time'])
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
    print("=" * 140)
    print("CROSS-MODEL COMPARISON (at each temperature)")
    print("=" * 140)

    for temp, desc in selected_temps:
        temp_display = temp if temp is not None else f"default ({config_temp})"
        temp_results = [r for r in all_results if r['temperature'] == temp_display]

        if not temp_results:
            continue

        print(f"\nTemperature: {temp_display} - {desc}")
        print("-" * 140)

        # Table header
        header = f"{'Model':<35} {'TPS':<10} {'Tokens':<10} {'Time':<15} {'Total Duration':<15} {'Response Preview':<50}"
        print(header)
        print("-" * 140)

        for r in temp_results:
            model_name = r['model']
            tps = r['metrics'].get('tokens_per_second', 0)
            tokens = r['metrics'].get('completion_tokens', 0)
            elapsed = format_duration(r['elapsed_time'])
            total_dur = format_duration(r['metrics'].get('total_duration_s', 0))
            response_preview = (r['response'][:47] + "...") if len(r['response']) > 50 else r['response']

            row = f"{model_name:<35} {tps:<10.2f} {tokens:<10} {elapsed:<15} {total_dur:<15} {response_preview:<50}"
            print(row)

def display_summary(results_data, selected_models):
    """Display summary statistics and insights"""
    print()
    print("=" * 140)
    print("SUMMARY & INSIGHTS")
    print("=" * 140)

    total_duration = results_data['test_metadata']['total_duration_s']
    all_results = results_data['results']
    summary = results_data['summary']

    print(f"\nTest Duration: {format_duration(total_duration)}")
    print(f"Total Tests: {len(all_results)}")

    print(f"\nPerformance:")
    print(f"  • Fastest Model: {summary['fastest_model']} ({summary['highest_tps']} TPS)")
    print(f"  • Average TPS: {summary['average_tps']}")
    print(f"  • Average Tokens: {summary['average_tokens']}")

    print(f"\nResponse Lengths:")
    print(f"  • Min: {summary['min_response_length']} chars")
    print(f"  • Max: {summary['max_response_length']} chars")
    print(f"  • Avg: {summary['avg_response_length']} chars")

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
    print("=" * 140)
    print("FULL RESPONSES")
    print("=" * 140)

    for model in selected_models:
        for temp, desc in selected_temps:
            temp_display = temp if temp is not None else f"default ({config_temp})"

            # Find matching result
            matching = [r for r in all_results if r['model'] == model and r['temperature'] == temp_display]

            if not matching:
                continue

            r = matching[0]

            print(f"\nModel: {model} | Temperature: {temp_display} ({desc})")
            print("-" * 140)
            print(f"Response ({len(r['response'])} chars):")
            print(r['response'])
            print()
            print(f"Metrics: {r['metrics'].get('completion_tokens', 0)} tokens @ {r['metrics'].get('tokens_per_second', 0):.2f} TPS")

def main():
    print("=" * 120)
    print("ENHANCED TEMPERATURE TEST - MULTI-MODEL COMPARISON")
    print("=" * 120)

    # Get prompt (from command line arg, file, or interactive input)
    prompt_arg = sys.argv[1] if len(sys.argv) > 1 else None
    prompt = get_prompt_from_file_or_input(prompt_arg)

    # Get available models
    print("\nFetching available models from Ollama...")
    available_models = get_available_models()

    if not available_models:
        print("No models available. Please ensure Ollama is running and has models installed.")
        sys.exit(1)

    # Let user select models and temperatures
    selected_models = select_models(available_models)
    if not selected_models:
        print("No models selected.")
        sys.exit(1)

    selected_temps = select_temperatures()
    output_filename = get_output_filename()

    # Display test configuration
    config_temp = get_config_temperature()
    print(f"\n{'='*120}")
    print("TEST CONFIGURATION")
    print(f"{'='*120}")
    print(f"Models: {', '.join(selected_models)}")
    print(f"Temperatures: {', '.join(str(t[0] if t[0] is not None else 'default') for t in selected_temps)}")
    print(f"Prompt: {prompt[:80]}..." if len(prompt) > 80 else f"Prompt: {prompt}")
    print(f"Output file: {output_filename}")
    print(f"{'='*120}")
    print("Note: Results are saved after each test to prevent data loss on interruption.")
    print(f"{'='*120}\n")

    # Run all tests (with incremental saving)
    test_results = run_tests(selected_models, selected_temps, prompt, output_filename, config_temp)

    if not test_results['results']:
        print("\nNo results collected. Please check if the API is running.")
        sys.exit(1)

    # Build final results data structure
    results_data = {
        "test_metadata": {
            "start_timestamp": test_results['start_timestamp'],
            "end_timestamp": test_results['end_timestamp'],
            "total_duration_s": test_results['total_duration'],
            "total_duration_readable": format_duration(test_results['total_duration']),
            "prompt": prompt,
            "models_tested": selected_models,
            "temperatures_tested": [t[0] if t[0] is not None else f"default ({config_temp})" for t in selected_temps],
            "total_tests": len(test_results['results']),
            "status": "completed"
        },
        "results": test_results['results'],
        "summary": calculate_summary_stats(test_results['results'])
    }

    # Final save with completed status and markdown export
    save_results_to_json(results_data, output_filename)
    export_results_to_markdown(results_data, output_filename)

    # Display results
    display_results_by_model(results_data['results'], selected_models)
    display_cross_model_comparison(results_data['results'], selected_models, selected_temps)
    display_summary(results_data, selected_models)

    # Footer
    print()
    print("=" * 140)
    print(f"Results saved to: {output_filename}")
    print(f"Markdown export: {output_filename.replace('.json', '.md')}")
    print("=" * 140)
    print()
    print("Demo complete!")

if __name__ == "__main__":
    # Assuming you have a project with a virtual environment managed by uv
    # You can run this script in the background using:
    # nohup uv run python your_long_running_script.py &
    main()
