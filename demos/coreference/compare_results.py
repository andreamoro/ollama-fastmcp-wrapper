#!/usr/bin/env python3
"""
Coreference Test Results Comparison Tool

Compares 2-5 test result files side-by-side in a tabular format.
Shows overall accuracy, timing, and per-language success/fail breakdown.

Usage:
    python compare_results.py

    The tool will:
    1. List all available result JSON files from the results/ directory
    2. Prompt you to select 2-5 files for comparison
    3. Display comprehensive side-by-side comparison tables

    Selection options:
    - Press Enter to compare the 3 most recent results (default)
    - Enter comma-separated numbers (e.g., 1,3,5) to select specific files
    - Select between 2-5 files for comparison

Output includes:
    - Comparison Summary: model, prompt, temperature, accuracy, timing
    - Per-Language Breakdown: accuracy percentages for each language
    - Language Success/Fail Comparison: visual bars showing pass/fail distribution
"""

import json
import sys
from pathlib import Path
from datetime import datetime


def load_result_file(filepath):
    """Load and parse a JSON result file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def get_available_results(results_dir):
    """Get list of available result JSON files, sorted by modification time (newest first)."""
    results_path = Path(results_dir)
    if not results_path.exists():
        return []

    # Get JSON files, excluding comparison reports
    json_files = [
        f for f in results_path.glob("*.json")
        if f.is_file() and 'comparison' not in f.name.lower()
    ]
    # Sort by modification time, newest first
    json_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return json_files


def display_file_list(files):
    """Display numbered list of available result files."""
    print("\n" + "=" * 120)
    print("AVAILABLE RESULT FILES")
    print("=" * 120)
    print(f"{'#':<4} {'Date':<20} {'Model':<30} {'Prompt':<25} {'Source File':<30}")
    print("-" * 120)

    for i, filepath in enumerate(files, 1):
        try:
            data = load_result_file(filepath)
            metadata = data.get('metadata', {})

            # Parse timestamp
            timestamp = metadata.get('timestamp', '')
            try:
                dt = datetime.fromisoformat(timestamp)
                date_str = dt.strftime('%Y-%m-%d %H:%M')
            except:
                date_str = timestamp[:16] if timestamp else 'Unknown'

            model = metadata.get('model', 'Unknown')[:28]
            prompt = metadata.get('prompt', 'Unknown')[:23]

            # Get source files - could be in metadata or from results
            source_files = metadata.get('source_files', [])
            if source_files:
                source_str = ', '.join(source_files)[:28]
            else:
                # Fallback: check results for source_file field
                results = data.get('results', [])
                if results and results[0].get('source_file'):
                    source_str = results[0].get('source_file', '')[:28]
                else:
                    source_str = 'N/A'

            print(f"{i:<4} {date_str:<20} {model:<30} {prompt:<25} {source_str:<30}")
        except Exception as e:
            print(f"{i:<4} [Error reading file: {filepath.name}]")

    print("=" * 120)


def get_user_selection(max_index):
    """Prompt user to select files for comparison."""
    print("\nSelect files to compare (comma-separated numbers, e.g., 1,3,5):")
    print("Or press Enter to compare the 3 most recent files")

    user_input = input("> ").strip()

    # Default: compare 3 most recent
    if not user_input:
        return list(range(1, min(4, max_index + 1)))

    # Parse comma-separated selection
    try:
        selections = [int(x.strip()) for x in user_input.split(',')]

        # Validate - need at least 2 files
        if len(selections) < 2:
            print(f"Error: Please select at least 2 files (you selected {len(selections)})")
            return None

        if any(s < 1 or s > max_index for s in selections):
            print(f"Error: Invalid selection. Numbers must be between 1 and {max_index}")
            return None

        return selections
    except ValueError:
        print("Error: Invalid input format. Use comma-separated numbers (e.g., 1,3,5)")
        return None


def format_duration(seconds):
    """Format duration in human-readable format."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}m {secs:.0f}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        return f"{hours}h {minutes}m {secs:.0f}s"


def truncate_text(text, max_len):
    """Truncate text to max length with ellipsis."""
    if len(text) <= max_len:
        return text
    return text[:max_len-3] + "..."


def print_comparison_table(results_data):
    """Print side-by-side comparison table."""
    num_results = len(results_data)

    # Column width calculations
    label_width = 35
    col_width = max(20, (140 - label_width) // num_results)

    print("\n" + "=" * 140)
    print("COMPARISON SUMMARY")
    print("=" * 140)

    # Header row with result identifiers
    header = f"{'Metric':<{label_width}}"
    for i, (filepath, data) in enumerate(results_data, 1):
        result_name = f"Result #{i}"
        header += f"{result_name:^{col_width}}"
    print(header)
    print("-" * 140)

    # Helper function to print a row
    def print_row(label, values):
        row = f"{label:<{label_width}}"
        for val in values:
            row += f"{val:^{col_width}}"
        print(row)

    # Model info
    print_row("Model", [truncate_text(d.get('metadata', {}).get('model', 'Unknown'), col_width-2)
                        for _, d in results_data])

    # Prompt
    print_row("Prompt", [truncate_text(d.get('metadata', {}).get('prompt', 'Unknown'), col_width-2)
                         for _, d in results_data])

    # Temperature
    print_row("Temperature", [f"{d.get('metadata', {}).get('temperature', 0):.1f}"
                              for _, d in results_data])

    # Ollama instance
    print_row("Ollama Instance", [truncate_text(d.get('metadata', {}).get('ollama_label', 'Unknown'), col_width-2)
                                   for _, d in results_data])

    print("-" * 140)

    # Overall accuracy
    print_row("Overall Accuracy", [f"{d.get('metrics', {}).get('overall', {}).get('accuracy', 0):.1f}%"
                                    for _, d in results_data])

    # Correct / Total
    print_row("Correct / Total", [f"{d.get('metrics', {}).get('overall', {}).get('correct', 0)} / {d.get('metrics', {}).get('overall', {}).get('total', 0)}"
                                   for _, d in results_data])

    print("-" * 140)

    # Timing - Ollama (actual processing)
    def get_ollama_timing(d, key):
        timing = d.get('metrics', {}).get('timing', {})
        ollama = timing.get('ollama', {})
        if ollama:
            return ollama.get(key, 0)
        # Fallback to legacy format
        return timing.get(key, 0)

    def get_wall_timing(d, key):
        timing = d.get('metrics', {}).get('timing', {})
        wall = timing.get('wall_clock', {})
        return wall.get(key, 0) if wall else 0

    def get_queue_timing(d, key):
        timing = d.get('metrics', {}).get('timing', {})
        queue = timing.get('queue_wait', {})
        return queue.get(key, 0) if queue else 0

    print_row("--- Ollama Processing ---", ["" for _ in results_data])
    print_row("Avg (GPU)", [format_duration(get_ollama_timing(d, 'average_s'))
                            for _, d in results_data])
    print_row("Min (GPU)", [format_duration(get_ollama_timing(d, 'min_s'))
                            for _, d in results_data])
    print_row("Max (GPU)", [format_duration(get_ollama_timing(d, 'max_s'))
                            for _, d in results_data])
    print_row("Total (GPU)", [format_duration(get_ollama_timing(d, 'total_s'))
                              for _, d in results_data])

    # Check if any result has wall-clock timing (new format)
    has_wall_clock = any(d.get('metrics', {}).get('timing', {}).get('wall_clock') for _, d in results_data)
    if has_wall_clock:
        print_row("--- Wall Clock ---", ["" for _ in results_data])
        print_row("Avg (waited)", [format_duration(get_wall_timing(d, 'average_s'))
                                   for _, d in results_data])
        print_row("Total (waited)", [format_duration(get_wall_timing(d, 'total_s'))
                                     for _, d in results_data])

        print_row("--- Queue Wait ---", ["" for _ in results_data])
        print_row("Total queue", [format_duration(get_queue_timing(d, 'total_s'))
                                  for _, d in results_data])
        print_row("Avg queue", [format_duration(get_queue_timing(d, 'average_s'))
                                for _, d in results_data])

    print("=" * 140)


def get_accuracy_header(results_data):
    """Generate accuracy header string for all results with prompt info."""
    parts = []
    for _, data in results_data:
        acc = data.get('metrics', {}).get('overall', {}).get('accuracy', 0)
        prompt = data.get('metadata', {}).get('prompt', 'Unknown')
        # Extract just the filename from the path
        if '/' in prompt:
            prompt = prompt.split('/')[-1]
        # Truncate long prompt names
        if len(prompt) > 20:
            prompt = prompt[:17] + "..."
        parts.append(f"{prompt}: {acc:.1f}%")
    return " | ".join(parts)


def print_language_breakdown(results_data):
    """Print per-language accuracy breakdown."""
    # Collect all languages across all results
    all_languages = set()
    for _, data in results_data:
        by_lang = data.get('metrics', {}).get('by_language', {})
        all_languages.update(by_lang.keys())

    if not all_languages:
        return

    sorted_languages = sorted(all_languages)
    num_results = len(results_data)

    # Column widths
    lang_width = 12
    col_width = max(25, (140 - lang_width) // num_results)

    print("\n" + "=" * 140)
    print(f"PER-LANGUAGE BREAKDOWN  [Overall: {get_accuracy_header(results_data)}]")
    print("=" * 140)

    # Header
    header = f"{'Language':<{lang_width}}"
    for i in range(num_results):
        header += f"{'Result #' + str(i+1):^{col_width}}"
    print(header)
    print("-" * 140)

    # Print each language
    for lang in sorted_languages:
        values = []
        for _, data in results_data:
            by_lang = data.get('metrics', {}).get('by_language', {})
            lang_data = by_lang.get(lang, {})

            if lang_data:
                accuracy = lang_data.get('accuracy', 0)
                correct = lang_data.get('correct', 0)
                total = lang_data.get('total', 0)
                value = f"{accuracy:.1f}% ({correct}/{total})"
            else:
                value = "N/A"

            values.append(value)

        # Print row
        row = f"{lang:<{lang_width}}"
        for val in values:
            row += f"{val:^{col_width}}"
        print(row)

    print("=" * 140)


def print_per_test_timing(results_data):
    """Print per-test timing comparison table."""
    # Build a map of doc_key -> timing for each result
    # First, collect all doc_keys across all results
    all_doc_keys = set()
    for _, data in results_data:
        for result in data.get('results', []):
            all_doc_keys.add(result.get('doc_key', ''))

    if not all_doc_keys:
        return

    sorted_doc_keys = sorted(all_doc_keys)
    num_results = len(results_data)

    # Column widths
    key_width = 25
    col_width = max(18, (140 - key_width) // num_results)

    print("\n" + "=" * 140)
    print(f"PER-TEST TIMING COMPARISON (Ollama GPU time)  [Overall: {get_accuracy_header(results_data)}]")
    print("=" * 140)

    # Header
    header = f"{'Doc Key':<{key_width}}"
    for i, (filepath, data) in enumerate(results_data, 1):
        model = data.get('metadata', {}).get('model', 'Unknown')[:col_width-2]
        header += f"{model:^{col_width}}"
    print(header)
    print("-" * 140)

    # Build lookup for each result
    result_lookups = []
    for _, data in results_data:
        lookup = {}
        for result in data.get('results', []):
            doc_key = result.get('doc_key', '')
            metrics = result.get('metrics', {})
            # Prefer Ollama timing, fallback to elapsed_time
            ollama_time = metrics.get('ollama_duration_s', 0)
            if not ollama_time:
                ollama_time = result.get('elapsed_time', 0)
            queue_time = metrics.get('queue_wait_s', 0)
            correct = result.get('correct', False)
            lookup[doc_key] = {
                'time': ollama_time,
                'queue': queue_time,
                'correct': correct
            }
        result_lookups.append(lookup)

    # Print each test row
    for doc_key in sorted_doc_keys:
        row = f"{doc_key:<{key_width}}"
        for lookup in result_lookups:
            test_data = lookup.get(doc_key, {})
            if test_data:
                time_val = test_data['time']
                correct = test_data['correct']
                status = "✓" if correct else "✗"
                cell = f"{status} {time_val:.1f}s"
            else:
                cell = "N/A"
            row += f"{cell:^{col_width}}"
        print(row)

    print("-" * 140)

    # Summary row - averages
    row = f"{'AVERAGE':<{key_width}}"
    for lookup in result_lookups:
        times = [v['time'] for v in lookup.values() if v['time'] > 0]
        avg = sum(times) / len(times) if times else 0
        row += f"{format_duration(avg):^{col_width}}"
    print(row)

    # Fastest/slowest comparison
    row = f"{'FASTEST':<{key_width}}"
    for lookup in result_lookups:
        times = [v['time'] for v in lookup.values() if v['time'] > 0]
        fastest = min(times) if times else 0
        row += f"{format_duration(fastest):^{col_width}}"
    print(row)

    row = f"{'SLOWEST':<{key_width}}"
    for lookup in result_lookups:
        times = [v['time'] for v in lookup.values() if v['time'] > 0]
        slowest = max(times) if times else 0
        row += f"{format_duration(slowest):^{col_width}}"
    print(row)

    print("=" * 140)


def print_language_comparison_chart(results_data):
    """Print visual comparison of language performance."""
    # Collect all languages
    all_languages = set()
    for _, data in results_data:
        by_lang = data.get('metrics', {}).get('by_language', {})
        all_languages.update(by_lang.keys())

    if not all_languages:
        return

    sorted_languages = sorted(all_languages)

    print("\n" + "=" * 140)
    print(f"LANGUAGE SUCCESS/FAIL COMPARISON  [Overall: {get_accuracy_header(results_data)}]")
    print("=" * 140)

    for i, (filepath, data) in enumerate(results_data, 1):
        model = data.get('metadata', {}).get('model', 'Unknown')
        acc = data.get('metrics', {}).get('overall', {}).get('accuracy', 0)
        print(f"\nResult #{i}: {model} ({acc:.1f}%)")
        print("-" * 140)

        by_lang = data.get('metrics', {}).get('by_language', {})

        for lang in sorted_languages:
            lang_data = by_lang.get(lang)

            if not lang_data:
                print(f"  {lang:>4}: N/A")
                continue

            correct = lang_data.get('correct', 0)
            total = lang_data.get('total', 0)
            failed = total - correct
            accuracy = lang_data.get('accuracy', 0)

            # Visual bar (20 chars max)
            bar_length = 20
            if total > 0:
                correct_bar = int((correct / total) * bar_length)
                failed_bar = bar_length - correct_bar
            else:
                correct_bar = 0
                failed_bar = 0

            success_bar = '█' * correct_bar
            fail_bar = '░' * failed_bar

            print(f"  {lang:>4}: [{success_bar}{fail_bar}] {accuracy:>6.1f}% | ✓{correct:>2} ✗{failed:>2} (total: {total})")

    print("=" * 140)


def main():
    """Main entry point."""
    # Get script directory
    script_dir = Path(__file__).parent
    results_dir = script_dir / "results"

    if not results_dir.exists():
        print(f"Error: Results directory not found: {results_dir}")
        sys.exit(1)

    # Get available files
    available_files = get_available_results(results_dir)

    if not available_files:
        print(f"No result files found in {results_dir}")
        sys.exit(1)

    print(f"\nFound {len(available_files)} result file(s)")

    # Display list
    display_file_list(available_files)

    # Get user selection
    while True:
        selections = get_user_selection(len(available_files))
        if selections is not None:
            break

    # Load selected files
    selected_data = []
    for idx in selections:
        filepath = available_files[idx - 1]
        try:
            data = load_result_file(filepath)
            selected_data.append((filepath, data))
        except Exception as e:
            print(f"Error loading {filepath.name}: {e}")
            sys.exit(1)

    # Print comparisons
    print_comparison_table(selected_data)
    print_language_breakdown(selected_data)
    print_per_test_timing(selected_data)
    print_language_comparison_chart(selected_data)

    print("\nComparison complete!")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nCancelled by user.")
        sys.exit(0)
