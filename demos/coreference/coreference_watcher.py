#!/usr/bin/env python3
"""
Coreference Test Progress Watcher

Monitor the progress of a running coreference test from another terminal.
Reads the in-progress JSON file and displays real-time statistics.

================================================================================
HOW FILE SELECTION WORKS
================================================================================

The watcher determines which JSON file to process using these strategies:

1. DEFAULT (no arguments):
   Automatically finds the most recently modified .json file in the results/
   directory. This works because the test script saves progress after each test,
   so the active test file will have the newest modification time.

   The selection logic:
     json_files = list(RESULTS_DIR.glob("*.json"))
     json_files.sort(key=lambda f: f.stat().st_mtime, reverse=True)
     return json_files[0]  # Most recently modified

2. EXPLICIT FILE (--file argument):
   You can specify exactly which file to watch:
     python coreference_watcher.py --file results/20240101_120000-gemma3-4b.json

   If the file doesn't exist yet (e.g., you started the watcher before the test),
   it will keep waiting indefinitely until the file appears or you press Ctrl+C.
   This allows starting the watcher before the test creates its output file.

3. LIST MODE (--list):
   See all available result files with their status:
     python coreference_watcher.py --list

   Output example:
     Available result files (most recent first):
       20240101_120500-gemma3-4b-pronoun.json  [in_progress] 15/50
       20240101_110000-llama3.2-pronoun.json   [completed] 50/50

================================================================================
DATA SOURCES - What's Real vs Estimated
================================================================================

All progress data comes from REAL measurements, not fictional estimates:

REAL DATA (from completed tests):
  - Actual elapsed time per test (measured in Python)
  - Running average time (calculated from real completed test times)
  - Current accuracy (from actual test results)
  - Ollama metrics: tokens_per_second, eval_count, total_duration
    (extracted from Ollama API response for each test)

CALCULATED ESTIMATES:
  - ETA = running_average * remaining_tests
    (based on real timing data, but still an estimate)

The in-progress JSON file contains Ollama's metrics for each completed test.
Example from a result entry:
  {
    "elapsed_time": 3.42,
    "metrics": {
      "tokens_per_second": 1.8,
      "total_duration": 3200000000,
      "eval_count": 12
    }
  }

================================================================================
WHAT THE WATCHER DISPLAYS
================================================================================

  - Status: in_progress/completed with visual indicator
  - Model/prompt/temperature configuration
  - Progress bar with percentage and test count
  - Current accuracy (correct/total)
  - Timing stats:
    - Elapsed time
    - Average time per test
    - ETA (estimated time remaining)
    - Min/max times (when completed)
  - Per-language accuracy breakdown
  - Ollama performance: average tokens/sec across all tests
  - Last test result with model answer vs gold answer
  - Recent failures (last 5) with expected vs actual

================================================================================
USAGE EXAMPLES
================================================================================

  # One-shot: Show current status of the most recent test
  python coreference_watcher.py

  # Watch mode: Auto-refresh every 5 seconds (default)
  python coreference_watcher.py --watch

  # Watch mode: Custom refresh interval (10 seconds)
  python coreference_watcher.py --watch --interval 10
  python coreference_watcher.py -w -i 10

  # Watch a specific file
  python coreference_watcher.py --file results/20240101_120000-model-prompt.json
  python coreference_watcher.py -f results/xyz.json

  # List all available result files
  python coreference_watcher.py --list
  python coreference_watcher.py -l

================================================================================
TYPICAL WORKFLOW
================================================================================

Terminal 1: Start the test
  $ python coreference_test.py --model gemma3:4b --default

Terminal 2: Watch progress (in another terminal)
  $ python coreference_watcher.py --watch

The watcher will automatically pick up the currently running test's file
and display live progress. When the test completes, the watcher exits.

Press Ctrl+C in the watcher terminal to stop watching at any time.
"""

import sys
import json
import time
import argparse
from pathlib import Path
from datetime import datetime

SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR / "results"


def find_latest_results():
    """Find the most recently modified JSON file in results directory."""
    if not RESULTS_DIR.exists():
        return None

    json_files = list(RESULTS_DIR.glob("*.json"))
    if not json_files:
        return None

    # Sort by modification time, most recent first
    json_files.sort(key=lambda f: f.stat().st_mtime, reverse=True)
    return json_files[0]


def format_duration(seconds):
    """Format duration in human-readable format."""
    if seconds < 1:
        return f"{seconds*1000:.0f}ms"
    elif seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes}m {secs}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"{hours}h {minutes}m"


def display_progress(filepath, clear_screen=False):
    """Read and display progress from a results JSON file."""
    if clear_screen:
        print("\033[2J\033[H", end="")  # ANSI clear screen and move to top

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except json.JSONDecodeError:
        print(f"Error: Could not parse {filepath.name} (file may be mid-write)")
        return False
    except FileNotFoundError:
        print(f"Error: File not found: {filepath}")
        return False

    meta = data.get('metadata', {})
    metrics = data.get('metrics', {})
    results = data.get('results', [])
    overall = metrics.get('overall', {})
    timing = metrics.get('timing', {})

    status = meta.get('status', 'unknown')
    completed = meta.get('completed', len(results))
    total = meta.get('total', meta.get('test_cases_count', 0))
    progress_pct = (completed / total * 100) if total > 0 else 0

    print("=" * 60)
    print("COREFERENCE TEST PROGRESS")
    print("=" * 60)
    print()

    # Status
    status_symbol = "..." if status == 'in_progress' else "DONE" if status == 'completed' else "?"
    print(f"Status: {status.upper()} [{status_symbol}]")
    print(f"File: {filepath.name}")
    print()

    # Test info
    print(f"Model: {meta.get('model', 'N/A')}")
    print(f"Prompt: {meta.get('prompt', 'N/A')}")
    print(f"Temperature: {meta.get('temperature', 'N/A')}")
    print()

    # Progress bar
    bar_width = 40
    filled = int(bar_width * progress_pct / 100)
    bar = "=" * filled + "-" * (bar_width - filled)
    print(f"Progress: [{bar}] {progress_pct:.1f}%")
    print(f"          {completed}/{total} tests completed")
    print()

    # Accuracy
    accuracy = overall.get('accuracy', 0)
    correct = overall.get('correct', 0)
    print(f"Accuracy: {accuracy}% ({correct}/{completed} correct)")
    print()

    # Timing stats
    avg_time = timing.get('average_s', 0)
    total_time = timing.get('total_s', 0)
    remaining = total - completed

    if status == 'in_progress' and avg_time > 0:
        eta_seconds = avg_time * remaining
        print(f"Timing:")
        print(f"  Elapsed: {format_duration(total_time)}")
        print(f"  Avg/test: {avg_time:.2f}s")
        print(f"  ETA: ~{format_duration(eta_seconds)} ({remaining} tests remaining)")
    else:
        print(f"Timing:")
        print(f"  Total: {meta.get('total_duration_readable', format_duration(total_time))}")
        print(f"  Avg/test: {avg_time:.2f}s")
        print(f"  Min: {timing.get('min_s', 0):.2f}s | Max: {timing.get('max_s', 0):.2f}s")
    print()

    # Per-language breakdown
    by_lang = metrics.get('by_language', {})
    if by_lang:
        print("By Language:")
        for lang, stats in sorted(by_lang.items()):
            print(f"  {lang}: {stats['accuracy']}% ({stats['correct']}/{stats['total']})")
        print()

    # Ollama performance metrics (average tokens/sec)
    tps_values = [r.get('metrics', {}).get('tokens_per_second', 0) for r in results if r.get('metrics', {}).get('tokens_per_second')]
    if tps_values:
        avg_tps = sum(tps_values) / len(tps_values)
        print(f"Ollama Performance: {avg_tps:.2f} tokens/sec (avg)")
        print()

    # Last test result
    if results:
        last = results[-1]
        result_mark = "CORRECT" if last.get('correct') else "WRONG"
        print(f"Last Test: {last.get('doc_key', 'N/A')} ({last.get('lang', '?')})")
        print(f"           {result_mark} - '{last.get('model_answer', 'N/A')}' vs '{last.get('gold_referent', 'N/A')}'")
        print(f"           Time: {last.get('elapsed_time', 0):.2f}s")
        print()

    # Recent failures (last 5)
    failures = [r for r in results if not r.get('correct', False)]
    if failures:
        print(f"Recent Failures ({len(failures)} total):")
        for r in failures[-5:]:
            print(f"  - {r['doc_key']}: expected '{r['gold_referent']}', got '{r.get('model_answer', 'N/A')}'")
        if len(failures) > 5:
            print(f"  ... and {len(failures) - 5} more")
        print()

    # Timestamp
    last_update = meta.get('last_update', meta.get('end_timestamp', ''))
    if last_update:
        try:
            update_dt = datetime.fromisoformat(last_update)
            ago = (datetime.now() - update_dt).total_seconds()
            print(f"Last updated: {ago:.0f}s ago ({update_dt.strftime('%H:%M:%S')})")
        except:
            print(f"Last updated: {last_update}")

    print("=" * 60)
    return True


def watch_mode(filepath, interval):
    """Continuously watch a file and display progress."""
    print(f"Watching {filepath.name} (Ctrl+C to stop, refreshing every {interval}s)...")
    print()

    try:
        while True:
            success = display_progress(filepath, clear_screen=True)
            if not success:
                break

            # Check if test completed
            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)
                if data.get('metadata', {}).get('status') == 'completed':
                    print("\nTest completed!")
                    break
            except:
                pass

            time.sleep(interval)
    except KeyboardInterrupt:
        print("\n\nStopped watching.")


def main():
    parser = argparse.ArgumentParser(
        description='Monitor coreference test progress',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python coreference_watcher.py                    # Show latest status
    python coreference_watcher.py --watch            # Watch mode (5s refresh)
    python coreference_watcher.py --watch -i 10     # Watch with 10s interval
    python coreference_watcher.py --file results/xyz.json  # Specific file
    python coreference_watcher.py --list             # List available results
        """
    )

    parser.add_argument(
        '--file', '-f',
        type=str,
        help='Specific results file to watch'
    )
    parser.add_argument(
        '--watch', '-w',
        action='store_true',
        help='Watch mode - continuously refresh'
    )
    parser.add_argument(
        '--interval', '-i',
        type=int,
        default=5,
        help='Refresh interval in seconds for watch mode (default: 5)'
    )
    parser.add_argument(
        '--list', '-l',
        action='store_true',
        help='List available result files'
    )

    args = parser.parse_args()

    # List mode
    if args.list:
        if not RESULTS_DIR.exists():
            print(f"Results directory not found: {RESULTS_DIR}")
            sys.exit(1)

        json_files = sorted(RESULTS_DIR.glob("*.json"), key=lambda f: f.stat().st_mtime, reverse=True)
        if not json_files:
            print("No result files found.")
            sys.exit(0)

        print("Available result files (most recent first):")
        for f in json_files[:20]:  # Show last 20
            try:
                with open(f, 'r') as fp:
                    data = json.load(fp)
                meta = data.get('metadata', {})
                status = meta.get('status', '?')
                completed = meta.get('completed', len(data.get('results', [])))
                total = meta.get('total', '?')
                print(f"  {f.name}  [{status}] {completed}/{total}")
            except:
                print(f"  {f.name}  [error reading]")
        sys.exit(0)

    # Find file to watch
    if args.file:
        filepath = Path(args.file)
        if not filepath.is_absolute():
            # Try relative to results dir first
            if (RESULTS_DIR / filepath).exists():
                filepath = RESULTS_DIR / filepath
            elif not filepath.exists():
                filepath = RESULTS_DIR / filepath  # Assume results dir for retries

        # Wait for file to appear (useful when starting watcher before test)
        if not filepath.exists():
            interval = args.interval
            print(f"File not found: {filepath}")
            print("Waiting for file to appear (Ctrl+C to cancel)...")

            try:
                attempt = 0
                while not filepath.exists():
                    time.sleep(interval)
                    attempt += 1
                    # Update same line with carriage return
                    print(f"\r  Waiting... {attempt * interval}s elapsed", end="", flush=True)
                print(f"\r  File found after {attempt * interval}s.        ")  # Extra spaces to clear
            except KeyboardInterrupt:
                print("\n\nCancelled by user.")
                sys.exit(0)
    else:
        filepath = find_latest_results()
        if filepath is None:
            print("No result files found in results/ directory.")
            print("Start a test first, or specify a file with --file")
            sys.exit(1)

    # Display or watch
    if args.watch:
        watch_mode(filepath, args.interval)
    else:
        display_progress(filepath)


if __name__ == "__main__":
    main()
