#!/usr/bin/env python3
"""
Temperature Test Demo - Compare different temperature settings
Shows how temperature affects response consistency and creativity
"""

from temperature_test_utils import (
    test_temperature_model,
    format_duration,
    print_summary,
    TEMPERATURE_TESTS,
    DEFAULT_PROMPT
)

def main():
    print("=== Temperature Test Demo ===")
    print("Testing the same prompt with different temperature values")
    print(f"Prompt: '{DEFAULT_PROMPT}'")
    print()

    # Default model for single-model test
    model = "llama3.2:3b"

    # Run all tests
    results = []
    for temp, desc in TEMPERATURE_TESTS:
        print(f"Running test: {desc} (temp={temp})...")
        result = test_temperature_model(model, temp, desc)
        if result:
            results.append(result)

    print()
    print("=" * 120)
    print("RESULTS TABLE")
    print("=" * 120)

    # Table header
    header = f"{'Temperature':<20} {'TPS':<10} {'Tokens':<10} {'Time':<12} {'Total Duration':<15} {'Description':<25}"
    print(header)
    print("-" * 120)

    # Table rows
    for r in results:
        temp_str = str(r['temperature'])
        tps = r['metrics'].get('tokens_per_second', 0)
        tokens = r['metrics'].get('completion_tokens', 0)
        elapsed = format_duration(r['elapsed_time'])
        total_dur = format_duration(r['metrics'].get('total_duration_s', 0))
        desc = r['description']

        row = f"{temp_str:<20} {tps:<10.2f} {tokens:<10} {elapsed:<12} {total_dur:<15} {desc:<25}"
        print(row)

    print("=" * 120)
    print()

    # Show responses
    print("RESPONSES:")
    print("-" * 120)
    for i, r in enumerate(results, 1):
        print(f"\n{i}. {r['description']} (temp={r['temperature']}):")
        print(f"   {r['response']}")

    print()
    print("=" * 120)
    print("SUMMARY")
    print("=" * 120)
    print_summary()
    print()
    print("Demo complete!")

if __name__ == "__main__":
    main()
