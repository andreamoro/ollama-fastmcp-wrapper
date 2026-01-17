#!/usr/bin/env python3
"""
Compare Two Coreference Results for Specular (Equivalent) Output

Checks if two coreference analysis outputs are essentially the same,
identifying similarities and differences in:
- Entities detected
- Pronouns identified
- Coreference chains

Usage:
    # Interactive mode - select 2 files to compare
    uv run python compare_specular.py

    # Specify files directly
    uv run python compare_specular.py file1.json file2.json
"""

import sys
import json
import re
from pathlib import Path
from datetime import datetime
from collections import defaultdict

SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR / "results"


def load_result_file(filepath):
    """Load a result JSON file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def get_available_results():
    """Get list of available result JSON files, sorted by modification time (newest first)."""
    if not RESULTS_DIR.exists():
        return []

    # Get JSON files, excluding comparison reports
    json_files = [
        f for f in RESULTS_DIR.glob("*.json")
        if f.is_file() and 'comparison' not in f.name.lower()
    ]
    json_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return json_files


def display_file_list(files):
    """Display numbered list of available result files, grouped by source file and ordered by prompt."""
    print("\n" + "=" * 120)
    print("AVAILABLE RESULT FILES")
    print("=" * 120)

    # First pass: collect file info and group by source file
    file_info = []
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
            prompt = metadata.get('prompt', 'Unknown')

            # Get source files - could be in metadata or from results
            source_files = metadata.get('source_files', [])
            if source_files:
                source_str = ', '.join(source_files)
            else:
                # Fallback: check results for source_file field
                results = data.get('results', [])
                if results and results[0].get('source_file'):
                    source_str = results[0].get('source_file', '')
                else:
                    source_str = 'N/A'

            file_info.append({
                'index': i,
                'date_str': date_str,
                'model': model,
                'prompt': prompt,
                'source_file': source_str,
                'filepath': filepath
            })
        except Exception as e:
            file_info.append({
                'index': i,
                'date_str': 'Error',
                'model': 'Error',
                'prompt': 'Error',
                'source_file': 'Error',
                'filepath': filepath,
                'error': str(e)
            })

    # Group by source file
    grouped = defaultdict(list)
    for info in file_info:
        grouped[info['source_file']].append(info)

    # Sort groups by source file name, and entries within each group by prompt name
    sorted_groups = sorted(grouped.items(), key=lambda x: x[0].lower())

    # Print grouped results
    print(f"{'#':<4} {'Date':<20} {'Model':<30} {'Prompt':<25} {'Source File':<30}")
    print("-" * 120)

    first_group = True
    for source_file, entries in sorted_groups:
        if not first_group:
            print()  # Empty line between groups
        first_group = False

        # Sort entries by prompt name (ascending)
        sorted_entries = sorted(entries, key=lambda x: x['prompt'].lower())

        for info in sorted_entries:
            if 'error' in info:
                print(f"{info['index']:<4} [Error reading file: {info['filepath'].name}]")
            else:
                print(f"{info['index']:<4} {info['date_str']:<20} {info['model']:<30} {info['prompt'][:23]:<25} {info['source_file'][:28]:<30}")

    print("=" * 120)


def select_two_files(available_files):
    """Interactive selection of exactly 2 files."""
    display_file_list(available_files)

    print("\nSelect exactly 2 files to compare (comma-separated numbers, e.g., 1,3):")

    user_input = input("> ").strip()

    if not user_input:
        print("Error: You must select exactly 2 files.")
        return None

    try:
        indices = [int(x.strip()) - 1 for x in user_input.split(',')]

        if len(indices) != 2:
            print(f"Error: Please select exactly 2 files (you selected {len(indices)})")
            return None

        if any(idx < 0 or idx >= len(available_files) for idx in indices):
            print(f"Error: Invalid selection. Numbers must be between 1 and {len(available_files)}")
            return None

        return [available_files[idx] for idx in indices]
    except ValueError:
        print("Error: Invalid input format. Use comma-separated numbers (e.g., 1,3)")
        return None


def extract_json_from_response(response_text):
    """Extract JSON from model response (handles markdown code blocks)."""
    if not response_text:
        return None

    # Try to find JSON in markdown code block
    json_match = re.search(r'```(?:json)?\n(.*?)\n```', response_text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(1))
        except json.JSONDecodeError:
            pass

    # Try to parse entire response as JSON
    try:
        return json.loads(response_text.strip())
    except json.JSONDecodeError:
        pass

    return None


def normalize_text(text):
    """Normalize text for comparison (lowercase, strip, collapse whitespace)."""
    if not text:
        return ""
    return ' '.join(str(text).lower().strip().split())


def extract_data(result_data):
    """Extract entities, pronouns, and chains from a result file."""
    metadata = result_data.get('metadata', {})
    results = result_data.get('results', [])

    if not results:
        return None

    result = results[0]
    response = result.get('model_response', '')
    parsed = extract_json_from_response(response)

    if not parsed:
        return None

    # Normalize entities
    entities = set()
    for e in parsed.get('entities', []):
        entities.add(normalize_text(e))

    # Normalize pronouns
    pronouns = set()
    for p in parsed.get('pronouns', []):
        if isinstance(p, dict):
            pron = normalize_text(p.get('pronoun', ''))
            ref = normalize_text(p.get('referent', ''))
            if pron:
                pronouns.add(f"{pron} -> {ref}")
        else:
            pronouns.add(normalize_text(p))

    # Normalize chains
    chains = []
    for c in parsed.get('chains', []):
        if isinstance(c, dict):
            entity = normalize_text(c.get('entity', ''))
            mentions = sorted([normalize_text(m) for m in c.get('mentions', [])])
            chains.append({
                'entity': entity,
                'mentions': tuple(mentions)
            })

    # Sort chains for comparison
    chains.sort(key=lambda x: x['entity'])

    return {
        'model': metadata.get('model', 'unknown'),
        'prompt': metadata.get('prompt', 'unknown'),
        'source_file': result.get('source_file', 'unknown'),
        'entities': entities,
        'pronouns': pronouns,
        'chains': chains,
        'raw_entities': parsed.get('entities', []),
        'raw_pronouns': parsed.get('pronouns', []),
        'raw_chains': parsed.get('chains', [])
    }


def compare_sets(set1, set2, name):
    """Compare two sets and return similarity metrics."""
    common = set1 & set2
    only_in_1 = set1 - set2
    only_in_2 = set2 - set1

    total_unique = len(set1 | set2)
    if total_unique == 0:
        similarity = 100.0
    else:
        similarity = (len(common) / total_unique) * 100

    return {
        'name': name,
        'common': common,
        'only_in_1': only_in_1,
        'only_in_2': only_in_2,
        'count_1': len(set1),
        'count_2': len(set2),
        'common_count': len(common),
        'similarity': similarity
    }


def compare_chains(chains1, chains2):
    """Compare chains between two results."""
    # Convert to comparable format
    def chain_to_tuple(c):
        return (c['entity'], c['mentions'])

    set1 = set(chain_to_tuple(c) for c in chains1)
    set2 = set(chain_to_tuple(c) for c in chains2)

    common = set1 & set2
    only_in_1 = set1 - set2
    only_in_2 = set2 - set1

    total_unique = len(set1 | set2)
    if total_unique == 0:
        similarity = 100.0
    else:
        similarity = (len(common) / total_unique) * 100

    # Also check entity overlap (chains with same entity but different mentions)
    entities_1 = {c['entity'] for c in chains1}
    entities_2 = {c['entity'] for c in chains2}
    entity_overlap = entities_1 & entities_2

    return {
        'count_1': len(chains1),
        'count_2': len(chains2),
        'exact_match_count': len(common),
        'only_in_1': only_in_1,
        'only_in_2': only_in_2,
        'similarity': similarity,
        'entity_overlap': entity_overlap,
        'entity_overlap_count': len(entity_overlap)
    }


def print_specular_report(data1, data2, file1, file2):
    """Print detailed specular comparison report."""
    print("\n" + "=" * 100)
    print("SPECULAR COMPARISON REPORT")
    print("=" * 100)

    # File info
    print(f"\nFile 1: {file1.name}")
    print(f"  Model: {data1['model']}")
    print(f"  Prompt: {data1['prompt']}")
    print(f"  Source: {data1['source_file']}")

    print(f"\nFile 2: {file2.name}")
    print(f"  Model: {data2['model']}")
    print(f"  Prompt: {data2['prompt']}")
    print(f"  Source: {data2['source_file']}")

    # Check source file match
    if data1['source_file'] != data2['source_file']:
        print("\n" + "!" * 80)
        print("WARNING: Source files are different! Results may not be directly comparable.")
        print("!" * 80)

    # Entity comparison
    print("\n" + "-" * 100)
    print("ENTITY COMPARISON")
    print("-" * 100)

    entity_cmp = compare_sets(data1['entities'], data2['entities'], 'Entities')
    print(f"\n  File 1: {entity_cmp['count_1']} entities")
    print(f"  File 2: {entity_cmp['count_2']} entities")
    print(f"  Common: {entity_cmp['common_count']} entities")
    print(f"  Similarity: {entity_cmp['similarity']:.1f}%")

    if entity_cmp['only_in_1']:
        print(f"\n  Only in File 1 ({len(entity_cmp['only_in_1'])}):")
        for e in sorted(entity_cmp['only_in_1']):
            print(f"    - {e}")

    if entity_cmp['only_in_2']:
        print(f"\n  Only in File 2 ({len(entity_cmp['only_in_2'])}):")
        for e in sorted(entity_cmp['only_in_2']):
            print(f"    - {e}")

    # Pronoun comparison
    print("\n" + "-" * 100)
    print("PRONOUN COMPARISON")
    print("-" * 100)

    pronoun_cmp = compare_sets(data1['pronouns'], data2['pronouns'], 'Pronouns')
    print(f"\n  File 1: {pronoun_cmp['count_1']} pronouns")
    print(f"  File 2: {pronoun_cmp['count_2']} pronouns")
    print(f"  Common: {pronoun_cmp['common_count']} pronouns")
    print(f"  Similarity: {pronoun_cmp['similarity']:.1f}%")

    if pronoun_cmp['only_in_1']:
        print(f"\n  Only in File 1 ({len(pronoun_cmp['only_in_1'])}):")
        for p in sorted(pronoun_cmp['only_in_1']):
            print(f"    - {p}")

    if pronoun_cmp['only_in_2']:
        print(f"\n  Only in File 2 ({len(pronoun_cmp['only_in_2'])}):")
        for p in sorted(pronoun_cmp['only_in_2']):
            print(f"    - {p}")

    # Chain comparison
    print("\n" + "-" * 100)
    print("CHAIN COMPARISON")
    print("-" * 100)

    chain_cmp = compare_chains(data1['chains'], data2['chains'])
    print(f"\n  File 1: {chain_cmp['count_1']} chains")
    print(f"  File 2: {chain_cmp['count_2']} chains")
    print(f"  Exact matches: {chain_cmp['exact_match_count']} chains")
    print(f"  Entity overlap: {chain_cmp['entity_overlap_count']} (same entity, possibly different mentions)")
    print(f"  Similarity: {chain_cmp['similarity']:.1f}%")

    if chain_cmp['only_in_1']:
        print(f"\n  Chains only in File 1 ({len(chain_cmp['only_in_1'])}):")
        for entity, mentions in sorted(chain_cmp['only_in_1']):
            print(f"    - {entity}: {list(mentions)}")

    if chain_cmp['only_in_2']:
        print(f"\n  Chains only in File 2 ({len(chain_cmp['only_in_2'])}):")
        for entity, mentions in sorted(chain_cmp['only_in_2']):
            print(f"    - {entity}: {list(mentions)}")

    # Overall verdict
    print("\n" + "=" * 100)
    print("VERDICT")
    print("=" * 100)

    overall_similarity = (entity_cmp['similarity'] + pronoun_cmp['similarity'] + chain_cmp['similarity']) / 3

    print(f"\n  Entity similarity:  {entity_cmp['similarity']:6.1f}%")
    print(f"  Pronoun similarity: {pronoun_cmp['similarity']:6.1f}%")
    print(f"  Chain similarity:   {chain_cmp['similarity']:6.1f}%")
    print(f"  --------------------------------")
    print(f"  Overall similarity: {overall_similarity:6.1f}%")

    if overall_similarity == 100.0:
        print("\n  SPECULAR: The outputs are IDENTICAL (100% match)")
    elif overall_similarity >= 90.0:
        print("\n  NEAR-SPECULAR: The outputs are VERY SIMILAR (>= 90% match)")
    elif overall_similarity >= 70.0:
        print("\n  SIMILAR: The outputs have significant overlap (>= 70% match)")
    elif overall_similarity >= 50.0:
        print("\n  PARTIAL: The outputs have some overlap (>= 50% match)")
    else:
        print("\n  DIFFERENT: The outputs are substantially different (< 50% match)")

    print("=" * 100)

    return {
        'entity_similarity': entity_cmp['similarity'],
        'pronoun_similarity': pronoun_cmp['similarity'],
        'chain_similarity': chain_cmp['similarity'],
        'overall_similarity': overall_similarity
    }


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Compare two coreference results for specular (equivalent) output',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        'files',
        nargs='*',
        help='Two result JSON files to compare'
    )

    args = parser.parse_args()

    print("=" * 100)
    print("SPECULAR COMPARISON")
    print("=" * 100)

    # Get files to compare
    if args.files and len(args.files) == 2:
        selected_files = [Path(f) for f in args.files]
    else:
        available_files = get_available_results()
        if not available_files:
            print("Error: No result files found in results/ directory.")
            sys.exit(1)

        if len(available_files) < 2:
            print("Error: Need at least 2 result files to compare.")
            sys.exit(1)

        while True:
            selected_files = select_two_files(available_files)
            if selected_files is not None:
                break

    # Load and extract data
    file1, file2 = selected_files

    # Handle paths
    for i, filepath in enumerate([file1, file2]):
        if isinstance(filepath, str):
            filepath = Path(filepath)
        if not filepath.is_absolute() and not filepath.exists():
            if (RESULTS_DIR / filepath.name).exists():
                if i == 0:
                    file1 = RESULTS_DIR / filepath.name
                else:
                    file2 = RESULTS_DIR / filepath.name

    if not file1.exists():
        print(f"Error: File not found: {file1}")
        sys.exit(1)
    if not file2.exists():
        print(f"Error: File not found: {file2}")
        sys.exit(1)

    print(f"\nLoading {file1.name}...")
    result1 = load_result_file(file1)
    data1 = extract_data(result1)

    print(f"Loading {file2.name}...")
    result2 = load_result_file(file2)
    data2 = extract_data(result2)

    if not data1:
        print(f"Error: Could not parse coreference data from {file1.name}")
        sys.exit(1)
    if not data2:
        print(f"Error: Could not parse coreference data from {file2.name}")
        sys.exit(1)

    # Run comparison
    print_specular_report(data1, data2, file1, file2)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nComparison cancelled by user (Ctrl+C).")
        sys.exit(0)
