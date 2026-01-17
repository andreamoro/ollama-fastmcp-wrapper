#!/usr/bin/env python3
"""
Compare pre-fix vs post-fix coreference results to detect invented entities.

Usage:
    python compare_pre_post.py --pre <pre_folder> --post <post_folder> [--export <output.json>]

Example:
    python compare_pre_post.py --pre results/20260116-coref-llama3.2-pre-fix --post results/
"""

import json
import argparse
import sys
from pathlib import Path
from collections import defaultdict

RESULTS_DIR = Path(__file__).parent / 'results'


def load_result_file(filepath):
    """Load and parse a result JSON file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def extract_entities_from_result(result_data):
    """Extract entities, pronouns, and chains from a result file."""
    metadata = result_data.get('metadata', {})
    results = result_data.get('results', [])

    if not results:
        return None

    result = results[0]
    response = result.get('model_response', '')

    # Try to parse JSON
    parsed = None
    try:
        parsed = json.loads(response)
    except json.JSONDecodeError:
        import re
        json_match = re.search(r'```(?:json)?\n(.*?)\n```', response, re.DOTALL)
        if json_match:
            try:
                parsed = json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass

    if not parsed:
        return None

    return {
        'prompt': metadata.get('prompt', 'unknown'),
        'model': metadata.get('model', 'unknown'),
        'timestamp': metadata.get('timestamp', ''),
        'entities': parsed.get('entities', []),
        'pronouns': parsed.get('pronouns', []),
        'chains': parsed.get('chains', []),
        'raw_response': response
    }


def find_matching_files(pre_folder, post_folder):
    """Find matching prompt files between pre and post folders."""
    pre_path = Path(pre_folder)
    post_path = Path(post_folder)

    if not pre_path.exists():
        print(f"Error: Pre folder not found: {pre_path}")
        sys.exit(1)
    if not post_path.exists():
        print(f"Error: Post folder not found: {post_path}")
        sys.exit(1)

    # Get all json files
    pre_files = {f.name: f for f in pre_path.glob('*-textfile.json')}
    post_files = {f.name: f for f in post_path.glob('*-textfile.json')}

    # Match by prompt name (extract prompt from filename)
    def get_prompt_from_filename(filename):
        # Format: 20260116_081729-llama3.2-latest-coref_1_ita-textfile.json
        parts = filename.replace('-textfile.json', '').split('-')
        if len(parts) >= 2:
            return parts[-1]  # Last part before -textfile is the prompt
        return filename

    pre_by_prompt = {}
    for name, path in pre_files.items():
        prompt = get_prompt_from_filename(name)
        pre_by_prompt[prompt] = path

    post_by_prompt = {}
    for name, path in post_files.items():
        prompt = get_prompt_from_filename(name)
        post_by_prompt[prompt] = path

    # Find common prompts
    common_prompts = set(pre_by_prompt.keys()) & set(post_by_prompt.keys())

    matches = []
    for prompt in sorted(common_prompts):
        matches.append({
            'prompt': prompt,
            'pre_file': pre_by_prompt[prompt],
            'post_file': post_by_prompt[prompt]
        })

    return matches


def compare_entities(pre_entities, post_entities):
    """Compare two lists of entities and find differences."""
    pre_set = set(e.lower().strip() for e in pre_entities if isinstance(e, str))
    post_set = set(e.lower().strip() for e in post_entities if isinstance(e, str))

    return {
        'pre_only': sorted(pre_set - post_set),
        'post_only': sorted(post_set - pre_set),
        'common': sorted(pre_set & post_set),
        'pre_count': len(pre_entities),
        'post_count': len(post_entities)
    }


def check_for_contamination(entities, known_contaminants=None):
    """Check if entities contain known contaminants from prompt examples."""
    if known_contaminants is None:
        known_contaminants = [
            'marco', 'giulia', 'john', 'sarah', 'alice', 'bob',
            'il bar', 'the cafe', 'caffè', 'coffee', 'un libro', 'a book'
        ]

    contaminated = []
    for entity in entities:
        entity_lower = entity.lower().strip() if isinstance(entity, str) else str(entity).lower()
        for contaminant in known_contaminants:
            if contaminant in entity_lower:
                contaminated.append(entity)
                break

    return contaminated


def print_comparison_report(matches, results):
    """Print a detailed comparison report."""
    print("\n" + "=" * 100)
    print("PRE-FIX vs POST-FIX COMPARISON REPORT")
    print("=" * 100)

    summary = {
        'total_prompts': len(results),
        'pre_contaminated': 0,
        'post_contaminated': 0,
        'entity_count_increased': 0,
        'entity_count_decreased': 0,
        'entity_count_same': 0
    }

    for result in results:
        prompt = result['prompt']
        pre = result['pre']
        post = result['post']
        diff = result['diff']

        print(f"\n{'─' * 100}")
        print(f"PROMPT: {prompt}")
        print(f"{'─' * 100}")

        # Entity counts
        pre_count = diff['pre_count']
        post_count = diff['post_count']
        delta = post_count - pre_count
        delta_str = f"+{delta}" if delta > 0 else str(delta)

        print(f"\nEntity counts: PRE={pre_count}, POST={post_count} ({delta_str})")

        if delta > 0:
            summary['entity_count_increased'] += 1
        elif delta < 0:
            summary['entity_count_decreased'] += 1
        else:
            summary['entity_count_same'] += 1

        # Contamination check
        pre_contaminated = result.get('pre_contaminated', [])
        post_contaminated = result.get('post_contaminated', [])

        if pre_contaminated:
            summary['pre_contaminated'] += 1
            print(f"\n⚠️  PRE-FIX CONTAMINATION DETECTED:")
            for c in pre_contaminated:
                print(f"    - {c}")

        if post_contaminated:
            summary['post_contaminated'] += 1
            print(f"\n⚠️  POST-FIX CONTAMINATION DETECTED:")
            for c in post_contaminated:
                print(f"    - {c}")

        if not pre_contaminated and not post_contaminated:
            print(f"\n✓ No contamination detected")

        # Entity differences
        if diff['pre_only']:
            print(f"\nEntities ONLY in PRE ({len(diff['pre_only'])}):")
            for e in diff['pre_only'][:10]:
                print(f"    - {e}")
            if len(diff['pre_only']) > 10:
                print(f"    ... and {len(diff['pre_only']) - 10} more")

        if diff['post_only']:
            print(f"\nEntities ONLY in POST ({len(diff['post_only'])}):")
            for e in diff['post_only'][:10]:
                print(f"    - {e}")
            if len(diff['post_only']) > 10:
                print(f"    ... and {len(diff['post_only']) - 10} more")

        if diff['common']:
            print(f"\nCommon entities: {len(diff['common'])}")

    # Summary
    print("\n" + "=" * 100)
    print("SUMMARY")
    print("=" * 100)

    print(f"\nTotal prompts compared: {summary['total_prompts']}")
    print(f"\nContamination:")
    print(f"  - PRE-FIX files with contamination: {summary['pre_contaminated']}")
    print(f"  - POST-FIX files with contamination: {summary['post_contaminated']}")

    print(f"\nEntity count changes:")
    print(f"  - Increased (POST > PRE): {summary['entity_count_increased']}")
    print(f"  - Decreased (POST < PRE): {summary['entity_count_decreased']}")
    print(f"  - Same: {summary['entity_count_same']}")

    return summary


def export_comparison(results, summary, output_file):
    """Export comparison results to JSON."""
    export_data = {
        'summary': summary,
        'comparisons': []
    }

    for result in results:
        export_data['comparisons'].append({
            'prompt': result['prompt'],
            'pre_file': str(result['pre_file']),
            'post_file': str(result['post_file']),
            'pre_entity_count': result['diff']['pre_count'],
            'post_entity_count': result['diff']['post_count'],
            'pre_contaminated': result.get('pre_contaminated', []),
            'post_contaminated': result.get('post_contaminated', []),
            'entities_only_in_pre': result['diff']['pre_only'],
            'entities_only_in_post': result['diff']['post_only'],
            'common_entities': result['diff']['common']
        })

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(export_data, f, indent=2, ensure_ascii=False)

    print(f"\nComparison exported to: {output_file}")


def export_markdown(results, summary, output_file):
    """Export comparison results to Markdown."""
    lines = []

    # Header
    lines.append("# Pre-Fix vs Post-Fix Comparison Report\n")
    lines.append(f"**Date:** {results[0]['pre'].get('timestamp', '')[:10] if results else 'N/A'}")
    lines.append(f"**Model:** {results[0]['pre'].get('model', 'N/A') if results else 'N/A'}")
    lines.append("\n---\n")

    # Summary table
    lines.append("## Summary\n")
    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    lines.append(f"| Total prompts compared | {summary['total_prompts']} |")
    lines.append(f"| PRE files with contamination | {summary['pre_contaminated']} |")
    lines.append(f"| POST files with contamination | {summary['post_contaminated']} |")
    lines.append(f"| Entity count increased (POST > PRE) | {summary['entity_count_increased']} |")
    lines.append(f"| Entity count decreased (POST < PRE) | {summary['entity_count_decreased']} |")
    lines.append("\n---\n")

    # Entity count overview
    lines.append("## Entity Count Overview\n")
    lines.append("| Prompt | PRE | POST | Delta | Trend |")
    lines.append("|--------|-----|------|-------|-------|")
    for result in results:
        prompt = result['prompt']
        pre_count = result['diff']['pre_count']
        post_count = result['diff']['post_count']
        delta = post_count - pre_count
        delta_str = f"+{delta}" if delta > 0 else str(delta)
        trend = ":arrow_up:" if delta > 0 else (":arrow_down:" if delta < 0 else "-")
        lines.append(f"| {prompt} | {pre_count} | {post_count} | {delta_str} | {trend} |")
    lines.append("\n---\n")

    # Detailed entity comparison
    lines.append("## Detailed Entity Comparison by Prompt\n")

    for result in results:
        prompt = result['prompt']
        diff = result['diff']

        lines.append(f"### {prompt}\n")

        # Collect all entities for this prompt
        all_entities = sorted(set(diff['pre_only'] + diff['post_only'] + diff['common']))
        pre_set = set(diff['common'] + diff['pre_only'])
        post_set = set(diff['common'] + diff['post_only'])

        lines.append("| Entity | PRE | POST |")
        lines.append("|--------|-----|------|")
        for entity in all_entities:
            pre_mark = "X" if entity in pre_set else ""
            post_mark = "X" if entity in post_set else ""
            lines.append(f"| {entity} | {pre_mark} | {post_mark} |")
        lines.append(f"| **TOTAL** | **{diff['pre_count']}** | **{diff['post_count']}** |")
        lines.append("\n---\n")

    # Key observations
    lines.append("## Key Observations\n")
    if summary['pre_contaminated'] == 0 and summary['post_contaminated'] == 0:
        lines.append("1. **No contamination detected** - the prompt fixes successfully eliminated invented entities.\n")

    # Find biggest changes
    if results:
        max_increase = max(results, key=lambda r: r['diff']['post_count'] - r['diff']['pre_count'])
        max_decrease = min(results, key=lambda r: r['diff']['post_count'] - r['diff']['pre_count'])

        increase_delta = max_increase['diff']['post_count'] - max_increase['diff']['pre_count']
        decrease_delta = max_decrease['diff']['post_count'] - max_decrease['diff']['pre_count']

        if increase_delta > 0:
            lines.append(f"2. **{max_increase['prompt']}** showed the most improvement post-fix (+{increase_delta} entities).\n")
        if decrease_delta < 0:
            lines.append(f"3. **{max_decrease['prompt']}** had the largest decrease ({decrease_delta} entities).\n")

    # Write file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))

    print(f"\nMarkdown exported to: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description='Compare pre-fix vs post-fix coreference results',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python compare_pre_post.py --pre results/20260116-coref-llama3.2-pre-fix --post results/
    python compare_pre_post.py --pre results/pre-fix --post results/post-fix --export comparison.json
        """
    )

    parser.add_argument('--pre', required=True, help='Folder containing pre-fix result files')
    parser.add_argument('--post', required=True, help='Folder containing post-fix result files')
    parser.add_argument('--output', type=str, default=None, help='Base filename for output (will create .json and .md files)')

    args = parser.parse_args()

    print("=" * 100)
    print("PRE-FIX vs POST-FIX COMPARISON")
    print("=" * 100)

    print(f"\nPRE folder: {args.pre}")
    print(f"POST folder: {args.post}")

    # Find matching files
    matches = find_matching_files(args.pre, args.post)

    if not matches:
        print("\nError: No matching prompt files found between the two folders.")
        sys.exit(1)

    print(f"\nFound {len(matches)} matching prompt(s):")
    for m in matches:
        print(f"  - {m['prompt']}")

    # Load and compare each pair
    results = []
    for match in matches:
        print(f"\nLoading {match['prompt']}...")

        pre_data = load_result_file(match['pre_file'])
        post_data = load_result_file(match['post_file'])

        pre_extracted = extract_entities_from_result(pre_data)
        post_extracted = extract_entities_from_result(post_data)

        if not pre_extracted or not post_extracted:
            print(f"  Warning: Could not parse results for {match['prompt']}")
            continue

        # Compare entities
        diff = compare_entities(pre_extracted['entities'], post_extracted['entities'])

        # Check for contamination
        pre_contaminated = check_for_contamination(pre_extracted['entities'])
        post_contaminated = check_for_contamination(post_extracted['entities'])

        results.append({
            'prompt': match['prompt'],
            'pre_file': match['pre_file'],
            'post_file': match['post_file'],
            'pre': pre_extracted,
            'post': post_extracted,
            'diff': diff,
            'pre_contaminated': pre_contaminated,
            'post_contaminated': post_contaminated
        })

    # Print report
    summary = print_comparison_report(matches, results)

    # Export results (both JSON and Markdown)
    if args.output:
        export_comparison(results, summary, args.output + '.json')
        export_markdown(results, summary, args.output + '.md')

    print("\n" + "=" * 100)
    print("COMPARISON COMPLETE")
    print("=" * 100)


if __name__ == '__main__':
    main()
