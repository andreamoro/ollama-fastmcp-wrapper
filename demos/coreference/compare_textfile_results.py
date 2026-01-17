#!/usr/bin/env python3
"""
Compare Coreference Analysis Results from Text Files

Compares multiple result files to evaluate:
- Entities found
- Pronouns identified
- Coreference chains
- Processing time
- Model response quality vs original text

What Makes a Good Coreference Analysis?
---------------------------------------
Coreference resolution identifies when different expressions in a text refer to the
same real-world entity. Quality is measured by several factors:

1. **Chain Density** (mentions per chain):
   - Higher density indicates richer, more connected chains
   - A chain with 5 mentions linking "Maria", "lei", "la ragazza", "sua" is more
     valuable than 5 separate single-mention chains
   - Target: density > 2.0 indicates meaningful entity tracking

2. **Coreference Type Coverage**:
   Good analysis should detect multiple coreference types:
   - Pronominal: pronouns referring to entities ("lui", "lei", "esso", "loro")
   - Nominal: noun phrases referring to same entity ("il cliente" → "l'utente")
   - Proper names: name variations ("La Roche-Posay" → "LRP")
   - Definite descriptions: descriptive references ("il prodotto" → "la crema")
   - Demonstratives: deictic references ("questo", "quella soluzione")

3. **Entity-Chain Balance**:
   - Too many entities with few chains = fragmented analysis (missed links)
   - Few entities with dense chains = focused, well-connected analysis
   - Optimal: entities are grouped into meaningful chains, not isolated

4. **Processing Time** (secondary metric):
   - Faster is better when quality is comparable
   - Excessive time may indicate model confusion or over-analysis

5. **Scoring Formula**:
   The comparison score weighs these factors:
   - Base: number of coreference chains detected
   - Bonus: chain density (rewards connected chains over isolated entities)
   - Bonus: type coverage (rewards detecting multiple coreference types)
   - Penalty: excessive processing time

Usage:
    # Interactive mode - select files to compare
    uv run python compare_textfile_results.py

    # Specify files to compare
    uv run python compare_textfile_results.py file1.json file2.json file3.json

    # Compare all textfile results
    uv run python compare_textfile_results.py --all
"""

import sys
import json
import re
from pathlib import Path
from datetime import datetime
from collections import defaultdict

SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR / "results"
DATASETS_DIR = SCRIPT_DIR / "datasets"


def load_result_file(filepath):
    """Load a result JSON file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def get_available_textfile_results():
    """Get list of available textfile result JSON files, sorted by modification time (newest first)."""
    if not RESULTS_DIR.exists():
        return []

    # Get JSON files, excluding comparison reports
    json_files = [
        f for f in RESULTS_DIR.glob("*.json")
        if f.is_file() and 'comparison' not in f.name.lower()
    ]
    # Sort by modification time, newest first
    json_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return json_files


def display_file_list(files):
    """Display numbered list of available result files, grouped by source → model → prompt → date."""
    print("\n" + "=" * 120)
    print("AVAILABLE RESULT FILES")
    print("=" * 120)

    # First pass: collect file info
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
                sort_date = dt  # For sorting
            except:
                date_str = timestamp[:16] if timestamp else 'Unknown'
                sort_date = datetime.min  # Put unknowns at the end

            model = metadata.get('model', 'Unknown')
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
                'sort_date': sort_date,
                'model': model,
                'prompt': prompt,
                'source_file': source_str,
                'filepath': filepath
            })
        except Exception as e:
            file_info.append({
                'index': i,
                'date_str': 'Error',
                'sort_date': datetime.min,
                'model': 'Error',
                'prompt': 'Error',
                'source_file': 'Error',
                'filepath': filepath,
                'error': str(e)
            })

    # Group by source file → model → prompt
    # Structure: {source: {model: {prompt: [entries]}}}
    grouped = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for info in file_info:
        grouped[info['source_file']][info['model']][info['prompt']].append(info)

    # Sort and print
    print(f"{'#':<4} {'Date':<20} {'Model':<30} {'Prompt':<25} {'Source File':<30}")
    print("-" * 120)

    first_source = True
    for source_file in sorted(grouped.keys(), key=str.lower):
        if not first_source:
            print()  # Empty line between source groups
        first_source = False

        first_model = True
        for model in sorted(grouped[source_file].keys(), key=str.lower):
            if not first_model:
                print()  # Empty line between model groups
            first_model = False

            first_prompt = True
            for prompt in sorted(grouped[source_file][model].keys(), key=str.lower):
                if not first_prompt:
                    print()  # Empty line between prompt groups
                first_prompt = False

                # Sort entries by date (newest first)
                entries = sorted(grouped[source_file][model][prompt], key=lambda x: x['sort_date'], reverse=True)

                for info in entries:
                    if 'error' in info:
                        print(f"{info['index']:<4} [Error reading file: {info['filepath'].name}]")
                    else:
                        print(f"{info['index']:<4} {info['date_str']:<20} {info['model'][:28]:<30} {info['prompt'][:23]:<25} {info['source_file'][:28]:<30}")

    print("=" * 120)


def select_files(available_files):
    """Interactive file selection."""
    display_file_list(available_files)

    print("\nSelect files to compare (comma-separated numbers, e.g., 1,3,5):")
    print("Or press Enter to compare the 3 most recent files")

    user_input = input("> ").strip()

    # Default: compare 3 most recent
    if not user_input:
        return available_files[:min(3, len(available_files))]

    # Parse comma-separated selection
    try:
        indices = [int(x.strip()) - 1 for x in user_input.split(',')]

        # Validate - need at least 2 files
        if len(indices) < 2:
            print(f"Error: Please select at least 2 files (you selected {len(indices)})")
            return None

        if any(idx < 0 or idx >= len(available_files) for idx in indices):
            print(f"Error: Invalid selection. Numbers must be between 1 and {len(available_files)}")
            return None

        return [available_files[idx] for idx in indices]
    except ValueError:
        print("Error: Invalid input format. Use comma-separated numbers (e.g., 1,3,5)")
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


def analyze_result(result_data):
    """Analyze a single result file and extract key metrics."""
    metadata = result_data.get('metadata', {})
    results = result_data.get('results', [])

    if not results:
        return None

    result = results[0]  # Single file analysis
    response = result.get('model_response', '')

    # Parse JSON response
    parsed = extract_json_from_response(response)

    # Use Ollama's actual processing time (total_duration_s) instead of elapsed_time
    # to avoid network latency affecting performance comparisons
    metrics = result.get('metrics', {})
    processing_time = metrics.get('total_duration_s', result.get('elapsed_time', 0))

    analysis = {
        'model': metadata.get('model', 'unknown'),
        'prompt': metadata.get('prompt', 'unknown'),
        'timestamp': metadata.get('timestamp', ''),
        'source_file': result.get('source_file', 'unknown'),
        'text_length': result.get('text_length', 0),
        'elapsed_time': result.get('elapsed_time', 0),  # Keep for reference
        'processing_time': processing_time,  # Actual model processing time
        'metrics': metrics,
        'raw_response': response,
        'parsed_response': parsed,
        'entity_count': 0,
        'pronoun_count': 0,
        'chain_count': 0,
        'entities': [],
        'pronouns': [],
        'chains': [],
        'filepath': ''  # Will be set by caller
    }

    if parsed:
        analysis['entities'] = parsed.get('entities', [])
        analysis['pronouns'] = parsed.get('pronouns', [])
        analysis['chains'] = parsed.get('chains', [])
        analysis['entity_count'] = len(analysis['entities'])
        analysis['pronoun_count'] = len(analysis['pronouns'])
        analysis['chain_count'] = len(analysis['chains'])

    return analysis


def load_source_text(source_file):
    """Load the original text that was analyzed."""
    filepath = DATASETS_DIR / source_file
    if not filepath.exists():
        return None

    with open(filepath, 'r', encoding='utf-8') as f:
        return f.read()


def analyze_chain_quality(chains):
    """Analyze quality metrics for coreference chains."""
    if not chains:
        return {
            'chain_density': 0,
            'avg_mentions_per_chain': 0,
            'singleton_ratio': 0,
            'long_chain_count': 0
        }

    mention_counts = []
    singleton_count = 0
    long_chain_count = 0

    for chain in chains:
        if isinstance(chain, dict):
            mentions = chain.get('mentions', [])
            mention_count = len(mentions)
        else:
            mention_count = 1

        mention_counts.append(mention_count)

        if mention_count == 1:
            singleton_count += 1
        elif mention_count >= 3:
            long_chain_count += 1

    total_mentions = sum(mention_counts)
    avg_mentions = total_mentions / len(chains) if chains else 0
    singleton_ratio = singleton_count / len(chains) if chains else 0

    return {
        'chain_density': avg_mentions,
        'avg_mentions_per_chain': avg_mentions,
        'singleton_ratio': singleton_ratio,
        'long_chain_count': long_chain_count
    }


def detect_coreference_types(parsed_response, source_text):
    """Detect what types of coreference relations were identified."""
    types_found = {
        'pronominal': False,
        'nominal': False,
        'proper_names': False,
        'definite_descriptions': False,
        'demonstratives': False,
        'has_chains': False
    }

    if not parsed_response:
        return types_found

    # Check for chains
    chains = parsed_response.get('chains', [])
    if chains:
        types_found['has_chains'] = True

    # Check pronouns field for pronominal and demonstrative coreference
    pronouns = parsed_response.get('pronouns', [])

    # Personal/possessive pronouns (pronominal coreference)
    personal_pronouns = [
        # English
        'he', 'she', 'it', 'they', 'him', 'her', 'his', 'their', 'its', 'we', 'us', 'our',
        # Italian
        'lui', 'lei', 'essi', 'esse', 'essa', 'loro', 'lo', 'la', 'li', 'le', 'ne', 'ci', 'vi',
        'suo', 'sua', 'suoi', 'sue', 'nostro', 'nostra', 'nostri', 'nostre',
        'i nostri', 'le nostre', 'il nostro', 'la nostra',
        # Spanish
        'él', 'ella', 'ellos', 'ellas', 'su', 'sus', 'nuestro', 'nuestra'
    ]

    # Demonstratives
    demonstratives = [
        # English
        'this', 'that', 'these', 'those',
        # Italian
        'questo', 'questa', 'questi', 'queste',
        'quello', 'quella', 'quelli', 'quelle',
        'ciò', 'tale', 'tali',
        # Spanish
        'este', 'esta', 'estos', 'estas',
        'ese', 'esa', 'esos', 'esas',
        'aquel', 'aquella', 'aquellos', 'aquellas'
    ]

    for pron_entry in pronouns:
        if pron_entry is None:
            continue
        if isinstance(pron_entry, dict):
            pron_text = (pron_entry.get('pronoun') or '').lower().strip()
        else:
            pron_text = str(pron_entry).lower().strip()

        # Check for personal/possessive pronouns
        if pron_text in personal_pronouns or any(p in pron_text for p in personal_pronouns):
            types_found['pronominal'] = True

        # Check for demonstratives
        if pron_text in demonstratives or any(d in pron_text.split() for d in demonstratives):
            types_found['demonstratives'] = True

    # Analyze chain content for proper names and nominal coreference
    for chain in chains:
        if isinstance(chain, dict):
            entity = chain.get('entity', '')
            mentions = chain.get('mentions', [])

            # Proper names - entity name starts with uppercase
            if entity and len(entity) > 1:
                words = str(entity).split()
                if any(w[0].isupper() for w in words if len(w) > 0):
                    types_found['proper_names'] = True

            # Nominal coreference - multiple different noun phrases referring to same entity
            if len(mentions) >= 2:
                types_found['nominal'] = True

            # Check mentions for definite descriptions (the X, il X, la X - but not just articles in names)
            for mention in mentions:
                mention_lower = str(mention).lower().strip()
                # More strict check: must be "the/il/la + noun phrase", not just starting with article
                if (mention_lower.startswith('the ') and len(mention_lower) > 5 and
                    not any(c.isupper() for c in mention[4:])):  # Not a proper name after "the"
                    types_found['definite_descriptions'] = True
                # For Italian, check for definite article + common noun (not proper names)
                elif (mention_lower.startswith(('il ', 'lo ', 'la ', "l'", 'i ', 'gli ', 'le ')) and
                      len(mention_lower) > 4):
                    # Skip if it looks like a proper name (has uppercase after article)
                    rest = mention[2:].strip() if mention[1] == ' ' else mention[3:].strip()
                    if rest and not rest[0].isupper():
                        types_found['definite_descriptions'] = True

    return types_found


def calculate_quality_scores(analyses, source_text=None):
    """Calculate quality scores for each analysis based on coreference quality."""
    for analysis in analyses:
        score = 0
        parsed = analysis.get('parsed_response')

        # 1. Format Compliance (30 points)
        if parsed:
            score += 20  # Valid JSON
            if parsed.get('entities'):
                score += 5
            if parsed.get('chains'):
                score += 5  # Chains are more important than pronoun list

        # 2. Chain Quality (40 points)
        chains = analysis.get('chains', [])
        chain_metrics = analyze_chain_quality(chains)

        # Chain density (avg mentions per entity) - ideal is 2-4
        density = chain_metrics['chain_density']
        if 2 <= density <= 4:
            score += 15  # Optimal density
        elif 1.5 <= density < 2 or 4 < density <= 5:
            score += 10  # Acceptable
        elif density > 1:
            score += 5   # Suboptimal but present

        # Singleton ratio (lower is better - means entities are linked)
        singleton_ratio = chain_metrics['singleton_ratio']
        if singleton_ratio < 0.3:
            score += 10  # Good linking
        elif singleton_ratio < 0.5:
            score += 5   # Acceptable

        # Long chains (3+ mentions)
        long_chains = chain_metrics['long_chain_count']
        if long_chains > 0:
            score += min(long_chains * 3, 15)  # Up to 15 points

        # 3. Coreference Type Coverage (20 points)
        coref_types = detect_coreference_types(parsed, source_text)
        type_score = 0
        if coref_types['has_chains']:
            type_score += 5
        if coref_types['pronominal']:
            type_score += 3
        if coref_types['nominal']:
            type_score += 4
        if coref_types['proper_names']:
            type_score += 4
        if coref_types['definite_descriptions']:
            type_score += 2
        if coref_types['demonstratives']:
            type_score += 2
        score += type_score

        # 4. Efficiency (10 points) - based on actual processing time, not network latency
        processing_time = analysis.get('processing_time', analysis.get('elapsed_time', 0))
        if processing_time > 0:
            time_score = max(0, 10 - (processing_time / 20))
            score += time_score

        analysis['quality_score'] = round(score, 1)
        analysis['chain_metrics'] = chain_metrics
        analysis['coref_types'] = coref_types

    return analyses


def get_file_id(analysis):
    """Extract a short identifier from the filepath for display."""
    filepath = analysis.get('filepath', '')
    if not filepath:
        timestamp = analysis.get('timestamp', '')
        return timestamp[:10] if timestamp else 'unknown'

    # Extract date from filename (e.g., 20260116_083441 -> 0116_0834)
    parts = filepath.replace('.json', '').split('/')
    filename = parts[-1]

    # Try to extract timestamp from filename like "20260116_083441-..."
    if '_' in filename and len(filename) > 15:
        date_part = filename[:15]  # "20260116_083441"
        short_date = date_part[4:8] + '_' + date_part[9:13]  # "0116_0834"

        # Check if in subfolder (pre-fix vs post-fix)
        if len(parts) > 1:
            folder = parts[-2]
            if 'pre-fix' in folder.lower():
                return f"{short_date}/PRE"
            elif 'post' in folder.lower():
                return f"{short_date}/POST"
        return short_date

    return filename[:12]


def print_summary_table(analyses):
    """Print executive summary table with rankings."""
    print("\n" + "=" * 160)
    print("EXECUTIVE SUMMARY - RANKED BY QUALITY SCORE")
    print("=" * 160)

    # Sort by quality score
    ranked = sorted(analyses, key=lambda x: x.get('quality_score', 0), reverse=True)

    # Header
    print(f"\n{'Rank':<6} {'ID':<14} {'Score':<8} {'Model':<18} {'Prompt':<16} {'Chains':<8} {'Density':<9} {'Types':<22} {'Time':<8} {'Status':<10}")
    print("-" * 160)

    # Rows
    for i, analysis in enumerate(ranked, 1):
        rank = f"#{i}"
        file_id = get_file_id(analysis)
        score = f"{analysis.get('quality_score', 0):.1f}/100"
        model = analysis['model'][:17]
        prompt = analysis['prompt'].replace('.txt', '')[:15]
        chains = analysis['chain_count']

        chain_metrics = analysis.get('chain_metrics', {})
        density = f"{chain_metrics.get('avg_mentions_per_chain', 0):.1f}"

        # Coreference types detected
        coref_types = analysis.get('coref_types', {})
        types_list = []
        if coref_types.get('pronominal'):
            types_list.append('Pron')
        if coref_types.get('nominal'):
            types_list.append('Nom')
        if coref_types.get('proper_names'):
            types_list.append('Names')
        if coref_types.get('definite_descriptions'):
            types_list.append('Def')
        if coref_types.get('demonstratives'):
            types_list.append('Demo')
        types_str = ', '.join(types_list) if types_list else '-'
        types_str = types_str[:21]

        processing_time = analysis.get('processing_time', analysis.get('elapsed_time', 0))
        time = f"{processing_time:.1f}s"
        status = "✓ Valid" if analysis['parsed_response'] else "✗ Failed"

        print(f"{rank:<6} {file_id:<14} {score:<8} {model:<18} {prompt:<16} {chains:<8} {density:<9} {types_str:<22} {time:<8} {status:<10}")

    print()

    # Key insights
    print("\nKEY INSIGHTS:")

    # Helper to count actual coref types (excluding 'has_chains' which is just a flag)
    def count_coref_types(coref_types):
        real_types = ['pronominal', 'nominal', 'proper_names', 'definite_descriptions', 'demonstratives']
        return sum(1 for t in real_types if coref_types.get(t, False))

    valid_analyses = [a for a in analyses if a['parsed_response']]
    if valid_analyses:
        best = max(valid_analyses, key=lambda x: x['quality_score'])
        best_types = best.get('coref_types', {})
        types_count = count_coref_types(best_types)
        best_id = get_file_id(best)
        print(f"  🏆 Best overall: [{best_id}] {best['prompt']} (score: {best['quality_score']:.1f}, {types_count} coref types)")

        # Best chain quality
        best_chain = max(valid_analyses, key=lambda x: x.get('chain_metrics', {}).get('avg_mentions_per_chain', 0))
        chain_density = best_chain.get('chain_metrics', {}).get('avg_mentions_per_chain', 0)
        chain_id = get_file_id(best_chain)
        print(f"  🔗 Best chain quality: [{chain_id}] {best_chain['prompt']} (density: {chain_density:.1f} mentions/entity)")

        # Most diverse (types covered) - only count actual coref types, not 'has_chains'
        most_diverse = max(valid_analyses, key=lambda x: count_coref_types(x.get('coref_types', {})))
        diverse_count = count_coref_types(most_diverse.get('coref_types', {}))
        diverse_id = get_file_id(most_diverse)
        # Only show "Most comprehensive" if it actually has MORE types than "Best overall"
        if diverse_count > types_count:
            print(f"  📊 Most comprehensive: [{diverse_id}] {most_diverse['prompt']} ({diverse_count} coref types)")

        fastest = min(valid_analyses, key=lambda x: x.get('processing_time', x.get('elapsed_time', 0)))
        fastest_time = fastest.get('processing_time', fastest.get('elapsed_time', 0))
        fastest_id = get_file_id(fastest)
        print(f"  ⚡ Fastest: [{fastest_id}] {fastest['prompt']} ({fastest_time:.1f}s)")

    failed = [a for a in analyses if not a['parsed_response']]
    if failed:
        print(f"\n  ⚠️  {len(failed)} result(s) failed to parse correctly")

    # Coreference type coverage summary
    print("\nCOREFERENCE TYPE COVERAGE:")
    type_names = {
        'pronominal': 'Pronominal (he, she, it)',
        'nominal': 'Nominal (doctor → physician)',
        'proper_names': 'Proper Names (Obama → President)',
        'definite_descriptions': 'Definite Descriptions (the man)',
        'demonstratives': 'Demonstratives (this, that)'
    }

    for type_key, type_label in type_names.items():
        count = sum(1 for a in valid_analyses if a.get('coref_types', {}).get(type_key, False))
        coverage = count / len(valid_analyses) * 100 if valid_analyses else 0
        bar = '█' * int(coverage / 5)  # 20 bars max
        print(f"  {type_label:<40} {bar} {coverage:.0f}% ({count}/{len(valid_analyses)})")


def print_comparison_table(analyses):
    """Print detailed comparison table."""
    print("\n" + "=" * 140)
    print("DETAILED COMPARISON")
    print("=" * 140)

    # Header
    print(f"\n{'File/Date':<45} {'Prompt':<18} {'Entities':<10} {'Pronouns':<10} {'Chains':<10} {'Time (s)':<10} {'Tokens/s':<10}")
    print("-" * 140)

    # Rows
    for analysis in analyses:
        # Extract date from filepath or timestamp
        filepath = analysis.get('filepath', '')
        timestamp = analysis.get('timestamp', '')
        if filepath:
            # Extract filename without extension, show folder if present
            parts = filepath.replace('.json', '').split('/')
            if len(parts) > 1:
                file_info = f"{parts[-2][:20]}/{parts[-1][:22]}"
            else:
                file_info = parts[-1][:44]
        elif timestamp:
            file_info = timestamp[:19]  # Just date and time
        else:
            file_info = 'unknown'

        prompt = analysis['prompt'][:17]
        entities = analysis['entity_count']
        pronouns = analysis['pronoun_count']
        chains = analysis['chain_count']
        processing_time = analysis.get('processing_time', analysis.get('elapsed_time', 0))
        time = f"{processing_time:.1f}"
        tps = analysis['metrics'].get('tokens_per_second', 0)
        tps_str = f"{tps:.1f}" if tps else "-"

        print(f"{file_info:<45} {prompt:<18} {entities:<10} {pronouns:<10} {chains:<10} {time:<10} {tps_str:<10}")

    print()


def print_detailed_comparison(analyses, source_text):
    """Print detailed comparison including entities, pronouns, and chains."""
    print("\n" + "=" * 100)
    print("DETAILED ANALYSIS")
    print("=" * 100)

    # Source text
    if source_text:
        print("\n" + "-" * 100)
        print("ORIGINAL TEXT")
        print("-" * 100)
        print(source_text[:500] + ("..." if len(source_text) > 500 else ""))
        print(f"\n[Total length: {len(source_text)} characters]")

    for i, analysis in enumerate(analyses, 1):
        print("\n" + "-" * 100)
        filepath = analysis.get('filepath', '')
        print(f"RESULT {i}: {analysis['model']} / {analysis['prompt']}")
        if filepath:
            print(f"File: {filepath}")
        print("-" * 100)

        processing_time = analysis.get('processing_time', analysis.get('elapsed_time', 0))
        elapsed_time = analysis.get('elapsed_time', 0)
        print(f"\nProcessing time: {processing_time:.2f}s (Ollama server)")
        if abs(elapsed_time - processing_time) > 1.0:
            print(f"Wall-clock time: {elapsed_time:.2f}s (includes network latency)")
        print(f"Text length: {analysis['text_length']} chars")

        if analysis['metrics']:
            metrics = analysis['metrics']
            print(f"Prompt tokens: {metrics.get('prompt_tokens', 'N/A')}")
            print(f"Completion tokens: {metrics.get('completion_tokens', 'N/A')}")
            print(f"Tokens/second: {metrics.get('tokens_per_second', 'N/A'):.1f}")

        if analysis['parsed_response']:
            print(f"\n✓ Response parsed successfully")

            # Entities
            print(f"\nEntities ({analysis['entity_count']}):")
            for entity in analysis['entities'][:10]:  # Show first 10
                if isinstance(entity, dict):
                    print(f"  - {entity.get('text', entity)}")
                else:
                    print(f"  - {entity}")
            if len(analysis['entities']) > 10:
                print(f"  ... and {len(analysis['entities']) - 10} more")

            # Pronouns
            print(f"\nPronouns ({analysis['pronoun_count']}):")
            for pronoun in analysis['pronouns'][:10]:  # Show first 10
                if isinstance(pronoun, dict):
                    pron = pronoun.get('pronoun', '')
                    ref = pronoun.get('referent', '')
                    print(f"  - {pron} → {ref}")
                else:
                    print(f"  - {pronoun}")
            if len(analysis['pronouns']) > 10:
                print(f"  ... and {len(analysis['pronouns']) - 10} more")

            # Chains
            print(f"\nCoreference Chains ({analysis['chain_count']}):")
            for chain in analysis['chains'][:5]:  # Show first 5
                if isinstance(chain, dict):
                    entity = chain.get('entity', 'unknown')
                    mentions = chain.get('mentions', [])
                    print(f"  - {entity}: {len(mentions)} mentions")
                    for mention in mentions[:3]:
                        print(f"      • {mention}")
                    if len(mentions) > 3:
                        print(f"      • ... and {len(mentions) - 3} more")
                else:
                    print(f"  - {chain}")
            if len(analysis['chains']) > 5:
                print(f"  ... and {len(analysis['chains']) - 5} more")
        else:
            print(f"\n✗ Response could not be parsed as JSON")
            print(f"\nRaw response (first 500 chars):")
            print(analysis['raw_response'][:500])


def print_entity_overlap(analyses):
    """Print entity overlap analysis across all results."""
    print("\n" + "=" * 100)
    print("ENTITY OVERLAP ANALYSIS")
    print("=" * 100)

    # Collect all entities from all analyses
    all_entities = defaultdict(list)

    for i, analysis in enumerate(analyses, 1):
        label = f"{analysis['model']}/{analysis['prompt']}"
        for entity in analysis['entities']:
            entity_text = entity if isinstance(entity, str) else entity.get('text', str(entity))
            entity_normalized = entity_text.lower().strip()
            all_entities[entity_normalized].append((i, label))

    # Find common entities
    common_entities = {k: v for k, v in all_entities.items() if len(v) >= 2}

    if common_entities:
        print(f"\nEntities found in multiple results ({len(common_entities)}):")
        for entity, sources in sorted(common_entities.items(), key=lambda x: len(x[1]), reverse=True)[:20]:
            print(f"\n  '{entity}' - found in {len(sources)} results:")
            for idx, label in sources:
                print(f"    [{idx}] {label}")

    # Find unique entities
    unique_entities = {k: v for k, v in all_entities.items() if len(v) == 1}
    if unique_entities:
        print(f"\n\nUnique entities (found in only one result): {len(unique_entities)}")
        by_source = defaultdict(list)
        for entity, sources in unique_entities.items():
            idx, label = sources[0]
            by_source[label].append(entity)

        for label, entities in by_source.items():
            print(f"\n  {label} ({len(entities)} unique):")
            for entity in entities[:10]:
                print(f"    - {entity}")
            if len(entities) > 10:
                print(f"    ... and {len(entities) - 10} more")


def export_comparison_json(analyses, output_file):
    """Export comparison to JSON file."""
    comparison_data = {
        'timestamp': Path(output_file).stem,
        'files_compared': len(analyses),
        'summary': {
            'models': list(set(a['model'] for a in analyses)),
            'prompts': list(set(a['prompt'] for a in analyses)),
            'avg_entities': sum(a['entity_count'] for a in analyses) / len(analyses),
            'avg_pronouns': sum(a['pronoun_count'] for a in analyses) / len(analyses),
            'avg_chains': sum(a['chain_count'] for a in analyses) / len(analyses),
            'avg_time': sum(a.get('processing_time', a.get('elapsed_time', 0)) for a in analyses) / len(analyses),
        },
        'analyses': analyses
    }

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(comparison_data, f, indent=2, ensure_ascii=False)

    print(f"\nJSON exported to: {output_file}")


def export_comparison_markdown(analyses, output_file):
    """Export comparison to Markdown file."""
    lines = []

    # Header
    timestamp = analyses[0].get('timestamp', '')[:10] if analyses else 'N/A'
    model = analyses[0].get('model', 'N/A') if analyses else 'N/A'
    lines.append("# Coreference Analysis Comparison Report\n")
    lines.append(f"**Date:** {timestamp}")
    lines.append(f"**Model:** {model}")
    lines.append(f"**Files compared:** {len(analyses)}")
    lines.append("\n---\n")

    # Ranking summary
    ranked = sorted(analyses, key=lambda x: x.get('quality_score', 0), reverse=True)
    lines.append("## Ranking Summary\n")
    lines.append("| Rank | Prompt | Score | Entities | Chains | Density | Time |")
    lines.append("|------|--------|-------|----------|--------|---------|------|")
    for i, a in enumerate(ranked, 1):
        prompt = a['prompt'].replace('.txt', '')
        score = a.get('quality_score', 0)
        entities = a['entity_count']
        chains = a['chain_count']
        density = a.get('chain_metrics', {}).get('avg_mentions_per_chain', 0)
        time = a.get('processing_time', a.get('elapsed_time', 0))
        lines.append(f"| #{i} | {prompt} | {score:.1f} | {entities} | {chains} | {density:.1f} | {time:.1f}s |")
    lines.append("\n---\n")

    # Entity comparison across all prompts
    lines.append("## Entity Comparison Across All Prompts\n")

    # Collect all entities
    all_entities = set()
    for a in analyses:
        for e in a.get('entities', []):
            entity_text = e if isinstance(e, str) else str(e)
            all_entities.add(entity_text.lower())

    # Build header
    prompt_names = [a['prompt'].replace('.txt', '').replace('coref_', '').replace('_ita', '') for a in ranked]
    header = "| Entity | " + " | ".join(prompt_names) + " |"
    separator = "|--------" + "|---" * len(ranked) + "|"
    lines.append(header)
    lines.append(separator)

    # Build entity rows
    for entity in sorted(all_entities):
        row = f"| {entity} |"
        for a in ranked:
            entities_lower = [e.lower() if isinstance(e, str) else str(e).lower() for e in a.get('entities', [])]
            mark = " X" if entity in entities_lower else ""
            row += f" {mark} |"
        lines.append(row)

    # Total row
    total_row = "| **TOTAL** |"
    for a in ranked:
        total_row += f" **{a['entity_count']}** |"
    lines.append(total_row)

    lines.append("\n---\n")

    # Key observations
    lines.append("## Key Observations\n")
    if ranked:
        best = ranked[0]
        lines.append(f"1. **Best overall:** {best['prompt']} (score: {best.get('quality_score', 0):.1f})\n")

        fastest = min(analyses, key=lambda x: x.get('processing_time', x.get('elapsed_time', float('inf'))))
        lines.append(f"2. **Fastest:** {fastest['prompt']} ({fastest.get('processing_time', fastest.get('elapsed_time', 0)):.1f}s)\n")

        best_density = max(analyses, key=lambda x: x.get('chain_metrics', {}).get('avg_mentions_per_chain', 0))
        density = best_density.get('chain_metrics', {}).get('avg_mentions_per_chain', 0)
        lines.append(f"3. **Best chain quality:** {best_density['prompt']} (density: {density:.1f})\n")

    # Write file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))

    print(f"Markdown exported to: {output_file}")


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Compare coreference analysis results from text files',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        'files',
        nargs='*',
        help='Result JSON files to compare'
    )
    parser.add_argument(
        '--all',
        action='store_true',
        help='Compare all textfile results'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Base filename for output files (creates .json and .md)'
    )
    parser.add_argument(
        '--verbose',
        '-v',
        action='store_true',
        help='Show detailed comparison with full entity lists and chains'
    )

    args = parser.parse_args()

    print("=" * 100)
    print("COREFERENCE RESULTS COMPARISON")
    print("=" * 100)

    # Get files to compare
    if args.files:
        selected_files = [Path(f) for f in args.files]
    else:
        available_files = get_available_textfile_results()
        if not available_files:
            print("Error: No result files found in results/ directory.")
            sys.exit(1)

        if args.all:
            selected_files = available_files
            print(f"\nComparing all {len(selected_files)} results...")
        else:
            while True:
                selected_files = select_files(available_files)
                if selected_files is not None:
                    break

    # Load and analyze results
    analyses = []
    source_text = None

    for filepath in selected_files:
        # Handle both Path objects and strings
        if isinstance(filepath, str):
            filepath = Path(filepath)

        # Check if it's absolute or relative
        if not filepath.is_absolute():
            if not filepath.exists():
                # Try with RESULTS_DIR prefix
                if (RESULTS_DIR / filepath.name).exists():
                    filepath = RESULTS_DIR / filepath.name

        if not filepath.exists():
            print(f"Warning: File not found: {filepath}")
            continue

        if filepath.is_dir():
            continue  # Skip directories silently

        print(f"Loading {filepath.name}...")
        result_data = load_result_file(filepath)
        analysis = analyze_result(result_data)

        if analysis:
            analysis['filepath'] = str(filepath)
            analyses.append(analysis)

            # Load source text (only once, assumes all results are from same source)
            if source_text is None:
                source_text = load_source_text(analysis['source_file'])

    if not analyses:
        print("Error: No valid results to compare.")
        sys.exit(1)

    print(f"\nLoaded {len(analyses)} results for comparison.")

    # Calculate quality scores
    analyses = calculate_quality_scores(analyses, source_text)

    # Print comparisons
    print_summary_table(analyses)

    if args.verbose:
        print_comparison_table(analyses)
        print_detailed_comparison(analyses, source_text)
        print_entity_overlap(analyses)

    # Export results (both JSON and Markdown)
    if args.output:
        base = args.output
        export_comparison_json(analyses, base + '.json')
        export_comparison_markdown(analyses, base + '.md')

    print("\n" + "=" * 100)
    print("COMPARISON COMPLETE")
    print("=" * 100)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nComparison cancelled by user (Ctrl+C).")
        sys.exit(0)
