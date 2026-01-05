"""
Coreference Resolution Evaluation Utilities

Provides normalized string matching, metrics calculation (F1, accuracy),
and result aggregation for pronoun resolution testing.
"""

import json
import re
import time
import requests
from pathlib import Path
from collections import defaultdict

# Import shared config
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from demo_config import API_URL

HOST = API_URL


def set_host(url):
    """Override the default API host URL."""
    global HOST
    HOST = url

# Articles to strip for normalized matching (multilingual)
ARTICLES = {
    'en': ['the', 'a', 'an'],
    'es': ['el', 'la', 'los', 'las', 'un', 'una', 'unos', 'unas'],
    'fr': ['le', 'la', 'les', 'un', 'une', 'des', "l'"],
    'de': ['der', 'die', 'das', 'ein', 'eine', 'einen', 'einem', 'einer'],
    'it': ['il', 'lo', 'la', 'i', 'gli', 'le', 'un', 'uno', 'una', "l'"],
    'ru': [],  # Russian has no articles
    'zh': [],  # Chinese has no articles
}


def normalize_text(text, lang='en', strip_articles=True):
    """
    Normalize text for comparison.

    Args:
        text: The text to normalize
        lang: Language code for article stripping
        strip_articles: Whether to remove leading articles

    Returns:
        Normalized lowercase string with stripped whitespace
    """
    if not text:
        return ""

    # Lowercase and strip whitespace
    normalized = text.lower().strip()

    # Remove extra whitespace
    normalized = re.sub(r'\s+', ' ', normalized)

    # Strip leading articles if requested
    if strip_articles:
        articles = ARTICLES.get(lang, [])
        for article in articles:
            pattern = rf'^{re.escape(article)}\s+'
            normalized = re.sub(pattern, '', normalized, flags=re.IGNORECASE)

    return normalized


def levenshtein_distance(s1, s2):
    """
    Calculate the Levenshtein (edit) distance between two strings.
    Returns the minimum number of single-character edits (insertions,
    deletions, substitutions) required to transform s1 into s2.
    """
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            # Cost is 0 if characters match, 1 otherwise
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


def match_referent(model_answer, gold_referent, lang='en'):
    """
    Check if model answer matches gold referent using normalized comparison.

    Args:
        model_answer: The model's extracted referent
        gold_referent: The gold standard referent
        lang: Language code

    Returns:
        tuple: (is_match: bool, normalized_model: str, normalized_gold: str, match_type: str)
        match_type can be: 'exact', 'partial', 'fuzzy', or 'none'
    """
    norm_model = normalize_text(model_answer, lang)
    norm_gold = normalize_text(gold_referent, lang)
    match_type = 'none'

    # Exact normalized match
    is_match = norm_model == norm_gold
    if is_match:
        match_type = 'exact'

    # Partial match: only if model answer is a meaningful substring (min 3 chars)
    # and covers a significant portion of the gold answer
    if not is_match and len(norm_model) >= 3:
        # Model contains gold (e.g., "the brown suitcase" contains "suitcase")
        if norm_gold in norm_model:
            is_match = True
            match_type = 'partial'
        # Gold contains model, but model must be substantial (>50% of gold length)
        elif norm_model in norm_gold and len(norm_model) > len(norm_gold) * 0.5:
            is_match = True
            match_type = 'partial'

    # Fuzzy match for typos using Levenshtein distance
    # e.g., "councilmmen" vs "councilmen" (model typo with 1 extra char)
    if not is_match and len(norm_model) >= 3 and len(norm_gold) >= 3:
        edit_dist = levenshtein_distance(norm_model, norm_gold)
        max_len = max(len(norm_model), len(norm_gold))
        # Allow up to 2 edits for short strings, or ~10% of length for longer ones
        max_allowed = max(2, int(max_len * 0.1))
        if edit_dist <= max_allowed:
            is_match = True
            match_type = f'fuzzy (edit_dist={edit_dist})'

    return is_match, norm_model, norm_gold, match_type


def extract_referent_from_response(response_text):
    """
    Extract the referent from the model's response.

    Tries multiple strategies:
    1. Parse as complete JSON
    2. Find JSON object with "referent" key
    3. Extract value after "referent": with regex
    4. Plain text fallback - extract noun phrase from natural language response

    Args:
        response_text: Raw response from the model

    Returns:
        str: Extracted referent or empty string if parsing fails
    """
    if not response_text:
        return ""

    # Try to find JSON in the response
    try:
        # First, try to parse the whole response as JSON
        data = json.loads(response_text.strip())
        return data.get('referent', '')
    except json.JSONDecodeError:
        pass

    # Try to find JSON object in the response
    json_match = re.search(r'\{[^{}]*"referent"[^{}]*\}', response_text, re.DOTALL)
    if json_match:
        try:
            data = json.loads(json_match.group())
            return data.get('referent', '')
        except json.JSONDecodeError:
            pass

    # Try to extract referent with regex as fallback
    referent_match = re.search(r'"referent"\s*:\s*"([^"]*)"', response_text)
    if referent_match:
        return referent_match.group(1)

    # Plain text fallback: try to extract a noun phrase from natural language
    # This handles cases where the model ignores JSON format
    text = response_text.strip()

    # Remove common prefixes that models add
    prefixes_to_remove = [
        r'^the\s+(?:pronoun\s+)?(?:refers?\s+to|is\s+referring\s+to)\s+',
        r'^it\s+refers?\s+to\s+',
        r'^the\s+answer\s+is\s+',
        r'^answer:\s*',
        r'^referent:\s*',
        r'^the\s+referent\s+is\s+',
    ]
    for pattern in prefixes_to_remove:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE)

    # Remove trailing punctuation and quotes
    text = text.strip('."\',:;!? ')

    # If the result is a reasonable length (likely a noun phrase), use it
    # Reject if too long (probably an explanation) or too short
    if 2 <= len(text) <= 50 and '\n' not in text:
        return text

    return ""


def extract_entity_type_from_response(response_text):
    """
    Extract the entity type from the model's response if provided.

    Some prompts ask for entity type (PERSON, ORG, LOC, etc.) alongside the referent.
    This is useful for NER integration.

    Args:
        response_text: Raw response from the model

    Returns:
        str: Entity type (e.g., "PERSON", "ORG") or empty string if not found
    """
    if not response_text:
        return ""

    # Try to parse as JSON and extract type
    try:
        data = json.loads(response_text.strip())
        return data.get('type', '')
    except json.JSONDecodeError:
        pass

    # Try to find JSON object with type
    json_match = re.search(r'\{[^{}]*"type"[^{}]*\}', response_text, re.DOTALL)
    if json_match:
        try:
            data = json.loads(json_match.group())
            return data.get('type', '')
        except json.JSONDecodeError:
            pass

    # Try regex extraction
    type_match = re.search(r'"type"\s*:\s*"([^"]*)"', response_text)
    if type_match:
        return type_match.group(1)

    return ""


def load_prompt_template(template_path=None):
    """
    Load the prompt template for pronoun resolution.

    Args:
        template_path: Path to template file (optional)

    Returns:
        str: The prompt template with {text} and {pronoun} placeholders
    """
    if template_path is None:
        template_path = Path(__file__).parent / 'prompts' / 'pronoun_resolution.txt'

    with open(template_path, 'r', encoding='utf-8') as f:
        return f.read()


def load_test_dataset(dataset_path=None):
    """
    Load the test dataset from JSON file.

    Args:
        dataset_path: Path to dataset file (optional)

    Returns:
        list: List of test case dictionaries
    """
    if dataset_path is None:
        dataset_path = Path(__file__).parent / 'datasets' / 'coreference_testset.json'

    with open(dataset_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def calculate_adaptive_timeout(text_length, base_timeout=60, chars_per_second=5, max_timeout=600):
    """
    Calculate adaptive timeout based on text length.

    Args:
        text_length: Number of characters in the text
        base_timeout: Minimum timeout in seconds (default: 60)
        chars_per_second: Estimated processing rate (default: 5 chars/sec for small models)
        max_timeout: Maximum timeout cap (default: 600 = 10 minutes)

    Returns:
        int: Timeout in seconds
    """
    # Estimate: base time + time proportional to text length
    estimated_time = base_timeout + (text_length / chars_per_second)
    return min(int(estimated_time), max_timeout)


def send_to_model(prompt, model, temperature=None, timeout=None, keep_alive="30m"):
    """
    Send a prompt to the model and get the response.

    Args:
        prompt: The formatted prompt to send
        model: Model name to use
        temperature: Optional temperature setting
        timeout: Optional timeout in seconds (if None, uses adaptive timeout)
        keep_alive: How long to keep model loaded (default: "30m")

    Returns:
        tuple: (response_text, elapsed_time, metrics_dict)
            - elapsed_time: Wall-clock time (includes queue wait)
            - metrics_dict: Contains both raw Ollama metrics and calculated timing:
                - ollama_duration_s: Ollama's total_duration in seconds (actual processing)
                - queue_wait_s: Estimated queue/network wait time
                - tokens_per_second, eval_count, etc.
    """
    payload = {
        "message": prompt,
        "model": model,
        "mcp_server": "",
        "stateless": True,
        "keep_alive": keep_alive
    }

    if temperature is not None:
        payload["temperature"] = temperature

    # Calculate timeout: use provided value, or calculate adaptively
    if timeout is None:
        timeout = calculate_adaptive_timeout(len(prompt))

    start_time = time.time()
    try:
        response = requests.post(f"{HOST}/chat", json=payload, timeout=timeout)
        elapsed_time = time.time() - start_time

        if response.status_code != 200:
            return None, elapsed_time, {}

        result = response.json()
        metrics = result.get('metrics', {})

        # Calculate Ollama's internal duration (convert from nanoseconds to seconds)
        # total_duration is the actual processing time, not including queue wait
        ollama_total_ns = metrics.get('total_duration', 0)
        ollama_duration_s = ollama_total_ns / 1_000_000_000 if ollama_total_ns else 0

        # Calculate queue/network wait time (wall-clock minus actual processing)
        queue_wait_s = max(0, elapsed_time - ollama_duration_s) if ollama_duration_s else 0

        # Add calculated timing to metrics
        metrics['ollama_duration_s'] = round(ollama_duration_s, 3)
        metrics['queue_wait_s'] = round(queue_wait_s, 3)

        return result.get('response', ''), elapsed_time, metrics

    except Exception as e:
        elapsed_time = time.time() - start_time
        print(f"Error calling model: {e}")
        return None, elapsed_time, {}


def check_wrapper_running():
    """
    Check if the wrapper API is accessible and compatible.

    Returns:
        tuple: (is_running: bool, error_message: str or None)
    """
    try:
        # First check if server is reachable
        response = requests.get(f"{HOST}/", timeout=5)
        if response.status_code != 200:
            return False, f"Server returned status {response.status_code}"

        # Check if it has the expected API structure (try to get models)
        models_response = requests.get(f"{HOST}/model/list", timeout=5)
        if models_response.status_code == 200:
            data = models_response.json()
            if 'models' in data:
                return True, None
            else:
                return False, "API responded but 'models' key not found - is this the ollama-wrapper?"
        elif models_response.status_code == 404:
            # Maybe it's raw Ollama API? Try their endpoint
            ollama_response = requests.get(f"{HOST}/api/tags", timeout=5)
            if ollama_response.status_code == 200:
                return False, f"This appears to be raw Ollama API at {HOST}. This script requires the ollama-wrapper. Start it with: uv run python ollama_wrapper.py api"
            return False, f"API endpoint /model/list not found at {HOST}"
        else:
            return False, f"Failed to get models: status {models_response.status_code}"

    except requests.exceptions.ConnectionError:
        return False, f"Cannot connect to {HOST}. Is the wrapper running? Start with: uv run python ollama_wrapper.py api"
    except requests.exceptions.Timeout:
        return False, f"Connection to {HOST} timed out"
    except Exception as e:
        return False, f"Error checking API: {e}"


def get_available_models():
    """Fetch available models from the API."""
    try:
        response = requests.get(f"{HOST}/model/list", timeout=5)
        if response.status_code == 200:
            data = response.json()
            return [m['name'] for m in data['models']]
        return []
    except:
        return []


def get_ollama_config():
    """Fetch Ollama instance configuration from the wrapper API.

    Returns:
        dict: Ollama configuration with 'host' key, or empty dict if unavailable
    """
    try:
        response = requests.get(f"{HOST}/ollama/config", timeout=5)
        if response.status_code == 200:
            return response.json()
        return {}
    except:
        return {}


def calculate_metrics(results):
    """
    Calculate evaluation metrics from test results.

    Args:
        results: List of result dictionaries with 'correct' boolean

    Returns:
        dict: Metrics including accuracy, per-language breakdown, timing stats
    """
    if not results:
        return {}

    total = len(results)
    correct = sum(1 for r in results if r.get('correct', False))

    # Overall accuracy
    accuracy = correct / total if total > 0 else 0

    # Per-language breakdown
    by_language = defaultdict(lambda: {'total': 0, 'correct': 0})
    for r in results:
        lang = r.get('lang', 'unknown')
        by_language[lang]['total'] += 1
        if r.get('correct', False):
            by_language[lang]['correct'] += 1

    language_accuracy = {}
    for lang, counts in by_language.items():
        lang_acc = counts['correct'] / counts['total'] if counts['total'] > 0 else 0
        language_accuracy[lang] = {
            'accuracy': round(lang_acc * 100, 2),
            'correct': counts['correct'],
            'total': counts['total']
        }

    # Wall-clock timing statistics (includes queue wait)
    wall_times = [r.get('elapsed_time', 0) for r in results if r.get('elapsed_time')]
    wall_avg = sum(wall_times) / len(wall_times) if wall_times else 0
    wall_min = min(wall_times) if wall_times else 0
    wall_max = max(wall_times) if wall_times else 0

    # Ollama internal timing statistics (actual processing, no queue wait)
    ollama_times = [r.get('metrics', {}).get('ollama_duration_s', 0) for r in results]
    ollama_times = [t for t in ollama_times if t > 0]  # Filter zeros
    ollama_avg = sum(ollama_times) / len(ollama_times) if ollama_times else 0
    ollama_min = min(ollama_times) if ollama_times else 0
    ollama_max = max(ollama_times) if ollama_times else 0

    # Queue wait statistics
    queue_times = [r.get('metrics', {}).get('queue_wait_s', 0) for r in results]
    queue_times = [t for t in queue_times if t >= 0]
    queue_total = sum(queue_times) if queue_times else 0
    queue_avg = queue_total / len(queue_times) if queue_times else 0

    return {
        'overall': {
            'accuracy': round(accuracy * 100, 2),
            'correct': correct,
            'total': total
        },
        'by_language': language_accuracy,
        'timing': {
            # Wall-clock timing (what you actually waited)
            'wall_clock': {
                'average_s': round(wall_avg, 3),
                'min_s': round(wall_min, 3),
                'max_s': round(wall_max, 3),
                'total_s': round(sum(wall_times), 3)
            },
            # Ollama internal timing (actual GPU processing - use for benchmarks)
            'ollama': {
                'average_s': round(ollama_avg, 3),
                'min_s': round(ollama_min, 3),
                'max_s': round(ollama_max, 3),
                'total_s': round(sum(ollama_times), 3)
            },
            # Queue/network wait time
            'queue_wait': {
                'total_s': round(queue_total, 3),
                'average_s': round(queue_avg, 3)
            },
            # Legacy fields for backwards compatibility
            'average_s': round(ollama_avg, 3),  # Now uses Ollama timing
            'min_s': round(ollama_min, 3),
            'max_s': round(ollama_max, 3),
            'total_s': round(sum(ollama_times), 3)
        }
    }


def format_duration(seconds):
    """Format duration in human-readable format."""
    if seconds < 1:
        return f"{seconds*1000:.0f}ms"
    elif seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes}m {secs}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours}h {minutes}m {secs}s"


def export_results_to_json(results_data, filename):
    """Export results to JSON file."""
    filepath = Path(filename)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(results_data, f, indent=2, ensure_ascii=False)


def export_results_to_markdown(results_data, filename):
    """Export results to Markdown file."""
    filepath = Path(filename)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    with open(filepath, 'w', encoding='utf-8') as f:
        f.write("# Coreference Resolution Test Results\n\n")

        # Metadata
        meta = results_data.get('metadata', {})
        f.write("## Test Information\n\n")
        f.write(f"- **Model**: {meta.get('model', 'N/A')}\n")
        f.write(f"- **Prompt**: {meta.get('prompt', 'N/A')}\n")
        f.write(f"- **Temperature**: {meta.get('temperature', 'N/A')}\n")
        f.write(f"- **Timestamp**: {meta.get('timestamp', 'N/A')}\n")
        f.write(f"- **Total Duration**: {meta.get('total_duration_readable', 'N/A')}\n\n")

        # Overall metrics
        metrics = results_data.get('metrics', {})
        overall = metrics.get('overall', {})
        f.write("## Overall Results\n\n")
        f.write(f"- **Accuracy**: {overall.get('accuracy', 0)}%\n")
        f.write(f"- **Correct**: {overall.get('correct', 0)} / {overall.get('total', 0)}\n\n")

        # Per-language breakdown
        by_lang = metrics.get('by_language', {})
        if by_lang:
            f.write("## Results by Language\n\n")
            f.write("| Language | Accuracy | Correct | Total |\n")
            f.write("|----------|----------|---------|-------|\n")
            for lang, stats in sorted(by_lang.items()):
                f.write(f"| {lang} | {stats['accuracy']}% | {stats['correct']} | {stats['total']} |\n")
            f.write("\n")

        # Timing
        timing = metrics.get('timing', {})
        f.write("## Timing Statistics\n\n")

        # Ollama timing (actual processing - use for benchmarks)
        ollama_timing = timing.get('ollama', {})
        if ollama_timing:
            f.write("### Ollama Processing Time (GPU)\n\n")
            f.write("*Actual model inference time - use these for benchmarks*\n\n")
            f.write(f"- **Average**: {ollama_timing.get('average_s', 0)}s\n")
            f.write(f"- **Min**: {ollama_timing.get('min_s', 0)}s\n")
            f.write(f"- **Max**: {ollama_timing.get('max_s', 0)}s\n")
            f.write(f"- **Total**: {ollama_timing.get('total_s', 0)}s\n\n")

        # Wall-clock timing
        wall_timing = timing.get('wall_clock', {})
        if wall_timing:
            f.write("### Wall-Clock Time\n\n")
            f.write("*Total wait time including queue delays*\n\n")
            f.write(f"- **Average**: {wall_timing.get('average_s', 0)}s\n")
            f.write(f"- **Min**: {wall_timing.get('min_s', 0)}s\n")
            f.write(f"- **Max**: {wall_timing.get('max_s', 0)}s\n")
            f.write(f"- **Total**: {wall_timing.get('total_s', 0)}s\n\n")

        # Queue wait time
        queue_timing = timing.get('queue_wait', {})
        if queue_timing and queue_timing.get('total_s', 0) > 0:
            f.write("### Queue/Network Wait\n\n")
            f.write(f"- **Total**: {queue_timing.get('total_s', 0)}s\n")
            f.write(f"- **Average**: {queue_timing.get('average_s', 0)}s\n\n")
        elif not ollama_timing:
            # Fallback for old format
            f.write(f"- **Average**: {timing.get('average_s', 0)}s\n")
            f.write(f"- **Min**: {timing.get('min_s', 0)}s\n")
            f.write(f"- **Max**: {timing.get('max_s', 0)}s\n")
            f.write(f"- **Total**: {timing.get('total_s', 0)}s\n\n")

        # Detailed results
        f.write("## Detailed Results\n\n")
        f.write("| # | Doc Key | Lang | Correct | Model Answer | Gold Answer | Ollama Time | Queue |\n")
        f.write("|---|---------|------|---------|--------------|-------------|-------------|-------|\n")

        for i, r in enumerate(results_data.get('results', []), 1):
            correct_mark = "Y" if r.get('correct') else "N"
            model_ans = r.get('model_answer') or ''
            model_ans = model_ans[:30] if model_ans else 'N/A'
            gold_ans = (r.get('gold_referent') or '')[:30]
            # Use Ollama timing (actual processing) as primary
            metrics = r.get('metrics', {})
            ollama_time = metrics.get('ollama_duration_s', r.get('elapsed_time', 0))
            queue_time = metrics.get('queue_wait_s', 0)
            time_s = f"{ollama_time:.2f}s"
            queue_s = f"{queue_time:.2f}s" if queue_time > 0.1 else "-"
            f.write(f"| {i} | {r.get('doc_key', '')} | {r.get('lang', '')} | {correct_mark} | {model_ans} | {gold_ans} | {time_s} | {queue_s} |\n")
