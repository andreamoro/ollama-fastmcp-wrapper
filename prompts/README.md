# Prompts Directory

This directory contains custom prompts for temperature testing and model evaluation.

## File Format

Prompt files are plain text files (`.txt` or `.md`) containing the complete prompt you want to test.

**Simple prompts:**
```
Explain what a binary search algorithm does in one sentence.
```

**Complex prompts with formatting:**
```
You are an NLP system. Perform strict coreference resolution on the following Italian text.

REQUIREMENTS (follow ALL of them):
1. Identify every entity...
2. For each entity...
...

TEXT:
	"Your text here..."
```

## Usage

### With `temperature_test_multi_model.py`:

```bash
# Load prompt from file
python demos/temperature_test_multi_model.py prompts/coreference_resolution.txt

# Or select interactively when prompted
python demos/temperature_test_multi_model.py
# Then enter: prompts/coreference_resolution.txt
```

### Prompt File Examples

- `coreference_resolution.txt` - NLP coreference resolution task
- `code_generation.txt` - Programming task
- `creative_writing.txt` - Creative writing task
- `translation.txt` - Translation task
- `reasoning.txt` - Logical reasoning task

## Tips

1. **One prompt per file** - Each file should contain a complete, self-contained prompt
2. **Include context** - Add all necessary instructions and examples in the prompt
3. **Preserve formatting** - Indentation and line breaks are preserved
4. **Test variations** - Create multiple versions to test different phrasings
5. **Name descriptively** - Use clear filenames that describe the task

## Privacy

This directory is `.gitignore`d to protect your intellectual property. Prompt files will not be committed to the repository.
