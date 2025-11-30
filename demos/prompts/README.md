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

The script supports **flexible path resolution** - it automatically searches multiple locations:

```bash
# Method 1: Just the filename (easiest - auto-searches demos/prompts/)
python demos/temperature_test_multi_model.py coreference_resolution.txt

# Method 2: Relative path from project root
python demos/temperature_test_multi_model.py demos/prompts/coreference_resolution.txt

# Method 3: Relative path from current directory
cd demos
python temperature_test_multi_model.py prompts/coreference_resolution.txt

# Method 4: Absolute path
python demos/temperature_test_multi_model.py /full/path/to/prompt.txt

# Method 5: Interactive selection
python demos/temperature_test_multi_model.py
# Then enter any of the above formats when prompted
```

**Path Resolution Order:**
1. Absolute paths (if provided)
2. Relative to current directory
3. Relative to `demos/prompts/` (auto-search)
4. Relative to project root

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

This directory is `.gitignore`d to protect the intellectual property. Prompt files will not be committed to the repository.
