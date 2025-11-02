# legiscope

Automated analysis of municipal codes for legal epidemiology.

## Getting started

### Environment Setup

This project uses `uv` for dependency management.

```bash
# Install uv if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Set up the development environment
make env

# Activate the environment
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### Setting up a local postgres database

#### 1. Install postgres.

* MacOS: I find [Postgres.app](https://www.postgresql.org/download/macosx/) is an easy
way to get set up for local development instances on the Mac.

* Linux, etc.: [This site](https://www.postgresql.org/download/) gives instructions for
different distros, and other operating systems. You will also need to install the
`vector` extension.

#### 2. Start psql and create a user

```sh
psql -U postgres
```

then

```sql
CREATE USER muni WITH PASSWORD 'muni';
```

#### 2. Create a database and set up permissions

```sql
CREATE DATABASE muni;
```

then

```sh
psql -U postgres -d muni
```

then

```sql
CREATE EXTENSION IF NOT EXISTS vector;
GRANT CREATE ON SCHEMA public TO muni;
```

#### 3. Run the initialization script

```sh
. scripts/reset.sh localhost muni muni
```

## Development

### Running Tests

```bash
make test
```

### Code Quality

```bash
# Run linting and formatting checks
make lint

# Format code
make format

# Fix linting issues
make fix
```

## Municipal Code Analysis

This project uses [Marimo](https://marimo.io/) notebooks for interactive municipal code analysis.

### Getting Started with Marimo

Run the analysis notebook:
```bash
uv run marimo edit notebooks/template-workflow.py
```

This will open the notebook in your browser at `http://localhost:2718`

### Key Features

- **Interactive UI Elements**: Text areas for queries, sliders for result limits, buttons for actions
- **Reactive Cells**: Automatic updates when inputs change
- **Real-time Results**: Query results update as you type
- **Separate Interfaces**: Dedicated interfaces for semantic, full-text, and hybrid queries
- **Interactive Reports**: Generate and upload reports with button controls

### Usage Workflow

1. **Data Preparation**: Create a jurisdiction directory structure using the utility function:
   ```python
   from legiscope.utils import create_jurisdiction_structure
   
   # Create directory structure for a new jurisdiction
   base_path = create_jurisdiction_structure("CA", "LosAngeles")
   # Creates: data/laws/CA-LosAngeles/{raw,processed,tables}/
   ```
   Then place DOCX files containing the municipal code in the `raw/` subdirectory and run `scripts/convert_docx.sh` to convert them to a single text file.

2. **Command-Line Processing**: Use the provided script to convert jurisdictions to Markdown:
   ```bash
   # Basic usage
   python scripts/convert_to_markdown.py data/laws/IL-WindyCity
   
   # With verbose output
   python scripts/convert_to_markdown.py data/laws/CA-LosAngeles --verbose
   
   # Using Makefile
   make convert-to-markdown JURISDICTION=data/laws/IL-WindyCity
   
   # Custom options
   python scripts/convert_to_markdown.py data/laws/NY-NewYork \
     --input-file municipal_code.txt \
     --output-file code.md \
     --max-lines 200 \
     --model gpt-4o \
     --verbose
   ```
2. **Configure**: Update the `heading_examples` dictionary in the notebook with your jurisdiction's heading patterns
3. **Process**: The notebook automatically parses and processes the municipal code
4. **Query**: Use the interactive query interfaces to search the code
5. **Report**: Generate and upload reports using the interactive buttons

### Creating Jurisdiction-Specific Notebooks

To create a version for a new jurisdiction:

```bash
cp notebooks/template-workflow.py notebooks/<jurisdiction>.py
uv run marimo edit notebooks/<jurisdiction>.py
```

Then update the jurisdiction-specific parameters in the notebook.

### Structured LLM Outputs

This project includes the [Instructor](https://github.com/jxnl/instructor) library for structured outputs from language models. The `legiscope.utils` module provides the core `ask` function for getting type-safe, structured responses from LLMs, while `legiscope.convert` provides conversion-specific utilities.

#### Legal Text Analysis

The `scan_legal_text` function analyzes municipal ordinances and statutes to identify heading structure:

```python
import instructor
from openai import OpenAI
from legiscope.convert import scan_legal_text

# Setup instructor client
client = instructor.from_openai(OpenAI())

# Analyze legal text structure
structure = scan_legal_text(
    client=client,
    file_path="data/laws/IL-WindyCity/processed/code.txt",
    max_lines=150  # Optional: limit lines for analysis
)

print(f"Found {structure.total_levels} heading levels:")
for level in structure.levels:
    print(f"Level {level.level}: {level.example_heading}")
    print(f"  Regex: {level.regex_pattern}")
    print(f"  Markdown: {level.markdown_prefix}")
```

**Features:**
- **Automatic heading detection**: Identifies hierarchical structure (CHAPTER, SECTION, ARTICLE, etc.)
- **Regex pattern generation**: Creates patterns for programmatic heading detection
- **Markdown formatting**: Suggests appropriate Markdown prefixes (#, ##, ###)
- **Configurable sampling**: Limit analysis with `max_lines` parameter
- **Error handling**: Validates file access and regex patterns

**Use Cases:**
- **Document parsing**: Convert legal texts to structured formats
- **Content analysis**: Understand document organization
- **Template generation**: Create parsing templates for new jurisdictions
- **Quality control**: Validate heading consistency across documents

#### Text to Markdown Conversion

The `text2md` function converts legal text files to Markdown using heading structure analysis:

```python
import instructor
from openai import OpenAI
from legiscope.convert import scan_legal_text, text2md

# Setup instructor client
client = instructor.from_openai(OpenAI())

# Analyze heading structure
structure = scan_legal_text(client, "municipal_code.txt")

# Convert to Markdown
text2md(structure, "municipal_code.txt", "municipal_code.md")
print("Conversion completed!")
```

**Features:**
- **Automatic heading conversion**: Transforms legal headings to Markdown format
- **Hierarchical processing**: Handles multi-level heading structures
- **Pattern matching**: Uses regex patterns from HeadingStructure analysis
- **Content preservation**: Maintains non-heading text unchanged
- **YAML frontmatter**: Includes jurisdiction metadata and heading patterns
- **Error handling**: Validates inputs and handles file access issues

**Complete Workflow:**
```python
# End-to-end conversion pipeline
client = instructor.from_openai(OpenAI())

# Step 1: Analyze document structure
structure = scan_legal_text(
    client=client,
    file_path="data/laws/IL-WindyCity/processed/code.txt",
    model="gpt-4o"  # Optional: specify model
)

# Step 2: Convert to Markdown
text2md(
    structure=structure,
    input_path="data/laws/IL-WindyCity/processed/code.txt",
    output_path="data/laws/IL-WindyCity/processed/code.md",
    state="IL",
    municipality="WindyCity"
)

print(f"Converted {structure.total_levels} heading levels to Markdown")
```

**Command-Line Alternative:**
```bash
# One-step conversion using the provided script
python scripts/convert_to_markdown.py data/laws/IL-WindyCity --verbose
```

**Integration Benefits:**
- **Consistent formatting**: Standardized Markdown output across jurisdictions
- **Automated processing**: Batch conversion of multiple documents
- **Version control**: Markdown files are easier to track and diff
- **Documentation ready**: Output suitable for documentation systems
- **Web compatible**: Markdown can be rendered directly in browsers
- **Rich metadata**: YAML frontmatter provides structured data for automation
- **Audit trails**: Creation timestamps and pattern tracking for compliance

**Frontmatter Structure:**
```yaml
---
jurisdiction:
  state: "IL"
  municipality: "WindyCity"
  full_name: "IL - WindyCity"
heading_patterns:
  - level: 1
    regex_pattern: "^CHAPTER\\s+\\d+:\\s+.+$"
    markdown_prefix: "#"
    example_heading: "CHAPTER 1: GENERAL PROVISIONS"
  - level: 2
    regex_pattern: "^(SECTION|ARTICLE)\\s+[\\d.]+:\\s+.+$"
    markdown_prefix: "##"
    example_heading: "SECTION 1.1: PURPOSE"
created_at: "2025-11-02T16:20:15.256802+00:00"
---
```

**Complete Workflow:**
```python
# End-to-end conversion pipeline with frontmatter
client = instructor.from_openai(OpenAI())

# Step 1: Analyze structure
structure = scan_legal_text(
    client=client,
    file_path="data/laws/IL-WindyCity/processed/code.txt"
)

# Step 2: Convert to Markdown with frontmatter
text2md(
    structure=structure,
    input_path="data/laws/IL-WindyCity/processed/code.txt",
    output_path="data/laws/IL-WindyCity/processed/code.md",
    state="IL",
    municipality="WindyCity"
)

print(f"Converted {structure.total_levels} heading levels to Markdown with frontmatter")
```

#### Example Usage

```python
import instructor
from openai import OpenAI
from legiscope.utils import ask
from legiscope.convert import LegalAnalysis

# Setup instructor client
client = instructor.from_openai(OpenAI())

# Extract structured legal analysis
analysis = ask(
    client=client,
    prompt="Analyze this municipal code for zoning provisions...",
    response_model=LegalAnalysis,
    system="You are a helpful legal assistant.",
    model="gpt-5-mini",
    temperature=0.1
)

print(f"Found {len(analysis.provisions)} provisions")
print(f"Summary: {analysis.summary}")
```

#### Content Logging

The `ask` function supports optional content logging for debugging and audit purposes:

```bash
# Enable content logging
export LOG_ASK_CONTENT=true

# Run your Python script
python your_script.py
```

When enabled, content logging creates a separate log file `logs/ask_content.log` that contains:
- **User prompts**: Full text of user queries
- **System prompts**: Full text of system instructions  
- **Model responses**: Complete structured responses in JSON format

**Log Files:**
- `logs/ask_function.log` - Operational metadata (always created)
- `logs/ask_content.log` - Prompt/response content (only when `LOG_ASK_CONTENT=true`)

**Content Logging Configuration:**
- **Rotation**: 50 MB (larger due to content)
- **Retention**: 30 days (longer for audit trails)
- **Format**: Timestamp + message content
- **Security**: Disabled by default to protect sensitive data

**Use Cases:**
- **Development**: Debug prompt/response interactions
- **Testing**: Verify model behavior with specific inputs
- **Compliance**: Audit trails for legal/medical applications
- **Analysis**: Review conversation patterns and model performance

### Data Directory Structure

The project organizes municipal code data in a structured hierarchy:

```
data/
├── laws/                           # Municipal code data
│   └── {state}-{municipality}/     # Jurisdiction-specific directories
│       ├── raw/                    # Original source files (DOCX, PDF, etc.)
│       ├── processed/              # Processed text files and intermediate results
│       └── tables/                 # Structured data tables and exports
└── queries/                        # Database queries and search templates
```

**Example:**
```
data/
├── laws/
│   ├── IL-WindyCity/              # Chicago municipal code
│   │   ├── raw/                   # Original DOCX files
│   │   ├── processed/             # Converted text files
│   │   └── tables/                # Structured data
│   └── CA-LosAngeles/             # Los Angeles municipal code
│       ├── raw/
│       ├── processed/
│       └── tables/
└── queries/
```

### Project Structure

```
.
├── src/
│   └── legiscope/       # Main package source code
│       ├── convert.py   # Conversion utilities and response models
│       ├── utils.py     # Core utility functions (ask function, directory creation)
│       └── code.py      # Core municipal code parsing
├── tests/               # Test files
├── notebooks/           # Marimo notebooks for analysis
├── scripts/             # Utility scripts
├── data/                # Data directory (not tracked by git)
├── pyproject.toml       # Project configuration and dependencies
├── Makefile            # Development commands
└── AGENTS.md           # Detailed development documentation
```

Instructions for the bots: [AGENTS.md](AGENTS.md).
