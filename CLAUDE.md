# Project Configuration for Claude Code

## Project Overview
A LangChain-based application that extracts knowledge from PDF files and generates Turtle files and CSV files for Neo4j

## General Guidelines
- **Language**: All code comments, commit messages, documentation, and Claude Code outputs must be in English
- **Tone**: Use a friendly, casual, and polite tone in all responses. Be conversational and approachable while maintaining professionalism

## Development Environment
- **Language**: Python
- **Framework**: LangChain
- **Database**: Neo4j
- **Container**: Docker Compose

## Key Files and Structure
```
/workspace/
├── app/                    # Main application
│   ├── src/               # Source code (stage-based architecture)
│   │   ├── extract_workflow.py     # Entry point
│   │   ├── extract/       # Extract workflow modules
│   │   │   ├── core/      # Application core (graph.py, nodes.py, state.py)
│   │   │   ├── classification/  # File classification stage
│   │   │   ├── text/      # Text extraction stage (document/presentation)
│   │   │   ├── view/      # View extraction stage
│   │   │   ├── class/     # Class extraction stage
│   │   │   ├── property/  # Property extraction stage
│   │   │   └── output/    # Output generation stage (turtle_generator.py, neo4j_csv_generator.py)
│   │   ├── evaluate/      # Evaluation workflow modules
│   │   ├── optimize/      # Optimization workflow modules
│   │   └── utils/         # Common utilities (logger.py, llm_factory.py, chunk_creator.py)
│   ├── schemas/           # Input/output schema definitions
│   │   ├── *-chain/
│   │   └── common/        # Common schemas
│   ├── prompts/           # Prompt text for tasks
│   ├── metamodel/         # Metamodel definitions
│   ├── output/            # Generated file output directory
│   ├── log/               # Log files
│   └── requirements.txt   # Python dependencies
├── neo4j/                 # Neo4j data
└── doc/                   # Documentation
```

## Schema Structure
`/workspace/app/schemas/` defines JSON schemas for the input/output of each chain and LLM. To improve reusability, schemas are defined by reference when the input/output is the same as the parent level.
- **Chains**: doc-text-chain, view-chain, class-chain, property-chain
- **Extractors**: text-extractor, view-extractor, class-extractor, property-extractor
- **Filters**: text-filter, view-filter
- **Generators**: turtle-generator, neo4j-csv-generator
- **Common**: Shared schema definitions (metamodel.schema.json)

## Prompt Templates
`/workspace/app/prompts/` contains the prompt text used by the LLM

## Common Commands

### Development
```bash
# Install dependencies
cd app && pip install -r requirements.txt

# Run main application
cd app && python src/extract_workflow.py [PDF_PATH]

# Using the wrapper script (recommended)
cd app && ./extract.sh [PDF_PATH] --project [PROJECT_NAME]

# All options
cd app && python src/extract_workflow.py [PDF_PATH] --project [PROJECT_NAME] --model [MODEL_NAME] --output-dir [OUTPUT_DIR]

# Run specific chain (for development/debugging)
cd app && python -c "import sys; sys.path.insert(0, '.'); from src.extract.text.doc_text_chain import *"
cd app && python -c "import sys; sys.path.insert(0, '.'); from src.extract.view.view_chain import *"
cd app && python -c "import sys; sys.path.insert(0, '.'); from src.extract.class.class_chain import *"
cd app && python -c "import sys; sys.path.insert(0, '.'); from src.extract.property.property_chain import *"
```

### Batch Extraction
Extract knowledge from multiple PDF files using a batch file:

```bash
# Basic batch extraction
cd app && ./extract.sh --batch ../doc/batch-sample.txt --project myproject

# With model override
cd app && ./extract.sh --batch ../doc/batch-sample.txt --project myproject --model gpt-5-nano

# With retry configuration (retry failed files up to 2 times)
cd app && ./extract.sh --batch ../doc/batch-sample.txt --project myproject --max-retries 2

# Full example with all options
cd app && ./extract.sh --batch ../doc/batch-sample.txt --project myproject --model gpt-5-nano --max-retries 2
```

**Batch file format** (plain text, one PDF path per line):
- Lines starting with `#` are comments
- Blank lines are ignored
- File paths are relative to batch file location (or use absolute paths)
- Example batch file (`/workspace/doc/batch-sample.txt`):
  ```
  # My documents
  sample.pdf
  csample1.pdf

  # You can also use absolute paths
  /workspace/doc/psample1.pdf
  ```

**Batch processing behavior**:
- Files are processed sequentially (one after another)
- Failed files can be automatically retried (configure with `--max-retries`)
- Non-existent files are skipped with a warning
- Processing continues even if some files fail
- All successfully extracted knowledge is accumulated into the project's `knowledge.json`
- Summary report shows total files, successful count, and failed count
```

### Docker Environment
```bash
# Start environment
docker-compose up -d

# Enter application container
docker exec -it app_container bash

# Access Neo4j browser
# http://localhost:7474 (neo4j/neo4jpassword)
```

## Testing and Quality
Use **gpt-5-nano** when testing the entire process to reduce costs

### Single file extraction
```bash
# Default project
python src/extract_workflow.py ../doc/sample.pdf --model gpt-5-nano

# Specific project
python src/extract_workflow.py ../doc/sample.pdf --project test-project --model gpt-5-nano
```

### Multiple files extraction
```bash
python src/extract_workflow.py ../doc/sample.pdf --project test-project --model gpt-5-nano
python src/extract_workflow.py ../doc/sample2.pdf --project test-project --model gpt-5-nano
```

## Environment Variables
- `NEO4J_AUTH=neo4j/neo4jpassword` (Docker environment)

## Output Files
Output files are managed by project, specified with the `--project` parameter.

### Directory Structure
```
/workspace/app/output/
├── default/                           # Default project
│   ├── knowledge_graph_YYYYMMDD_HHMMSS.ttl
│   ├── nodes.csv
│   ├── relationships.csv
│   ├── knowledge.json                 # Accumulated knowledge (ACCEPT classes + valid properties)
│   └── extraction_result_YYYYMMDD_HHMMSS.json
├── project1/                          # Custom project
│   ├── knowledge_graph_YYYYMMDD_HHMMSS.ttl
│   ├── nodes.csv
│   ├── relationships.csv
│   ├── knowledge.json
│   └── extraction_result_YYYYMMDD_HHMMSS.json
└── project2/
    └── ...
```

### File Types
- **Turtle files**: `knowledge_graph_*.ttl` - Knowledge graph in RDF/Turtle format
- **Neo4j CSV files**: `nodes.csv`, `relationships.csv` - CSV files for Neo4j import
- **Knowledge JSON**: `knowledge.json` - Accumulated knowledge (lightweight version, used as input for next execution)
  - Only ACCEPT-judged classes
  - Only properties with strength > 0
  - Automatically accumulated across multiple executions
- **Result JSON**: `extraction_result_YYYYMMDD_HHMMSS.json` - Complete processing results for each execution (with timestamp, for history/audit)

## Key Dependencies
- langchain: LLM chain construction
- langchain-openai: OpenAI API integration
- pdfplumber, pypdf: PDF processing
- rdflib: RDF/Turtle generation
- neo4j: Graph database

## Notes
- PDF files are placed in `/workspace/doc/`
- Output files are generated in `/workspace/app/output/{project}/`
- Log files are generated in `/workspace/app/log/`
- Neo4j data is persisted (`./neo4j/data`)
- Schema definitions clarify input/output specifications for each chain/task
- Prompt text is managed in external files for easy adjustment and maintenance

## Command Line Arguments
```bash
python src/extract_workflow.py [PDF_PATH] [OPTIONS]

Arguments:
  PDF_PATH              Target PDF file path (default: ../doc/sample.pdf)

Options:
  --project PROJECT     Project name (default: default)
  --model MODEL         LLM model name (e.g., gpt-5-mini, gpt-5-nano)
  --output-dir DIR      Output directory (default: /workspace/app/output)
```