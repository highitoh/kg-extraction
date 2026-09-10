# Project Configuration for Claude Code

## Project Overview
A LangChain-based system that extracts knowledge from documents (PDF, DOCX, PPTX, TXT, Markdown) and sources (Miro, Confluence, GitHub) and generates Turtle files and CSV files for Neo4j.

The workspace is a polyglot monorepo, split by deployment target:
- **`app/`** — Core extraction engine and CLI (Python, uv/pyproject). This is where most day-to-day work happens.
- **`lambda/`** — AWS Lambda handlers for the serverless API (Python, boto3). Self-contained; does not import from `app/`.
- **`infra/`** — AWS CDK infrastructure-as-code (TypeScript, npm).
- **`tool/`** — Operational shell scripts (e.g. Neo4j bulk import).
- **`neo4j/`** — Neo4j data (persisted).
- **`doc/`** — Input documents and documentation.

## General Guidelines
- **Language**: All code comments, commit messages, documentation, and Claude Code outputs must be in English
- **Tone**: Use a friendly, casual, and polite tone in all responses. Be conversational and approachable while maintaining professionalism
- **Code Comments**: Do not include "NEW" or time-conscious descriptions (e.g., "recently added", "old version") in code comments. Comments should describe what the code does, not when it was added or modified
- **Code Quality**: After modifying Python files, ALWAYS run the following commands to ensure code quality:
  - **IMPORTANT**: Change directory to `/workspace/app` first before running ruff commands and run each ruff command separately (one by one), NOT combined with `&&`
  ```bash
  cd /workspace/app
  uv run ruff check
  uv run ruff format
  ```
  - **Lambda code** (`/workspace/lambda/`) has its own [`ruff.toml`](lambda/ruff.toml), matched to `app/pyproject.toml`. When editing files under `lambda/`, run ruff from `/workspace/lambda` so the correct config applies:
  ```bash
  cd /workspace/lambda
  uv run ruff check
  uv run ruff format
  ```
  - **Infra code** (`/workspace/infra/`) is TypeScript — use `npm run build` (from `infra/`) to type-check; it is not covered by ruff.
- **Git Commits**: **CRITICAL RULE** - NEVER commit code to the git repository without explicit user permission. ALWAYS ask the user "Should I commit these changes?" and wait for explicit approval before running `git commit`. This applies to ALL commits, no exceptions

## Development Environment
- **Language**: Python 3.14
  - **IMPORTANT**: Only use Python features available in Python 3.14 or earlier. Do NOT use features introduced in Python 3.15 or later versions
- **Framework**: LangChain
- **Database**: Neo4j
- **Container**: Docker Compose

## Key Files and Structure
```
/workspace/
├── app/                    # Core extraction engine & CLI (Python, uv)
│   ├── src/               # Source code (stage-based architecture)
│   ├── prompts/           # Prompt text for tasks
│   ├── metamodel/         # Metamodel definitions
│   ├── schemas/           # JSON schemas for chain/LLM I/O
│   ├── output/            # Generated file output directory
│   ├── log/               # Log files
│   ├── *.sh               # Wrapper scripts (extract, snapshot, changeset, ...)
│   └── pyproject.toml     # Python dependencies (uv-managed)
├── lambda/                 # AWS Lambda handlers (Python, boto3)
│   ├── job_management/    # Upload URL, submit, status, download results
│   ├── knowledge_api/     # Knowledge class/property read endpoints + normalize
│   ├── project_management/# Projects, snapshots, changeset generation
│   ├── mcp_server/        # MCP server handler
│   └── ruff.toml          # Lint config (py314 runtime)
├── infra/                  # AWS CDK infrastructure (TypeScript, npm)
│   ├── bin/               # CDK app entrypoint
│   ├── lib/stacks/        # api / compute / events / storage stacks
│   ├── docker/            # Dockerfile for batch jobs
│   ├── docs/              # Architecture, API, DB schema, operations docs
│   └── package.json       # npm dependencies
├── tool/                   # Operational scripts (e.g. neo4j_bulk_import.sh)
├── neo4j/                  # Neo4j data (persisted)
└── doc/                    # Input documents & documentation
```

**Deployment architecture** (see [`infra/README.md`](infra/README.md)): API Gateway + Lambda for the REST API and MCP server, AWS Batch (Fargate Spot) for extraction workflows, S3 for file storage, DynamoDB for job/KG/session tracking, EventBridge for scheduled jobs, ECR for batch images.

## Schema Structure
`/workspace/app/schemas/` defines JSON schemas for the input/output of each chain and LLM.

## Prompt Templates
`/workspace/app/prompts/` contains the prompt text used by the LLM

## Common Commands

### Development
```bash
# Install dependencies
cd app && uv sync

# Run main application (PDF, DOCX, PPTX, TXT, or Markdown extraction)
cd app && uv run python src/extract_workflow.py [FILE_PATH]

# Using the wrapper script (recommended)
cd app && ./extract.sh [FILE_PATH] --project [PROJECT_NAME]

# All options
cd app && uv run python src/extract_workflow.py [FILE_PATH] --project [PROJECT_NAME] --model [MODEL_NAME] --output-dir [OUTPUT_DIR] --timestamp [YYYYMMDD]

# Miro board extraction
cd app && ./extract.sh "https://miro.com/app/board/BOARD_ID" --project [PROJECT_NAME]

# Confluence space extraction
cd app && ./extract.sh "https://company.atlassian.net/wiki/spaces/DOCS" --project [PROJECT_NAME]
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
cd app && uv run python src/extract_workflow.py ../doc/sample.pdf --model gpt-5-nano

# Specific project
cd app && uv run python src/extract_workflow.py ../doc/sample.pdf --project test-project --model gpt-5-nano
```

### Multiple files extraction
```bash
cd app && uv run python src/extract_workflow.py ../doc/sample.pdf --project test-project --model gpt-5-nano
cd app && uv run python src/extract_workflow.py ../doc/sample2.pdf --project test-project --model gpt-5-nano
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
- **Result JSON**: `extraction_result_YYYYMMDD_HHMMSS.json` - Complete processing results for each execution (with timestamp, for history/audit)

## Snapshot Management

The system automatically creates snapshots of the knowledge graph after each extraction, enabling historical review and rollback.

### Snapshot Directory Structure
```
/workspace/app/output/{project}/snapshots/
├── metadata.json              # Snapshot index
├── 20260425120000/           # Snapshot (timestamp-based)
│   ├── snapshot.json         # Metadata
│   └── knowledge.json        # Full KG snapshot
└── 20260425130000/
    └── ...
```

### Automatic Snapshot Creation

Snapshots are created automatically after each successful extraction:
```bash
cd app && ./extract.sh ../doc/sample.pdf --project myproject
# Snapshot created automatically
```

Disable snapshots with `--no-snapshot` flag:
```bash
cd app && ./extract.sh ../doc/sample.pdf --no-snapshot
```

### Snapshot Management Commands

```bash
# List all snapshots
./snapshot.sh list --project myproject

# Show specific snapshot
./snapshot.sh show --project myproject --snapshot 20260425120000

# Show latest snapshot
./snapshot.sh latest --project myproject

# Rollback to a snapshot (creates backup first)
./snapshot.sh rollback --project myproject --snapshot 20260425120000 --backup
```

## Changeset Management

Track changes between knowledge graph snapshots with automatic business driver extraction.

### Changeset Directory Structure
```
/workspace/app/output/{project}/changesets/
├── metadata.json                    # Changeset index
├── cs_20260501_120000/             # Changeset
│   └── changeset.json              # Instance graph (changes, drivers, links)
└── cs_20260501_130000/
    └── ...
```

### Changeset Management Commands

```bash
# List all changesets (one per extraction)
./changeset.sh list --project myproject

# Show specific changeset details
./changeset.sh show --project myproject --changeset cs_20260501_120000

# Show the changes recorded between two snapshots (aggregated, no LLM)
./changeset.sh range --project myproject --before 20260501110000 --after 20260501120000

# Validate against the JSON schema (schemas/changeset-generator/output.schema.json)
./changeset.sh validate --project myproject --changeset cs_20260501_120000
```

## Key Dependencies
- langchain: LLM chain construction
- langchain-openai: OpenAI API integration
- pdfplumber, pypdf: PDF processing
- rdflib: RDF/Turtle generation
- neo4j: Graph database
- requests: HTTP client for Miro API

## Notes
- PDF files are placed in `/workspace/doc/`
- Output files are generated in `/workspace/app/output/{project}/`
- Log files are generated in `/workspace/app/log/`
- Neo4j data is persisted (`./neo4j/data`)
- Schema definitions clarify input/output specifications for each chain/task
- Prompt text is managed in external files for easy adjustment and maintenance
