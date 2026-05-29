# PDF Knowledge Graph Extractor

A LangChain-based application that automatically extracts knowledge from PDF files and generates knowledge graphs in RDF/Turtle format and Neo4j CSV files.

## Features

- **PDF Analysis**: Automatic classification of document and presentation formats
- **Knowledge Extraction**: Automated class and property extraction using LLMs
- **Multi-format Output**: Simultaneous generation of RDF/Turtle and Neo4j CSV
- **Metamodel-driven**: Structured extraction management using JSON schemas
- **Pipeline Processing**: Consistent workflow from text extraction to graph generation

## Architecture

```
PDF Input → File Type Classification → Text Extraction → View Extraction
  → Class Extraction → Property Extraction → Output Generation
                                              ├─ Turtle (.ttl)
                                              └─ Neo4j CSV
```

## Prerequisites

- Python 3.8+
- Docker & Docker Compose (optional)
- OpenAI API Key

## Installation

### 1. Clone the repository

```bash
git clone <repository-url>
cd workspace
```

### 2. Install dependencies

```bash
cd app
pip install -r requirements.txt
```

### 3. Set up environment variables

```bash
export OPENAI_API_KEY=your_api_key_here
```

If you use GCP Vertex AI (`LLM_PROVIDER=gcp_vertexai`), authenticate on the host machine with `gcloud` before starting the app:

```bash
gcloud auth application-default login
```

The container relies on the host's Application Default Credentials, so this host-side `gcloud` authentication step is required for GCP usage.

### 4. Start Neo4j (optional)

```bash
docker-compose up -d neo4j
```

Neo4j Browser: http://localhost:7474
Credentials: neo4j / neo4jpassword (described in docker-compose.yml)

## Quick Start

### Basic Usage

```bash
cd app
./extract.sh ../doc/sample.pdf
```

### With Project Organization

Organize output files by project:

```bash
./extract.sh ../doc/sample.pdf --project my-project
```

### With Custom Model

For development, use `gpt-5-nano` to reduce costs:

```bash
./extract.sh ../doc/sample.pdf --project test --model gpt-5-nano
```

### All Options

```bash
./extract.sh ../doc/sample.pdf \
  --project healthcare \
  --model gpt-5-nano \
  --output-dir ./custom_output
```

### Miro Board Extraction

```bash
cd app
./extract.sh "https://miro.com/app/board/BOARD_ID" ../doc/sample.pdf --project my-miro-project
```

## Project Structure

```
/workspace/
├── app/                         # Main application
│   ├── src/                     # Source code (stage-based organization)
│   │   ├── extract_workflow.py  # Entry point
│   │   ├── core/                # Application core (graph.py, nodes.py, state.py)
│   │   ├── utils/               # Shared utilities (logger.py, llm_factory.py, chunk_creator.py)
│   │   ├── classification/      # Stage 0: File type classification
│   │   ├── text/                # Stage 1: Text extraction (document/presentation)
│   │   ├── view/                # Stage 2: View extraction
│   │   ├── class/               # Stage 3: Class extraction
│   │   ├── property/            # Stage 4: Property extraction
│   │   └── output/              # Stage 5: Output generation (Turtle/Neo4j CSV)
│   ├── schemas/                 # JSON schema definitions
│   ├── prompts/                 # Prompt templates
│   ├── metamodel/               # Metamodel definitions
│   ├── output/                  # Generated file output
│   ├── log/                     # Log files
│   └── requirements.txt         # Python dependencies
├── neo4j/                       # Neo4j data (persistent)
│   ├── data/
│   ├── import/
│   ├── logs/
│   └── plugins/
├── doc/                         # Documentation
└── docker-compose.yml
```

## Output Files

After processing, files are organized by project in the output directory:

```
app/output/
├── default/                            # Default project
│   ├── knowledge_graph_YYYYMMDD_HHMMSS.ttl  # Knowledge graph in RDF/Turtle format
│   ├── nodes.csv                       # Neo4j node CSV
│   ├── relationships.csv               # Neo4j relationship CSV
│   ├── knowledge.json                  # Accumulated knowledge (lightweight)
│   └── extraction_result_YYYYMMDD_HHMMSS.json  # Full extraction result (timestamped)
├── project1/                           # Custom project
│   ├── knowledge_graph_YYYYMMDD_HHMMSS.ttl
│   ├── nodes.csv
│   ├── relationships.csv
│   ├── knowledge.json
│   └── extraction_result_YYYYMMDD_HHMMSS.json
└── project2/
    └── ...
```

### File Types

- **Turtle files** (`knowledge_graph_*.ttl`): Knowledge graph in RDF/Turtle format with timestamp
- **Neo4j CSV files** (`nodes.csv`, `relationships.csv`): CSV files for Neo4j bulk import
- **Knowledge JSON** (`knowledge.json`): Accumulated knowledge for incremental extraction
  - Contains only ACCEPT'ed classes and valid properties (strength > 0)
  - Automatically merged across multiple runs
  - Lightweight format for efficient loading
- **Result JSON** (`extraction_result_YYYYMMDD_HHMMSS.json`): Full processing result with timestamp for audit trail

## Snapshot Management

The system automatically creates snapshots of the knowledge graph after each extraction, enabling historical review and rollback capabilities.

### Snapshot Structure

```
app/output/{project}/snapshots/
├── metadata.json              # Snapshot index
├── 20260425120000/           # Snapshot (timestamp)
│   ├── snapshot.json         # Metadata
│   └── knowledge.json        # Full KG snapshot
└── 20260425130000/
    └── ...
```

### Usage

**List snapshots**:
```bash
cd app && ./snapshot.sh list --project myproject
```

**View latest snapshot**:
```bash
cd app && ./snapshot.sh latest --project myproject
```

**Rollback to a snapshot**:
```bash
cd app && ./snapshot.sh rollback --project myproject --snapshot 20260425120000 --backup
```

**Disable snapshots** (when running extraction):
```bash
cd app && ./extract.sh ../doc/sample.pdf --no-snapshot
```

## Configuration

### Metamodel

The metamodel definition ([app/metamodel/metamodel.json](app/metamodel/metamodel.json)) constrains the structure of extracted classes and properties.

### Schema Structure

Input/output for each chain is defined by JSON schemas in [app/schemas/](app/schemas/). Common schema references are used to enhance reusability.

### Prompt Templates

Prompts used by LLMs are managed as text files in [app/prompts/](app/prompts/) for easy adjustment and maintenance.

### MCP Server Configuration

This project uses MCP servers.

1. MCP server configuration is defined in `.mcp.json`
2. Create `.claude/settings.local.json` to enable the MCP server:

```json
{
  "enableAllProjectMcpServers": true
}
```

## Development


### Docker Environment

```bash
# Start entire environment
docker-compose up -d

# Enter application container
docker exec -it app_container bash

# Start Neo4j only
docker-compose up -d neo4j
```

## Technologies

- **LangChain**: LLM chain construction framework
- **OpenAI API**: Knowledge extraction using GPT-4/5 models
- **pdfplumber / pypdf**: PDF parsing and text extraction
- **rdflib**: RDF/Turtle generation
- **Neo4j**: Graph database (optional)

## Command Line Arguments

```bash
./extract.sh [PDF_PATH] [OPTIONS]
```

### Arguments

- `PDF_PATH` - Path to the PDF file to extract knowledge from (default: `../doc/sample.pdf`)

### Options

- `--project PROJECT` - Project name for organizing output files (default: `default`)
- `--model MODEL` - LLM model name (e.g., `gemini-3.1-flash-lite`, `gpt-5-nano`)
- `--output-dir DIR` - Output directory path (default: `/workspace/app/output`)

### Examples

```bash
# Use default settings
./extract.sh

# Specify project name
./extract.sh ../doc/sample.pdf --project test

# Development testing with cost-effective model
./extract.sh ../doc/sample.pdf --project test --model gpt-5-nano

# All options
./extract.sh ../doc/sample.pdf --project test --model gpt-5-nano --output-dir /custom/output
```

## Tips

- Use `--model gpt-5-nano` during development to reduce costs
- Use `--project` to organize outputs for different analysis projects
- Place PDF files in `/workspace/doc/`
- Output files are generated in `/workspace/app/output/{project}/`
- Log files are generated in `/workspace/app/log/`
- Neo4j data is persisted in `./neo4j/data`

## License

MIT License
