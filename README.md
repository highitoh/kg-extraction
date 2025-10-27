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
python main.py ../doc/sample.pdf
```

### With Custom Model

For development, use `gpt-5-nano` to reduce costs:

```bash
python main.py ../doc/sample.pdf --model gpt-5-nano
```

### Specify Output Directory

```bash
python main.py ../doc/sample.pdf --output-dir ./custom_output
```

## Project Structure

```
/workspace/
├── app/                         # Main application
│   ├── main.py                  # Main entry point
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

After processing, the following files are generated:

```
app/output/
├── knowledge_graph.ttl           # Knowledge graph in RDF/Turtle format
├── nodes.csv                     # Neo4j node CSV
├── relationships.csv             # Neo4j relationship CSV
└── extraction_result.json        # Processing result summary
```

## Configuration

### Metamodel

The metamodel definition ([app/metamodel/metamodel.json](app/metamodel/metamodel.json)) constrains the structure of extracted classes and properties.

### Schema Structure

Input/output for each chain is defined by JSON schemas in [app/schemas/](app/schemas/). Common schema references are used to enhance reusability.

### Prompt Templates

Prompts used by LLMs are managed as text files in [app/prompts/](app/prompts/) for easy adjustment and maintenance.

## Development

### Run Individual Chains

Test individual chains before integration:

```bash
cd app
python doc_text_chain.py
python view_chain.py
python class_chain.py
python property_chain.py
```

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

## Tips

- Use `--model gpt-5-nano` during development to reduce costs
- Place PDF files in `/workspace/doc/`
- Output files are generated in `/workspace/app/output/`
- Log files are generated in `/workspace/app/log/`
- Neo4j data is persisted in `./neo4j/data`

## License

MIT License

