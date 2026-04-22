"""
LangGraph State Definition for Knowledge Extraction
LangGraph State definition for knowledge extraction processing
"""

from typing import TypedDict, List, Dict, Any, Optional


class KnowledgeExtractionState(TypedDict, total=False):
    """
    State definition for the entire knowledge extraction process

    Each node receives this State and reads/updates necessary fields.
    With total=False, all fields are treated as optional,
    allowing each node to update only the fields it needs.
    """

    # ========== Input Information ==========
    files: List[Dict[str, str]]  # [{"path": "/path/to/file.pdf"}]
    file_path: str  # File path to process (value extracted from files)
    file_type: str  # "document" | "presentation"

    # ========== Metamodel ==========
    metamodel: Dict[str, Any]  # Metamodel definition (referenced by all nodes)

    # ========== LLM Model Settings ==========
    model: Optional[str]  # LLM model name (e.g., "gemini-3.1-flash-lite-preview", "gpt-5-nano")

    # ========== Output Destination Settings ==========
    output_dir: str  # Output directory
    project: str  # Project name (output files classified by project)

    # ========== Intermediate Results: Text Extraction Stage ==========
    file_id: str  # Stable file ID
    file_name: str  # File name
    sentences: List[Dict[str, Any]]  # List of extracted sentences
    # sentence: {"text": str, ...}

    # ========== Intermediate Results: Chunk Splitting Stage ==========
    chunks: List[Dict[str, str]]  # Text chunks
    # chunk: {"text": str}

    # ========== Intermediate Results: Image Extraction Stage ==========
    image_chunks: List[Dict[str, Any]]  # Image descriptions as chunks (parallel to text chunks)
    # image_chunk: {
    #   "text": str,              # Description text (drawio format or natural language)
    #   "file_id": str,
    #   "page_number": int,
    #   "provenance": {
    #     "source": "image",
    #     "image_id": str,
    #     "relevance_score": float,
    #     "has_structural_content": bool,
    #     "image_description": str,
    #     "drawio_description": str | None
    #   }
    # }
    image_statistics: Dict[str, Any]  # Image pipeline statistics

    # ========== Intermediate Results: View Extraction Stage ==========
    views: List[Dict[str, Any]]  # View description extraction results
    # view: {
    #   "type": str,  # View type
    #   "texts": [{"file_id": str, "text": str}, ...]
    # }

    # ========== Intermediate Results: Class Extraction Stage ==========
    classes: List[Dict[str, Any]]  # Class extraction results
    # class: {
    #   "id": str,         # Class instance ID
    #   "class_iri": str,  # Metamodel class IRI
    #   "label": str,      # Extracted class name
    #   "sources": List[str],   # Source sentences
    #   "file_ids": List[str]   # Source file IDs
    # }

    # ========== Intermediate Results: Property Extraction Stage ==========
    properties: List[Dict[str, Any]]  # Property (relationship) extraction results
    # property: {
    #   "id": str,           # Property ID
    #   "src_id": str,       # Source class instance ID
    #   "property_iri": str, # Property IRI
    #   "dest_id": str       # Destination class instance ID
    # }

    # ========== Final Output ==========
    turtle_file: str  # Turtle file path
    turtle_content: str  # Turtle content (usually not saved to JSON due to size)
    nodes_csv: str  # Neo4j node CSV file path
    relationships_csv: str  # Neo4j relationship CSV file path

    # ========== Meta Information ==========
    current_stage: str  # Current processing stage (for debugging)
    errors: List[str]  # Error message list

    # ========== Summary ==========
    summary: Dict[str, Any]  # Final summary information
    # {
    #   "classes_count": int,
    #   "properties_count": int,
    #   "files_generated": List[str]
    # }
