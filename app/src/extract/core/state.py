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
    file_type: str  # "document" | "presentation" | "miro_board" | "github_repo"

    # ========== Metamodel ==========
    metamodel: Dict[str, Any]  # Metamodel definition (referenced by all nodes)

    # ========== LLM Model Settings ==========
    model: Optional[str]  # LLM model name (e.g., "gemini-3.1-flash-lite", "gpt-5-nano")

    # ========== Output Destination Settings ==========
    output_dir: str  # Output directory
    project: str  # Project name (output files classified by project)

    # ========== Intermediate Results: Text Extraction Stage ==========
    file_id: str  # Stable file ID
    file_name: str  # File name
    sentences: List[Dict[str, Any]]  # List of extracted sentences
    # sentence: {"text": str, ...}
    miro_raw_data: Optional[Dict[str, Any]]  # Raw Miro board data (for Miro sources)
    github_raw_data: Optional[Dict[str, Any]]  # Raw GitHub repository data (for GitHub sources)

    # ========== Miro PDF-Specific Fields ==========
    miro_board_url: Optional[str]  # Miro board URL (for miro_board file type with PDF)
    miro_board_id: Optional[str]  # Extracted board ID (e.g., "o9J_abc123")
    miro_visual_items: Optional[List[Dict[str, Any]]]  # Frame + Diagram items from API
    # Visual item structure (Frame or Diagram): {
    #   "item_id": str,
    #   "item_type": str,  # "frame" or "diagram"
    #   "child_items": List[Dict],  # Items with parent=item_id
    #   "item_text": str,  # Concatenated text from all child items
    #   "child_item_types": Dict[str, int]  # {"diagram": 2, "doc": 1, "sticky_note": 5, ...}
    # }
    # Note: item_title is NOT included (Frame/Diagram titles not visible in PDF)
    item_page_mappings: Optional[List[Dict[str, Any]]]  # PDF page ↔ Frame/Diagram matches
    # Mapping structure: {
    #   "page_number": int,
    #   "item_id": str,
    #   "item_type": str,  # "frame" or "diagram"
    #   "similarity_score": float
    # }

    # ========== Incremental Extraction ==========
    incremental_mode: bool  # If True, only process changed items
    previous_snapshot_id: Optional[str]  # Previous snapshot for comparison
    changed_item_ids: Optional[List[str]]  # Item IDs to process (new/modified)
    deleted_item_ids: Optional[List[str]]  # Item IDs deleted from board

    # ========== GitHub-Specific Fields ==========
    date: Optional[str]  # Date in YYYYMMDD format for GitHub extraction at specific point in time
    github_scope: Optional[str]  # Extraction scope: "readme-only", "documentation", "diff", etc.
    github_profile: Optional[Dict[str, Any]]  # Active profile definition
    github_patterns: Optional[List[str]]  # Include patterns from profile
    github_exclusions: Optional[List[str]]  # Exclude patterns from profile
    github_from_tag: Optional[str]  # For diff extraction (base tag/branch)
    github_to_tag: Optional[str]  # For diff extraction (head tag/branch)
    github_file_map: Optional[Dict[str, str]]  # Map file_id → file metadata for change tracking

    # ========== Intermediate Results: Chunk Splitting Stage ==========
    chunks: List[Dict[str, str]]  # Text chunks
    # chunk: {"text": str}

    # ========== Intermediate Results: Image Extraction Stage ==========
    image_chunks: List[Dict[str, Any]]  # Image descriptions as chunks (parallel to text chunks)
    # image_chunk: {
    #   "text": str,              # Description text (Mermaid diagram or natural language)
    #   "file_id": str,
    #   "page_number": int,
    #   "provenance": {
    #     "source": "image",
    #     "image_id": str,
    #     "relevance_score": float,
    #     "has_structural_content": bool,
    #     "image_description": str,
    #     "structural_description": str | None
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
