"""
LangGraph State Definition for Knowledge Extraction
LangGraph State definition for knowledge extraction processing
"""

from typing import Any, Dict, List, Optional, TypedDict


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

    # Classification fields (two-dimensional)
    source_type: str  # Information source: "pdf_file" | "docx_file" | "pptx_file" | "txt_file" | "md_file" |
    # "miro_board" | "github_repo" | "jira_project" | "confluence_space"
    content_type: Optional[str]  # Content structure (optional): "document" | "presentation" | None
    # Only set for file sources (PDFs) that need layout-specific processing

    # ========== Metamodel ==========
    metamodel: Dict[str, Any]  # Metamodel definition (referenced by all nodes)

    # ========== LLM Model Settings ==========
    model: Optional[str]  # LLM model name

    # ========== Output Destination Settings ==========
    output_dir: str  # Output directory
    project: str  # Project name (output files classified by project)
    use_cache: Optional[bool]  # Reuse the source read cached under {output_dir}/{project}/cache/ (Miro)

    # ========== Project Profile ==========
    project_profile: Optional[Dict[str, Any]]  # {output_dir}/{project}/project_profile.json; absent when none
    # profile: {"description": str, "products": [{"name": str, "aliases": List[str]}]}

    # ========== Intermediate Results: Text Extraction Stage ==========
    file_id: str  # Stable file ID
    file_name: str  # File name
    sentences: List[Dict[str, Any]]  # List of extracted sentences
    # sentence: {"text": str, "section": str (optional, for MD files), ...}
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
    stale_file_ids: Optional[List[str]]  # file_ids whose accumulated knowledge is no longer valid
    edited_sentences: Optional[List[Dict[str, Any]]]  # Text of the items added or edited since the previous snapshot

    # ========== Jira-Specific Fields ==========
    jira_instance_url: Optional[str]  # Jira instance URL (e.g., "https://company.atlassian.net")
    jira_project_key: Optional[str]  # Project key (e.g., "PROJ")
    jira_raw_data: Optional[Dict[str, Any]]  # Raw Jira project data (for Jira sources)
    # Raw data structure: {
    #   "domain": str,
    #   "project_key": str,
    #   "issue_map": Dict[str, Any],  # issue_key → issue metadata for change tracking
    #   "total_issues": int,
    #   "fetched_at": str  # ISO timestamp
    # }

    # ========== Confluence-Specific Fields ==========
    confluence_space_key: Optional[str]  # Space key (e.g., "DOCS")
    confluence_raw_data: Optional[Dict[str, Any]]  # Raw Confluence space data (for Confluence sources)
    # Raw data structure: {
    #   "domain": str,
    #   "space_key": str,
    #   "page_map": Dict[str, Any],  # page_id → page metadata for change tracking
    #   "total_pages": int,
    #   "watermark": str  # Latest lastModified timestamp
    # }

    # ========== GitHub-Specific Fields ==========
    as_of: Optional[str]  # YYYYMMDD; extract the source as of 23:59:59 UTC that day (GitHub, Miro)
    github_scope: Optional[str]  # Extraction scope: "readme-only", "documentation", "diff", etc.
    github_profile: Optional[Dict[str, Any]]  # Active profile definition
    github_patterns: Optional[List[str]]  # Include patterns from profile
    github_exclusions: Optional[List[str]]  # Exclude patterns from profile
    github_from_tag: Optional[str]  # For diff extraction (base tag/branch)
    github_to_tag: Optional[str]  # For diff extraction (head tag/branch)
    github_file_map: Optional[Dict[str, str]]  # Map file_id → file metadata for change tracking

    # ========== Intermediate Results: Chunk Splitting Stage ==========
    chunks: List[Dict[str, Any]]  # Text chunks
    # chunk: {"text": str}
    # Miro chunks also carry "file_id" (region), "elements" (the board items packed into
    # the chunk, URL-only mode) and "title" (the region's frame title, when it has one)

    # ========== Intermediate Results: Image Extraction Stage ==========
    image_chunks: List[Dict[str, Any]]  # Image descriptions as chunks (parallel to text chunks)
    # image_chunk: {
    #   "text": str,              # The image's structural (Mermaid) form; non-structural images yield no chunk
    #   "file_id": str,
    #   "source": {               # Where the image came from
    #     "kind": "image",
    #     "page_number": int,
    #     "image_id": str,        # PDF embedded images only
    #     "item_id": str,         # Miro board items only
    #     "item_type": str        # Miro board items only
    #   },
    #   "analysis": {             # What the pipeline derived from it
    #     "has_structural_content": bool,
    #     "structural_description": str | None,   # Mermaid
    #     "relevance_score": float                # only when screening ran
    #   }
    # }
    image_statistics: Dict[str, Any]  # Image pipeline statistics

    extracted_structure: List[Dict[str, Any]]  # Format-agnostic structural edges from diagrams
    # Parsed from diagram chunks (Mermaid today, other sources later). Consumed by
    # PropertyExtractor to derive deterministic relationship candidates.
    # record: {"src_label": str, "dest_label": str, "edge_label": str, "file_id": str}

    # ========== Intermediate Results: Structured Block Stage (tables) ==========
    structured_blocks: List[Dict[str, Any]]  # Structured blocks (tables) preserved before flattening
    # Each block nests a source-independent table core under "table" while the
    # surrounding source-dependent context (file_id/section_path/context_window)
    # is supplied by the source extractor. Consumed by ClassExtractor (class
    # candidates from headers) and build_table_edges (deterministic property edges).
    # block: {
    #   "block_type": "table",
    #   "file_id": str,
    #   "scope_ref": str,          # e.g. page_id (provenance/context)
    #   "section_path": str,       # heading path above the table
    #   "context_window": str,     # surrounding text (a few hundred chars)
    #   "table": {                 # source-independent core (table_extractor output)
    #     "row_headers": List[str],
    #     "col_headers": List[str],
    #     "cells": List[List[str]],   # row-major; "" = empty cell
    #     "caption": Optional[str]
    #   }
    # }

    # ========== Intermediate Results: Relevance Screening Stage ==========
    # Set only when a project profile is present
    kept_sentences: List[Dict[str, Any]]  # Text elements the screener kept, in order ({"text": str})
    screening: Dict[str, Any]  # Screening summary
    # {
    #   "segments": int,   # LLM judgments made (one per region, page window, image or table)
    #   "elements": int,   # Elements judged (board items, sentences, images, tables)
    #   "kept": int,
    #   "unjudged": int,   # Elements the LLM returned no valid judgment for (kept)
    #   "dropped": [{"file_id": str, "text": str, "reason": str}]
    # }

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
    absorbed_class_ids: Optional[Dict[str, str]]  # Carried-over class ID merged away -> ID it went into

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
