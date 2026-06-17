"""Build structured blocks from normalized table cores.

This module provides shared utilities for wrapping normalized tables
(from table_extractor) with source-specific context metadata.
Used by both PDF and Confluence extractors to ensure consistent
structured block format across all sources.
"""

from typing import Any, Dict, List


def build_table_blocks(
    tables: List[Dict[str, Any]],
    file_id: str,
    scope_ref: str,
    section_path: str,
    context_window: str,
) -> List[Dict[str, Any]]:
    """Build structured blocks from normalized table cores.

    Generic helper used by both PDF and Confluence extractors.
    Takes normalized tables (with row_headers, col_headers, cells, caption)
    and wraps them with source-specific context metadata.

    Args:
        tables: List of normalized table cores from table_extractor
            Each table has: row_headers, col_headers, cells, caption
        file_id: Source identifier for provenance tracking
            Examples: "pdf:doc.pdf:page_3", "confluence:domain:space:page_id"
        scope_ref: Scope identifier for change tracking
            Examples: "page_3", "confluence_page_id"
        section_path: Human-readable section path for context
            Examples: "requirements.pdf - Page 3", "Architecture Overview"
        context_window: Surrounding text for LLM typing context
            First ~200 chars of surrounding text, whitespace normalized

    Returns:
        List of structured blocks ready for state["structured_blocks"].
        Each block has:
            - block_type: "table"
            - file_id: Source identifier
            - scope_ref: Scope for change tracking
            - section_path: Human-readable context
            - context_window: Surrounding text
            - table: Normalized table core

    Examples:
        >>> from src.utils.table_extractor import extract_tables_from_xhtml
        >>> tables = extract_tables_from_xhtml("<table>...</table>")
        >>> blocks = build_table_blocks(
        ...     tables,
        ...     file_id="pdf:doc.pdf:page_1",
        ...     scope_ref="page_1",
        ...     section_path="doc.pdf - Page 1",
        ...     context_window="The following table shows..."
        ... )
        >>> len(blocks) == len(tables)
        True
        >>> blocks[0]["block_type"]
        'table'
    """
    blocks = []
    for table in tables:
        blocks.append(
            {
                "block_type": "table",
                "file_id": file_id,
                "scope_ref": scope_ref,
                "section_path": section_path,
                "context_window": context_window,
                "table": table,
            }
        )
    return blocks
