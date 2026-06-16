"""Derive deterministic structural edges from table blocks.

A matrix table encodes relationships explicitly: a non-empty data cell at
(row_header_i, col_header_j) means those two entities are related. This module
turns such filled cells into format-agnostic edge records -- the same shape
``build_extracted_structure`` (mermaid_parser) produces for diagrams -- so they
flow through the existing ``_extract_properties_from_structure`` resolution
(label -> class, pair_map for the property type). No LLM is involved.

Edge direction/type is NOT decided here: it is resolved downstream from the
class types of the resolved endpoints via the metamodel pair_map (both
directions are tried), exactly like diagram edges.
"""

from typing import Any, Dict, List


def build_table_edges(structured_blocks: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """Build structural edge records from table blocks.

    For each table, every non-empty data cell ``cells[i][j]`` emits an edge
    between ``row_headers[i]`` and ``col_headers[j]``. Empty cells (and any
    header that is blank) are skipped. Non-table blocks are ignored.

    Args:
        structured_blocks: List of structured blocks (see
            ``KnowledgeExtractionState.structured_blocks``).

    Returns:
        List of ``{"src_label", "dest_label", "edge_label", "file_id"}`` records
        (deduplicated). Compatible with the ``extracted_structure`` channel.
    """
    edges: List[Dict[str, str]] = []
    seen = set()

    for block in structured_blocks or []:
        if block.get("block_type") != "table":
            continue
        table = block.get("table") or {}
        row_headers = table.get("row_headers") or []
        col_headers = table.get("col_headers") or []
        cells = table.get("cells") or []
        file_id = block.get("file_id", "")

        for i, row in enumerate(cells):
            if i >= len(row_headers):
                break
            src_label = (row_headers[i] or "").strip()
            if not src_label:
                continue
            for j, value in enumerate(row):
                if j >= len(col_headers):
                    break
                # A non-empty cell marks a relationship between the row and column.
                if not (value or "").strip():
                    continue
                dest_label = (col_headers[j] or "").strip()
                if not dest_label:
                    continue

                key = (src_label, dest_label, file_id)
                if key in seen:
                    continue
                seen.add(key)
                edges.append(
                    {
                        "src_label": src_label,
                        "dest_label": dest_label,
                        "edge_label": "",
                        "file_id": file_id,
                    }
                )

    return edges


if __name__ == "__main__":
    import json

    blocks = [
        {
            "block_type": "table",
            "file_id": "confluence:example:DOCS:123",
            "scope_ref": "123",
            "section_path": "システム要件",
            "context_window": "...",
            "table": {
                "row_headers": ["ユーザー認証要件", "アクセス制御要件", "データ暗号化要件"],
                "col_headers": ["ログイン機能", "SSO連携機能", "監査ログ機能"],
                "cells": [
                    ["○", "○", ""],
                    ["", "○", "○"],
                    ["", "", "○"],
                ],
                "caption": "要件・機能対応表",
            },
        }
    ]
    edges = build_table_edges(blocks)
    print(json.dumps(edges, ensure_ascii=False, indent=2))
    assert len(edges) == 5, f"expected 5 edges, got {len(edges)}"
    assert {
        "src_label": "ユーザー認証要件",
        "dest_label": "ログイン機能",
        "edge_label": "",
        "file_id": "confluence:example:DOCS:123",
    } in edges
    print("\nOK")
