"""Detect changes in source data (e.g., Miro board items) between snapshots."""

import json
from pathlib import Path
from typing import Dict, List, Optional


def load_source_raw(project: str, snapshot_id: str, output_dir: str = "/workspace/app/output") -> Optional[dict]:
    """Load source_raw.json from a snapshot.

    Args:
        project: Project name
        snapshot_id: Snapshot ID
        output_dir: Base output directory

    Returns:
        Raw source data dict, or None if not found
    """
    snapshot_path = Path(output_dir) / project / "snapshots" / snapshot_id / "source_raw.json"

    if not snapshot_path.exists():
        return None

    with open(snapshot_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def detect_source_changes(
    before_snapshot_id: str,
    after_snapshot_id: str,
    project: str,
    output_dir: str = "/workspace/app/output"
) -> dict:
    """Compare source_raw.json between two snapshots.

    Args:
        before_snapshot_id: Earlier snapshot ID
        after_snapshot_id: Later snapshot ID
        project: Project name
        output_dir: Base output directory

    Returns:
        Dictionary with:
        {
            "new_items": [...],      # Items added
            "modified_items": [...], # Items changed
            "deleted_items": [...],  # Items removed
            "source_type": str       # Type of source (e.g., "miro", "pdf")
        }

    Example:
        >>> changes = detect_source_changes("20260501110000", "20260501120000", "myproject")
        >>> print(len(changes["new_items"]))
        3
    """
    before_raw = load_source_raw(project, before_snapshot_id, output_dir)
    after_raw = load_source_raw(project, after_snapshot_id, output_dir)

    # If no source_raw.json in either snapshot, return empty changes
    if not before_raw and not after_raw:
        return {
            "new_items": [],
            "modified_items": [],
            "deleted_items": [],
            "source_type": "none"
        }

    # Handle case where only one has source_raw
    if not before_raw:
        before_raw = {"items": []}
    if not after_raw:
        after_raw = {"items": []}

    # Detect source type (Miro, PDF, etc.)
    source_type = "unknown"
    if "id" in after_raw and "items" in after_raw:
        source_type = "miro"

    # Build item maps by ID
    before_items = {item["id"]: item for item in before_raw.get("items", [])}
    after_items = {item["id"]: item for item in after_raw.get("items", [])}

    # Find new, deleted, and common items
    before_ids = set(before_items.keys())
    after_ids = set(after_items.keys())

    new_ids = after_ids - before_ids
    deleted_ids = before_ids - after_ids
    common_ids = before_ids & after_ids

    # Find modified items (content changed)
    modified_ids = []
    for item_id in common_ids:
        before_item = before_items[item_id]
        after_item = after_items[item_id]

        # Compare content (for Miro items)
        if _item_content_changed(before_item, after_item):
            modified_ids.append(item_id)

    return {
        "new_items": [after_items[id] for id in new_ids],
        "modified_items": [after_items[id] for id in modified_ids],
        "deleted_items": [before_items[id] for id in deleted_ids],
        "source_type": source_type
    }


def _item_content_changed(before_item: dict, after_item: dict) -> bool:
    """Check if item content has changed.

    Args:
        before_item: Item from earlier snapshot
        after_item: Item from later snapshot

    Returns:
        True if content changed
    """
    # For Miro items, compare the data.content field
    before_content = before_item.get("data", {}).get("content", "")
    after_content = after_item.get("data", {}).get("content", "")

    # Also check title for cards/documents
    before_title = before_item.get("data", {}).get("title", "")
    after_title = after_item.get("data", {}).get("title", "")

    return before_content != after_content or before_title != after_title
