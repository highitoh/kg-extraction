"""Compute diffs between knowledge.json snapshots."""

import json
from pathlib import Path
from typing import Dict, List


def load_knowledge(project: str, snapshot_id: str, output_dir: str = "/workspace/app/output") -> dict:
    """Load knowledge.json from a snapshot.

    Args:
        project: Project name
        snapshot_id: Snapshot ID
        output_dir: Base output directory

    Returns:
        Knowledge graph dict with classes and properties

    Raises:
        FileNotFoundError: If knowledge.json not found
    """
    snapshot_path = Path(output_dir) / project / "snapshots" / snapshot_id / "knowledge.json"

    if not snapshot_path.exists():
        raise FileNotFoundError(f"knowledge.json not found at {snapshot_path}")

    with open(snapshot_path, "r", encoding="utf-8") as f:
        return json.load(f)


def compute_diff(before_kg: dict, after_kg: dict) -> dict:
    """Compute diff between two knowledge graphs.

    Args:
        before_kg: Earlier knowledge graph
        after_kg: Later knowledge graph

    Returns:
        Diff result with:
        {
            "added_classes": [...],
            "updated_classes": [...],
            "deleted_classes": [...],
            "added_properties": [...],
            "deleted_properties": [...],
            "summary": {counts}
        }

    Example:
        >>> diff = compute_diff(kg1, kg2)
        >>> print(len(diff["added_classes"]))
        5
    """
    # Build maps by ID for efficient lookup
    before_classes = {cls["id"]: cls for cls in before_kg.get("classes", [])}
    after_classes = {cls["id"]: cls for cls in after_kg.get("classes", [])}

    before_props = {_prop_key(p): p for p in before_kg.get("properties", [])}
    after_props = {_prop_key(p): p for p in after_kg.get("properties", [])}

    # Find added/deleted/updated classes
    before_class_ids = set(before_classes.keys())
    after_class_ids = set(after_classes.keys())

    added_class_ids = after_class_ids - before_class_ids
    deleted_class_ids = before_class_ids - after_class_ids
    common_class_ids = before_class_ids & after_class_ids

    added_classes = [after_classes[id] for id in added_class_ids]
    deleted_classes = [before_classes[id] for id in deleted_class_ids]

    # Check for updated classes (label, judgment, etc. changed)
    updated_classes = []
    for class_id in common_class_ids:
        before_cls = before_classes[class_id]
        after_cls = after_classes[class_id]

        if _class_content_changed(before_cls, after_cls):
            updated_classes.append(
                {"before": before_cls, "after": after_cls, "diff": _compute_class_diff(before_cls, after_cls)}
            )

    # Find added/deleted properties
    before_prop_keys = set(before_props.keys())
    after_prop_keys = set(after_props.keys())

    added_prop_keys = after_prop_keys - before_prop_keys
    deleted_prop_keys = before_prop_keys - after_prop_keys

    added_properties = [after_props[key] for key in added_prop_keys]
    deleted_properties = [before_props[key] for key in deleted_prop_keys]

    # Build summary
    summary = {
        "total_changes": (
            len(added_classes)
            + len(updated_classes)
            + len(deleted_classes)
            + len(added_properties)
            + len(deleted_properties)
        ),
        "added_classes": len(added_classes),
        "updated_classes": len(updated_classes),
        "deleted_classes": len(deleted_classes),
        "added_properties": len(added_properties),
        "deleted_properties": len(deleted_properties),
    }

    return {
        "added_classes": added_classes,
        "updated_classes": updated_classes,
        "deleted_classes": deleted_classes,
        "added_properties": added_properties,
        "deleted_properties": deleted_properties,
        "summary": summary,
    }


def _prop_key(prop: dict) -> str:
    """Generate unique key for a property.

    Uses: property_iri + src_id + dest_id

    Args:
        prop: Property dict

    Returns:
        Unique key string
    """
    return f"{prop.get('property_iri', '')}|{prop.get('src_id', '')}|{prop.get('dest_id', '')}"


def _class_content_changed(before_cls: dict, after_cls: dict) -> bool:
    """Check if class content has changed.

    Args:
        before_cls: Class from earlier snapshot
        after_cls: Class from later snapshot

    Returns:
        True if content changed
    """
    # Compare key fields
    fields_to_compare = ["label", "judgment", "class_iri", "sources"]

    for field in fields_to_compare:
        before_val = before_cls.get(field)
        after_val = after_cls.get(field)

        if before_val != after_val:
            return True

    return False


def _compute_class_diff(before_cls: dict, after_cls: dict) -> dict:
    """Compute detailed diff for a single class.

    Args:
        before_cls: Class from earlier snapshot
        after_cls: Class from later snapshot

    Returns:
        Diff dict with modified_fields, old_values, new_values
    """
    fields_to_compare = ["label", "judgment", "class_iri", "sources"]
    modified_fields = []
    old_values = {}
    new_values = {}

    for field in fields_to_compare:
        before_val = before_cls.get(field)
        after_val = after_cls.get(field)

        if before_val != after_val:
            modified_fields.append(field)
            old_values[field] = before_val
            new_values[field] = after_val

    return {"modified_fields": modified_fields, "old_values": old_values, "new_values": new_values}
