"""Compute diffs between knowledge.json snapshots."""

import json
from pathlib import Path

# Fields compared to detect content changes on surviving objects. Relationship
# endpoints (src_label/dest_label) are deliberately excluded: a node rename
# already surfaces as a node UPDATE, and repeating it on every attached
# relationship would only add noise.
CLASS_DIFF_FIELDS = ["label", "judgment", "class_iri", "sources"]
PROPERTY_DIFF_FIELDS = ["strength", "judgment", "justification"]


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
            "updated_properties": [...],
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

    added_classes = [after_classes[id] for id in after_class_ids - before_class_ids]
    deleted_classes = [before_classes[id] for id in before_class_ids - after_class_ids]
    updated_classes = _collect_updates(
        before_classes, after_classes, before_class_ids & after_class_ids, CLASS_DIFF_FIELDS
    )

    # Find added/deleted/updated properties
    before_prop_keys = set(before_props.keys())
    after_prop_keys = set(after_props.keys())

    added_properties = [after_props[key] for key in after_prop_keys - before_prop_keys]
    deleted_properties = [before_props[key] for key in before_prop_keys - after_prop_keys]
    updated_properties = _collect_updates(
        before_props, after_props, before_prop_keys & after_prop_keys, PROPERTY_DIFF_FIELDS
    )

    # Build summary
    summary = {
        "total_changes": (
            len(added_classes)
            + len(updated_classes)
            + len(deleted_classes)
            + len(added_properties)
            + len(updated_properties)
            + len(deleted_properties)
        ),
        "added_classes": len(added_classes),
        "updated_classes": len(updated_classes),
        "deleted_classes": len(deleted_classes),
        "added_properties": len(added_properties),
        "updated_properties": len(updated_properties),
        "deleted_properties": len(deleted_properties),
    }

    return {
        "added_classes": added_classes,
        "updated_classes": updated_classes,
        "deleted_classes": deleted_classes,
        "added_properties": added_properties,
        "updated_properties": updated_properties,
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


def _collect_updates(before_map: dict, after_map: dict, common_keys: set, fields: list) -> list:
    """Collect updated objects among those present in both snapshots.

    Args:
        before_map: key -> object from earlier snapshot
        after_map: key -> object from later snapshot
        common_keys: Keys present in both maps
        fields: Field names to compare

    Returns:
        List of {"before": ..., "after": ..., "diff": ...} entries
    """
    updates = []
    for key in common_keys:
        before_obj = before_map[key]
        after_obj = after_map[key]
        diff = _compute_field_diff(before_obj, after_obj, fields)
        if diff["modified_fields"]:
            updates.append({"before": before_obj, "after": after_obj, "diff": diff})
    return updates


def _compute_field_diff(before_obj: dict, after_obj: dict, fields: list) -> dict:
    """Compute a field-level diff between two objects.

    Args:
        before_obj: Object from earlier snapshot
        after_obj: Object from later snapshot
        fields: Field names to compare

    Returns:
        Diff dict with modified_fields, old_values, new_values
    """
    modified_fields = []
    old_values = {}
    new_values = {}

    for field in fields:
        before_val = before_obj.get(field)
        after_val = after_obj.get(field)

        if before_val != after_val:
            modified_fields.append(field)
            old_values[field] = before_val
            new_values[field] = after_val

    return {"modified_fields": modified_fields, "old_values": old_values, "new_values": new_values}
