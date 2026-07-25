"""Build the deterministic part of a changeset instance graph.

Metamodel entries marked ``"extraction": "deterministic"`` are computed from
the structural diff of two snapshots instead of being extracted by the LLM
chains. Entries are looked up by their metamodel ``name``, so this module
carries no IRIs of its own.

Building happens in two phases: class instances first, then the property
links, which need the instance IDs to be settled.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

from src.utils.id_generator import generate_class_id, generate_property_id


@dataclass(frozen=True)
class ObjectKind:
    """How one kind of knowledge graph object appears in the changeset metamodel."""

    class_name: str  # Metamodel class, e.g. "KGNode"
    link_suffix: str  # Link name suffix, e.g. "Node" in beforeNode / hasNode
    noun: str  # Wording used in the change description
    label: Callable[[dict], str] = field(repr=False)


NODE = ObjectKind(
    class_name="KGNode",
    link_suffix="Node",
    noun="node",
    label=lambda obj: obj.get("label", ""),
)

RELATIONSHIP = ObjectKind(
    class_name="KGRelationship",
    link_suffix="Relationship",
    noun="relationship",
    label=lambda obj: f"{obj.get('src_label', '?')} -> {obj.get('dest_label', '?')}",
)

CHANGE_VERBS = {"ADD": "Added", "UPDATE": "Updated", "DELETE": "Deleted"}


@dataclass
class ChangeRecord:
    """A derived change together with the snapshot objects it was computed from."""

    instance: dict
    kind: ObjectKind
    before: Optional[dict] = None
    after: Optional[dict] = None


def build_deterministic_graph(
    metamodel: Dict[str, Any],
    changeset_id: str,
    before_id: str,
    after_id: str,
    diff: Dict[str, Any],
) -> Dict[str, List[dict]]:
    """Build the deterministic instances and links for one change set.

    Raises:
        KeyError: If the metamodel is missing a class or property this module
            builds — the metamodel and this module are revised together.

    Args:
        metamodel: Changeset metamodel dict (classes/properties)
        changeset_id: ID of the change set being built
        before_id: Earlier snapshot ID
        after_id: Later snapshot ID
        diff: change_diff_engine.compute_diff() result

    Returns:
        {"classes": [instance, ...], "properties": [link, ...]}
    """
    defs = deterministic_defs(metamodel)

    # Phase 1: class instances
    change_set = build_change_set(defs, changeset_id)
    snapshots = build_snapshots(defs, before_id, after_id)
    changes = build_changes(defs, diff)
    objects = build_objects(defs, changes, before_id, after_id)

    classes = [change_set] + snapshots + [c.instance for c in changes] + objects

    # Phase 2: property links, now that every instance ID is settled
    properties = (
        build_snapshot_links(defs, changeset_id, before_id, after_id)
        + build_has_change_links(defs, changeset_id, changes)
        + build_change_object_links(defs, changes, before_id, after_id)
        + build_snapshot_object_links(defs, changes, before_id, after_id)
    )

    return {"classes": classes, "properties": properties}


def deterministic_defs(metamodel: Dict[str, Any]) -> Dict[str, dict]:
    """Index the metamodel's deterministic classes and properties by name."""
    entries = metamodel.get("classes", []) + metamodel.get("properties", [])
    return {e["name"]: e for e in entries if e.get("extraction") == "deterministic" and e.get("name")}


# ---------------------------------------------------------------------------
# Class instances
# ---------------------------------------------------------------------------


def build_change_set(defs: Dict[str, dict], changeset_id: str) -> dict:
    """Build the KGChangeSet instance."""
    definition = defs["KGChangeSet"]
    return {"id": changeset_id, "class_iri": definition["iri"], "label": changeset_id}


def build_snapshots(defs: Dict[str, dict], before_id: str, after_id: str) -> List[dict]:
    """Build the KGSnapshot instances for both sides of the diff."""
    definition = defs["KGSnapshot"]
    return [{"id": sid, "class_iri": definition["iri"], "label": sid} for sid in (before_id, after_id)]


def build_changes(defs: Dict[str, dict], diff: Dict[str, Any]) -> List[ChangeRecord]:
    """Build a KGChange instance for every entry of the diff."""
    definition = defs["KGChange"]

    records = []
    for change_type, kind, before, after, field_diff in _diff_entries(diff):
        subject = after or before
        label = f"{change_type} {kind.class_name}: {kind.label(subject)}"
        records.append(
            ChangeRecord(
                instance={
                    "id": generate_class_id(definition["iri"], label, subject.get("id", "")),
                    "class_iri": definition["iri"],
                    "label": label,
                    "sources": [_describe(change_type, kind, subject, field_diff)],
                    "attributes": {
                        "change_type": change_type,
                        "object_type": kind.class_name,
                        "diff": field_diff,
                    },
                },
                kind=kind,
                before=before,
                after=after,
            )
        )
    return records


def build_objects(defs: Dict[str, dict], changes: List[ChangeRecord], before_id: str, after_id: str) -> List[dict]:
    """Materialize the object states the changes refer to, deduplicated.

    Relationships become instances too: a property graph edge cannot be the
    endpoint of another edge, so beforeRelationship/afterRelationship need the
    relationship state to exist as a node.
    """
    objects: Dict[str, dict] = {}
    for kind, snapshot_id, raw in _referenced_objects(changes, before_id, after_id):
        object_id = _object_id(snapshot_id, raw)
        if object_id not in objects:
            objects[object_id] = {
                "id": object_id,
                "class_iri": defs[kind.class_name]["iri"],
                "label": kind.label(raw),
                "attributes": dict(raw),
            }
    return list(objects.values())


# ---------------------------------------------------------------------------
# Property links
# ---------------------------------------------------------------------------


def build_snapshot_links(defs: Dict[str, dict], changeset_id: str, before_id: str, after_id: str) -> List[dict]:
    """Build the beforeSnapshot / afterSnapshot links."""
    return [
        _link(defs, "beforeSnapshot", changeset_id, before_id),
        _link(defs, "afterSnapshot", changeset_id, after_id),
    ]


def build_has_change_links(defs: Dict[str, dict], changeset_id: str, changes: List[ChangeRecord]) -> List[dict]:
    """Build the hasChange links from the change set to each change."""
    return [_link(defs, "hasChange", changeset_id, record.instance["id"]) for record in changes]


def build_change_object_links(
    defs: Dict[str, dict], changes: List[ChangeRecord], before_id: str, after_id: str
) -> List[dict]:
    """Build the before<Kind> / after<Kind> links from each change to its object states."""
    links = []
    for record in changes:
        if record.before is not None:
            links.append(
                _link(
                    defs,
                    f"before{record.kind.link_suffix}",
                    record.instance["id"],
                    _object_id(before_id, record.before),
                )
            )
        if record.after is not None:
            links.append(
                _link(
                    defs,
                    f"after{record.kind.link_suffix}",
                    record.instance["id"],
                    _object_id(after_id, record.after),
                )
            )
    return links


def build_snapshot_object_links(
    defs: Dict[str, dict], changes: List[ChangeRecord], before_id: str, after_id: str
) -> List[dict]:
    """Build the has<Kind> links from each snapshot to the object states it holds.

    Only objects referenced by a change are materialized, so this covers the
    diff rather than the whole snapshot.
    """
    links: Dict[str, dict] = {}
    for kind, snapshot_id, raw in _referenced_objects(changes, before_id, after_id):
        link = _link(defs, f"has{kind.link_suffix}", snapshot_id, _object_id(snapshot_id, raw))
        links.setdefault(link["id"], link)
    return list(links.values())


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _diff_entries(diff: Dict[str, Any]) -> Iterator[Tuple[str, ObjectKind, Optional[dict], Optional[dict], dict]]:
    """Yield (change_type, kind, before, after, field_diff) for every diff entry."""
    for cls in diff.get("added_classes", []):
        yield "ADD", NODE, None, cls, {}
    for update in diff.get("updated_classes", []):
        yield "UPDATE", NODE, update["before"], update["after"], update["diff"]
    for cls in diff.get("deleted_classes", []):
        yield "DELETE", NODE, cls, None, {}
    for prop in diff.get("added_properties", []):
        yield "ADD", RELATIONSHIP, None, prop, {}
    for update in diff.get("updated_properties", []):
        yield "UPDATE", RELATIONSHIP, update["before"], update["after"], update["diff"]
    for prop in diff.get("deleted_properties", []):
        yield "DELETE", RELATIONSHIP, prop, None, {}


def _referenced_objects(
    changes: List[ChangeRecord], before_id: str, after_id: str
) -> Iterator[Tuple[ObjectKind, str, dict]]:
    """Yield (kind, snapshot_id, raw_object) for every object a change refers to."""
    for record in changes:
        if record.before is not None:
            yield record.kind, before_id, record.before
        if record.after is not None:
            yield record.kind, after_id, record.after


def _object_id(snapshot_id: str, raw: dict) -> str:
    """Snapshot-scoped object ID: the same object holds different state per snapshot."""
    return f"{snapshot_id}#{raw.get('id', '')}"


def _link(defs: Dict[str, dict], name: str, src_id: str, dest_id: str) -> dict:
    """Build one link between two instances."""
    iri = defs[name]["iri"]
    return {
        "id": generate_property_id(src_id, iri, dest_id),
        "src_id": src_id,
        "property_iri": iri,
        "dest_id": dest_id,
    }


def _describe(change_type: str, kind: ObjectKind, subject: dict, field_diff: dict) -> str:
    """Render the change as a sentence, used as the change's source text."""
    verb = CHANGE_VERBS.get(change_type, change_type)
    label = kind.label(subject)
    if change_type == "UPDATE":
        return f"{verb} {kind.noun} '{label}' ({_field_changes(field_diff)})"
    if kind is NODE:
        return f"{verb} {kind.noun} '{label}' (type: {_local_name(subject.get('class_iri', ''))})"
    return f"{verb} {kind.noun} '{label}'"


def _field_changes(field_diff: dict) -> str:
    """Render modified fields as 'field: old -> new' pairs."""
    parts = []
    for name in field_diff.get("modified_fields", []):
        old = field_diff.get("old_values", {}).get(name)
        new = field_diff.get("new_values", {}).get(name)
        parts.append(f"{name}: {old!r} -> {new!r}")
    return "; ".join(parts)


def _local_name(iri: str) -> str:
    """Extract the local name from an IRI."""
    return iri.rsplit("#", 1)[-1] if "#" in iri else iri
