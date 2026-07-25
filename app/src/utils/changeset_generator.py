"""Generate changesets between knowledge graph snapshots.

A changeset records one extraction: what it changed in the knowledge graph
and why. It is generated right after the extraction that produced the later
snapshot, while the source text is still at hand — the text is the only place
the reason for a change is written, and it is not kept in the snapshots.
Ranges spanning several extractions are answered by aggregating the
changesets in between (see ``aggregate_changesets``), which needs no LLM.

The pipeline is driven by the changeset metamodel
(``metamodel/changeset-metamodel.json``):

1. Deterministic phase — changeset_builder computes instances of
   ``extraction: "deterministic"`` classes/properties from the structural diff
   of the two snapshots (changes, object states, structural links).
2. LLM class phase — the standard class chain extracts instances of
   ``extraction: "llm"`` classes (e.g. change drivers) from the passages the
   changed knowledge came from, with consolidation and review.
3. LLM property phase — the standard property chain extracts ``llm``
   properties (e.g. isDrivenBy) between the derived and extracted instances,
   with ranking, review, and src/dest validation.

The output is an instance graph (knowledge.json-like classes/properties
arrays) conforming to the changeset metamodel, so metamodel changes need no
format or code changes.
"""

import importlib
import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

from src.extract.property.property_chain import create_property_chain
from src.utils import change_diff_engine, changeset_builder, changeset_inputs

# "class" is a reserved word, so the class chain package cannot be imported
# with a regular from-import.
_class_extractor_module = importlib.import_module("src.extract.class.class_extractor")
create_class_chain = importlib.import_module("src.extract.class.class_chain").create_class_chain
view_types_from_metamodel = _class_extractor_module.view_types_from_metamodel

load_dotenv()

DEFAULT_METAMODEL_PATH = str(Path(__file__).parent.parent.parent / "metamodel" / "changeset-metamodel.json")


def generate_changeset(
    before_snapshot_id: str,
    after_snapshot_id: str,
    project: str,
    output_dir: str = "/workspace/app/output",
    model: Optional[str] = None,
    changeset_id: Optional[str] = None,
    metamodel_path: str = DEFAULT_METAMODEL_PATH,
    max_review_iterations: int = 2,
    source_sentences: Optional[list] = None,
    file_id: str = "unknown",
) -> str:
    """Generate a metamodel-driven changeset for one extraction.

    Args:
        before_snapshot_id: Snapshot before the extraction
        after_snapshot_id: Snapshot the extraction produced
        project: Project name
        output_dir: Base output directory
        model: Optional LLM model for the extraction chains
        changeset_id: Optional ID override; generated from the after-snapshot
            ID when omitted
        metamodel_path: Changeset metamodel file
        max_review_iterations: Review-loop iterations for the LLM chains
        source_sentences: Sentences of the extraction, used to extract the
            change drivers. Without them the changeset holds the structural
            changes only.
        file_id: Provenance ID recorded on the extracted drivers

    Returns:
        Changeset ID

    Example:
        >>> changeset_id = generate_changeset("20260501110000", "20260501120000", "myproject")
        >>> print(changeset_id)
        "cs_20260501_120000"
    """
    metamodel = _load_metamodel(metamodel_path)
    changeset_id = changeset_id or _generate_changeset_id(after_snapshot_id)

    # 1. Load snapshots and compute the structural diff
    before_kg = change_diff_engine.load_knowledge(project, before_snapshot_id, output_dir)
    after_kg = change_diff_engine.load_knowledge(project, after_snapshot_id, output_dir)
    kg_diff = change_diff_engine.compute_diff(before_kg, after_kg)

    # 2. Deterministic phase
    derived = changeset_builder.build_deterministic_graph(
        metamodel, changeset_id, before_snapshot_id, after_snapshot_id, kg_diff
    )

    # 3. LLM class phase (e.g. KGChangeDriver from the text behind the changes)
    texts = changeset_inputs.build_change_context(kg_diff, source_sentences or [], file_id)
    extracted_classes = _extract_llm_classes(metamodel, metamodel_path, texts, model, max_review_iterations)

    # 4. LLM property phase (e.g. isDrivenBy between changes and drivers)
    all_instances = derived["classes"] + extracted_classes
    extracted_properties = _extract_llm_properties(
        metamodel, metamodel_path, all_instances, model, max_review_iterations
    )

    # 5. Assemble the instance graph
    summary = {**kg_diff["summary"], "drivers": len(extracted_classes)}
    changeset = {
        "changeSetId": changeset_id,
        "beforeSnapshotId": before_snapshot_id,
        "afterSnapshotId": after_snapshot_id,
        "timestamp": datetime.now().isoformat(),
        "summary": summary,
        "classes": _strip_internal_flags(all_instances),
        "properties": derived["properties"] + extracted_properties,
    }

    # 6. Save the changeset
    _save_changeset(project, changeset_id, changeset, output_dir)

    return changeset_id


def aggregate_changesets(
    before_snapshot_id: str,
    after_snapshot_id: str,
    project: str,
    output_dir: str = "/workspace/app/output",
) -> dict:
    """Combine the changesets recorded between two snapshots.

    Each extraction records its own changeset, so a range is answered by
    replaying the ones in between rather than diffing the endpoints. That
    keeps the reason behind every step: a node added for one reason and later
    revised for another shows both, where a direct diff of the endpoints would
    show neither.

    Args:
        before_snapshot_id: Start of the range (exclusive)
        after_snapshot_id: End of the range (inclusive)
        project: Project name
        output_dir: Base output directory

    Returns:
        A changeset dict spanning the range, in the same instance-graph shape
        as a single changeset

    Raises:
        FileNotFoundError: If the project has no changeset index
    """
    entries = _changesets_in_range(project, before_snapshot_id, after_snapshot_id, output_dir)

    classes: dict = {}
    properties: dict = {}
    summary: dict = {}
    for entry in entries:
        changeset = _load_changeset(project, entry["changeSetId"], output_dir)
        for instance in changeset.get("classes", []):
            classes.setdefault(instance["id"], instance)
        for link in changeset.get("properties", []):
            properties.setdefault(link["id"], link)
        for key, value in changeset.get("summary", {}).items():
            summary[key] = summary.get(key, 0) + value

    # The aggregate is itself a change set spanning the whole range, so the
    # per-extraction change set instances are replaced by a single one.
    changeset_id = f"cs_{before_snapshot_id}_{after_snapshot_id}"
    metamodel = _load_metamodel()
    change_set_iri = changeset_builder.deterministic_defs(metamodel)["KGChangeSet"]["iri"]
    for instance in list(classes.values()):
        if instance.get("class_iri") == change_set_iri:
            del classes[instance["id"]]
    classes[changeset_id] = {"id": changeset_id, "class_iri": change_set_iri, "label": changeset_id}
    properties = {
        link_id: ({**link, "src_id": changeset_id} if link["src_id"] not in classes else link)
        for link_id, link in properties.items()
    }

    return {
        "changeSetId": changeset_id,
        "beforeSnapshotId": before_snapshot_id,
        "afterSnapshotId": after_snapshot_id,
        "timestamp": datetime.now().isoformat(),
        "summary": summary,
        "classes": list(classes.values()),
        "properties": list(properties.values()),
        "aggregatedFrom": [entry["changeSetId"] for entry in entries],
    }


def _changesets_in_range(project: str, before_id: str, after_id: str, output_dir: str) -> list:
    """List the recorded changesets covering (before_id, after_id], in order."""
    metadata_file = Path(output_dir) / project / "changesets" / "metadata.json"
    if not metadata_file.exists():
        raise FileNotFoundError(f"No changesets recorded for project '{project}'")

    with open(metadata_file, "r", encoding="utf-8") as f:
        entries = json.load(f).get("changesets", [])

    in_range = [e for e in entries if before_id < e["afterSnapshotId"] <= after_id]
    return sorted(in_range, key=lambda e: e["afterSnapshotId"])


def _load_changeset(project: str, changeset_id: str, output_dir: str) -> dict:
    """Load one recorded changeset."""
    path = Path(output_dir) / project / "changesets" / changeset_id / "changeset.json"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_metamodel(metamodel_path: str = DEFAULT_METAMODEL_PATH) -> dict:
    """Load the changeset metamodel."""
    with open(metamodel_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _extract_llm_classes(
    metamodel: dict, metamodel_path: str, texts: list, model: Optional[str], max_review_iterations: int
) -> list:
    """Extract instances of the metamodel's LLM classes from the change context."""
    view_types = view_types_from_metamodel(metamodel)
    if not view_types:
        return []

    if not texts:
        print("[ChangesetGenerator] No source text for the changes; skipping LLM class extraction")
        return []

    print(f"[ChangesetGenerator] Extracting LLM classes from {len(texts)} source spans")
    chain = create_class_chain(model=model, max_review_iterations=max_review_iterations, metamodel_path=metamodel_path)
    result = chain.invoke(
        {
            "view_info": changeset_inputs.as_view_info(texts, view_types),
            "metamodel": metamodel,
            "existing_classes": [],
        }
    )

    return [cls for cls in result.get("classes", []) if cls.get("judgment") != "REJECT"]


def _extract_llm_properties(
    metamodel: dict, metamodel_path: str, instances: list, model: Optional[str], max_review_iterations: int
) -> list:
    """Extract the metamodel's LLM properties between the given instances."""
    llm_properties = [p for p in metamodel.get("properties", []) if p.get("extraction", "llm") != "deterministic"]
    if not llm_properties:
        return []

    # Skip the chain entirely when no LLM property has instances on both ends
    iris_with_instances = {inst.get("class_iri") for inst in instances}
    if not any(
        p.get("src_class") in iris_with_instances and p.get("dest_class") in iris_with_instances for p in llm_properties
    ):
        print("[ChangesetGenerator] No instance pairs for LLM properties; skipping LLM property extraction")
        return []

    print(f"[ChangesetGenerator] Extracting LLM properties between {len(instances)} instances")
    chain = create_property_chain(
        model=model, max_review_iterations=max_review_iterations, metamodel_path=metamodel_path
    )
    result = chain.invoke(
        {
            "class_info": changeset_inputs.as_class_info(instances),
            "metamodel": metamodel,
        }
    )

    return [prop for prop in result.get("properties", []) if prop.get("judgment") != "REJECT"]


def _strip_internal_flags(instances: list) -> list:
    """Remove chain-internal control flags from output instances."""
    cleaned = []
    for inst in instances:
        cleaned.append({k: v for k, v in inst.items() if not k.startswith("_")})
    return cleaned


def _generate_changeset_id(snapshot_id: str) -> str:
    """Generate changeset ID from snapshot ID.

    Args:
        snapshot_id: Snapshot ID (YYYYMMDDHHMMSS)

    Returns:
        Changeset ID (cs_YYYYMMDD_HHMMSS)
    """
    if len(snapshot_id) >= 14:
        date_part = snapshot_id[:8]
        time_part = snapshot_id[8:14]
        return f"cs_{date_part}_{time_part}"
    # Fallback to UUID if snapshot_id is malformed
    return f"cs_{uuid.uuid4().hex[:16]}"


def _save_changeset(project: str, changeset_id: str, changeset: dict, output_dir: str):
    """Save the changeset JSON and update the metadata index.

    Args:
        project: Project name
        changeset_id: Changeset ID
        changeset: Changeset dict
        output_dir: Base output directory
    """
    changeset_dir = Path(output_dir) / project / "changesets" / changeset_id
    changeset_dir.mkdir(parents=True, exist_ok=True)

    changeset_file = changeset_dir / "changeset.json"
    with open(changeset_file, "w", encoding="utf-8") as f:
        json.dump(changeset, f, ensure_ascii=False, indent=2)

    _update_changesets_metadata(project, changeset, output_dir)


def _update_changesets_metadata(project: str, changeset: dict, output_dir: str):
    """Update changesets/metadata.json index.

    Args:
        project: Project name
        changeset: Changeset dict
        output_dir: Base output directory
    """
    changesets_dir = Path(output_dir) / project / "changesets"
    metadata_file = changesets_dir / "metadata.json"

    if metadata_file.exists():
        with open(metadata_file, "r", encoding="utf-8") as f:
            metadata = json.load(f)
    else:
        metadata = {"changesets": []}

    changeset_entry = {
        "changeSetId": changeset["changeSetId"],
        "beforeSnapshotId": changeset["beforeSnapshotId"],
        "afterSnapshotId": changeset["afterSnapshotId"],
        "timestamp": changeset["timestamp"],
        "total_changes": changeset["summary"]["total_changes"],
        "drivers_count": changeset["summary"].get("drivers", 0),
    }

    # Replace an existing entry on regeneration instead of duplicating it
    metadata["changesets"] = [
        entry for entry in metadata["changesets"] if entry.get("changeSetId") != changeset["changeSetId"]
    ]
    metadata["changesets"].append(changeset_entry)

    with open(metadata_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
