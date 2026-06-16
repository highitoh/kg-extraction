"""
LangGraph Node Functions for Knowledge Extraction
LangGraph node function definitions for knowledge extraction processing
"""

import importlib
import json
import os
from typing import Any, Dict, List, Set, Tuple

from src.extract.classification.file_type_classifier import FileTypeClassifier
from src.extract.core.state import KnowledgeExtractionState
from src.extract.image.image_extractor import ImageExtractor
from src.extract.output.neo4j_csv_generator import Neo4jCSVGenerator
from src.extract.output.turtle_generator import TurtleGenerator
from src.extract.property.property_chain import create_property_chain
from src.extract.text.doc_text_chain import DocTextChain
from src.extract.text.presentation_text_chain import PresentationTextChain
from src.extract.view.view_chain import create_view_chain
from src.utils.chunk_creator import ChunkCreator
from src.utils.mermaid_parser import build_extracted_structure
from src.utils.table_structure import build_table_edges

# Note: 'class' is a reserved keyword, so we use importlib for this import
_class_chain_module = importlib.import_module("src.extract.class.class_chain")
create_class_chain = _class_chain_module.create_class_chain


def _load_metamodel() -> Dict[str, Any]:
    """Load metamodel file"""
    # Go up 3 levels from src/extract/core/ to /workspace/app/
    app_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    metamodel_path = os.path.join(app_dir, "metamodel", "metamodel.json")
    with open(metamodel_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _detect_miro_item_changes(
    prev_raw: Dict[str, Any], curr_raw: Dict[str, Any]
) -> Tuple[Set[str], Set[str], Set[str]]:
    """Detect changes in Miro board items by comparing snapshots.

    Args:
        prev_raw: Previous snapshot raw data (with "items" key)
        curr_raw: Current snapshot raw data (with "items" key)

    Returns:
        Tuple of (new_ids, modified_ids, deleted_ids)
    """
    prev_items = {item["id"]: item for item in prev_raw.get("items", [])}
    curr_items = {item["id"]: item for item in curr_raw.get("items", [])}

    prev_ids = set(prev_items.keys())
    curr_ids = set(curr_items.keys())

    new_ids = curr_ids - prev_ids
    deleted_ids = prev_ids - curr_ids
    common_ids = prev_ids & curr_ids

    # Find modified items (compare content/title)
    modified_ids = set()
    for item_id in common_ids:
        prev_item = prev_items[item_id]
        curr_item = curr_items[item_id]

        prev_content = prev_item.get("data", {}).get("content", "")
        curr_content = curr_item.get("data", {}).get("content", "")
        prev_title = prev_item.get("data", {}).get("title", "")
        curr_title = curr_item.get("data", {}).get("title", "")

        if prev_content != curr_content or prev_title != curr_title:
            modified_ids.add(item_id)

    return new_ids, modified_ids, deleted_ids


def _create_chunks_by_source(sentences: List[Dict[str, Any]], file_type: str) -> List[str]:
    """Create chunks by grouping sentences by source (page/issue/file).

    For Confluence, Jira, and GitHub extractions, sentences are grouped by their
    source identifier (page_id, issue_id, or file_path) to ensure that content
    from different sources is not mixed in the same chunk.

    If a source's content is large, it will be split into multiple chunks
    using ChunkCreator (with overlap to preserve context).

    Args:
        sentences: List of sentence dicts with metadata
        file_type: File type (confluence_space, jira_project, github_repo)

    Returns:
        List of chunk texts (one or more chunks per source)
    """
    from collections import defaultdict

    # Determine metadata key based on file type
    if file_type == "confluence_space":
        key_name = "page_id"
    elif file_type == "jira_project":
        key_name = "issue_id"
    elif file_type == "github_repo":
        key_name = "file_path"
    else:
        key_name = "page_id"

    # Group sentences by source
    source_groups = defaultdict(list)
    for sentence in sentences:
        metadata = sentence.get("metadata", {})
        source_id = metadata.get(key_name, "unknown")
        source_groups[source_id].append(sentence["text"])

    # Create chunks for each source using ChunkCreator
    chunk_creator = ChunkCreator(chunk_mode="char", max_chars=2000, overlap_chars=100)
    chunks = []

    for source_id in sorted(source_groups.keys()):
        source_sentences = source_groups[source_id]
        # Combine sentences with space (same as standard char mode)
        source_text = " ".join(source_sentences)
        # Split using ChunkCreator (handles long text automatically)
        source_chunks = chunk_creator.create(source_text)
        chunks.extend(source_chunks)

    return chunks


# ========== Node Function Definitions ==========


def classify_file_type_node(state: KnowledgeExtractionState) -> KnowledgeExtractionState:
    """
    Node that determines file type
    files -> file_type, file_path, metamodel
    """
    try:
        classifier = FileTypeClassifier(model=state.get("model"))
        classifier_input = {"files": state["files"]}
        if state.get("miro_board_url"):
            classifier_input["miro_board_url"] = state["miro_board_url"]
        result = classifier.invoke(classifier_input)

        # Get file_type and file_path
        state["file_type"] = result.get("file_type", "document")
        state["file_path"] = state["files"][0]["path"]

        # Load metamodel
        state["metamodel"] = _load_metamodel()

        # Initialize error list and stage information
        state.setdefault("errors", [])
        state["current_stage"] = "file_type_classified"

    except Exception as e:
        state.setdefault("errors", [])
        state["errors"].append(f"File type classification failed: {e}")
        state["file_type"] = "document"  # Fallback to default
        state["current_stage"] = "file_type_classification_failed"

    return state


def extract_text_document_node(state: KnowledgeExtractionState) -> KnowledgeExtractionState:
    """
    Node that extracts text from documents
    file_path -> sentences, file_id, file_name
    """
    try:
        doc_chain = DocTextChain(model=state.get("model"))

        # Match DocTextChain input format
        input_data = {"files": state["files"]}

        result = doc_chain.invoke(input_data)

        # Store results in State
        state["sentences"] = result["sentences"]
        state["file_id"] = result["file_id"]
        state["file_name"] = result["file_name"]
        state["current_stage"] = "text_extracted_document"

    except Exception as e:
        state.setdefault("errors", [])
        state["errors"].append(f"Document text extraction failed: {e}")
        state["sentences"] = []
        state["current_stage"] = "text_extraction_failed"

    return state


def extract_text_presentation_node(state: KnowledgeExtractionState) -> KnowledgeExtractionState:
    """
    Node that extracts text from presentations
    file_path -> sentences, file_id, file_name
    """
    try:
        presentation_chain = PresentationTextChain(model=state.get("model"))

        # Match PresentationTextChain input format
        input_data = {"files": state["files"]}

        result = presentation_chain.invoke(input_data)

        # Store results in State
        state["sentences"] = result["sentences"]
        state["file_id"] = result["file_id"]
        state["file_name"] = result["file_name"]
        state["current_stage"] = "text_extracted_presentation"

    except Exception as e:
        state.setdefault("errors", [])
        state["errors"].append(f"Presentation text extraction failed: {e}")
        state["sentences"] = []
        state["current_stage"] = "text_extraction_failed"

    return state


def detect_miro_changes_node(state: KnowledgeExtractionState) -> KnowledgeExtractionState:
    """
    Node that detects changes in Miro board.
    Sets incremental mode and identifies changed/deleted items.
    Supports both URL-only mode and PDF+URL mode.
    """
    try:
        from src.utils.snapshot_repository import SnapshotRepository

        # Get board ID from either PDF+URL mode or URL-only mode
        miro_board_url = state.get("miro_board_url")
        if miro_board_url:
            # PDF+URL mode: board URL is in state
            from src.extract.miro.miro_board_text_extractor import parse_board_id_from_url

            board_id = parse_board_id_from_url(miro_board_url)
        else:
            # URL-only mode: URL is in files path
            from src.extract.miro.miro_text_extractor import parse_board_id_from_url

            miro_url = state["files"][0]["path"]
            board_id = parse_board_id_from_url(miro_url)

        project = state.get("project", "default")
        output_dir = state.get("output_dir", "/workspace/app/output")

        # Find previous snapshot for this board
        repo = SnapshotRepository(output_dir=output_dir)
        prev_snapshot_id = repo.find_latest_for_board(board_id, project)

        if not prev_snapshot_id:
            print("[DetectMiroChanges] No previous snapshot found, doing full extraction")
            state["incremental_mode"] = False
            return state

        print(f"[DetectMiroChanges] Found previous snapshot: {prev_snapshot_id}")
        state["previous_snapshot_id"] = prev_snapshot_id
        state["incremental_mode"] = True
        state["current_stage"] = "miro_changes_detected"

    except Exception as e:
        print(f"[DetectMiroChanges] Error: {e}, falling back to full extraction")
        state["incremental_mode"] = False
        state["current_stage"] = "miro_change_detection_failed"

    return state


def extract_text_miro_node(state: KnowledgeExtractionState) -> KnowledgeExtractionState:
    """
    Node that extracts text from Miro board.
    Supports incremental mode (only extract changed items).
    miro_url -> text_chunks, miro_raw_data
    """
    try:
        from src.extract.miro.miro_text_extractor import extract_text_from_miro_board
        from src.utils.source_change_detector import load_source_raw

        miro_url = state["files"][0]["path"]
        chunks = extract_text_from_miro_board(miro_url)

        # Extract miro_raw_data from first chunk
        miro_raw_data = chunks[0].pop("_miro_raw_data", None)

        # Apply incremental filtering if enabled
        if state.get("incremental_mode"):
            prev_snapshot_id = state.get("previous_snapshot_id")
            project = state.get("project", "default")
            output_dir = state.get("output_dir", "/workspace/app/output")

            # Load previous snapshot data
            prev_raw = load_source_raw(project, prev_snapshot_id, output_dir)

            if prev_raw and miro_raw_data:
                # Detect changes using helper function
                new_ids, modified_ids, deleted_ids = _detect_miro_item_changes(prev_raw, miro_raw_data)
                changed_ids = new_ids | modified_ids

                # Filter chunks to only changed items
                filtered_chunks = [chunk for chunk in chunks if chunk.get("metadata", {}).get("item_id") in changed_ids]

                print(
                    f"[ExtractTextMiroNode] Incremental mode: "
                    f"new={len(new_ids)}, modified={len(modified_ids)}, "
                    f"deleted={len(deleted_ids)}, processing={len(filtered_chunks)} chunks"
                )

                # Store deleted/changed item IDs for cleanup and merging
                state["deleted_item_ids"] = list(deleted_ids)
                state["changed_item_ids"] = list(changed_ids)

                chunks = filtered_chunks
            else:
                print("[ExtractTextMiroNode] Could not load previous data, using full extraction")
                state["incremental_mode"] = False

        # Convert chunks to sentences format expected by downstream nodes
        sentences = []

        for chunk in chunks:
            if chunk.get("text"):
                sentences.append(
                    {
                        "text": chunk["text"],
                        "metadata": chunk.get("metadata", {}),
                        "file_id": chunk.get("file_id"),  # Preserve item-level file_id
                    }
                )

        # Store results in State
        state["sentences"] = sentences
        state["file_id"] = chunks[0].get("file_id", "") if chunks else ""
        state["file_name"] = miro_raw_data.get("name", "Miro Board") if miro_raw_data else "Miro Board"
        state["miro_raw_data"] = miro_raw_data
        state["current_stage"] = "text_extracted_miro"

        print(f"[ExtractTextMiroNode] Extracted {len(sentences)} items from Miro board")

    except Exception as e:
        state.setdefault("errors", [])
        state["errors"].append(f"Miro text extraction failed: {e}")
        state["sentences"] = []
        state["current_stage"] = "text_extraction_failed"

    return state


def detect_github_changes_node(state: KnowledgeExtractionState) -> KnowledgeExtractionState:
    """
    Node that detects changes in GitHub repository.
    Sets incremental mode and identifies changed/deleted files.
    Only runs for GitHub file_type.
    """
    try:
        from src.extract.github.github_url_parser import parse_github_url
        from src.utils.snapshot_repository import SnapshotRepository

        github_url = state["files"][0]["path"]
        parsed = parse_github_url(github_url)
        owner = parsed["owner"]
        repo = parsed["repo"]

        repo_identifier = f"github:{owner}:{repo}"
        project = state.get("project", "default")
        output_dir = state.get("output_dir", "/workspace/app/output")

        # Find previous snapshot for this repository
        repo_obj = SnapshotRepository(output_dir=output_dir)

        # Look for previous snapshot with matching source identifier
        try:
            snapshots = repo_obj.list_snapshots(project)
            prev_snapshot_id = None

            for snapshot in snapshots:
                trigger = snapshot.get("trigger", {})
                source_id = trigger.get("sourceIdentifier", "")
                if source_id == repo_identifier:
                    prev_snapshot_id = snapshot["snapshot_id"]
                    break

            if not prev_snapshot_id:
                print("[DetectGitHubChanges] No previous snapshot found, doing full extraction")
                state["incremental_mode"] = False
                return state

            print(f"[DetectGitHubChanges] Found previous snapshot: {prev_snapshot_id}")
            state["previous_snapshot_id"] = prev_snapshot_id
            state["incremental_mode"] = True
            state["current_stage"] = "github_changes_detected"

        except FileNotFoundError:
            print("[DetectGitHubChanges] No snapshots directory, doing full extraction")
            state["incremental_mode"] = False

    except Exception as e:
        print(f"[DetectGitHubChanges] Error: {e}, falling back to full extraction")
        state["incremental_mode"] = False
        state["current_stage"] = "github_change_detection_failed"

    return state


def extract_text_github_node(state: KnowledgeExtractionState) -> KnowledgeExtractionState:
    """
    Node that extracts text from GitHub repository.
    Supports incremental mode (only extract changed files).
    github_url -> text_chunks, github_raw_data
    """
    try:
        from src.extract.github.github_text_extractor import extract_text_from_github
        from src.utils.source_change_detector import load_source_raw

        github_url = state["files"][0]["path"]
        date = state.get("date")
        chunks = extract_text_from_github(github_url, date=date)

        # Extract github_raw_data from first chunk
        github_raw_data = chunks[0].pop("_github_raw_data", None) if chunks else None

        # Apply incremental filtering if enabled
        if state.get("incremental_mode") and github_raw_data:
            prev_snapshot_id = state.get("previous_snapshot_id")
            project = state.get("project", "default")
            output_dir = state.get("output_dir", "/workspace/app/output")

            # Load previous snapshot data
            prev_raw = load_source_raw(project, prev_snapshot_id, output_dir)

            if prev_raw and "file_map" in prev_raw:
                prev_file_map = prev_raw.get("file_map", {})
                curr_file_map = github_raw_data.get("file_map", {})

                prev_ids = set(prev_file_map.keys())
                curr_ids = set(curr_file_map.keys())

                new_ids = curr_ids - prev_ids
                deleted_ids = prev_ids - curr_ids

                # Find modified files (check SHA changes)
                modified_ids = set()
                for file_id in prev_ids & curr_ids:
                    prev_sha = prev_file_map[file_id].get("sha", "")
                    curr_sha = curr_file_map[file_id].get("sha", "")
                    if prev_sha != curr_sha:
                        modified_ids.add(file_id)

                changed_ids = new_ids | modified_ids

                # Filter chunks to only changed files
                filtered_chunks = [chunk for chunk in chunks if chunk.get("file_id") in changed_ids]

                print(
                    f"[ExtractTextGitHubNode] Incremental mode: "
                    f"new={len(new_ids)}, modified={len(modified_ids)}, "
                    f"deleted={len(deleted_ids)}, processing={len(filtered_chunks)} chunks"
                )

                # Store deleted/changed file IDs for cleanup and merging
                state["deleted_item_ids"] = list(deleted_ids)
                state["changed_item_ids"] = list(changed_ids)

                chunks = filtered_chunks
            else:
                print("[ExtractTextGitHubNode] Could not load previous data, using full extraction")
                state["incremental_mode"] = False

        # Convert chunks to sentences format expected by downstream nodes
        sentences = []

        for chunk in chunks:
            if chunk.get("text"):
                sentences.append(
                    {
                        "text": chunk["text"],
                        "metadata": chunk.get("metadata", {}),
                        "file_id": chunk.get("file_id"),  # Preserve file-level file_id
                    }
                )

        # Store results in State
        state["sentences"] = sentences
        state["file_id"] = chunks[0].get("file_id", "") if chunks else ""
        state["file_name"] = (
            f"{github_raw_data['owner']}/{github_raw_data['repo']}" if github_raw_data else "GitHub Repository"
        )
        state["github_raw_data"] = github_raw_data
        state["current_stage"] = "text_extracted_github"

        print(f"[ExtractTextGitHubNode] Extracted {len(sentences)} items from GitHub repository")

    except Exception as e:
        import traceback

        traceback.print_exc()
        state.setdefault("errors", [])
        state["errors"].append(f"GitHub text extraction failed: {e}")
        state["sentences"] = []
        state["current_stage"] = "text_extraction_failed"

    return state


def detect_jira_changes_node(state: KnowledgeExtractionState) -> KnowledgeExtractionState:
    """
    Node that detects changes in Jira project.
    Sets incremental mode and identifies changed/deleted issues.
    Only runs for jira_project file_type.
    """
    try:
        from src.extract.jira.jira_url_parser import get_project_identifier, parse_jira_url
        from src.utils.snapshot_repository import SnapshotRepository

        jira_url = state["files"][0]["path"]
        parsed = parse_jira_url(jira_url)
        domain = parsed["domain"]
        project_key = parsed["project_key"]

        project_identifier = get_project_identifier(domain, project_key)
        project = state.get("project", "default")
        output_dir = state.get("output_dir", "/workspace/app/output")

        # Find previous snapshot for this Jira project
        repo = SnapshotRepository(output_dir=output_dir)

        try:
            snapshots = repo.list_snapshots(project)
            prev_snapshot_id = None

            for snapshot in snapshots:
                # Check both old format (trigger.sourceIdentifier) and new format (source)
                trigger = snapshot.get("trigger", {})
                source_id = trigger.get("sourceIdentifier", "") or snapshot.get("source", "")
                if source_id == project_identifier:
                    prev_snapshot_id = snapshot["snapshot_id"]
                    break

            if not prev_snapshot_id:
                print("[DetectJiraChanges] No previous snapshot found, doing full extraction")
                state["incremental_mode"] = False
                return state

            print(f"[DetectJiraChanges] Found previous snapshot: {prev_snapshot_id}")
            state["previous_snapshot_id"] = prev_snapshot_id
            state["incremental_mode"] = True
            state["current_stage"] = "jira_changes_detected"

        except FileNotFoundError:
            print("[DetectJiraChanges] No snapshots directory, doing full extraction")
            state["incremental_mode"] = False

    except Exception as e:
        print(f"[DetectJiraChanges] Error: {e}, falling back to full extraction")
        state["incremental_mode"] = False
        state["current_stage"] = "jira_change_detection_failed"

    return state


def extract_text_jira_node(state: KnowledgeExtractionState) -> KnowledgeExtractionState:
    """
    Node that extracts text from Jira project.
    Supports incremental mode (only extract changed issues).
    jira_url -> text_chunks, jira_raw_data
    """
    try:
        from src.extract.jira.jira_text_extractor import extract_text_from_jira
        from src.utils.source_change_detector import load_source_raw

        jira_url = state["files"][0]["path"]

        # Load previous snapshot data if in incremental mode
        since_watermark = None
        prev_raw = None
        if state.get("incremental_mode"):
            prev_snapshot_id = state.get("previous_snapshot_id")
            project = state.get("project", "default")
            output_dir = state.get("output_dir", "/workspace/app/output")
            prev_raw = load_source_raw(project, prev_snapshot_id, output_dir)

            if prev_raw and prev_raw.get("watermark"):
                since_watermark = prev_raw["watermark"]
                print(f"[ExtractTextJiraNode] Incremental mode: fetching issues updated since {since_watermark}")
            else:
                print("[ExtractTextJiraNode] Previous snapshot has no watermark, falling back to full fetch")

        chunks = extract_text_from_jira(jira_url, since_watermark=since_watermark)

        # Extract jira_raw_data from first chunk
        jira_raw_data = chunks[0].pop("_jira_raw_data", None) if chunks else None

        # Apply incremental filtering if enabled
        if state.get("incremental_mode") and jira_raw_data:
            if prev_raw and "issue_map" in prev_raw:
                prev_issue_map = prev_raw.get("issue_map", {})
                curr_issue_map = jira_raw_data.get("issue_map", {})

                prev_ids = set(prev_issue_map.keys())
                curr_ids = set(curr_issue_map.keys())

                # The incremental fetch only returns issues updated since the watermark,
                # so curr_issue_map contains just the changed/new issues -- it is NOT the
                # full project state. Deletion detection is therefore not possible here
                # (an absent issue may simply be unchanged), and is out of scope for now.
                new_ids = curr_ids - prev_ids
                modified_ids = curr_ids & prev_ids
                changed_ids = new_ids | modified_ids

                # Filter chunks to only changed issues
                filtered_chunks = [
                    chunk for chunk in chunks if chunk.get("metadata", {}).get("issue_key") in changed_ids
                ]

                print(
                    f"[ExtractTextJiraNode] Incremental mode: "
                    f"new={len(new_ids)}, modified={len(modified_ids)}, "
                    f"processing={len(filtered_chunks)} chunks"
                )

                # Store changed issue IDs for downstream merging
                state["changed_item_ids"] = list(changed_ids)

                # Merge the new issue_map over the previous one so the snapshot keeps the
                # full project baseline (unchanged issues are carried forward). The watermark
                # advances to the latest "updated" seen across the merged set.
                merged_issue_map = {**prev_issue_map, **curr_issue_map}
                jira_raw_data["issue_map"] = merged_issue_map
                jira_raw_data["total_issues"] = len(merged_issue_map)
                merged_watermark = max(
                    (entry.get("updated", "") for entry in merged_issue_map.values() if entry.get("updated")),
                    default=jira_raw_data.get("watermark"),
                )
                jira_raw_data["watermark"] = merged_watermark

                chunks = filtered_chunks
            else:
                print("[ExtractTextJiraNode] Could not load previous data, using full extraction")
                state["incremental_mode"] = False
        elif state.get("incremental_mode") and not jira_raw_data and prev_raw:
            # Incremental run that fetched zero changed issues: no new chunks, so
            # jira_raw_data is None. Carry the previous snapshot's baseline forward
            # (issue_map + watermark) so the snapshot chain stays intact for next time.
            jira_raw_data = prev_raw
            state["changed_item_ids"] = []
            print("[ExtractTextJiraNode] Incremental mode: no changes, carrying forward previous baseline")

        # Convert chunks to sentences format expected by downstream nodes
        sentences = []

        for chunk in chunks:
            if chunk.get("text"):
                sentences.append(
                    {
                        "text": chunk["text"],
                        "metadata": chunk.get("metadata", {}),
                        "file_id": chunk.get("file_id"),  # Preserve issue-level file_id
                    }
                )

        # Store results in State
        state["sentences"] = sentences
        state["file_id"] = chunks[0].get("file_id", "") if chunks else ""

        # Extract domain and project_key for file_name
        if jira_raw_data:
            domain = jira_raw_data.get("domain", "unknown")
            project_key = jira_raw_data.get("project_key", "unknown")
            state["file_name"] = f"Jira:{domain}:{project_key}"
        else:
            state["file_name"] = "Jira Project"

        state["jira_raw_data"] = jira_raw_data
        state["current_stage"] = "text_extracted_jira"

        print(f"[ExtractTextJiraNode] Extracted {len(sentences)} issues from Jira project")

    except Exception as e:
        import traceback

        traceback.print_exc()
        state.setdefault("errors", [])
        state["errors"].append(f"Jira text extraction failed: {e}")
        state["sentences"] = []
        state["current_stage"] = "text_extraction_failed"

    return state


def detect_confluence_changes_node(state: KnowledgeExtractionState) -> KnowledgeExtractionState:
    """
    Node that detects changes in Confluence space.
    Sets incremental mode and identifies changed/deleted pages.
    Only runs for confluence_space file_type.
    """
    try:
        from src.extract.confluence.confluence_url_parser import get_space_identifier, parse_confluence_url
        from src.utils.snapshot_repository import SnapshotRepository

        confluence_url = state["files"][0]["path"]
        parsed = parse_confluence_url(confluence_url)
        domain = parsed["domain"]
        space_key = parsed["space_key"]

        space_identifier = get_space_identifier(domain, space_key)
        project = state.get("project", "default")
        output_dir = state.get("output_dir", "/workspace/app/output")

        # Find previous snapshot for this Confluence space
        repo = SnapshotRepository(output_dir=output_dir)

        try:
            snapshots = repo.list_snapshots(project)
            prev_snapshot_id = None

            for snapshot in snapshots:
                # Check both old format (trigger.sourceIdentifier) and new format (source)
                trigger = snapshot.get("trigger", {})
                source_id = trigger.get("sourceIdentifier", "") or snapshot.get("source", "")
                if source_id == space_identifier:
                    prev_snapshot_id = snapshot["snapshot_id"]
                    break

            if not prev_snapshot_id:
                print("[DetectConfluenceChanges] No previous snapshot found, doing full extraction")
                state["incremental_mode"] = False
                return state

            print(f"[DetectConfluenceChanges] Found previous snapshot: {prev_snapshot_id}")
            state["previous_snapshot_id"] = prev_snapshot_id
            state["incremental_mode"] = True
            state["current_stage"] = "confluence_changes_detected"

        except FileNotFoundError:
            print("[DetectConfluenceChanges] No snapshots directory, doing full extraction")
            state["incremental_mode"] = False

    except Exception as e:
        print(f"[DetectConfluenceChanges] Error: {e}, falling back to full extraction")
        state["incremental_mode"] = False
        state["current_stage"] = "confluence_change_detection_failed"

    return state


def extract_text_confluence_node(state: KnowledgeExtractionState) -> KnowledgeExtractionState:
    """
    Node that extracts text from Confluence space.
    Supports incremental mode (only extract changed pages).
    confluence_url -> sentences (line-level), confluence_raw_data
    """
    try:
        from src.extract.confluence.confluence_text_extractor import extract_text_from_confluence
        from src.utils.source_change_detector import load_source_raw

        confluence_url = state["files"][0]["path"]

        # Load previous snapshot data if in incremental mode
        since_watermark = None
        prev_raw = None
        if state.get("incremental_mode"):
            prev_snapshot_id = state.get("previous_snapshot_id")
            project = state.get("project", "default")
            output_dir = state.get("output_dir", "/workspace/app/output")
            prev_raw = load_source_raw(project, prev_snapshot_id, output_dir)

            if prev_raw and prev_raw.get("watermark"):
                since_watermark = prev_raw["watermark"]
                print(f"[ExtractTextConfluenceNode] Incremental mode: fetching pages updated since {since_watermark}")
            else:
                print("[ExtractTextConfluenceNode] Previous snapshot has no watermark, falling back to full fetch")

        sentences = extract_text_from_confluence(confluence_url, since_watermark=since_watermark)

        # Extract confluence_raw_data and structured table blocks from first sentence
        confluence_raw_data = sentences[0].pop("_confluence_raw_data", None) if sentences else None
        structured_blocks = sentences[0].pop("_confluence_structured_blocks", []) if sentences else []

        # Apply incremental filtering if enabled
        if state.get("incremental_mode") and confluence_raw_data:
            if prev_raw and "page_map" in prev_raw:
                prev_page_map = prev_raw.get("page_map", {})
                curr_page_map = confluence_raw_data.get("page_map", {})

                prev_ids = set(prev_page_map.keys())
                curr_ids = set(curr_page_map.keys())

                # The incremental fetch only returns pages updated since the watermark,
                # so curr_page_map contains just the changed/new pages -- it is NOT the
                # full space state. Deletion detection is therefore not possible here
                # (an absent page may simply be unchanged).
                new_ids = curr_ids - prev_ids
                modified_ids = curr_ids & prev_ids
                changed_ids = new_ids | modified_ids

                # Filter sentences to only changed pages (by page_id in metadata)
                filtered_sentences = [
                    sentence for sentence in sentences if sentence.get("metadata", {}).get("page_id") in changed_ids
                ]

                # Filter structured blocks to only changed pages (by scope_ref=page_id)
                structured_blocks = [b for b in structured_blocks if b.get("scope_ref") in changed_ids]

                print(
                    f"[ExtractTextConfluenceNode] Incremental mode: "
                    f"new={len(new_ids)}, modified={len(modified_ids)}, "
                    f"processing={len(filtered_sentences)} sentences from {len(changed_ids)} pages"
                )

                # Store changed page IDs for downstream merging
                state["changed_item_ids"] = list(changed_ids)

                # Merge the new page_map over the previous one so the snapshot keeps the
                # full space baseline (unchanged pages are carried forward). The watermark
                # advances to the latest "updated" seen across the merged set.
                merged_page_map = {**prev_page_map, **curr_page_map}
                confluence_raw_data["page_map"] = merged_page_map
                confluence_raw_data["total_pages"] = len(merged_page_map)
                merged_watermark = max(
                    (entry.get("updated", "") for entry in merged_page_map.values() if entry.get("updated")),
                    default=confluence_raw_data.get("watermark"),
                )
                confluence_raw_data["watermark"] = merged_watermark

                sentences = filtered_sentences
            else:
                print("[ExtractTextConfluenceNode] Could not load previous data, using full extraction")
                state["incremental_mode"] = False
        elif state.get("incremental_mode") and not confluence_raw_data and prev_raw:
            # Incremental run that fetched zero changed pages: no new sentences, so
            # confluence_raw_data is None. Carry the previous snapshot's baseline forward
            # (page_map + watermark) so the snapshot chain stays intact for next time.
            confluence_raw_data = prev_raw
            state["changed_item_ids"] = []
            print("[ExtractTextConfluenceNode] Incremental mode: no changes, carrying forward previous baseline")

        # Store results in State (sentences already in correct format - line-level)
        state["sentences"] = sentences
        state["structured_blocks"] = structured_blocks

        # Generate file_id for space
        if confluence_raw_data:
            domain = confluence_raw_data.get("domain", "unknown")
            space_key = confluence_raw_data.get("space_key", "unknown")
            state["file_id"] = f"confluence:{domain}:{space_key}"
            state["file_name"] = f"{domain}/wiki/spaces/{space_key}"
        else:
            state["file_id"] = "confluence:unknown"
            state["file_name"] = "Confluence Space"

        state["confluence_raw_data"] = confluence_raw_data
        state["current_stage"] = "text_extracted_confluence"

        print(f"[ExtractTextConfluenceNode] Extracted {len(sentences)} sentences from Confluence space")

    except Exception as e:
        import traceback

        traceback.print_exc()
        state.setdefault("errors", [])
        state["errors"].append(f"Confluence text extraction failed: {e}")
        state["sentences"] = []
        state["structured_blocks"] = []
        state["current_stage"] = "text_extraction_failed"

    return state


def transform_to_chunks_node(state: KnowledgeExtractionState) -> KnowledgeExtractionState:
    """
    Node that splits extracted text into chunks
    sentences -> chunks
    """
    try:
        file_type = state["file_type"]
        sentences = state.get("sentences", []) or []

        # For Confluence/Jira/GitHub, group sentences by page/issue/file and create chunks per source
        if file_type in ("confluence_space", "jira_project", "github_repo"):
            chunk_texts = _create_chunks_by_source(sentences, file_type)
            state["chunks"] = [{"text": chunk} for chunk in chunk_texts]
            state["current_stage"] = "chunks_created"

            total_sentences = len(sentences)
            print(
                f"[TransformToChunksNode] Starting chunk creation: "
                f"file_type={file_type} mode=by_source sentences={total_sentences}"
            )

            if chunk_texts:
                chunk_lens = [len(c) for c in chunk_texts]
                avg_len = sum(chunk_lens) / len(chunk_lens)
                print(
                    f"[TransformToChunksNode] done: chunks={len(chunk_texts)} "
                    f"avg_chars={avg_len:.0f} min_chars={min(chunk_lens)} max_chars={max(chunk_lens)}"
                )
            else:
                print("[TransformToChunksNode] done: chunks=0 (no sentences to split)")

        else:
            # Select chunk mode based on file type
            if file_type == "presentation":
                chunk_mode = "page"
                chunk_creator = ChunkCreator(chunk_mode=chunk_mode)
                combined_text = "\n".join([s["text"] for s in sentences])
            else:
                chunk_mode = "char"
                chunk_creator = ChunkCreator(chunk_mode=chunk_mode)
                combined_text = " ".join([s["text"] for s in sentences])

            print(
                f"[TransformToChunksNode] Starting chunk creation: "
                f"file_type={file_type} mode={chunk_mode} sentences={len(sentences)} "
                f"combined_chars={len(combined_text)}"
            )

            # Split into chunks
            chunk_texts = chunk_creator.create(combined_text)
            state["chunks"] = [{"text": chunk} for chunk in chunk_texts]
            state["current_stage"] = "chunks_created"

            if chunk_texts:
                chunk_lens = [len(c) for c in chunk_texts]
                avg_len = sum(chunk_lens) / len(chunk_lens)
                print(
                    f"[TransformToChunksNode] done: chunks={len(chunk_texts)} "
                    f"avg_chars={avg_len:.0f} min_chars={min(chunk_lens)} max_chars={max(chunk_lens)}"
                )
            else:
                print("[TransformToChunksNode] done: chunks=0 (no sentences to split)")

    except Exception as e:
        state.setdefault("errors", [])
        state["errors"].append(f"Chunk creation failed: {e}")
        state["chunks"] = []
        state["current_stage"] = "chunk_creation_failed"

    return state


def extract_images_node(state: KnowledgeExtractionState) -> Dict[str, Any]:
    """
    Node that extracts image descriptions from the PDF and converts them into chunks.
    files, file_id -> image_chunks, image_statistics

    Runs in parallel with the text-extraction/chunking path and converges at
    merge_chunks_node. Returns only the keys this node writes to avoid
    concurrent-update conflicts with the parallel text branch. Applies to both
    document and presentation file types (extracts embedded image XObjects such
    as pasted diagrams; does not re-read rendered page text).
    """
    try:
        extractor = ImageExtractor(model=state.get("model"))

        input_data = {
            "files": state["files"],
            "file_id": state.get("file_id"),
        }

        result = extractor.invoke(input_data)

        image_chunks = result.get("image_chunks", [])
        return {
            "image_chunks": image_chunks,
            "image_statistics": result.get("image_statistics", {}),
            "extracted_structure": build_extracted_structure(image_chunks),
        }

    except Exception as e:
        return {
            "image_chunks": [],
            "image_statistics": {"error": str(e), "failed_message": f"Image extraction failed: {e}"},
            "extracted_structure": [],
        }


def merge_chunks_node(state: KnowledgeExtractionState) -> KnowledgeExtractionState:
    """
    Node that merges text chunks and image chunks into a single unified list.
    chunks + image_chunks -> chunks
    """
    try:
        text_chunks = state.get("chunks", []) or []
        image_chunks = state.get("image_chunks", []) or []

        merged = list(text_chunks) + list(image_chunks)
        state["chunks"] = merged
        state["current_stage"] = "chunks_merged"

        print(
            f"[MergeChunksNode] merged text_chunks={len(text_chunks)} + "
            f"image_chunks={len(image_chunks)} = {len(merged)} total"
        )

    except Exception as e:
        state.setdefault("errors", [])
        state["errors"].append(f"Chunk merge failed: {e}")
        state["current_stage"] = "chunk_merge_failed"

    return state


def extract_views_node(state: KnowledgeExtractionState) -> KnowledgeExtractionState:
    """
    Node that extracts view descriptions
    chunks, metamodel -> views
    """
    try:
        view_chain = create_view_chain(model=state.get("model"))

        # Match ViewChain input format
        input_data = {
            "target": {"file_id": state["file_id"], "file_name": state["file_name"], "chunks": state["chunks"]},
            "metamodel": state["metamodel"],
        }

        result = view_chain.invoke(input_data)

        # Store results in State
        state["views"] = result["views"]
        state["current_stage"] = "views_extracted"

    except Exception as e:
        state.setdefault("errors", [])
        state["errors"].append(f"View extraction failed: {e}")
        state["views"] = []
        state["current_stage"] = "view_extraction_failed"

    return state


def extract_classes_node(state: KnowledgeExtractionState) -> KnowledgeExtractionState:
    """
    Node that extracts classes (including extraction, filtering, and merging)
    views, metamodel -> classes

    If existing knowledge.json exists, load ACCEPT'ed classes
    and merge with newly extracted classes
    """
    try:
        # Load existing classes
        existing_classes = []
        output_dir = state.get("output_dir", "/workspace/app/output")
        project = state.get("project", "default")
        extraction_result_path = os.path.join(output_dir, project, "knowledge.json")

        if os.path.exists(extraction_result_path):
            try:
                with open(extraction_result_path, "r", encoding="utf-8") as f:
                    previous_result = json.load(f)
                    # Load only ACCEPT'ed classes
                    all_existing_classes = previous_result.get("classes", [])
                    existing_classes = []
                    for c in all_existing_classes:
                        if c.get("judgment") != "ACCEPT":
                            continue
                        c_copy = c.copy()
                        # Internal flags for ClassChain to skip re-review
                        # when a carry-over ACCEPT class label remains unchanged.
                        c_copy["_frozen_from_existing_accept"] = True
                        c_copy["_frozen_label"] = c_copy.get("label", "")
                        existing_classes.append(c_copy)
                    if existing_classes:
                        print(
                            f"[ExtractClassesNode] Loaded {len(existing_classes)} existing ACCEPT'ed classes "
                            f"from previous extraction"
                        )
            except (json.JSONDecodeError, KeyError) as e:
                print(
                    f"[ExtractClassesNode] Warning: Could not load existing classes from {extraction_result_path}: {e}"
                )
                existing_classes = []

        class_chain = create_class_chain(model=state.get("model"))

        # Match ClassChain input format
        # Use ViewChain output directly as view_info
        # Pass existing classes as well
        input_data = {
            "view_info": {"views": state["views"]},
            "metamodel": state["metamodel"],
            "existing_classes": existing_classes,  # Add existing classes
            "structured_blocks": state.get("structured_blocks", []),  # Table header candidates (step A)
        }

        result = class_chain.invoke(input_data)

        # Mark newly extracted classes (those NOT from existing_classes)
        # This flag is used by PropertyExtractor to filter class pairs for incremental updates
        existing_ids = {c.get("id") for c in existing_classes if c.get("id")}
        for cls in result["classes"]:
            if cls.get("id") not in existing_ids:
                cls["_is_new"] = True
            else:
                cls["_is_new"] = False

        # Store results in State
        state["classes"] = result["classes"]
        state["current_stage"] = "classes_extracted"

    except Exception as e:
        state.setdefault("errors", [])
        state["errors"].append(f"Class extraction failed: {e}")
        state["classes"] = []
        state["current_stage"] = "class_extraction_failed"

    return state


def extract_properties_node(state: KnowledgeExtractionState) -> KnowledgeExtractionState:
    """
    Node that extracts properties (relationships) (including extraction, filtering, and validation)
    classes, metamodel -> properties

    If existing knowledge.json exists, load properties with strength > 0
    and merge with newly extracted properties
    """
    try:
        # Load existing properties
        existing_properties = []
        output_dir = state.get("output_dir", "/workspace/app/output")
        project = state.get("project", "default")
        extraction_result_path = os.path.join(output_dir, project, "knowledge.json")

        if os.path.exists(extraction_result_path):
            try:
                with open(extraction_result_path, "r", encoding="utf-8") as f:
                    previous_result = json.load(f)
                    # Load only properties with strength > 0
                    all_existing_properties = previous_result.get("properties", [])
                    existing_properties = []
                    for p in all_existing_properties:
                        strength = p.get("strength", 0)
                        if strength <= 0:
                            continue
                        p_copy = p.copy()
                        # Internal flags for PropertyChain to skip re-review
                        # when carry-over properties keep the same strength after ranking.
                        p_copy["_frozen_from_existing_accept"] = True
                        p_copy["_frozen_strength"] = strength
                        existing_properties.append(p_copy)
                    if existing_properties:
                        print(
                            f"[ExtractPropertiesNode] Loaded {len(existing_properties)} existing properties "
                            f"(strength > 0) from previous extraction"
                        )
            except (json.JSONDecodeError, KeyError) as e:
                print(
                    f"[ExtractPropertiesNode] Warning: Could not load existing properties from "
                    f"{extraction_result_path}: {e}"
                )
                existing_properties = []

        property_chain = create_property_chain(model=state.get("model"), existing_properties=existing_properties)

        # Combine diagram-derived structural edges with deterministic table edges
        # (non-empty matrix cells -> row_header x col_header). Both flow through the
        # same structure resolution (label -> class, pair_map for the property type).
        table_edges = build_table_edges(state.get("structured_blocks", []))
        extracted_structure = list(state.get("extracted_structure", []) or []) + table_edges
        if table_edges:
            print(f"[ExtractPropertiesNode] Added {len(table_edges)} table-derived structural edges")

        # Match PropertyChain input format
        input_data = {
            "class_info": {"classes": state["classes"]},
            "metamodel": state["metamodel"],
            "extracted_structure": extracted_structure,
        }

        result = property_chain.invoke(input_data)

        # Store results in State
        state["properties"] = result["properties"]
        state["current_stage"] = "properties_extracted"

    except Exception as e:
        state.setdefault("errors", [])
        state["errors"].append(f"Property extraction failed: {e}")
        state["properties"] = []
        state["current_stage"] = "property_extraction_failed"

    return state


def cleanup_deleted_items_node(state: KnowledgeExtractionState) -> KnowledgeExtractionState:
    """
    Node that removes knowledge from deleted items.
    Runs only in incremental mode when items were deleted.
    Supports Miro, GitHub, and Jira sources.
    """
    try:
        # Only run in incremental mode with deletions
        if not state.get("incremental_mode"):
            return state

        deleted_item_ids = state.get("deleted_item_ids", [])
        if not deleted_item_ids:
            return state

        file_type = state.get("file_type")

        # Build file_ids for deleted items based on source type
        if file_type == "miro_board":
            from src.extract.miro.miro_text_extractor import parse_board_id_from_url

            miro_url = state["files"][0]["path"]
            board_id = parse_board_id_from_url(miro_url)
            deleted_file_ids = {f"miro:{board_id}:{item_id}" for item_id in deleted_item_ids}

        elif file_type == "jira_project":
            from src.extract.jira.jira_url_parser import get_issue_identifier, parse_jira_url

            jira_url = state["files"][0]["path"]
            parsed = parse_jira_url(jira_url)
            domain = parsed["domain"]
            # deleted_item_ids are issue keys like "PROJ-123"
            deleted_file_ids = {get_issue_identifier(domain, issue_key) for issue_key in deleted_item_ids}

        elif file_type == "github_repo":
            # Deleted file_ids already in format "github:owner:repo:path:sha"
            deleted_file_ids = set(deleted_item_ids)

        else:
            print(f"[CleanupDeletedItems] Unsupported file_type: {file_type}, skipping")
            return state

        print(f"[CleanupDeletedItems] Removing knowledge from {len(deleted_file_ids)} deleted items")

        # Load existing knowledge.json
        output_dir = state.get("output_dir", "/workspace/app/output")
        project = state.get("project", "default")
        knowledge_path = os.path.join(output_dir, project, "knowledge.json")

        if not os.path.exists(knowledge_path):
            return state  # Nothing to clean up

        with open(knowledge_path, "r", encoding="utf-8") as f:
            knowledge = json.load(f)

        # Clean up classes: remove deleted file_ids
        cleaned_classes = []
        removed_class_ids = set()

        for cls in knowledge.get("classes", []):
            file_ids = cls.get("file_ids", [])
            sources = cls.get("sources", [])

            # Filter out deleted file_ids (maintain alignment with sources)
            filtered_pairs = [
                (src, fid) for src, fid in zip(sources, file_ids, strict=True) if fid not in deleted_file_ids
            ]

            if filtered_pairs:
                # Class still has sources from non-deleted items
                new_sources, new_file_ids = zip(*filtered_pairs, strict=True)
                cls["sources"] = list(new_sources)
                cls["file_ids"] = list(new_file_ids)
                cleaned_classes.append(cls)
            else:
                # Class has no remaining sources → remove it
                removed_class_ids.add(cls["id"])
                print(f"[CleanupDeletedItems] Removing class: {cls.get('label')} (no sources remain)")

        # Clean up properties: remove those referencing deleted classes
        cleaned_properties = [
            prop
            for prop in knowledge.get("properties", [])
            if prop["src_id"] not in removed_class_ids and prop["dest_id"] not in removed_class_ids
        ]

        # Save cleaned knowledge back
        cleaned_knowledge = {"classes": cleaned_classes, "properties": cleaned_properties}

        with open(knowledge_path, "w", encoding="utf-8") as f:
            json.dump(cleaned_knowledge, f, ensure_ascii=False, indent=2)

        print(
            f"[CleanupDeletedItems] Removed {len(removed_class_ids)} classes, "
            f"{len(knowledge['properties']) - len(cleaned_properties)} properties"
        )

        state["current_stage"] = "deleted_items_cleaned"

    except Exception as e:
        print(f"[CleanupDeletedItems] Warning: Cleanup failed: {e}")
        state.setdefault("errors", [])
        state["errors"].append(f"Deleted item cleanup failed: {e}")

    return state


def save_accumulated_knowledge_node(state: KnowledgeExtractionState) -> KnowledgeExtractionState:
    """
    Node that saves accumulated knowledge to knowledge.json
    Saves only ACCEPT'ed classes and properties with strength > 0

    This node executes before generate_outputs_node,
    making the latest accumulated knowledge available during output generation
    """
    try:
        from src.utils.knowledge_filter import filter_accumulated_knowledge

        output_dir = state.get("output_dir", "/workspace/app/output")
        project = state.get("project", "default")
        project_output_dir = os.path.join(output_dir, project)
        os.makedirs(project_output_dir, exist_ok=True)

        knowledge_path = os.path.join(project_output_dir, "knowledge.json")

        # Extract ACCEPT'ed classes and valid properties from current execution result
        current_classes, current_properties = filter_accumulated_knowledge(
            state.get("classes", []), state.get("properties", [])
        )

        # Load existing knowledge.json if it exists
        existing_knowledge = {"classes": [], "properties": []}
        if os.path.exists(knowledge_path):
            try:
                with open(knowledge_path, "r", encoding="utf-8") as f:
                    existing_knowledge = json.load(f)
            except (json.JSONDecodeError, KeyError) as e:
                print(f"[Warning] Could not load existing knowledge.json: {e}")

        # Merge classes (deduplicate/replace based on semantic key)
        # Keep existing classes to record UUID mapping when UUIDs change
        current_class_map = {(c.get("class_iri"), c.get("label")): c for c in current_classes}
        existing_class_map = {(c.get("class_iri"), c.get("label")): c for c in existing_knowledge.get("classes", [])}

        # UUID mapping: old_uuid -> new_uuid (used when replacing classes)
        uuid_map = {}
        for key, new_class in current_class_map.items():
            old_class = existing_class_map.get(key)
            if old_class:
                # Merge file_ids from both sources
                new_file_ids = new_class.get("file_ids", [])
                old_file_ids = old_class.get("file_ids", [])
                new_sources = new_class.get("sources", [])
                old_sources = old_class.get("sources", [])

                # Combine sources and file_ids
                # Use dict to deduplicate by file_id while preserving source mapping
                source_file_map = {}
                for src, fid in zip(old_sources, old_file_ids, strict=True):
                    source_file_map[fid] = src
                for src, fid in zip(new_sources, new_file_ids, strict=True):
                    source_file_map[fid] = src  # New sources override old if same file_id

                # Rebuild arrays
                merged_file_ids = list(source_file_map.keys())
                merged_sources = [source_file_map[fid] for fid in merged_file_ids]

                new_class["file_ids"] = merged_file_ids
                new_class["sources"] = merged_sources

                # Track UUID changes
                if old_class.get("id") != new_class.get("id"):
                    uuid_map[old_class["id"]] = new_class["id"]

            existing_class_map[key] = new_class
        merged_classes = list(existing_class_map.values())

        # Remap property src_id/dest_id (handle class UUID changes)
        def remap_property_ids(properties, uuid_mapping):
            """Remap property src_id/dest_id to new UUIDs"""
            remapped = []
            for prop in properties:
                p = prop.copy()
                p["src_id"] = uuid_mapping.get(p["src_id"], p["src_id"])
                p["dest_id"] = uuid_mapping.get(p["dest_id"], p["dest_id"])
                remapped.append(p)
            return remapped

        # Remap UUIDs of existing properties
        existing_properties = remap_property_ids(existing_knowledge.get("properties", []), uuid_map)
        # Remap current properties as well (just in case)
        current_properties = remap_property_ids(current_properties, uuid_map)

        # Merge properties (deduplicate by ID, update strength)
        existing_property_map = {p.get("id"): p for p in existing_properties}
        for prop in current_properties:
            prop_id = prop.get("id")
            if prop_id in existing_property_map:
                if prop.get("strength", 0) > existing_property_map[prop_id].get("strength", 0):
                    existing_property_map[prop_id] = prop
            else:
                existing_property_map[prop_id] = prop
        merged_properties = list(existing_property_map.values())

        # Remove orphaned properties (those referencing non-existent classes)
        valid_class_ids = {c["id"] for c in merged_classes}
        orphaned_count = 0
        validated_properties = []
        for prop in merged_properties:
            if prop["src_id"] in valid_class_ids and prop["dest_id"] in valid_class_ids:
                validated_properties.append(prop)
            else:
                orphaned_count += 1
        merged_properties = validated_properties

        if orphaned_count > 0:
            print(f"[Info] Removed {orphaned_count} orphaned properties (referencing non-existent classes)")

        # Save to knowledge.json
        accumulated_knowledge = {"classes": merged_classes, "properties": merged_properties}

        with open(knowledge_path, "w", encoding="utf-8") as f:
            json.dump(accumulated_knowledge, f, ensure_ascii=False, indent=2)

        print(f"[Info] Knowledge accumulated to: {knowledge_path}")
        print(f"[Info] Total classes: {len(merged_classes)}, Total properties: {len(merged_properties)}")

        state["current_stage"] = "knowledge_saved"

    except Exception as e:
        state.setdefault("errors", [])
        state["errors"].append(f"Knowledge save failed: {e}")
        state["current_stage"] = "knowledge_save_failed"

    return state


def generate_outputs_node(state: KnowledgeExtractionState) -> KnowledgeExtractionState:
    """
    Node that generates Turtle file and neo4j CSV files

    Note: Loads accumulated knowledge from knowledge.json to generate output
    Since save_accumulated_knowledge_node executes before this node,
    knowledge.json contains the latest accumulated knowledge
    """
    try:
        from src.utils.knowledge_filter import filter_accumulated_knowledge

        output_dir = state.get("output_dir", "/workspace/app/output")
        project = state.get("project", "default")
        project_output_dir = os.path.join(output_dir, project)
        os.makedirs(project_output_dir, exist_ok=True)

        # Load accumulated knowledge from knowledge.json
        knowledge_path = os.path.join(project_output_dir, "knowledge.json")
        if os.path.exists(knowledge_path):
            with open(knowledge_path, "r", encoding="utf-8") as f:
                knowledge = json.load(f)
            accumulated_classes = knowledge.get("classes", [])
            accumulated_properties = knowledge.get("properties", [])
        else:
            # For first execution (when knowledge.json doesn't exist yet), filter current results
            accumulated_classes, accumulated_properties = filter_accumulated_knowledge(
                state["classes"], state["properties"]
            )

        # TurtleGenerator
        turtle_generator = TurtleGenerator(output_dir=project_output_dir)
        turtle_input = {
            "class_info": {"classes": accumulated_classes},
            "property_info": {"properties": accumulated_properties},
        }
        turtle_path, turtle_result = turtle_generator.generate_and_save(turtle_input)

        # Neo4jCSVGenerator
        csv_generator = Neo4jCSVGenerator(output_dir=project_output_dir)
        csv_result = csv_generator.generate(turtle_input)

        # Store results in State
        state["turtle_file"] = turtle_path
        state["turtle_content"] = turtle_result["turtle"]
        state["nodes_csv"] = csv_result["nodes_path"]
        state["relationships_csv"] = csv_result["properties_path"]

        # Generate summary information
        state["summary"] = {
            "classes_count": len(accumulated_classes),
            "properties_count": len(accumulated_properties),
            "files_generated": [turtle_path, csv_result["nodes_path"], csv_result["properties_path"]],
        }

        state["current_stage"] = "outputs_generated"

    except Exception as e:
        state.setdefault("errors", [])
        state["errors"].append(f"Output generation failed: {e}")
        state["current_stage"] = "output_generation_failed"

    return state


# ========== Routing Functions ==========


def extract_text_miro_board_node(state: KnowledgeExtractionState) -> KnowledgeExtractionState:
    """
    Extract text from Miro board (PDF + API) and match pages to Frames/Diagrams.
    Supports incremental mode: only process changed items.
    """
    try:
        from src.extract.miro.miro_board_text_extractor import extract_text_from_miro_board
        from src.utils.source_change_detector import load_source_raw

        pdf_path = state["files"][0]["path"]
        miro_board_url = state["miro_board_url"]

        result = extract_text_from_miro_board(pdf_path, miro_board_url)

        # Apply incremental filtering if enabled
        if state.get("incremental_mode"):
            prev_snapshot_id = state.get("previous_snapshot_id")
            project = state.get("project", "default")
            output_dir = state.get("output_dir", "/workspace/app/output")

            # Load previous snapshot data
            prev_raw = load_source_raw(project, prev_snapshot_id, output_dir)
            miro_raw_data = result["miro_raw_data"]

            if prev_raw and miro_raw_data:
                print("[ExtractTextMiroBoard] Incremental mode: detecting changes...")

                # Detect changes using helper function
                new_ids, modified_ids, deleted_ids = _detect_miro_item_changes(prev_raw, miro_raw_data)
                changed_ids = new_ids | modified_ids

                print(
                    f"[ExtractTextMiroBoard] Changes detected: {len(new_ids)} new, {len(modified_ids)} modified, "
                    f"{len(deleted_ids)} deleted"
                )

                if changed_ids:
                    # Filter visual_items to only include Frames/Diagrams with changed children
                    changed_visual_items = []
                    for visual_item in result["miro_visual_items"]:
                        # Check if this Frame/Diagram or any of its children changed
                        item_id = visual_item["item_id"]
                        child_ids = {child["id"] for child in visual_item.get("child_items", [])}

                        if item_id in changed_ids or child_ids & changed_ids:
                            changed_visual_items.append(visual_item)

                    result["miro_visual_items"] = changed_visual_items

                    changed_visual_item_ids = {item["item_id"] for item in changed_visual_items}

                    # Filter item_page_mappings to only include changed items
                    result["item_page_mappings"] = [
                        mapping
                        for mapping in result["item_page_mappings"]
                        if mapping["item_id"] in changed_visual_item_ids
                    ]

                    # Filter text chunks to changed items. A chunk's file_id ends
                    # with the id of its container (Frame/Diagram) or, for
                    # standalone Documents/Tables, the item's own id. Keep a chunk
                    # if that id is a changed visual item OR is itself a changed id
                    # (covers standalone items not wrapped in a Frame/Diagram).
                    keep_ids = changed_visual_item_ids | changed_ids

                    def _chunk_belongs_to_changed(sent):
                        chunk_item_id = sent["file_id"].rsplit(":", 1)[-1]
                        return chunk_item_id in keep_ids

                    result["sentences"] = [sent for sent in result["sentences"] if _chunk_belongs_to_changed(sent)]

                    state["changed_item_ids"] = list(changed_ids)
                    state["deleted_item_ids"] = list(deleted_ids)

                    print(f"[ExtractTextMiroBoard] Filtered to {len(result['miro_visual_items'])} changed visual items")
                    print(
                        f"[ExtractTextMiroBoard] Filtered to {len(result['item_page_mappings'])} changed page mappings"
                    )
                else:
                    print("[ExtractTextMiroBoard] No changes detected, skipping extraction")
                    state["incremental_mode"] = False
                    result["sentences"] = []
                    result["miro_visual_items"] = []
                    result["item_page_mappings"] = []
                    state["deleted_item_ids"] = list(deleted_ids) if deleted_ids else []
            else:
                print("[ExtractTextMiroBoard] Could not load previous snapshot, doing full extraction")
                state["incremental_mode"] = False

        # Update state with results
        state["sentences"] = result["sentences"]
        state["miro_visual_items"] = result["miro_visual_items"]
        state["item_page_mappings"] = result["item_page_mappings"]
        state["miro_board_id"] = result["miro_board_id"]
        state["miro_raw_data"] = result["miro_raw_data"]
        state["file_path"] = pdf_path
        state["file_name"] = os.path.basename(pdf_path)
        state["file_id"] = f"miro_board:{result['miro_board_id']}"
        state["current_stage"] = "extract_text_miro_board"

        print(f"[ExtractTextMiroBoard] Extracted {len(result['sentences'])} text chunks")
        print(f"[ExtractTextMiroBoard] Found {len(result['item_page_mappings'])} page-to-item mappings")

        # Save extracted text to log for inspection
        from src.utils.logger import Logger

        logger = Logger("/workspace/app/log/miro_text_extractor")
        logger.save_log(
            {
                "board_id": result["miro_board_id"],
                "sentences": result["sentences"],
                "item_page_mappings": result["item_page_mappings"],
            },
            "miro_extracted_text_",
        )

    except Exception as e:
        print(f"[ExtractTextMiroBoard] Error: {e}")
        import traceback

        traceback.print_exc()
        state["errors"].append(f"ExtractTextMiroBoard: {e}")
        state["current_stage"] = "extract_text_miro_board_failed"

    return state


def extract_images_miro_board_node(state: KnowledgeExtractionState) -> KnowledgeExtractionState:
    """
    Extract images from ALL PDF pages matched to Frames/Diagrams (no screening).
    Converts PDF pages to images and sends to LLM for analysis.
    """
    try:
        from src.extract.miro.miro_board_image_extractor import MiroBoardImageExtractor

        model = state.get("model")
        extractor = MiroBoardImageExtractor(model=model) if model else MiroBoardImageExtractor()

        result = extractor.invoke(
            {
                "pdf_path": state["file_path"],
                "item_page_mappings": state["item_page_mappings"],
                "file_id": state["file_id"],
            }
        )

        state["image_chunks"] = result["image_chunks"]
        state["image_statistics"] = result["image_statistics"]
        state["extracted_structure"] = build_extracted_structure(result["image_chunks"])
        state["current_stage"] = "extract_images_miro_board"

        print(f"[ExtractImagesMiroBoard] Generated {len(result['image_chunks'])} image chunks")
        print(f"[ExtractImagesMiroBoard] Parsed {len(state['extracted_structure'])} structural edges")

    except Exception as e:
        print(f"[ExtractImagesMiroBoard] Error: {e}")
        import traceback

        traceback.print_exc()
        state["errors"].append(f"ExtractImagesMiroBoard: {e}")
        state["current_stage"] = "extract_images_miro_board_failed"

    return state


def route_by_file_type(state: KnowledgeExtractionState) -> str:
    """
    Routing function that determines the next node based on file type
    """
    file_type = state.get("file_type")
    if file_type == "presentation":
        return "extract_text_presentation"
    elif file_type == "miro_board":
        return "extract_text_miro"
    else:
        return "extract_text_document"
