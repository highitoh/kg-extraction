"""
LangGraph Node Functions for Knowledge Extraction
LangGraph node function definitions for knowledge extraction processing
"""

import os
import json
import importlib
from typing import Dict, Any

from src.core.state import KnowledgeExtractionState
from src.classification.file_type_classifier import FileTypeClassifier
from src.text.doc_text_chain import DocTextChain
from src.text.presentation_text_chain import PresentationTextChain
from src.view.view_chain import create_view_chain
# Note: 'class' is a reserved keyword, so we use importlib for this import
_class_chain_module = importlib.import_module('src.class.class_chain')
create_class_chain = _class_chain_module.create_class_chain
from src.property.property_chain import create_property_chain
from src.output.turtle_generator import TurtleGenerator
from src.output.neo4j_csv_generator import Neo4jCSVGenerator
from src.utils.chunk_creator import ChunkCreator


def _load_metamodel() -> Dict[str, Any]:
    """Load metamodel file"""
    # Go up 2 levels from src/core/ to /workspace/app/
    app_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    metamodel_path = os.path.join(app_dir, "metamodel", "metamodel.json")
    with open(metamodel_path, "r", encoding="utf-8") as f:
        return json.load(f)


# ========== Node Function Definitions ==========

def classify_file_type_node(state: KnowledgeExtractionState) -> KnowledgeExtractionState:
    """
    Node that determines file type
    files -> file_type, file_path, metamodel
    """
    try:
        classifier = FileTypeClassifier()
        result = classifier.invoke({"files": state["files"]})

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
    file_path -> sentences, id, file_id, file_name
    """
    try:
        doc_chain = DocTextChain(model=state.get("model"))

        # Match DocTextChain input format
        input_data = {
            "files": state["files"]
        }

        result = doc_chain.invoke(input_data)

        # Store results in State
        state["sentences"] = result["sentences"]
        state["id"] = result["id"]
        state["file_id"] = result.get("file_id", result["id"])
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
    file_path -> sentences, id, file_id, file_name
    """
    try:
        presentation_chain = PresentationTextChain(model=state.get("model"))

        # Match PresentationTextChain input format
        input_data = {
            "files": state["files"]
        }

        result = presentation_chain.invoke(input_data)

        # Store results in State
        state["sentences"] = result["sentences"]
        state["id"] = result["id"]
        state["file_id"] = result.get("file_id", result["id"])
        state["file_name"] = result["file_name"]
        state["current_stage"] = "text_extracted_presentation"

    except Exception as e:
        state.setdefault("errors", [])
        state["errors"].append(f"Presentation text extraction failed: {e}")
        state["sentences"] = []
        state["current_stage"] = "text_extraction_failed"

    return state


def transform_to_chunks_node(state: KnowledgeExtractionState) -> KnowledgeExtractionState:
    """
    Node that splits extracted text into chunks
    sentences -> chunks
    """
    try:
        # Select chunk mode based on file type
        if state["file_type"] == "presentation":
            chunk_creator = ChunkCreator(chunk_mode="page")
            combined_text = "\n".join([s["text"] for s in state["sentences"]])
        else:
            chunk_creator = ChunkCreator(chunk_mode="char")
            combined_text = " ".join([s["text"] for s in state["sentences"]])

        # Split into chunks
        chunk_texts = chunk_creator.create(combined_text)
        state["chunks"] = [{"text": chunk} for chunk in chunk_texts]
        state["current_stage"] = "chunks_created"

    except Exception as e:
        state.setdefault("errors", [])
        state["errors"].append(f"Chunk creation failed: {e}")
        state["chunks"] = []
        state["current_stage"] = "chunk_creation_failed"

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
            "target": {
                "id": state["id"],
                "file_id": state["file_id"],
                "file_name": state["file_name"],
                "chunks": state["chunks"]
            },
            "metamodel": state["metamodel"]
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
                    existing_classes = [
                        c for c in all_existing_classes
                        if c.get("judgment") == "ACCEPT"
                    ]
                    if existing_classes:
                        print(f"[ExtractClassesNode] Loaded {len(existing_classes)} existing ACCEPT'ed classes from previous extraction")
            except (json.JSONDecodeError, KeyError) as e:
                print(f"[ExtractClassesNode] Warning: Could not load existing classes from {extraction_result_path}: {e}")
                existing_classes = []

        class_chain = create_class_chain(model=state.get("model"))

        # Match ClassChain input format
        # Use ViewChain output directly as view_info
        # Pass existing classes as well
        input_data = {
            "view_info": {
                "id": state["id"],
                "views": state["views"]
            },
            "metamodel": state["metamodel"],
            "existing_classes": existing_classes  # Add existing classes
        }

        result = class_chain.invoke(input_data)

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
                    existing_properties = [
                        p for p in all_existing_properties
                        if p.get("strength", 0) > 0
                    ]
                    if existing_properties:
                        print(f"[ExtractPropertiesNode] Loaded {len(existing_properties)} existing properties (strength > 0) from previous extraction")
            except (json.JSONDecodeError, KeyError) as e:
                print(f"[ExtractPropertiesNode] Warning: Could not load existing properties from {extraction_result_path}: {e}")
                existing_properties = []

        property_chain = create_property_chain(
            model=state.get("model"),
            existing_properties=existing_properties
        )

        # Match PropertyChain input format
        input_data = {
            "class_info": {
                "id": state["id"],
                "classes": state["classes"]
            },
            "metamodel": state["metamodel"]
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
            state.get("classes", []),
            state.get("properties", [])
        )

        # Load existing knowledge.json if it exists
        existing_knowledge = {"id": state.get("id", ""), "classes": [], "properties": []}
        if os.path.exists(knowledge_path):
            try:
                with open(knowledge_path, "r", encoding="utf-8") as f:
                    existing_knowledge = json.load(f)
            except (json.JSONDecodeError, KeyError) as e:
                print(f"[Warning] Could not load existing knowledge.json: {e}")

        # Merge classes (deduplicate/replace based on semantic key)
        # Keep existing classes to record UUID mapping when UUIDs change
        current_class_map = {
            (c.get("class_iri"), c.get("label")): c
            for c in current_classes
        }
        existing_class_map = {
            (c.get("class_iri"), c.get("label")): c
            for c in existing_knowledge.get("classes", [])
        }

        # UUID mapping: old_uuid -> new_uuid (used when replacing classes)
        uuid_map = {}
        for key, new_class in current_class_map.items():
            old_class = existing_class_map.get(key)
            if old_class and old_class.get("id") != new_class.get("id"):
                # Same semantic key but different UUID → UUID has changed
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
        existing_properties = remap_property_ids(
            existing_knowledge.get("properties", []),
            uuid_map
        )
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
        accumulated_knowledge = {
            "id": state.get("id", ""),
            "classes": merged_classes,
            "properties": merged_properties
        }

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
                state["classes"],
                state["properties"]
            )

        # TurtleGenerator
        turtle_generator = TurtleGenerator(output_dir=project_output_dir)
        turtle_input = {
            "class_info": {
                "id": state["id"],
                "classes": accumulated_classes
            },
            "property_info": {
                "id": state["id"],
                "properties": accumulated_properties
            }
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
            "files_generated": [turtle_path, csv_result["nodes_path"], csv_result["properties_path"]]
        }

        state["current_stage"] = "outputs_generated"

    except Exception as e:
        state.setdefault("errors", [])
        state["errors"].append(f"Output generation failed: {e}")
        state["current_stage"] = "output_generation_failed"

    return state


# ========== Routing Functions ==========

def route_by_file_type(state: KnowledgeExtractionState) -> str:
    """
    Routing function that determines the next node based on file type
    """
    if state.get("file_type") == "presentation":
        return "extract_text_presentation"
    else:
        return "extract_text_document"
