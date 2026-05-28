"""
LangGraph Graph Construction for Knowledge Extraction
LangGraph graph definition for knowledge extraction processing
"""

from langgraph.graph import StateGraph, END
from src.extract.core.state import KnowledgeExtractionState
from src.extract.core.nodes import (
    classify_file_type_node,
    detect_miro_changes_node,
    detect_github_changes_node,
    extract_text_document_node,
    extract_text_presentation_node,
    extract_text_miro_node,
    extract_text_miro_board_node,
    extract_images_miro_board_node,
    extract_text_github_node,
    transform_to_chunks_node,
    extract_images_node,
    merge_chunks_node,
    extract_views_node,
    extract_classes_node,
    extract_properties_node,
    cleanup_deleted_items_node,
    save_accumulated_knowledge_node,
    generate_outputs_node,
    route_by_file_type
)


def create_knowledge_extraction_graph():
    """
    Build the LangGraph for knowledge extraction processing

    Graph structure:
    START
      ↓
    classify_file_type
      ↓ (conditional routing)
      ├─→ extract_text_document ─┐
      ├─→ extract_text_presentation ─┤
      ├─→ extract_text_miro_board (Miro PDF+API) → extract_images_miro_board ─┤
      ├─→ detect_miro_changes (Miro URL-only) → extract_text_miro ─┤
      └─→ detect_github_changes (GitHub only) → extract_text_github ─┘
              ↓                     ↘
         transform_to_chunks     extract_images (parallel - image-to-text)
              ↓                     ↙
              └─→ merge_chunks ←───┘
                       ↓
                  extract_views
                       ↓
                  extract_classes
                       ↓
                  extract_properties
                       ↓
                  cleanup_deleted_items (incremental mode only)
                       ↓
                  save_accumulated_knowledge
                       ↓
                  generate_outputs
                       ↓
                      END
    """

    # Create StateGraph
    workflow = StateGraph(KnowledgeExtractionState)

    # ========== Add Nodes ==========
    workflow.add_node("classify_file_type", classify_file_type_node)
    workflow.add_node("detect_miro_changes", detect_miro_changes_node)
    workflow.add_node("detect_github_changes", detect_github_changes_node)
    workflow.add_node("extract_text_document", extract_text_document_node)
    workflow.add_node("extract_text_presentation", extract_text_presentation_node)
    workflow.add_node("extract_text_miro", extract_text_miro_node)
    workflow.add_node("extract_text_miro_board", extract_text_miro_board_node)
    workflow.add_node("extract_images_miro_board", extract_images_miro_board_node)
    workflow.add_node("extract_text_github", extract_text_github_node)
    workflow.add_node("transform_to_chunks", transform_to_chunks_node)
    workflow.add_node("extract_images", extract_images_node)
    workflow.add_node("merge_chunks", merge_chunks_node)
    workflow.add_node("extract_views", extract_views_node)
    workflow.add_node("extract_classes", extract_classes_node)
    workflow.add_node("extract_properties", extract_properties_node)
    workflow.add_node("cleanup_deleted_items", cleanup_deleted_items_node)
    workflow.add_node("save_accumulated_knowledge", save_accumulated_knowledge_node)
    workflow.add_node("generate_outputs", generate_outputs_node)

    # ========== Set Entry Point ==========
    workflow.set_entry_point("classify_file_type")

    # ========== Add Conditional Edges ==========
    # Select text extraction node based on file type
    # For Miro board with PDF+URL, route directly to new path
    # For old Miro URL-only mode, route to change detection first
    def route_after_classify(state):
        file_type = state.get("file_type")
        if file_type == "github_repo":
            return "detect_github_changes"
        elif file_type == "miro_board":
            # Both PDF+API and URL-only modes go through change detection
            return "detect_miro_changes"
        elif file_type == "document":
            return "extract_text_document"
        elif file_type == "presentation":
            return "extract_text_presentation"
        else:
            return "extract_text_document"  # default

    workflow.add_conditional_edges(
        "classify_file_type",
        route_after_classify,
        {
            "detect_github_changes": "detect_github_changes",
            "detect_miro_changes": "detect_miro_changes",
            "extract_text_miro_board": "extract_text_miro_board",
            "extract_text_document": "extract_text_document",
            "extract_text_presentation": "extract_text_presentation",
        }
    )

    # detect_github_changes → extract_text_github
    workflow.add_edge("detect_github_changes", "extract_text_github")

    # detect_miro_changes → route based on mode (PDF+URL vs URL-only)
    def route_after_miro_changes(state):
        miro_board_url = state.get("miro_board_url")
        if miro_board_url:
            # PDF + URL mode: proceed to PDF+API extraction
            return "extract_text_miro_board"
        else:
            # URL-only mode: use old extraction path
            return "extract_text_miro"

    workflow.add_conditional_edges(
        "detect_miro_changes",
        route_after_miro_changes,
        {
            "extract_text_miro_board": "extract_text_miro_board",
            "extract_text_miro": "extract_text_miro",
        }
    )

    # New miro_board path: extract_text_miro_board → extract_images_miro_board → transform_to_chunks
    workflow.add_edge("extract_text_miro_board", "extract_images_miro_board")
    workflow.add_edge("extract_images_miro_board", "transform_to_chunks")
    workflow.add_edge("extract_images_miro_board", "merge_chunks")

    # ========== Add Unconditional Edges ==========
    # Text path: extract → chunk
    workflow.add_edge("extract_text_document", "transform_to_chunks")
    workflow.add_edge("extract_text_presentation", "transform_to_chunks")
    workflow.add_edge("extract_text_miro", "transform_to_chunks")
    workflow.add_edge("extract_text_github", "transform_to_chunks")

    # Image path (parallel to text chunking): text extracted → extract images → merge
    # Fan out from both text-extraction nodes so image extraction runs after file_id is known
    workflow.add_edge("extract_text_document", "extract_images")
    workflow.add_edge("extract_text_presentation", "extract_images")
    workflow.add_edge("extract_text_miro", "extract_images")
    workflow.add_edge("extract_text_github", "extract_images")

    # Both paths converge at merge_chunks
    workflow.add_edge("transform_to_chunks", "merge_chunks")
    workflow.add_edge("extract_images", "merge_chunks")

    # Merged chunks → View extraction → Class extraction → Property extraction → Cleanup deleted → Accumulated save → Output generation
    workflow.add_edge("merge_chunks", "extract_views")
    workflow.add_edge("extract_views", "extract_classes")
    workflow.add_edge("extract_classes", "extract_properties")
    workflow.add_edge("extract_properties", "cleanup_deleted_items")
    workflow.add_edge("cleanup_deleted_items", "save_accumulated_knowledge")
    workflow.add_edge("save_accumulated_knowledge", "generate_outputs")

    # After output generation, end
    workflow.add_edge("generate_outputs", END)

    # ========== Compile Graph ==========
    app = workflow.compile()

    return app
