"""
LangGraph Graph Construction for Knowledge Extraction
LangGraph graph definition for knowledge extraction processing
"""

from langgraph.graph import END, StateGraph

from src.extract.core.nodes import (
    classify_file_type_node,
    detect_confluence_changes_node,
    detect_github_changes_node,
    detect_jira_changes_node,
    detect_miro_changes_node,
    extract_classes_node,
    extract_images_miro_board_node,
    extract_images_miro_node,
    extract_images_node,
    extract_properties_node,
    extract_text_confluence_node,
    extract_text_document_node,
    extract_text_docx_node,
    extract_text_github_node,
    extract_text_jira_node,
    extract_text_md_node,
    extract_text_miro_board_node,
    extract_text_miro_node,
    extract_text_pptx_node,
    extract_text_presentation_node,
    extract_text_txt_node,
    extract_views_node,
    generate_outputs_node,
    merge_chunks_node,
    save_accumulated_knowledge_node,
    screen_relevance_node,
    transform_to_chunks_node,
)
from src.extract.core.state import KnowledgeExtractionState


def route_after_properties(state: KnowledgeExtractionState) -> str:
    """Stop a failed run before it touches knowledge.json, so no snapshot or changeset is recorded."""
    return END if state.get("errors") else "save_accumulated_knowledge"


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
              ↓                       or extract_images_miro (Miro URL-only)
              └─→ merge_chunks ←───┘
                       ↓
                  screen_relevance (passes through without a project profile)
                       ↓
                  extract_views
                       ↓
                  extract_classes
                       ↓
                  extract_properties
                       ↓ (END if any stage recorded an error)
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
    workflow.add_node("detect_jira_changes", detect_jira_changes_node)
    workflow.add_node("detect_confluence_changes", detect_confluence_changes_node)
    workflow.add_node("extract_text_docx", extract_text_docx_node)
    workflow.add_node("extract_text_pptx", extract_text_pptx_node)
    workflow.add_node("extract_text_txt", extract_text_txt_node)
    workflow.add_node("extract_text_md", extract_text_md_node)
    workflow.add_node("extract_text_document", extract_text_document_node)
    workflow.add_node("extract_text_presentation", extract_text_presentation_node)
    workflow.add_node("extract_text_miro", extract_text_miro_node)
    workflow.add_node("extract_text_miro_board", extract_text_miro_board_node)
    workflow.add_node("extract_images_miro_board", extract_images_miro_board_node)
    workflow.add_node("extract_text_github", extract_text_github_node)
    workflow.add_node("extract_text_jira", extract_text_jira_node)
    workflow.add_node("extract_text_confluence", extract_text_confluence_node)
    workflow.add_node("transform_to_chunks", transform_to_chunks_node)
    workflow.add_node("extract_images", extract_images_node)
    workflow.add_node("extract_images_miro", extract_images_miro_node)
    workflow.add_node("merge_chunks", merge_chunks_node)
    workflow.add_node("screen_relevance", screen_relevance_node)
    workflow.add_node("extract_views", extract_views_node)
    workflow.add_node("extract_classes", extract_classes_node)
    workflow.add_node("extract_properties", extract_properties_node)
    workflow.add_node("save_accumulated_knowledge", save_accumulated_knowledge_node)
    workflow.add_node("generate_outputs", generate_outputs_node)

    # ========== Set Entry Point ==========
    workflow.set_entry_point("classify_file_type")

    # ========== Add Conditional Edges ==========
    # Select text extraction node based on source_type and content_type
    def route_after_classify(state):
        """
        Route based on source_type and content_type.

        Priority:
        1. Check source_type for API sources (GitHub, Jira, Confluence, Miro)
        2. For pdf_file source, route by content_type (document vs presentation)
        """
        source_type = state.get("source_type")
        content_type = state.get("content_type")

        # Route API sources by source_type
        if source_type == "github_repo":
            return "detect_github_changes"
        elif source_type == "jira_project":
            return "detect_jira_changes"
        elif source_type == "confluence_space":
            return "detect_confluence_changes"
        elif source_type == "miro_board":
            # Both PDF+API and URL-only modes go through change detection
            return "detect_miro_changes"
        elif source_type == "docx_file":
            return "extract_text_docx"
        elif source_type == "pptx_file":
            return "extract_text_pptx"
        elif source_type == "txt_file":
            return "extract_text_txt"
        elif source_type == "md_file":
            return "extract_text_md"

        # Route PDF files by content_type
        elif source_type == "pdf_file":
            if content_type == "presentation":
                return "extract_text_presentation"
            else:  # content_type == "document" or None
                return "extract_text_document"

        # Default fallback
        else:
            return "extract_text_document"

    workflow.add_conditional_edges(
        "classify_file_type",
        route_after_classify,
        {
            "detect_github_changes": "detect_github_changes",
            "detect_jira_changes": "detect_jira_changes",
            "detect_confluence_changes": "detect_confluence_changes",
            "detect_miro_changes": "detect_miro_changes",
            "extract_text_miro_board": "extract_text_miro_board",
            "extract_text_docx": "extract_text_docx",
            "extract_text_pptx": "extract_text_pptx",
            "extract_text_txt": "extract_text_txt",
            "extract_text_md": "extract_text_md",
            "extract_text_document": "extract_text_document",
            "extract_text_presentation": "extract_text_presentation",
        },
    )

    # detect_github_changes → extract_text_github
    workflow.add_edge("detect_github_changes", "extract_text_github")

    # detect_jira_changes → extract_text_jira
    workflow.add_edge("detect_jira_changes", "extract_text_jira")

    # detect_confluence_changes → extract_text_confluence
    workflow.add_edge("detect_confluence_changes", "extract_text_confluence")

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
        },
    )

    # miro_board path: extract_text_miro_board → extract_images_miro_board → transform_to_chunks → merge_chunks
    workflow.add_edge("extract_text_miro_board", "extract_images_miro_board")
    workflow.add_edge("extract_images_miro_board", "transform_to_chunks")

    # ========== Add Unconditional Edges ==========
    # Text path: extract → chunk
    workflow.add_edge("extract_text_docx", "transform_to_chunks")
    workflow.add_edge("extract_text_pptx", "transform_to_chunks")
    workflow.add_edge("extract_text_txt", "transform_to_chunks")
    workflow.add_edge("extract_text_md", "transform_to_chunks")
    workflow.add_edge("extract_text_document", "transform_to_chunks")
    workflow.add_edge("extract_text_presentation", "transform_to_chunks")
    workflow.add_edge("extract_text_miro", "transform_to_chunks")
    workflow.add_edge("extract_text_github", "transform_to_chunks")
    workflow.add_edge("extract_text_jira", "transform_to_chunks")
    workflow.add_edge("extract_text_confluence", "transform_to_chunks")

    # Image path (parallel to text chunking): text extracted → extract images → merge
    # Fan out from both text-extraction nodes so image extraction runs after file_id is known
    workflow.add_edge("extract_text_docx", "extract_images")
    workflow.add_edge("extract_text_pptx", "extract_images")
    workflow.add_edge("extract_text_txt", "extract_images")
    workflow.add_edge("extract_text_md", "extract_images")
    workflow.add_edge("extract_text_document", "extract_images")
    workflow.add_edge("extract_text_presentation", "extract_images")
    workflow.add_edge("extract_text_miro", "extract_images_miro")
    workflow.add_edge("extract_images_miro", "merge_chunks")
    workflow.add_edge("extract_text_github", "extract_images")
    workflow.add_edge("extract_text_jira", "extract_images")
    workflow.add_edge("extract_text_confluence", "extract_images")

    # Both paths converge at merge_chunks
    workflow.add_edge("transform_to_chunks", "merge_chunks")
    workflow.add_edge("extract_images", "merge_chunks")

    # Merged chunks → Relevance screening → View extraction → Class extraction → Property extraction →
    # Accumulated save → Output generation
    workflow.add_edge("merge_chunks", "screen_relevance")
    workflow.add_edge("screen_relevance", "extract_views")
    workflow.add_edge("extract_views", "extract_classes")
    workflow.add_edge("extract_classes", "extract_properties")
    workflow.add_conditional_edges(
        "extract_properties",
        route_after_properties,
        {END: END, "save_accumulated_knowledge": "save_accumulated_knowledge"},
    )
    workflow.add_edge("save_accumulated_knowledge", "generate_outputs")

    # After output generation, end
    workflow.add_edge("generate_outputs", END)

    # ========== Compile Graph ==========
    app = workflow.compile()

    return app
