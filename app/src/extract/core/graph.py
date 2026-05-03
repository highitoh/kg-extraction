"""
LangGraph Graph Construction for Knowledge Extraction
LangGraph graph definition for knowledge extraction processing
"""

from langgraph.graph import StateGraph, END
from src.extract.core.state import KnowledgeExtractionState
from src.extract.core.nodes import (
    classify_file_type_node,
    extract_text_document_node,
    extract_text_presentation_node,
    extract_text_miro_node,
    transform_to_chunks_node,
    extract_images_node,
    merge_chunks_node,
    extract_views_node,
    extract_classes_node,
    extract_properties_node,
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
      └─→ extract_text_miro ─────────┘
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
    workflow.add_node("extract_text_document", extract_text_document_node)
    workflow.add_node("extract_text_presentation", extract_text_presentation_node)
    workflow.add_node("extract_text_miro", extract_text_miro_node)
    workflow.add_node("transform_to_chunks", transform_to_chunks_node)
    workflow.add_node("extract_images", extract_images_node)
    workflow.add_node("merge_chunks", merge_chunks_node)
    workflow.add_node("extract_views", extract_views_node)
    workflow.add_node("extract_classes", extract_classes_node)
    workflow.add_node("extract_properties", extract_properties_node)
    workflow.add_node("save_accumulated_knowledge", save_accumulated_knowledge_node)
    workflow.add_node("generate_outputs", generate_outputs_node)

    # ========== Set Entry Point ==========
    workflow.set_entry_point("classify_file_type")

    # ========== Add Conditional Edges ==========
    # Select text extraction node based on file type
    workflow.add_conditional_edges(
        "classify_file_type",
        route_by_file_type,
        {
            "extract_text_document": "extract_text_document",
            "extract_text_presentation": "extract_text_presentation",
            "extract_text_miro": "extract_text_miro"
        }
    )

    # ========== Add Unconditional Edges ==========
    # Text path: extract → chunk
    workflow.add_edge("extract_text_document", "transform_to_chunks")
    workflow.add_edge("extract_text_presentation", "transform_to_chunks")
    workflow.add_edge("extract_text_miro", "transform_to_chunks")

    # Image path (parallel to text chunking): text extracted → extract images → merge
    # Fan out from both text-extraction nodes so image extraction runs after file_id is known
    workflow.add_edge("extract_text_document", "extract_images")
    workflow.add_edge("extract_text_presentation", "extract_images")
    workflow.add_edge("extract_text_miro", "extract_images")

    # Both paths converge at merge_chunks
    workflow.add_edge("transform_to_chunks", "merge_chunks")
    workflow.add_edge("extract_images", "merge_chunks")

    # Merged chunks → View extraction → Class extraction → Property extraction → Accumulated save → Output generation
    workflow.add_edge("merge_chunks", "extract_views")
    workflow.add_edge("extract_views", "extract_classes")
    workflow.add_edge("extract_classes", "extract_properties")
    workflow.add_edge("extract_properties", "save_accumulated_knowledge")
    workflow.add_edge("save_accumulated_knowledge", "generate_outputs")

    # After output generation, end
    workflow.add_edge("generate_outputs", END)

    # ========== Compile Graph ==========
    app = workflow.compile()

    return app
