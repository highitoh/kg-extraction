"""
LangGraph Graph Construction for Knowledge Extraction
知識抽出処理のLangGraphグラフ定義
"""

from langgraph.graph import StateGraph, END
from state import KnowledgeExtractionState
from nodes import (
    classify_file_type_node,
    extract_text_document_node,
    extract_text_presentation_node,
    transform_to_chunks_node,
    extract_views_node,
    extract_classes_node,
    extract_properties_node,
    generate_outputs_node,
    route_by_file_type
)


def create_knowledge_extraction_graph():
    """
    知識抽出処理のLangGraphを構築する

    グラフ構造:
    START
      ↓
    classify_file_type
      ↓ (conditional routing)
      ├─→ extract_text_document ─┐
      └─→ extract_text_presentation ─┘
              ↓
         transform_to_chunks
              ↓
         extract_views
              ↓
         extract_classes
              ↓
         extract_properties
              ↓
         generate_outputs
              ↓
            END
    """

    # StateGraphを作成
    workflow = StateGraph(KnowledgeExtractionState)

    # ========== ノードを追加 ==========
    workflow.add_node("classify_file_type", classify_file_type_node)
    workflow.add_node("extract_text_document", extract_text_document_node)
    workflow.add_node("extract_text_presentation", extract_text_presentation_node)
    workflow.add_node("transform_to_chunks", transform_to_chunks_node)
    workflow.add_node("extract_views", extract_views_node)
    workflow.add_node("extract_classes", extract_classes_node)
    workflow.add_node("extract_properties", extract_properties_node)
    workflow.add_node("generate_outputs", generate_outputs_node)

    # ========== エントリーポイントを設定 ==========
    workflow.set_entry_point("classify_file_type")

    # ========== 条件付きエッジを追加 ==========
    # ファイルタイプに応じてテキスト抽出ノードを選択
    workflow.add_conditional_edges(
        "classify_file_type",
        route_by_file_type,
        {
            "extract_text_document": "extract_text_document",
            "extract_text_presentation": "extract_text_presentation"
        }
    )

    # ========== 無条件エッジを追加 ==========
    # テキスト抽出後はチャンク分割へ
    workflow.add_edge("extract_text_document", "transform_to_chunks")
    workflow.add_edge("extract_text_presentation", "transform_to_chunks")

    # チャンク分割 → ビュー抽出 → クラス抽出 → プロパティ抽出 → 出力生成
    workflow.add_edge("transform_to_chunks", "extract_views")
    workflow.add_edge("extract_views", "extract_classes")
    workflow.add_edge("extract_classes", "extract_properties")
    workflow.add_edge("extract_properties", "generate_outputs")

    # 出力生成後は終了
    workflow.add_edge("generate_outputs", END)

    # ========== グラフをコンパイル ==========
    app = workflow.compile()

    return app
