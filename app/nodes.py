"""
LangGraph Node Functions for Knowledge Extraction
知識抽出処理のLangGraphノード関数定義
"""

import os
import json
from typing import Dict, Any

from state import KnowledgeExtractionState
from file_type_classifier import FileTypeClassifier
from doc_text_chain import DocTextChain
from presentation_text_chain import PresentationTextChain
from view_chain import create_view_chain
from class_chain import create_class_chain
from property_chain import create_property_chain
from turtle_generator import TurtleGenerator
from neo4j_csv_generator import Neo4jCSVGenerator
from chunk_creator import ChunkCreator


def _load_metamodel() -> Dict[str, Any]:
    """メタモデルファイルを読み込む"""
    metamodel_path = os.path.join(os.path.dirname(__file__), "metamodel", "metamodel.json")
    with open(metamodel_path, "r", encoding="utf-8") as f:
        return json.load(f)


# ========== ノード関数定義 ==========

def classify_file_type_node(state: KnowledgeExtractionState) -> KnowledgeExtractionState:
    """
    ファイルタイプを判定するノード
    files -> file_type, file_path, metamodel
    """
    try:
        classifier = FileTypeClassifier()
        result = classifier.invoke({"files": state["files"]})

        # file_typeとfile_pathを取得
        state["file_type"] = result.get("file_type", "document")
        state["file_path"] = state["files"][0]["path"]

        # メタモデルをロード
        state["metamodel"] = _load_metamodel()

        # エラーリストとステージ情報を初期化
        state.setdefault("errors", [])
        state["current_stage"] = "file_type_classified"

    except Exception as e:
        state.setdefault("errors", [])
        state["errors"].append(f"File type classification failed: {e}")
        state["file_type"] = "document"  # デフォルトにフォールバック
        state["current_stage"] = "file_type_classification_failed"

    return state


def extract_text_document_node(state: KnowledgeExtractionState) -> KnowledgeExtractionState:
    """
    ドキュメントからテキストを抽出するノード
    file_path -> sentences, id, file_id, file_name
    """
    try:
        doc_chain = DocTextChain(model=state.get("model"))

        # DocTextChainの入力形式に合わせる
        input_data = {
            "files": state["files"]
        }

        result = doc_chain.invoke(input_data)

        # 結果をStateに格納
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
    プレゼンテーションからテキストを抽出するノード
    file_path -> sentences, id, file_id, file_name
    """
    try:
        presentation_chain = PresentationTextChain(model=state.get("model"))

        # PresentationTextChainの入力形式に合わせる
        input_data = {
            "files": state["files"]
        }

        result = presentation_chain.invoke(input_data)

        # 結果をStateに格納
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
    抽出されたテキストをチャンクに分割するノード
    sentences -> chunks
    """
    try:
        # ファイルタイプに応じてチャンクモードを選択
        if state["file_type"] == "presentation":
            chunk_creator = ChunkCreator(chunk_mode="page")
            combined_text = "\n".join([s["text"] for s in state["sentences"]])
        else:
            chunk_creator = ChunkCreator(chunk_mode="char")
            combined_text = " ".join([s["text"] for s in state["sentences"]])

        # チャンクに分割
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
    ビュー記述を抽出するノード
    chunks, metamodel -> views
    """
    try:
        view_chain = create_view_chain(model=state.get("model"))

        # ViewChainの入力形式に合わせる
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

        # 結果をStateに格納
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
    クラスを抽出するノード（抽出・フィルタ・統合を含む）
    views, metamodel -> classes
    """
    try:
        class_chain = create_class_chain(model=state.get("model"))

        # ClassChainの入力形式に合わせる
        # ViewChainの出力をそのまま view_info として使用
        input_data = {
            "view_info": {
                "id": state["id"],
                "views": state["views"]
            },
            "metamodel": state["metamodel"]
        }

        result = class_chain.invoke(input_data)

        # 結果をStateに格納
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
    プロパティ（関係）を抽出するノード（抽出・フィルタ・バリデーションを含む）
    classes, metamodel -> properties
    """
    try:
        property_chain = create_property_chain(model=state.get("model"))

        # PropertyChainの入力形式に合わせる
        input_data = {
            "class_info": {
                "id": state["id"],
                "classes": state["classes"]
            },
            "metamodel": state["metamodel"]
        }

        result = property_chain.invoke(input_data)

        # 結果をStateに格納
        state["properties"] = result["properties"]
        state["current_stage"] = "properties_extracted"

    except Exception as e:
        state.setdefault("errors", [])
        state["errors"].append(f"Property extraction failed: {e}")
        state["properties"] = []
        state["current_stage"] = "property_extraction_failed"

    return state


def generate_outputs_node(state: KnowledgeExtractionState) -> KnowledgeExtractionState:
    """
    Turtleファイルとneo4j CSVファイルを生成するノード
    classes, properties -> turtle_file, nodes_csv, relationships_csv, summary
    """
    try:
        output_dir = state.get("output_dir", "/workspace/app/output")

        # 出力ディレクトリを作成
        os.makedirs(output_dir, exist_ok=True)

        # TurtleGenerator
        turtle_generator = TurtleGenerator(output_dir=output_dir)
        turtle_input = {
            "class_info": {
                "id": state["id"],
                "classes": state["classes"]
            },
            "property_info": {
                "id": state["id"],
                "properties": state["properties"]
            }
        }
        turtle_path, turtle_result = turtle_generator.generate_and_save(turtle_input)

        # Neo4jCSVGenerator
        csv_generator = Neo4jCSVGenerator(output_dir=output_dir)
        csv_result = csv_generator.generate(turtle_input)

        # 結果をStateに格納
        state["turtle_file"] = turtle_path
        state["turtle_content"] = turtle_result["turtle"]
        state["nodes_csv"] = csv_result["nodes_path"]
        state["relationships_csv"] = csv_result["properties_path"]

        # サマリー情報を生成
        state["summary"] = {
            "classes_count": len(state["classes"]),
            "properties_count": len(state["properties"]),
            "files_generated": [turtle_path, csv_result["nodes_path"], csv_result["properties_path"]]
        }

        state["current_stage"] = "outputs_generated"

    except Exception as e:
        state.setdefault("errors", [])
        state["errors"].append(f"Output generation failed: {e}")
        state["current_stage"] = "output_generation_failed"

    return state


# ========== ルーティング関数 ==========

def route_by_file_type(state: KnowledgeExtractionState) -> str:
    """
    ファイルタイプに応じて次のノードを決定するルーティング関数
    """
    if state.get("file_type") == "presentation":
        return "extract_text_presentation"
    else:
        return "extract_text_document"
