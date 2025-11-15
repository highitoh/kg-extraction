"""
LangGraph Node Functions for Knowledge Extraction
知識抽出処理のLangGraphノード関数定義
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
    """メタモデルファイルを読み込む"""
    # Go up 2 levels from src/core/ to /workspace/app/
    app_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    metamodel_path = os.path.join(app_dir, "metamodel", "metamodel.json")
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

    既存のknowledge.jsonがあれば、ACCEPT済みクラスを読み込んで
    新規抽出クラスと統合する
    """
    try:
        # 既存クラスの読み込み
        existing_classes = []
        output_dir = state.get("output_dir", "/workspace/app/output")
        project = state.get("project", "default")
        extraction_result_path = os.path.join(output_dir, project, "knowledge.json")

        if os.path.exists(extraction_result_path):
            try:
                with open(extraction_result_path, "r", encoding="utf-8") as f:
                    previous_result = json.load(f)
                    # ACCEPT判定のクラスのみを読み込み
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

        # ClassChainの入力形式に合わせる
        # ViewChainの出力をそのまま view_info として使用
        # 既存クラスも渡す
        input_data = {
            "view_info": {
                "id": state["id"],
                "views": state["views"]
            },
            "metamodel": state["metamodel"],
            "existing_classes": existing_classes  # 既存クラスを追加
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

    既存のknowledge.jsonがあれば、強度>0のプロパティを読み込んで
    新規抽出プロパティと統合する
    """
    try:
        # 既存プロパティの読み込み
        existing_properties = []
        output_dir = state.get("output_dir", "/workspace/app/output")
        project = state.get("project", "default")
        extraction_result_path = os.path.join(output_dir, project, "knowledge.json")

        if os.path.exists(extraction_result_path):
            try:
                with open(extraction_result_path, "r", encoding="utf-8") as f:
                    previous_result = json.load(f)
                    # 強度>0のプロパティのみを読み込み
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


def save_accumulated_knowledge_node(state: KnowledgeExtractionState) -> KnowledgeExtractionState:
    """
    knowledge.jsonに累積知識を保存するノード
    ACCEPT判定のクラスと強度>0のプロパティのみを保存

    このノードはgenerate_outputs_nodeの前に実行され、
    出力生成時に最新の累積知識が利用可能になる
    """
    try:
        from src.utils.knowledge_filter import filter_accumulated_knowledge

        output_dir = state.get("output_dir", "/workspace/app/output")
        project = state.get("project", "default")
        project_output_dir = os.path.join(output_dir, project)
        os.makedirs(project_output_dir, exist_ok=True)

        knowledge_path = os.path.join(project_output_dir, "knowledge.json")

        # 現在の実行結果からACCEPT'edクラスと有効なプロパティを抽出
        current_classes, current_properties = filter_accumulated_knowledge(
            state.get("classes", []),
            state.get("properties", [])
        )

        # 既存のknowledge.jsonがあれば読み込み
        existing_knowledge = {"id": state.get("id", ""), "classes": [], "properties": []}
        if os.path.exists(knowledge_path):
            try:
                with open(knowledge_path, "r", encoding="utf-8") as f:
                    existing_knowledge = json.load(f)
            except (json.JSONDecodeError, KeyError) as e:
                print(f"[Warning] Could not load existing knowledge.json: {e}")

        # クラスをマージ（semantic keyベースで重複排除・置換）
        # UUIDが変わった場合のマッピングを記録するため、既存クラスを保持
        current_class_map = {
            (c.get("class_iri"), c.get("label")): c
            for c in current_classes
        }
        existing_class_map = {
            (c.get("class_iri"), c.get("label")): c
            for c in existing_knowledge.get("classes", [])
        }

        # UUIDマッピング: old_uuid -> new_uuid (クラス置換時に使用)
        uuid_map = {}
        for key, new_class in current_class_map.items():
            old_class = existing_class_map.get(key)
            if old_class and old_class.get("id") != new_class.get("id"):
                # 同じsemantic keyだがUUIDが異なる → UUIDが変わった
                uuid_map[old_class["id"]] = new_class["id"]
            existing_class_map[key] = new_class
        merged_classes = list(existing_class_map.values())

        # プロパティのsrc_id/dest_idをリマップ（クラスUUID変更に対応）
        def remap_property_ids(properties, uuid_mapping):
            """プロパティのsrc_id/dest_idを新しいUUIDにリマップ"""
            remapped = []
            for prop in properties:
                p = prop.copy()
                p["src_id"] = uuid_mapping.get(p["src_id"], p["src_id"])
                p["dest_id"] = uuid_mapping.get(p["dest_id"], p["dest_id"])
                remapped.append(p)
            return remapped

        # 既存プロパティのUUIDをリマップ
        existing_properties = remap_property_ids(
            existing_knowledge.get("properties", []),
            uuid_map
        )
        # 現在のプロパティもリマップ（念のため）
        current_properties = remap_property_ids(current_properties, uuid_map)

        # プロパティをマージ（IDベースで重複排除、強度更新）
        existing_property_map = {p.get("id"): p for p in existing_properties}
        for prop in current_properties:
            prop_id = prop.get("id")
            if prop_id in existing_property_map:
                if prop.get("strength", 0) > existing_property_map[prop_id].get("strength", 0):
                    existing_property_map[prop_id] = prop
            else:
                existing_property_map[prop_id] = prop
        merged_properties = list(existing_property_map.values())

        # 孤立したプロパティを除去（存在しないクラスを参照しているもの）
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

        # knowledge.jsonに保存
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
    Turtleファイルとneo4j CSVファイルを生成するノード

    Note: knowledge.jsonから累積知識を読み込んで出力を生成する
    このノードの前にsave_accumulated_knowledge_nodeが実行されているため、
    knowledge.jsonには最新の累積知識が含まれている
    """
    try:
        from src.utils.knowledge_filter import filter_accumulated_knowledge

        output_dir = state.get("output_dir", "/workspace/app/output")
        project = state.get("project", "default")
        project_output_dir = os.path.join(output_dir, project)
        os.makedirs(project_output_dir, exist_ok=True)

        # knowledge.jsonから累積知識を読み込む
        knowledge_path = os.path.join(project_output_dir, "knowledge.json")
        if os.path.exists(knowledge_path):
            with open(knowledge_path, "r", encoding="utf-8") as f:
                knowledge = json.load(f)
            accumulated_classes = knowledge.get("classes", [])
            accumulated_properties = knowledge.get("properties", [])
        else:
            # 初回実行時（knowledge.jsonがまだない場合）は現在の結果をフィルタリング
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

        # 結果をStateに格納
        state["turtle_file"] = turtle_path
        state["turtle_content"] = turtle_result["turtle"]
        state["nodes_csv"] = csv_result["nodes_path"]
        state["relationships_csv"] = csv_result["properties_path"]

        # サマリー情報を生成
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


# ========== ルーティング関数 ==========

def route_by_file_type(state: KnowledgeExtractionState) -> str:
    """
    ファイルタイプに応じて次のノードを決定するルーティング関数
    """
    if state.get("file_type") == "presentation":
        return "extract_text_presentation"
    else:
        return "extract_text_document"
