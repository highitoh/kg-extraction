"""
Knowledge Graph Extraction Main
PDFファイルから知識を抽出してTurtleとNeo4j用CSVファイルを生成する
"""

import os
import json
import sys
import argparse
from datetime import datetime
from typing import Dict, Any
from dotenv import load_dotenv

# Add the app directory to sys.path to enable src imports
app_dir = os.path.dirname(os.path.dirname(__file__))
if app_dir not in sys.path:
    sys.path.insert(0, app_dir)

# Load environment variables from .env file
load_dotenv()

from src.core.graph import create_knowledge_extraction_graph
from src.core.state import KnowledgeExtractionState


def _save_accumulated_knowledge(result: Dict[str, Any], project_output_dir: str) -> None:
    """
    knowledge.jsonに累積保存する
    ACCEPT判定のクラスと強度>0のプロパティのみを保存

    Args:
        result: 現在の実行結果
        project_output_dir: プロジェクト出力ディレクトリ
    """
    from src.utils.knowledge_filter import filter_accumulated_knowledge

    knowledge_path = os.path.join(project_output_dir, "knowledge.json")

    # 現在の実行結果からACCEPT'edクラスと有効なプロパティを抽出
    current_classes, current_properties = filter_accumulated_knowledge(
        result.get("classes", []),
        result.get("properties", [])
    )

    # 既存のknowledge.jsonがあれば読み込み
    existing_knowledge = {"id": result.get("id", ""), "classes": [], "properties": []}
    if os.path.exists(knowledge_path):
        try:
            with open(knowledge_path, "r", encoding="utf-8") as f:
                existing_knowledge = json.load(f)
        except (json.JSONDecodeError, KeyError) as e:
            print(f"[Warning] Could not load existing knowledge.json: {e}")

    # クラスをマージ（semantic keyベースで重複排除・置換）
    # Build map of current classes by semantic key
    current_class_map = {
        (c.get("class_iri"), c.get("label")): c
        for c in current_classes
    }

    # Build map of existing classes by semantic key
    existing_class_map = {
        (c.get("class_iri"), c.get("label")): c
        for c in existing_knowledge.get("classes", [])
    }

    # Merge: current classes override existing ones with same (class_iri, label)
    for key, cls in current_class_map.items():
        existing_class_map[key] = cls  # Replace or add

    merged_classes = list(existing_class_map.values())

    # プロパティをマージ（IDベースで重複排除、強度更新）
    existing_property_map = {p.get("id"): p for p in existing_knowledge.get("properties", [])}
    for prop in current_properties:
        prop_id = prop.get("id")
        if prop_id in existing_property_map:
            # 既存プロパティの強度を比較して高い方を保持
            if prop.get("strength", 0) > existing_property_map[prop_id].get("strength", 0):
                existing_property_map[prop_id] = prop
        else:
            existing_property_map[prop_id] = prop
    merged_properties = list(existing_property_map.values())

    # knowledge.jsonに保存
    accumulated_knowledge = {
        "id": result.get("id", ""),
        "classes": merged_classes,
        "properties": merged_properties
    }

    with open(knowledge_path, "w", encoding="utf-8") as f:
        json.dump(accumulated_knowledge, f, ensure_ascii=False, indent=2)

    print(f"  - Knowledge accumulated to: {knowledge_path}")
    print(f"    Total classes: {len(merged_classes)} (added {len(merged_classes) - len(existing_knowledge.get('classes', []))} new)")
    print(f"    Total properties: {len(merged_properties)} (added {len(merged_properties) - len(existing_knowledge.get('properties', []))} new)")


def run_knowledge_extraction(pdf_path: str, output_dir: str = "/workspace/app/output", model: str = None, project: str = "default") -> Dict[str, Any]:
    """
    PDFファイルから知識抽出を実行

    Args:
        pdf_path: 抽出対象のPDFファイルパス
        output_dir: 出力ディレクトリ
        model: LLMモデル名（Noneの場合は各ノードのデフォルトを使用）
        project: プロジェクト名（出力ファイルをプロジェクト別に分類）

    Returns:
        抽出結果辞書
    """
    # グラフを作成
    app = create_knowledge_extraction_graph()

    # 初期Stateを準備
    initial_state: KnowledgeExtractionState = {
        "files": [{"path": pdf_path}],
        "output_dir": output_dir,
        "project": project,
        "errors": []
    }

    # modelが指定されている場合はStateに追加
    if model:
        initial_state["model"] = model

    # グラフを実行
    result = app.invoke(initial_state)

    return result


if __name__ == "__main__":
    # コマンドライン引数のパース
    parser = argparse.ArgumentParser(description="PDFファイルから知識を抽出してTurtleとNeo4j用CSVファイルを生成")
    parser.add_argument("pdf_path", nargs="?", default="../doc/sample.pdf", help="抽出対象のPDFファイルパス")
    parser.add_argument("--model", type=str, default=None, help="使用するLLMモデル名（デフォルト: 各ノードのデフォルトモデル）")
    parser.add_argument("--output-dir", type=str, default="/workspace/app/output", help="出力ディレクトリ（デフォルト: /workspace/app/output）")
    parser.add_argument("--project", type=str, default="default", help="プロジェクト名（デフォルト: default）")

    args = parser.parse_args()
    pdf_path = args.pdf_path

    if not os.path.exists(pdf_path):
        print(f"Error: PDF file not found: {pdf_path}")
        sys.exit(1)

    print("\n=== Knowledge Extraction ===")
    print(f"PDF file: {pdf_path}")
    print(f"Project: {args.project}")
    if args.model:
        print(f"LLM model: {args.model}")
    else:
        print(f"LLM model: Using default models for each node")

    try:
        result = run_knowledge_extraction(pdf_path, output_dir=args.output_dir, model=args.model, project=args.project)

        # エラーチェック
        if result.get("errors"):
            print("\n=== Warnings/Errors ===")
            for error in result["errors"]:
                print(f"  - {error}")

        # 結果表示
        if "summary" in result:
            print("\n=== Extraction Complete ===")
            print(f"Final stage: {result.get('current_stage', 'unknown')}")
            print(f"\nGenerated files:")
            for file_path in result["summary"]["files_generated"]:
                print(f"  - {file_path}")

            print(f"\nSummary:")
            print(f"  - Classes extracted: {result['summary']['classes_count']}")
            print(f"  - Properties extracted: {result['summary']['properties_count']}")

            # 結果をJSONファイルに保存
            project_output_dir = os.path.join(args.output_dir, args.project)
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            output_json = os.path.join(project_output_dir, f"extraction_result_{timestamp}.json")
            with open(output_json, "w", encoding="utf-8") as f:
                # turtle_contentは長すぎるので保存時は除外
                result_copy = result.copy()
                result_copy.pop("turtle_content", None)
                json.dump(result_copy, f, ensure_ascii=False, indent=2)

            print(f"  - Result saved to: {output_json}")

            # knowledge.jsonに累積保存（ACCEPT'edクラス + 強度>0のプロパティのみ）
            _save_accumulated_knowledge(result, project_output_dir)
        else:
            print("\n=== Extraction Failed ===")
            print(f"Extraction did not complete successfully.")
            if result.get("errors"):
                print("\nErrors encountered:")
                for error in result["errors"]:
                    print(f"  - {error}")
            sys.exit(1)

    except Exception as e:
        print(f"Error during extraction: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)