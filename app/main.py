"""
Knowledge Graph Extraction Main
PDFファイルから知識を抽出してTurtleとNeo4j用CSVファイルを生成する
"""

import os
import json
import sys
import argparse
from typing import Dict, Any

from graph import create_knowledge_extraction_graph
from state import KnowledgeExtractionState




def run_knowledge_extraction(pdf_path: str, output_dir: str = "/workspace/app/output", model: str = None) -> Dict[str, Any]:
    """
    PDFファイルから知識抽出を実行

    Args:
        pdf_path: 抽出対象のPDFファイルパス
        output_dir: 出力ディレクトリ
        model: LLMモデル名（Noneの場合は各ノードのデフォルトを使用）

    Returns:
        抽出結果辞書
    """
    # グラフを作成
    app = create_knowledge_extraction_graph()

    # 初期Stateを準備
    initial_state: KnowledgeExtractionState = {
        "files": [{"path": pdf_path}],
        "output_dir": output_dir,
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

    args = parser.parse_args()
    pdf_path = args.pdf_path

    if not os.path.exists(pdf_path):
        print(f"Error: PDF file not found: {pdf_path}")
        sys.exit(1)

    print(f"Starting knowledge extraction from: {pdf_path}")
    if args.model:
        print(f"Using LLM model: {args.model}")
    else:
        print(f"Using default models for each node")

    try:
        result = run_knowledge_extraction(pdf_path, output_dir=args.output_dir, model=args.model)

        # エラーチェック
        if result.get("errors"):
            print("\n=== Warnings/Errors ===")
            for error in result["errors"]:
                print(f"  - {error}")

        # 最終ステージの確認
        print(f"\nFinal stage: {result.get('current_stage', 'unknown')}")

        # 結果表示
        if "summary" in result:
            print("\n=== Knowledge Extraction Complete ===")
            print(f"Generated files:")
            for file_path in result["summary"]["files_generated"]:
                print(f"  - {file_path}")

            print(f"\nSummary:")
            print(f"  Classes extracted: {result['summary']['classes_count']}")
            print(f"  Properties extracted: {result['summary']['properties_count']}")

            # 結果をJSONファイルに保存
            output_json = os.path.join(args.output_dir, "extraction_result.json")
            with open(output_json, "w", encoding="utf-8") as f:
                # turtle_contentは長すぎるので保存時は除外
                result_copy = result.copy()
                result_copy.pop("turtle_content", None)
                json.dump(result_copy, f, ensure_ascii=False, indent=2)

            print(f"  Result saved to: {output_json}")
        else:
            print("\n=== Knowledge Extraction Failed ===")
            print(f"Extraction did not complete successfully.")
            if result.get("errors"):
                print("Errors encountered:")
                for error in result["errors"]:
                    print(f"  - {error}")
            sys.exit(1)

    except Exception as e:
        print(f"Error during extraction: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)