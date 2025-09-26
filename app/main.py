"""
Knowledge Graph Extraction Main Chain
PDFファイルから知識を抽出してTurtleとNeo4j用CSVファイルを生成する連続チェイン
"""

import os
import json
from typing import Dict, Any
from langchain.schema.runnable import Runnable, RunnableSequence

from .pdf_text_chain import PDFTextChain
from .view_chain import create_view_chain
from .class_chain import create_class_chain
from .property_chain import create_property_chain
from .turtle_generator import TurtleGenerator
from .neo4j_csv_generator import Neo4jCSVGenerator


class DataTransformer(Runnable):
    """チェイン間のデータ変換を行うRunnable"""

    def __init__(self, transform_type: str):
        self.transform_type = transform_type

    def invoke(self, input: Dict[str, Any], config=None) -> Dict[str, Any]:
        if self.transform_type == "pdf_to_view":
            # PDFTextChainOutput -> ViewChainInput
            return {
                "target": {
                    "id": input["id"],
                    "file_id": input.get("file_id", input["id"]),
                    "file_name": input["file_name"],
                    "sentences": input["sentences"]
                },
                "metamodel": {}  # メタモデルは空で初期化
            }

        elif self.transform_type == "view_to_class":
            # ViewChainOutput -> ClassChainInput
            return {
                "view_info": input,
                "metamodel": {}
            }

        elif self.transform_type == "class_to_property":
            # ClassChainOutput -> PropertyChainInput
            return {
                "class_info": input,
                "metamodel": {}
            }

        elif self.transform_type == "prepare_output":
            # PropertyChainOutputとClassChainOutputを統合して出力準備
            class_info = input.get("class_info", {})
            property_info = input.get("property_info", input)  # inputがPropertyChainOutputの場合

            return {
                "class_info": class_info,
                "property_info": property_info
            }

        return input


class OutputGenerator(Runnable):
    """TurtleとNeo4j CSVファイルを生成するRunnable"""

    def __init__(self, output_dir: str = "/workspace/app/output"):
        self.turtle_generator = TurtleGenerator(output_dir=output_dir)
        self.csv_generator = Neo4jCSVGenerator(output_dir=output_dir)
        self.output_dir = output_dir

        # 出力ディレクトリを作成
        os.makedirs(output_dir, exist_ok=True)

    def invoke(self, input: Dict[str, Any], config=None) -> Dict[str, Any]:
        # Turtleファイル生成
        turtle_path, turtle_result = self.turtle_generator.generate_and_save(input)

        # Neo4j CSVファイル生成
        csv_result = self.csv_generator.generate(input)

        return {
            "turtle_file": turtle_path,
            "turtle_content": turtle_result["turtle"],
            "nodes_csv": csv_result["nodes_path"],
            "relationships_csv": csv_result["properties_path"],
            "summary": {
                "classes_count": len(input.get("class_info", {}).get("classes", [])),
                "properties_count": len(input.get("property_info", {}).get("properties", [])),
                "files_generated": [turtle_path, csv_result["nodes_path"], csv_result["properties_path"]]
            }
        }


class PropertyAndOutputChain(Runnable):
    """PropertyChainとOutputGeneratorを連結するカスタムチェイン"""

    def __init__(self, output_dir: str = "/workspace/app/output"):
        self.property_chain = create_property_chain()
        self.output_generator = OutputGenerator(output_dir)

    def invoke(self, input: Dict[str, Any], config=None) -> Dict[str, Any]:
        # PropertyChainを実行
        property_result = self.property_chain.invoke(input, config)

        # クラス情報とプロパティ情報を統合
        combined_input = {
            "class_info": input["class_info"],
            "property_info": property_result
        }

        # 出力生成
        return self.output_generator.invoke(combined_input, config)


def create_knowledge_extraction_chain(output_dir: str = "/workspace/app/output") -> RunnableSequence:
    """
    知識抽出チェインを作成

    実行順序:
    1. PDFTextChain - PDFからテキスト抽出
    2. ViewChain - ビュー記述抽出
    3. ClassChain - クラス抽出
    4. PropertyChain - プロパティ（関係）抽出
    5. OutputGenerator - TurtleとNeo4j CSV生成
    """

    # 各チェインの作成
    pdf_chain = PDFTextChain()
    view_chain = create_view_chain()
    class_chain = create_class_chain()
    property_output_chain = PropertyAndOutputChain(output_dir)

    # データ変換用のRunnableを作成
    pdf_to_view_transformer = DataTransformer("pdf_to_view")
    view_to_class_transformer = DataTransformer("view_to_class")
    class_to_property_transformer = DataTransformer("class_to_property")

    # チェインを連結
    return RunnableSequence(steps=[
        pdf_chain,                          # PDFTextChainInput -> PDFTextChainOutput
        pdf_to_view_transformer,            # PDFTextChainOutput -> ViewChainInput
        view_chain,                         # ViewChainInput -> ViewChainOutput
        view_to_class_transformer,          # ViewChainOutput -> ClassChainInput
        class_chain,                        # ClassChainInput -> ClassChainOutput
        class_to_property_transformer,      # ClassChainOutput -> PropertyChainInput
        property_output_chain,              # PropertyChainInput -> Final Output
    ])


def run_knowledge_extraction(pdf_path: str, output_dir: str = "/workspace/app/output") -> Dict[str, Any]:
    """
    PDFファイルから知識抽出を実行

    Args:
        pdf_path: 抽出対象のPDFファイルパス
        output_dir: 出力ディレクトリ

    Returns:
        抽出結果辞書
    """
    # チェインを作成
    chain = create_knowledge_extraction_chain(output_dir)

    # 入力データを準備
    input_data = {
        "files": [{"path": pdf_path}]
    }

    # チェインを実行
    result = chain.invoke(input_data)

    return result


if __name__ == "__main__":
    # サンプル実行
    import sys

    if len(sys.argv) > 1:
        pdf_path = sys.argv[1]
    else:
        pdf_path = "sample.pdf"  # デフォルトファイル

    if not os.path.exists(pdf_path):
        print(f"Error: PDF file not found: {pdf_path}")
        sys.exit(1)

    print(f"Starting knowledge extraction from: {pdf_path}")

    try:
        result = run_knowledge_extraction(pdf_path)

        print("\n=== Knowledge Extraction Complete ===")
        print(f"Generated files:")
        for file_path in result["summary"]["files_generated"]:
            print(f"  - {file_path}")

        print(f"\nSummary:")
        print(f"  Classes extracted: {result['summary']['classes_count']}")
        print(f"  Properties extracted: {result['summary']['properties_count']}")

        # 結果をJSONファイルに保存
        output_json = os.path.join("/workspace/app/output", "extraction_result.json")
        with open(output_json, "w", encoding="utf-8") as f:
            # turtle_contentは長すぎるので保存時は除外
            result_copy = result.copy()
            result_copy.pop("turtle_content", None)
            json.dump(result_copy, f, ensure_ascii=False, indent=2)

        print(f"  Result saved to: {output_json}")

    except Exception as e:
        print(f"Error during extraction: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)