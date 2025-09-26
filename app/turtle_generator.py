# turtle_generator.py
# Turtle RDF/OWL file generator based on class_extractor.py logic

import os
import re
import json
import uuid
import asyncio
from datetime import datetime
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple, Union

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage


# ---------- ユーティリティ ----------
def _normalize_ws(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip()

def _uuid8() -> str:
    return str(uuid.uuid4())[:8]

def _slug(s: str) -> str:
    s = _normalize_ws(s)
    s = re.sub(r"[^0-9A-Za-z\u00C0-\u024F\u3040-\u30FF\u4E00-\u9FFF\-_.]", "-", s)
    s = re.sub(r"-{2,}", "-", s).strip("-")
    return s[:48] or _uuid8()


class TurtleGenerator:
    """
    クラス抽出情報とプロパティ抽出情報を受け取り、RDF/OWL(Turtle) ファイルを生成する。
    """

    def __init__(
        self,
        *,
        base_prefix: str = "http://example.com/kg#",
        output_dir: str = "/workspace/app/output",
    ):
        self.base_prefix = base_prefix
        self.output_dir = output_dir

        # 出力ディレクトリを作成
        os.makedirs(self.output_dir, exist_ok=True)

    def _iri(self, path: str) -> str:
        """
        ex: のプレフィックス表記ではなく、<http://…> のフルIRIを返す。
        例) path="Extraction/0-aaaa" -> "<http://example.com/kg#Extraction/0-aaaa>"
        """
        base = self.base_prefix.rstrip("#/") + "#"
        return f"<{base}{path}>"

    def _escape_string(self, s: str) -> str:
        """
        Turtle文字列リテラルのエスケープ処理
        """
        return s.replace('\\', '\\\\').replace('"', '\\"').replace('\n', '\\n').replace('\r', '\\r')

    def _escape_turtle_text(self, s: str) -> str:
        """
        Turtle の長文字列リテラルのエスケープ処理
        """
        return s.replace('"""', '\\"""')

    def _generate_turtle_header(self) -> str:
        """
        Turtle ファイルのヘッダー部分を生成
        """
        header_lines = [
            f"# Generated on {datetime.now().isoformat()}",
            "",
            "@prefix ex: <http://example.com/kg#> .",
            "@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .",
            "@prefix owl: <http://www.w3.org/2002/07/owl#> .",
            "@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .",
            "@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .",
            "",
            "# Ontology Declaration",
            "ex: a owl:Ontology .",
            "",
        ]
        return "\n".join(header_lines)

    def _generate_class_declarations(self, class_iris: set) -> str:
        """
        クラス宣言部分を生成
        """
        lines = ["# Class Declarations"]
        for class_iri in sorted(class_iris):
            if class_iri.startswith("ex:"):
                lines.append(f"{class_iri} a owl:Class .")
        lines.append("")
        return "\n".join(lines)

    def _generate_property_declarations(self, property_iris: set) -> str:
        """
        プロパティ宣言部分を生成
        """
        lines = ["# Property Declarations"]
        for prop_iri in sorted(property_iris):
            if prop_iri.startswith("ex:"):
                lines.append(f"{prop_iri} a owl:ObjectProperty .")
        lines.append("")
        return "\n".join(lines)

    def _generate_instances(self, class_info: Dict[str, Any], property_info: Dict[str, Any]) -> Tuple[str, set, set]:
        """
        インスタンス部分を生成し、使用されているクラスIRIとプロパティIRIのセットを返す
        """
        lines = ["# Instances"]
        class_iris = set()
        property_iris = set()

        # クラス個体のマップを作成
        class_instances = {}
        for class_item in class_info.get("classes", []):
            class_instances[class_item["id"]] = class_item

        # クラス個体を出力
        for class_item in class_info.get("classes", []):
            class_id = class_item["id"]
            class_iri = class_item["class_iri"]
            label = class_item["label"]
            file_id = class_item.get("file_id", "")

            # インスタンスIRIを生成
            instance_iri = self._iri(f"instance/{_slug(label)}-{_uuid8()}")

            # クラスIRIを記録
            class_iris.add(class_iri)

            # インスタンス定義
            lines.append(f"{instance_iri} a {class_iri} ;")
            lines.append(f'  rdfs:label "{self._escape_string(label)}"@ja ;')
            if file_id:
                lines.append(f'  ex:fileId "{self._escape_string(file_id)}" ;')
            lines.append(f'  ex:classInstanceId "{self._escape_string(class_id)}" .')
            lines.append("")

        # プロパティ関係を出力
        property_map = {}
        for prop_item in property_info.get("properties", []):
            src_id = prop_item["src_id"]
            dest_id = prop_item["dest_id"]
            property_iri = prop_item["property_iri"]

            # プロパティIRIを記録
            property_iris.add(property_iri)

            # クラス個体IDからインスタンスIRIへのマッピング
            if src_id not in property_map:
                property_map[src_id] = []
            property_map[src_id].append((property_iri, dest_id))

        # プロパティ関係を追加
        for src_id, relations in property_map.items():
            if src_id in class_instances:
                src_class = class_instances[src_id]
                src_instance_iri = self._iri(f"instance/{_slug(src_class['label'])}-{_uuid8()}")

                for property_iri, dest_id in relations:
                    if dest_id in class_instances:
                        dest_class = class_instances[dest_id]
                        dest_instance_iri = self._iri(f"instance/{_slug(dest_class['label'])}-{_uuid8()}")

                        lines.append(f"{src_instance_iri} {property_iri} {dest_instance_iri} .")

        if len(lines) > 1:  # "# Instances" 以外にも行がある場合
            lines.append("")

        return "\n".join(lines), class_iris, property_iris

    def generate(self, input_data: Dict[str, Any]) -> Dict[str, str]:
        """
        入力データからTurtleファイルを生成する

        Args:
            input_data: スキーマに従った入力データ

        Returns:
            生成されたTurtle文字列を含む辞書
        """
        class_info = input_data.get("class_info", {})
        property_info = input_data.get("property_info", {})

        # Turtleファイルの各部分を生成
        header = self._generate_turtle_header()
        instances_content, class_iris, property_iris = self._generate_instances(class_info, property_info)
        class_declarations = self._generate_class_declarations(class_iris)
        property_declarations = self._generate_property_declarations(property_iris)

        # 全体を結合
        turtle_content = header + class_declarations + property_declarations + instances_content

        return {"turtle": turtle_content}

    def generate_and_save(self, input_data: Dict[str, Any], filename: Optional[str] = None) -> Tuple[str, Dict[str, str]]:
        """
        Turtleファイルを生成してファイルに保存する

        Args:
            input_data: スキーマに従った入力データ
            filename: 出力ファイル名（省略時は自動生成）

        Returns:
            (保存ファイルパス, 生成結果辞書)
        """
        result = self.generate(input_data)

        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"knowledge_graph_{timestamp}.ttl"

        if not filename.endswith('.ttl'):
            filename += '.ttl'

        filepath = os.path.join(self.output_dir, filename)

        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(result["turtle"])

        return filepath, result


# -------------- 使い方（例） --------------
if __name__ == "__main__":
    # テスト用のサンプルデータ
    sample_input = {
        "class_info": {
            "id": "test-extraction-001",
            "classes": [
                {
                    "id": "cls-001",
                    "class_iri": "ex:Stakeholder",
                    "label": "店舗スタッフ",
                    "file_id": "doc-001"
                },
                {
                    "id": "cls-002",
                    "class_iri": "ex:Value",
                    "label": "待ち時間短縮",
                    "file_id": "doc-001"
                },
                {
                    "id": "cls-003",
                    "class_iri": "ex:Vision",
                    "label": "待たせない店舗体験",
                    "file_id": "doc-001"
                }
            ]
        },
        "property_info": {
            "id": "test-property-001",
            "properties": [
                {
                    "id": "prop-001",
                    "src_id": "cls-001",
                    "property_iri": "ex:hasValue",
                    "dest_id": "cls-002"
                },
                {
                    "id": "prop-002",
                    "src_id": "cls-003",
                    "property_iri": "ex:realizesValue",
                    "dest_id": "cls-002"
                }
            ]
        }
    }

    generator = TurtleGenerator()
    filepath, result = generator.generate_and_save(sample_input, "test_output.ttl")

    print(f"Generated Turtle file: {filepath}")
    print("\nTurtle content preview:")
    print(result["turtle"][:500] + "..." if len(result["turtle"]) > 500 else result["turtle"])