"""
Neo4j Bulk Import CSV Generator

JSON入力からNeo4j Bulk Import用のCSVファイルを生成する。
bk/rdf_to_property_graph.pyのロジックを参考に、JSONスキーマから情報を取得してCSVを生成。
"""

import os
import csv
import json
from typing import Dict, Set, Tuple, Any


class Neo4jCSVGenerator:
    """
    JSON入力からNeo4j Bulk Import用のCSVファイル（nodes.csv, relationships.csv）を生成する。

    Input: class_info (ClassChainOutput) と property_info (PropertyChainOutput)
    Output: nodes.csv と relationships.csv を app/output に生成
    """

    NODE_HEADERS = [
        "id:ID",             # クラス個体ID
        "iri",               # class_iri (表示・確認用)
        ":LABEL",            # クラスのローカル名
        "label",             # 抽出クラスラベル
        "file_id",           # 抽出元ファイルID
    ]

    REL_HEADERS = [
        ":START_ID",         # src_id
        ":END_ID",           # dest_id
        ":TYPE",             # プロパティのローカル名
        "property_iri",      # プロパティIRI
        "id",                # プロパティID
    ]

    def __init__(self, output_dir: str = "./output"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        # ノード情報
        self.nodes: Dict[str, Dict[str, str]] = {}
        # リレーション情報
        self.relationships: Set[Tuple[str, str, str, str, str]] = set()

    @staticmethod
    def _local_name(iri: str) -> str:
        """IRIからローカル名を抽出"""
        if "#" in iri:
            return iri.rsplit("#", 1)[-1]
        if "/" in iri:
            return iri.rstrip("/").rsplit("/", 1)[-1]
        if ":" in iri:
            return iri.rsplit(":", 1)[-1]
        return iri

    @staticmethod
    def _to_snake_case(name: str) -> str:
        """チェインケース（ハイフン区切り）をスネークケースに変換"""
        return name.replace("-", "_")

    def _process_classes(self, class_info: Dict[str, Any]):
        """クラス情報からノードを生成"""
        for class_item in class_info.get("classes", []):
            node_id = class_item["id"]
            class_iri = class_item["class_iri"]
            label = class_item["label"]

            # file_idsフィールドを取得（配列または文字列）
            file_ids = class_item.get("file_ids", [])
            if isinstance(file_ids, str):
                file_ids = [file_ids]
            # カンマ区切りで連結（複数の出典を保持）
            file_id = ",".join(file_ids) if file_ids else ""

            # ノードのラベル（クラスのローカル名をスネークケースに変換）
            class_label = self._to_snake_case(self._local_name(class_iri))

            self.nodes[node_id] = {
                "id:ID": node_id,
                "iri": class_iri,
                ":LABEL": class_label,
                "label": label,
                "file_id": file_id,
            }

    def _process_properties(self, property_info: Dict[str, Any]):
        """プロパティ情報からリレーションを生成"""
        for prop_item in property_info.get("properties", []):
            prop_id = prop_item["id"]
            src_id = prop_item["src_id"]
            dest_id = prop_item["dest_id"]
            property_iri = prop_item["property_iri"]

            # リレーションのタイプ（プロパティのローカル名をスネークケースに変換）
            rel_type = self._to_snake_case(self._local_name(property_iri))

            self.relationships.add((
                src_id,         # :START_ID
                dest_id,        # :END_ID
                rel_type,       # :TYPE
                property_iri,   # property_iri
                prop_id         # id
            ))

    def generate(self, input_data: Dict[str, Any]) -> Dict[str, str]:
        """
        JSON入力からCSVファイルを生成

        Args:
            input_data: {"class_info": {...}, "property_info": {...}}

        Returns:
            {"nodes_path": str, "properties_path": str}
        """
        # 入力データの処理
        class_info = input_data.get("class_info", {})
        property_info = input_data.get("property_info", {})

        # ノードとリレーションの処理
        self._process_classes(class_info)
        self._process_properties(property_info)

        # CSVファイルの生成
        nodes_path = self._write_nodes_csv()
        relationships_path = self._write_relationships_csv()

        return {
            "nodes_path": nodes_path,
            "properties_path": relationships_path
        }

    def _write_nodes_csv(self) -> str:
        """ノードCSVの書き出し"""
        nodes_path = os.path.join(self.output_dir, "nodes.csv")

        with open(nodes_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.NODE_HEADERS, quoting=csv.QUOTE_MINIMAL)
            writer.writeheader()

            for node_id in sorted(self.nodes.keys()):
                row = self.nodes[node_id]
                # 不足列は空文字で埋める
                for header in self.NODE_HEADERS:
                    row.setdefault(header, "")
                writer.writerow(row)

        return nodes_path

    def _write_relationships_csv(self) -> str:
        """リレーションCSVの書き出し"""
        rels_path = os.path.join(self.output_dir, "relationships.csv")

        with open(rels_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.REL_HEADERS, quoting=csv.QUOTE_MINIMAL)
            writer.writeheader()

            for (start_id, end_id, rel_type, prop_iri, prop_id) in sorted(self.relationships):
                writer.writerow({
                    ":START_ID": start_id,
                    ":END_ID": end_id,
                    ":TYPE": rel_type,
                    "property_iri": prop_iri,
                    "id": prop_id,
                })

        return rels_path


def load_input_from_file(file_path: str) -> Dict[str, Any]:
    """JSONファイルから入力データを読み込み"""
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    """サンプル実行"""
    # サンプルデータ
    sample_input = {
        "class_info": {
            "id": "extraction_001",
            "classes": [
                {
                    "id": "class_001",
                    "class_iri": "https://github.com/highitoh/kg-extraction#Stakeholder",
                    "label": "店舗スタッフ",
                    "file_id": "file_001"
                },
                {
                    "id": "class_002",
                    "class_iri": "https://github.com/highitoh/kg-extraction#Value",
                    "label": "待ち時間短縮",
                    "file_id": "file_001"
                }
            ]
        },
        "property_info": {
            "id": "extraction_001",
            "properties": [
                {
                    "id": "prop_001",
                    "src_id": "class_001",
                    "property_iri": "https://github.com/highitoh/kg-extraction#hasValue",
                    "dest_id": "class_002"
                }
            ]
        }
    }

    generator = Neo4jCSVGenerator()
    result = generator.generate(sample_input)

    print(f"Generated nodes CSV: {result['nodes_path']}")
    print(f"Generated relationships CSV: {result['properties_path']}")


if __name__ == "__main__":
    main()