# class_extractor_template.py
from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Dict, Any, List, Optional
import uuid

VIEW_TYPES = [
    "value_analysis",
    "business_concept",
    "business_requirement",
    "system_requirement",
    "system_component",
    "data_analysis",
]


@dataclass
class ExtractedClass:
    """ClassExtractOutput.classes の 1 要素"""
    id: str              # 必須: クラス個体ID
    class_iri: str       # 必須: URI 形式
    label: str           # 必須: 抽出クラス（表示ラベル）
    file_id: str         # 必須: 抽出元ファイルID

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ClassExtractor:
    """
    ビュー（value_analysis 等）を横断してクラス候補を抽出する
    - 入力/出力は添付スキーマに準拠（ClassExtractorInput → ClassExtractorOutput）
    - 実ロジック（LLM/規則/辞書等）は _extract_labels_from_view() に後で実装
    - class_iri の決定規則は _resolve_class_iri() に後で実装
    """

    def __init__(self, progress: bool = True):
        self.progress = progress

    def invoke(self, input_data: Dict[str, Any], config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Parameters
        ----------
        input_data: ClassExtractorInput に準拠
          - view_info: ビュー抽出結果（各ビューに texts 等が入っている想定）
          - metamodel: メタモデル（クラスIRI決定で利用する想定・詳細は後段で実装）
        Returns
        -------
        output: ClassCExtractorOutput に準拠
        """
        view_info = self._get_view_info(input_data)
        metamodel = self._get_metamodel(input_data)

        classes: List[ExtractedClass] = []

        # ビューごとに抽出
        for view_type in VIEW_TYPES:
            texts = self._get_texts_for_view(view_info, view_type)
            if self.progress:
                print(f"[ClassExtractor] view={view_type}, texts={len(texts)}")

            # --- (未実装) ビュー内の「ラベル候補（=クラス名）」を抽出 ----------------
            # 返却想定: List[Dict] 例: [{"label": "顧客", "file_id": "file-001"}, ...]
            label_items = self._extract_labels_from_view(view_type, texts, metamodel)

            # --- class_iri 付与＆スキーマ整形 ---------------------------------------
            for li in label_items:
                label = li["label"]
                file_id = li["file_id"]
                class_iri = self._resolve_class_iri(view_type, label, metamodel)  # URI を返す実装にする
                classes.append(ExtractedClass(
                    id=str(uuid.uuid4()),
                    class_iri=class_iri,
                    label=label,
                    file_id=file_id,
                ))

        # ---- 出力（ClassChainOutput 準拠） -----------------------------------------
        output: Dict[str, Any] = {
            "id": str(uuid.uuid4()),
            "classes": [c.to_dict() for c in classes],  # スキーマ: minItems=1 を満たすのは実装後
        }
        return output

    def _get_view_info(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        if "view_info" not in input_data or not isinstance(input_data["view_info"], dict):
            raise ValueError("ClassChainInput: 'view_info' が不足しています。")
        return input_data["view_info"]

    def _get_metamodel(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        if "metamodel" not in input_data or not isinstance(input_data["metamodel"], dict):
            raise ValueError("ClassChainInput: 'metamodel' が不足しています。")
        return input_data["metamodel"]

    def _get_texts_for_view(self, view_info: Dict[str, Any], view_type: str) -> List[Dict[str, Any]]:
        """
        view_info から特定 view_type の texts をフラットに取得するテンプレート。
        期待する最小構造（例）:
        view_info = {
          "views": [
            { "type": "business_concept",
              "texts": [ {"file_id": "file-001", "text": "..."}, ... ] },
            ...
          ]
        }
        """
        views = view_info.get("views", [])
        for v in views:
            if v.get("type") == view_type:
                return v.get("texts", []) or []
        return []

    # ---- Extension points (あとで実装) ------------------------------------------
    def _extract_labels_from_view(
        self,
        view_type: str,
        texts: List[Dict[str, Any]],
        metamodel: Dict[str, Any],
    ) -> List[Dict[str, str]]:
        """
        ビュー内の文章群から「クラス名候補（ラベル）」を抽出する処理を後で実装。
        Returns (想定):
          [{"label": "<名詞句など>", "file_id": "<抽出元ファイルID>"} , ...]
        """
        # TODO: LLM/規則/辞書等を用いた抽出ロジックを実装
        raise NotImplementedError("ラベル抽出ロジックは後で実装してください。")

    def _resolve_class_iri(
        self,
        view_type: str,
        label: str,
        metamodel: Dict[str, Any],
    ) -> str:
        """
        抽出したラベルに対応する class_iri(URI) を決める処理を後で実装。
        - metamodel 参照でクラスIRIを引く/生成するなど
        必ず URI を返すこと（出力スキーマ要件）。
        """
        # TODO: 例: metamodel からビュー種別→クラスIRI の既定マッピングを引く／なければ既定命名
        # 例の既定命名: f"http://example.com/metamodel#{view_type.capitalize()}Class"
        raise NotImplementedError("class_iri の決定ロジックは後で実装してください。")


if __name__ == "__main__":
    import json
    import glob
    import os

    # ログディレクトリから最新のview_filterログファイルを取得
    log_dir = "/workspace/app/log/view_filter"
    log_files = glob.glob(os.path.join(log_dir, "view_filter_output_*.json"))

    if not log_files:
        print("Error: No view_filter log files found")
        exit(1)

    # 最新のログファイルを取得
    latest_log = max(log_files, key=os.path.getmtime)
    print(f"Loading input from: {latest_log}")

    with open(latest_log, "r", encoding="utf-8") as f:
        view_info = json.load(f)

    sample_input = {
        "view_info": view_info,
        "metamodel": {
            # 実際はメタモデルの定義を入れる想定（スキーマ参照）
        },
    }

    extractor = ClassExtractor(progress=True)
    out = extractor.invoke(sample_input)
    print(out)
