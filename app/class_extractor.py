from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Dict, Any, List, Optional
import json
import os
import uuid

from langchain_openai import ChatOpenAI

from logger import Logger

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
    source: str          # 必須: 抽出元文章
    file_id: str         # 必須: 抽出元ファイルID

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ClassExtractor:
    """
    ビュー記述からクラス候補を抽出するクラス
    """

    def __init__(
        self,
        llm: Any = None,
        model: str = "gpt-5-mini",
        temperature: float = 0.0,
        progress: bool = True,
        log_dir: str = "log/class_extractor"
    ):
        self.llm = llm or ChatOpenAI(
            model=model,
            temperature=temperature,
            reasoning={"effort": "minimal"},
            output_version="responses/v1",
        )
        self.progress = progress
        self.logger = Logger(log_dir)

        # JSONスキーマを読み込み
        schema = self._load_schema()
        self.llm_json = self.llm.bind(response_format={"type": "json_schema", "json_schema": {"name": "class_extractor", "schema": schema}})

        # プロンプトテンプレートを読み込み
        self.prompt_template = self._load_prompt_template()

    def _load_schema(self) -> dict:
        """JSONスキーマファイルを読み込む"""
        schema_path = os.path.join(os.path.dirname(__file__), "schemas", "class-extractor", "llm.schema.json")
        with open(schema_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _load_prompt_template(self) -> str:
        """プロンプトテンプレートファイルを読み込む"""
        prompt_path = os.path.join(os.path.dirname(__file__), "prompts", "class_extractor.txt")
        with open(prompt_path, "r", encoding="utf-8") as f:
            return f.read()

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

            # ビュー記述から「ラベル候補（=クラス名）」を抽出
            label_items = self._extract_labels_from_view(view_type, texts, metamodel)

            # class_iri 付与＆スキーマ整形
            for li in label_items:
                c = li["class"]
                label = li["label"]
                source = li["source"]
                file_id = li["file_id"]
                class_iri = self._resolve_class_iri(view_type, c, metamodel)  # URI を返す実装にする
                classes.append(ExtractedClass(
                    id=str(uuid.uuid4()),
                    class_iri=class_iri,
                    label=label,
                    source=source,
                    file_id=file_id,
                ))

        # ---- 出力（ClassChainOutput 準拠） -----------------------------------------
        output: Dict[str, Any] = {
            "id": str(uuid.uuid4()),
            "classes": [c.to_dict() for c in classes],  # スキーマ: minItems=1 を満たすのは実装後
        }

        if self.progress:
            print(f"ClassExtractor: extracted {len(classes)} classes")

        # Loggerを使ってClassExtractorOutputをログ出力
        self.logger.save_log(output, filename_prefix="class_extractor_output_")

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
        """
        views = view_info.get("views", [])
        for v in views:
            if v.get("type") == view_type:
                return v.get("texts", []) or []
        return []

    def _extract_labels_from_view(
        self,
        view_type: str,
        texts: List[Dict[str, Any]],
        metamodel: Dict[str, Any],
    ) -> List[Dict[str, str]]:
        """
        対象ビューの記述からクラス候補を抽出する処理。
        Returns:
          [{"class": "<クラス>", "label": "<インスタンス>", "source": "<出所>", "file_id": "<抽出元ファイルID>"} , ...]
        """
        if view_type == "value_analysis":
            return self._extract_value_analysis_labels(texts, metamodel)

        # その他のビューは空リストを返す（未実装）
        return []

    def _extract_value_analysis_labels(self, texts: List[Dict[str, Any]], metamodel: Dict[str, Any] = None) -> List[Dict[str, str]]:
        """
        value_analysis ビューから stakeholder と value をLLMで抽出する
        """
        from langchain.schema import HumanMessage

        if not texts:
            return []

        # ビュー記述のテキストを取得
        chunk_lines = []
        for idx, t in enumerate(texts):
            chunk_lines.append(f"{t.get('text', '')}")
        chunks_text = "\n".join(chunk_lines)

        # metamodelからクラス定義とポリシーの文字列を生成
        class_definitions = self._get_class_definitions(metamodel, "value_analysis")
        class_policies = self._get_class_policies(metamodel, "value_analysis")

        # プロンプトテンプレートに値を埋め込む
        prompt = self.prompt_template.format(
            class_definitions=class_definitions,
            class_policies=class_policies
        )
        prompt = f"{prompt}\n\n【テキスト】\n{chunks_text}\n"

        response = self.llm_json.invoke([HumanMessage(content=prompt)])

        # response.content からテキストを抽出
        result_text = self._to_text(response.content)

        # JSON パース
        try:
            result = json.loads(result_text)
        except json.JSONDecodeError:
            if self.progress:
                print(f"[ClassExtractor] JSON parse error: {result_text}")
            return []

        # ラベル候補リストに変換
        labels = []
        value_items = result.get("classes", [])
        for item in value_items:
            if "class" in item and "label" in item and "source" in item:
                labels.append({
                    "class": item["class"],
                    "label": item["label"],
                    "source": item["source"],
                    "file_id": texts[0].get("file_id", "unknown")  # 便宜的に最初のファイルIDを使用
                })

        return labels

    def _get_class_definitions(self, metamodel: Dict[str, Any], view_type: str) -> str:
        """
        metamodelから指定されたview_typeのクラス定義文字列を生成

        Parameters
        ----------
        metamodel : Dict[str, Any]
            メタモデル定義
        view_type : str
            対象ビュータイプ

        Returns
        -------
        str
            クラス定義の文字列（例: "- ステークホルダ: 説明文\n- 価値: 説明文"）
        """
        if not metamodel or "classes" not in metamodel:
            return ""

        classes = metamodel.get("classes", [])
        view_classes = [cls for cls in classes if cls.get("view_type") == view_type]

        # クラス定義を構築
        definitions = []
        for cls in view_classes:
            name = cls.get("name", "")
            description = cls.get("description", "")
            definitions.append(f"- {name}: {description}")

        return "\n".join(definitions)

    def _get_class_policies(self, metamodel: Dict[str, Any], view_type: str) -> str:
        """
        metamodelから指定されたview_typeのクラスポリシー文字列を生成

        Parameters
        ----------
        metamodel : Dict[str, Any]
            メタモデル定義
        view_type : str
            対象ビュータイプ

        Returns
        -------
        str
            クラスポリシーの文字列
        """
        if not metamodel or "classes" not in metamodel:
            return ""

        classes = metamodel.get("classes", [])
        view_classes = [cls for cls in classes if cls.get("view_type") == view_type]

        # ポリシーを構築
        policies = []
        for cls in view_classes:
            name = cls.get("name", "")
            policy_items = []

            # 文字数制約
            min_len = cls.get("min_length")
            max_len = cls.get("max_length")
            if min_len is not None and max_len is not None:
                policy_items.append(f"  - 文字数: {min_len}～{max_len}字")

            # フォーマットガイドライン（配列対応）
            format_guidelines = cls.get("format_guidelines", [])
            for guideline in format_guidelines:
                policy_items.append(f"  - {guideline}")

            if policy_items:
                policies.append(f"- {name}")
                policies.extend(policy_items)

        return "\n".join(policies)

    @staticmethod
    def _to_text(content: Any) -> str:
        """LLMレスポンスからテキストを抽出"""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    return part.get("text", "")
        return ""

    def _resolve_class_iri(
        self,
        view_type: str,
        class_name: str,
        metamodel: Dict[str, Any],
    ) -> str:
        """
        抽出したクラスに対応する class_iri(URI) を決める処理。
        - metamodel 参照でクラスIRIを引く／なければ既定命名
        必ず URI を返すこと（出力スキーマ要件）。
        """
        # metamodel から class_iri のマッピングを探す
        classes = metamodel.get("classes", [])
        for cls in classes:
            if cls.get("view_type") == view_type and cls.get("label") == class_name:
                return cls.get("iri", "")

        # マッピングが見つからない場合は既定命名
        # ラベルをURLエンコード用に変換
        from urllib.parse import quote
        encoded_label = quote(class_name)
        return f"http://example.com/ontology/{view_type}/{encoded_label}"


if __name__ == "__main__":
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
            "classes": [
                {
                    "iri": "http://example.com/ontology/Stakeholder",
                    "name": "ステークホルダ",
                    "description": "開発するシステムやサービスにより、価値を提供または受け取る主体。組織や役割などで表す",
                    "view_type": "value_analysis",
                    "min_length": 2,
                    "max_length": 10,
                    "format_guidelines": [
                        "個人名は避け、役割名を優先"
                    ]
                },
                {
                    "iri": "http://example.com/ontology/Value",
                    "name": "価値",
                    "description": "開発するシステムやサービスにより、ステークホルダが得る便益や達成したい状態を表す",
                    "view_type": "value_analysis",
                    "min_length": 4,
                    "max_length": 30,
                    "format_guidelines": [
                        "『〜の向上』『効率化』など名詞句で簡潔に"
                    ]
                }
            ],
            "properties": []
        },
    }

    extractor = ClassExtractor(progress=True)
    out = extractor.invoke(sample_input)
    print(out)
