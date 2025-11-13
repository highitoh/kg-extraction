from typing import Dict, Any, List, Union
import asyncio
import json
import os
import uuid

from langchain_core.runnables import Runnable
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from openai import APIConnectionError, RateLimitError, APIError

from src.utils.logger import Logger
from src.utils.llm_factory import LLMFactory


class PropertyConsolidator(Runnable):
    """
    プロパティ候補を統合するクラス
    新規抽出プロパティと既存プロパティをLLMで類似性を判定し統合する

    処理フロー:
    1. クラスIDマッピングを構築（ClassConsolidatorで統合されたクラスに対応）
    2. 既存プロパティのsrc_id/dest_idを正規化
    3. 新規＋既存プロパティをproperty_iriごとにグループ化
    4. LLMで類似プロパティを検出
    5. 類似プロパティをマージ（最高強度を選択）
    """

    def __init__(
        self,
        llm: ChatOpenAI | None = None,
        model: str = "gpt-5-mini",
        temperature: float = 0.0,
        max_concurrency: int = 8,
        max_retries: int = 5,
        progress: bool = True,
        log_dir: str = "/workspace/app/log/property_consolidator"
    ):
        self.llm = llm or LLMFactory.create_llm(
            model=model,
            temperature=temperature,
            reasoning_effort="minimal",
        )
        self.progress = progress
        self.max_concurrency = max_concurrency
        self.max_retries = max_retries
        self.logger = Logger(log_dir)

        # メタモデルを読み込み
        self.metamodel = self._load_metamodel()
        # プロンプトテンプレートを読み込み
        self.prompt_template = self._load_prompt_template()

    def _load_metamodel(self) -> dict:
        """メタモデルファイルを読み込む"""
        # Go up 2 levels from src/XXX/ to /workspace/app/

        app_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

        metamodel_path = os.path.join(app_dir, "metamodel", "metamodel.json")
        with open(metamodel_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _load_prompt_template(self) -> str:
        """プロンプトテンプレートファイルを読み込み"""
        # Go up 2 levels from src/XXX/ to /workspace/app/

        app_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

        prompt_path = os.path.join(app_dir, "prompts", "property_consolidator.txt")
        with open(prompt_path, "r", encoding="utf-8") as f:
            return f.read()

    def _build_schema(self) -> dict:
        """LLM用のJSONスキーマを読み込む"""
        # Go up 2 levels from src/XXX/ to /workspace/app/

        app_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

        schema_path = os.path.join(app_dir, "schemas", "property-consolidator", "llm.schema.json")
        with open(schema_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def invoke(self, input: Dict[str, Any], config=None) -> Dict[str, Any]:
        """
        プロパティ統合処理のメインエントリーポイント

        Parameters
        ----------
        input: PropertyConsolidatorInput
          - id: 抽出ID
          - properties: 新規抽出プロパティリスト
          - existing_properties: 既存プロパティリスト（オプション）
          - class_info: クラス情報（ID正規化用）

        Returns
        -------
        output: PropertyConsolidatorOutput (= PropertyExtractor output format)
          - id: 抽出ID
          - properties: 統合済みプロパティリスト
        """
        new_properties = input.get("properties", [])
        existing_properties = input.get("existing_properties", [])
        class_info = input.get("class_info", {})

        if self.progress:
            print(f"[PropertyConsolidator] Processing {len(new_properties)} new + {len(existing_properties)} existing properties")

        # 既存プロパティがない場合は新規プロパティをそのまま返す
        if not existing_properties:
            if self.progress:
                print("[PropertyConsolidator] No existing properties to merge")
            return {
                "id": input.get("id", str(uuid.uuid4())),
                "properties": new_properties
            }

        # クラスIDマッピングを構築
        class_id_mapping = self._build_class_id_mapping(class_info)

        # 既存プロパティのIDを正規化
        normalized_existing = self._normalize_property_ids(existing_properties, class_id_mapping)

        # Filter out properties with invalid class references
        valid_normalized = []
        invalid_properties = []
        class_ids_set = set(class_id_mapping.values())  # All valid canonical IDs

        for prop in normalized_existing:
            src_id = prop.get("src_id", "")
            dest_id = prop.get("dest_id", "")

            if src_id in class_ids_set and dest_id in class_ids_set:
                valid_normalized.append(prop)
            else:
                # Track invalid properties for logging
                invalid_properties.append({
                    "property": prop,
                    "reason": f"src_id or dest_id not in current classes (src_id={src_id}, dest_id={dest_id})"
                })

        if invalid_properties and self.progress:
            print(f"[PropertyConsolidator] Filtered {len(invalid_properties)} properties with invalid class references")

        # 新規＋既存を統合
        all_properties = new_properties + valid_normalized

        if self.progress:
            print(f"[PropertyConsolidator] Total {len(all_properties)} properties to consolidate")

        # property_iriでグループ化
        property_groups = self._group_by_property_iri(all_properties)

        if self.progress:
            print(f"[PropertyConsolidator] {len(property_groups)} property types found")

        # プロパティごとに統合を並列実行
        all_consolidated_properties = asyncio.run(
            self._process_properties_parallel(property_groups, class_info)
        )

        # 出力を構築
        output = {
            "id": input.get("id", str(uuid.uuid4())),
            "properties": all_consolidated_properties
        }

        if self.progress:
            print(f"[PropertyConsolidator] Completed: {len(all_properties)} -> {len(all_consolidated_properties)} properties")

        self.logger.save_log(output, filename_prefix="property_consolidator_output_")

        return output

    def _build_class_id_mapping(self, class_info: Dict[str, Any]) -> Dict[str, str]:
        """
        クラスIDマッピングを構築
        consolidated_idsを使用して、マージされたクラスIDを正規IDに変換

        Returns:
            mapping: {class_id: canonical_id} for all classes
                     Includes both current classes (identity) and merged IDs
        """
        mapping = {}

        for cls in class_info.get("classes", []):
            canonical_id = cls.get("id")
            if canonical_id:
                # Current class: identity mapping
                mapping[canonical_id] = canonical_id

                # Merged classes: map old IDs to this canonical ID
                consolidated_ids = cls.get("consolidated_ids", [])
                for old_id in consolidated_ids:
                    mapping[old_id] = canonical_id

        return mapping

    def _normalize_property_ids(
        self,
        properties: List[Dict[str, Any]],
        class_mapping: Dict[str, str]
    ) -> List[Dict[str, Any]]:
        """
        プロパティのsrc_id/dest_idを正規化
        古いクラスIDを新しい正規IDに変換
        """
        normalized = []
        for prop in properties:
            normalized_prop = prop.copy()
            normalized_prop["src_id"] = class_mapping.get(
                prop.get("src_id", ""),
                prop.get("src_id", "")
            )
            normalized_prop["dest_id"] = class_mapping.get(
                prop.get("dest_id", ""),
                prop.get("dest_id", "")
            )
            normalized.append(normalized_prop)
        return normalized

    def _group_by_property_iri(self, properties: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """property_iriでグループ化"""
        groups = {}
        for prop in properties:
            prop_iri = prop.get("property_iri", "")
            if prop_iri not in groups:
                groups[prop_iri] = []
            groups[prop_iri].append(prop)
        return groups

    def _get_property_name_from_iri(self, property_iri: str) -> str:
        """property_iriからプロパティ名を取得"""
        for prop in self.metamodel.get("properties", []):
            if prop.get("iri", "") == property_iri:
                return prop.get("name", "unknown")
        return property_iri.split("#")[-1] if "#" in property_iri else property_iri

    async def _process_properties_parallel(
        self,
        property_groups: Dict[str, List[Dict[str, Any]]],
        class_info: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """プロパティグループを並列処理"""
        sem = asyncio.Semaphore(self.max_concurrency)
        total = len(property_groups)
        counter = 0
        lock = asyncio.Lock()

        if self.progress:
            print(f"[PropertyConsolidator] Starting parallel processing: {total} property types (concurrency={self.max_concurrency})")

        async def _wrapped(property_iri: str, properties: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
            nonlocal counter
            async with sem:
                property_name = self._get_property_name_from_iri(property_iri)

                consolidated = await self._consolidate_properties_async(
                    properties, property_iri, class_info
                )

                async with lock:
                    counter += 1
                    if self.progress:
                        print(f"[{counter}/{total}] PropertyConsolidator: {property_name} processed ({len(properties)} -> {len(consolidated)} properties)")

                return consolidated

        results = await asyncio.gather(
            *[_wrapped(iri, props) for iri, props in property_groups.items()]
        )

        # 結果をフラット化
        all_consolidated = []
        for result in results:
            all_consolidated.extend(result)

        return all_consolidated

    async def _consolidate_properties_async(
        self,
        properties: List[Dict[str, Any]],
        property_iri: str,
        class_info: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        同じproperty_iriのプロパティリストを統合（非同期版）

        Args:
            properties: 統合対象のプロパティリスト
            property_iri: 対象プロパティのIRI
            class_info: クラス情報（ラベル取得用）

        Returns:
            統合後のプロパティリスト
        """
        if not properties:
            return []

        # プロパティが1つだけの場合は統合不要
        if len(properties) == 1:
            return properties

        # クラスインデックスを作成
        class_index = self._index_classes_by_id(class_info)

        # LLMで類似プロパティを検出
        try:
            similar_groups = await self._detect_similar_properties_async(
                properties, property_iri, class_index
            )
        except Exception as e:
            if self.progress:
                print(f"[PropertyConsolidator] LLM detection failed for {property_iri}: {e}")
            # LLM失敗時は重複なしとして全て個別に返す
            return properties

        # 各グループをマージ
        consolidated_properties = []
        processed_indices = set()

        for group_indices in similar_groups:
            if not group_indices:
                continue

            group_properties = [properties[i] for i in group_indices if i < len(properties)]
            if not group_properties:
                continue

            # グループをマージ
            merged = self._merge_properties(group_properties)
            consolidated_properties.append(merged)

            # 処理済みインデックスを記録
            processed_indices.update(group_indices)

        # マージされなかったプロパティを追加
        for i, prop in enumerate(properties):
            if i not in processed_indices:
                consolidated_properties.append(prop)

        return consolidated_properties

    async def _detect_similar_properties_async(
        self,
        properties: List[Dict[str, Any]],
        property_iri: str,
        class_index: Dict[str, Dict[str, Any]]
    ) -> List[List[int]]:
        """
        LLMを使用して類似プロパティを検出

        Args:
            properties: プロパティリスト
            property_iri: プロパティIRI
            class_index: クラスID->クラス情報のマッピング

        Returns:
            類似プロパティのインデックスグループのリスト
            例: [[0, 2, 5], [1, 3], [4]] - 0,2,5は類似、1,3は類似、4は独立
        """
        # プロパティ情報をフォーマット
        property_list_text = self._format_properties_for_llm(
            properties, property_iri, class_index
        )

        # プロンプトを構築
        property_name = self._get_property_name_from_iri(property_iri)
        prompt = self.prompt_template.format(
            property_name=property_name,
            property_iri=property_iri,
            property_list=property_list_text
        )

        # LLMスキーマを構築
        schema = self._build_schema()
        llm_json = self.llm.bind(
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "property_consolidator",
                    "schema": schema
                }
            }
        )

        msg = HumanMessage(content=[{"type": "text", "text": prompt}])

        # LLMでグループ化を実行（リトライ付き）
        response = await self._invoke_with_retry(llm_json, msg)

        # レスポンスからテキストを抽出
        result_text = self._to_text(response.content)

        # JSON パース
        try:
            parsed = json.loads(result_text)
            groups = parsed.get("groups", [])
        except json.JSONDecodeError:
            if self.progress:
                print(f"[PropertyConsolidator] JSON parse error: {result_text}")
            return []

        # グループのインデックスリストを抽出
        similar_groups = []
        for group in groups:
            indices = group.get("indices", [])
            if len(indices) > 1:  # 2つ以上のプロパティが類似している場合のみ
                similar_groups.append(indices)

        return similar_groups

    def _format_properties_for_llm(
        self,
        properties: List[Dict[str, Any]],
        property_iri: str,
        class_index: Dict[str, Dict[str, Any]]
    ) -> str:
        """LLM用にプロパティリストをフォーマット"""
        lines = []
        for i, prop in enumerate(properties):
            src_id = prop.get("src_id", "")
            dest_id = prop.get("dest_id", "")

            src = class_index.get(src_id, {})
            dest = class_index.get(dest_id, {})

            src_label = src.get("label", "N/A")
            dest_label = dest.get("label", "N/A")

            # ソーステキストも含める（利用可能な場合）
            src_sources = src.get("sources", [])
            dest_sources = dest.get("sources", [])
            src_source = src_sources[0] if src_sources else ""
            dest_source = dest_sources[0] if dest_sources else ""

            lines.append(f"{i}:")
            lines.append(f"  source: {src_label}")
            lines.append(f"    context: {src_source[:100]}" if src_source else "")
            lines.append(f"  destination: {dest_label}")
            lines.append(f"    context: {dest_source[:100]}" if dest_source else "")
            lines.append(f"  strength: {prop.get('strength', 0)}")
            lines.append("")

        return "\n".join(lines)

    def _merge_properties(self, group_properties: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        プロパティグループをマージ
        最高強度のプロパティを選択

        Args:
            group_properties: マージ対象のプロパティリスト

        Returns:
            マージ後のプロパティ
        """
        if len(group_properties) == 1:
            return group_properties[0]

        # 強度でソート（降順）
        sorted_props = sorted(
            group_properties,
            key=lambda p: p.get("strength", 0),
            reverse=True
        )

        # 最高強度のプロパティを返す
        return sorted_props[0].copy()

    async def _invoke_with_retry(self, llm_json, msg: HumanMessage):
        """リトライ機能付きのLLM呼び出し"""
        delay = 1.0
        max_delay = 60.0
        last_exception = None

        for attempt in range(self.max_retries + 1):
            try:
                return await llm_json.ainvoke([msg])
            except (APIConnectionError, RateLimitError, APIError) as e:
                last_exception = e

                if attempt == self.max_retries:
                    raise

                wait_time = min(delay, max_delay)
                if self.progress:
                    print(f"[PropertyConsolidator Retry {attempt + 1}/{self.max_retries}] API error: {type(e).__name__}. Retrying in {wait_time:.1f}s...")
                await asyncio.sleep(wait_time)
                delay *= 2

        if last_exception:
            raise last_exception

    @staticmethod
    def _to_text(content: Union[str, List[Any]]) -> str:
        """LLMレスポンスからテキストを抽出"""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    return part.get("text", "")
        return ""

    @staticmethod
    def _index_classes_by_id(class_info: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """クラスID->クラス情報のマッピングを作成"""
        index = {}
        for cls in class_info.get("classes", []):
            cls_id = cls.get("id")
            if cls_id:
                index[cls_id] = cls
        return index


if __name__ == "__main__":
    import glob

    # ログディレクトリから最新のproperty_extractorログファイルを取得
    extractor_log_dir = "/workspace/app/log/property_extractor"
    extractor_log_files = glob.glob(os.path.join(extractor_log_dir, "property_extractor_output_*.json"))

    # クラスコンソリデータログを取得
    class_log_dir = "/workspace/app/log/class_consolidator"
    class_log_files = glob.glob(os.path.join(class_log_dir, "class_consolidator_output_*.json"))

    if not extractor_log_files or not class_log_files:
        print("Error: Required log files not found")
        print(f"  PropertyExtractor logs: {len(extractor_log_files)}")
        print(f"  ClassConsolidator logs: {len(class_log_files)}")
        exit(1)

    # 最新のログファイルを取得
    latest_extractor_log = max(extractor_log_files, key=os.path.getmtime)
    latest_class_log = max(class_log_files, key=os.path.getmtime)

    print(f"Loading PropertyExtractor output from: {latest_extractor_log}")
    print(f"Loading ClassConsolidator output from: {latest_class_log}")

    with open(latest_extractor_log, "r", encoding="utf-8") as f:
        property_data = json.load(f)

    with open(latest_class_log, "r", encoding="utf-8") as f:
        class_data = json.load(f)

    # テスト入力データを構築
    sample_input = {
        "id": property_data.get("id", "test-id"),
        "properties": property_data.get("properties", []),
        "existing_properties": [],  # 既存プロパティなしでテスト
        "class_info": class_data
    }

    print(f"\nInput data:")
    print(f"  - New properties: {len(sample_input['properties'])}")
    print(f"  - Existing properties: {len(sample_input['existing_properties'])}")
    print(f"  - Classes: {len(class_data.get('classes', []))}")

    # PropertyConsolidatorを実行
    consolidator = PropertyConsolidator(progress=True)
    result = consolidator.invoke(sample_input)

    print(f"\nOutput data:")
    print(f"  - Consolidated properties: {len(result.get('properties', []))}")
    print(json.dumps(result, ensure_ascii=False, indent=2)[:500])
