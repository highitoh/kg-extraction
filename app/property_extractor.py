from typing import Dict, Any, List, Union
import asyncio
import json
import os
import uuid

from langchain.schema.runnable import Runnable
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from openai import APIConnectionError, RateLimitError, APIError

from logger import Logger

class PropertyExtractor(Runnable):
    """
    プロパティ（関係）抽出
    入力: PropertyChainInput (class_info, metamodel)
    出力: PropertyChainOutput (id, properties)
    """

    def __init__(
        self,
        llm: ChatOpenAI | None = None,
        model: str = "gpt-5-mini",
        temperature: float = 0.0,
        max_concurrency: int = 8,
        max_retries: int = 5,
        progress: bool = True,
        log_dir: str = "log/property_extractor",
        batch_size: int = 20,
    ):
        self.llm = llm or ChatOpenAI(
            model=model,
            temperature=temperature,
            reasoning={"effort": "minimal"},
            output_version="responses/v1",
        )

        # JSONスキーマを読み込み
        schema = self._load_schema()
        self.llm_json = self.llm.bind(response_format={"type": "json_schema", "json_schema": {"name": "property_extractor", "schema": schema}})
        self.max_concurrency = max_concurrency
        self.max_retries = max_retries
        self.progress = progress
        self.batch_size = batch_size

        # メタモデルを読み込み
        self.metamodel = self._load_metamodel()

        # プロンプトを読み込み
        self.prompt = self._load_prompt()

        # Logger設定
        self.logger = Logger(log_dir)

    def _load_schema(self) -> dict:
        """JSONスキーマファイルを読み込む"""
        schema_path = os.path.join(os.path.dirname(__file__), "schemas", "property-extractor", "llm.schema.json")
        with open(schema_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _load_metamodel(self) -> dict:
        """メタモデルファイルを読み込む"""
        metamodel_path = os.path.join(os.path.dirname(__file__), "metamodel", "metamodel.json")
        with open(metamodel_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _load_prompt(self) -> str:
        """プロンプトを読み込み"""
        prompt_path = os.path.join(os.path.dirname(__file__), "prompts", "property_extractor.txt")
        with open(prompt_path, "r", encoding="utf-8") as f:
            return f.read()

    @staticmethod
    def _to_text(c: Union[str, List[Any]]) -> str:
        """LLMの出力をテキスト形式に変換"""
        if isinstance(c, str):
            return c
        for part in c:
            if isinstance(part, dict) and part.get("type") in ("output_text", "text"):
                return part.get("text", "")
        return ""

    async def _extract_properties_for_batch(
        self,
        property_label: str,
        property_iri: str,
        property_definition: str,
        src_class_label: str,
        dest_class_label: str,
        src_class_definition: str,
        dest_class_definition: str,
        src_batch: List[Dict[str, Any]],
        dest_batch: List[Dict[str, Any]]
    ) -> List[Dict[str, str]]:
        """バッチ単位でプロパティ抽出を実行"""

        # LLMに渡す入力を整形
        src_texts = [f"- ID: {c['id']}, Label: {c['label']}" for c in src_batch]
        dest_texts = [f"- ID: {c['id']}, Label: {c['label']}" for c in dest_batch]

        # プロンプトテンプレートにデータを埋め込み
        prompt_text = self.prompt.format(
            src_class_label=src_class_label,
            dest_class_label=dest_class_label,
            property_label=property_label,
            src_class_definition=src_class_definition,
            dest_class_definition=dest_class_definition,
            property_definition=property_definition,
            src_list=os.linesep.join(src_texts),
            dest_list=os.linesep.join(dest_texts)
        )

        msg = HumanMessage(content=[{"type": "text", "text": prompt_text}])

        ai = await self._invoke_with_retry(msg)
        raw = self._to_text(ai.content)
        try:
            data = json.loads(raw)
        except Exception:
            data = {}

        # プロパティ情報を整理
        properties = []
        extracted_properties = data.get("properties", [])
        if isinstance(extracted_properties, list):
            for prop in extracted_properties:
                if not isinstance(prop, dict):
                    continue
                src_id = prop.get("src_id", "")
                dest_id = prop.get("dest_id", "")
                if src_id and dest_id:
                    properties.append({
                        "src_id": src_id.strip(),
                        "property_iri": property_iri,
                        "dest_id": dest_id.strip()
                    })

        return properties

    async def _invoke_with_retry(self, msg: HumanMessage):
        """リトライ機能付きのLLM呼び出し"""
        delay = 1.0
        max_delay = 60.0
        last_exception = None

        for attempt in range(self.max_retries + 1):
            try:
                return await self.llm_json.ainvoke([msg])
            except (APIConnectionError, RateLimitError, APIError) as e:
                last_exception = e

                if attempt == self.max_retries:
                    raise

                wait_time = min(delay, max_delay)
                if self.progress:
                    print(f"[PropertyExtractor Retry {attempt + 1}/{self.max_retries}] API error: {type(e).__name__}. Retrying in {wait_time:.1f}s...")
                await asyncio.sleep(wait_time)
                delay *= 2

        if last_exception:
            raise last_exception

    async def _extract_properties_for_one_type(
        self,
        property_def: Dict[str, Any],
        classes: List[Dict[str, Any]]
    ) -> List[Dict[str, str]]:
        """特定のプロパティタイプに対してクラス間の関係を抽出"""

        property_label = property_def.get("name", "")
        property_iri = property_def.get("iri", "")
        property_definition = property_def.get("description", "")
        src_class_iri = property_def.get("src_class", "")
        dest_class_iri = property_def.get("dest_class", "")

        # メタモデルからクラス定義を取得
        classes_def = self.metamodel.get("classes", [])
        src_class_def = next((c for c in classes_def if c.get("iri") == src_class_iri), None)
        dest_class_def = next((c for c in classes_def if c.get("iri") == dest_class_iri), None)

        if not src_class_def or not dest_class_def:
            return []

        src_class_label = src_class_def.get("name", "")
        dest_class_label = dest_class_def.get("name", "")
        src_class_definition = src_class_def.get("description", "")
        dest_class_definition = dest_class_def.get("description", "")

        # 対象クラスのインスタンスを抽出
        src_classes = [c for c in classes if c.get("class_iri") == src_class_iri]
        dest_classes = [c for c in classes if c.get("class_iri") == dest_class_iri]

        # 該当するクラスインスタンスがない場合はスキップ
        if not src_classes or not dest_classes:
            return []

        # クラスをバッチサイズで分割
        src_batches = [
            src_classes[i:i + self.batch_size]
            for i in range(0, len(src_classes), self.batch_size)
        ]
        dest_batches = [
            dest_classes[i:i + self.batch_size]
            for i in range(0, len(dest_classes), self.batch_size)
        ]

        # 全バッチの組み合わせを生成
        tasks = []
        for src_batch in src_batches:
            for dest_batch in dest_batches:
                tasks.append(
                    self._extract_properties_for_batch(
                        property_label=property_label,
                        property_iri=property_iri,
                        property_definition=property_definition,
                        src_class_label=src_class_label,
                        dest_class_label=dest_class_label,
                        src_class_definition=src_class_definition,
                        dest_class_definition=dest_class_definition,
                        src_batch=src_batch,
                        dest_batch=dest_batch
                    )
                )

        # 並列実行
        results = await asyncio.gather(*tasks)

        # 全結果を統合
        all_properties = []
        for result in results:
            all_properties.extend(result)

        return all_properties

    async def _extract_properties_from_classes(self, classes: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        """クラス情報からプロパティ候補を抽出"""

        # メタモデルから全プロパティ定義を取得
        properties_def = self.metamodel.get("properties", [])

        if not properties_def:
            return []

        # 各プロパティタイプに対して並列処理
        tasks = [
            self._extract_properties_for_one_type(property_def, classes)
            for property_def in properties_def
        ]

        results = await asyncio.gather(*tasks)

        # 全結果を統合
        all_properties = []
        for result in results:
            all_properties.extend(result)

        return all_properties

    def invoke(self, input: Dict[str, Any], config=None) -> Dict[str, Any]:
        """PropertyChainInputからプロパティを抽出してPropertyChainOutputを出力"""
        class_info = input["class_info"]
        classes = class_info.get("classes", [])

        # プロパティ抽出実行
        extracted_properties = asyncio.run(self._extract_properties_from_classes(classes))

        # 出力形式に合わせて整理
        properties = []
        properties_with_labels = []
        for prop in extracted_properties:
            prop_id = str(uuid.uuid4())

            # スキーマ準拠のプロパティ情報
            property_item = {
                "id": prop_id,
                "src_id": prop["src_id"],
                "property_iri": prop["property_iri"],
                "dest_id": prop["dest_id"],
            }
            properties.append(property_item)

            # ログ出力用にラベル情報を追加
            src_class = next((c for c in classes if c["id"] == prop["src_id"]), None)
            dest_class = next((c for c in classes if c["id"] == prop["dest_id"]), None)

            src_label = src_class["label"] if src_class else prop["src_id"]
            dest_label = dest_class["label"] if dest_class else prop["dest_id"]

            properties_with_labels.append({
                **property_item,
                "src_label": src_label,
                "dest_label": dest_label,
            })

        output: Dict[str, Any] = {
            "id": str(uuid.uuid4()),
            "properties": properties,
        }

        if self.progress:
            total_properties = len(properties)
            print(f"PropertyExtractor: extracted {total_properties} properties")

        # Loggerでpropertyチェーンの出力を保存（ラベル情報付き）
        log_output = {
            "id": output["id"],
            "properties": properties_with_labels,
        }
        self.logger.save_log(log_output, filename_prefix="property_extractor_output_")

        return output


if __name__ == "__main__":
    import glob

    # Create PropertyExtractor instance
    extractor = PropertyExtractor()

    # Load the latest ClassFilter output JSON
    class_files = sorted(glob.glob("log/class_consolidator/class_consolidator_output_*.json"), reverse=True)

    if not class_files:
        print("Error: No ClassFilter output files found in log/class_consolidator/")
        exit(1)

    latest_file = class_files[0]
    print(f"Loading latest ClassFilter output: {latest_file}")

    with open(latest_file, "r", encoding="utf-8") as f:
        class_info = json.load(f)

    # Prepare input data for PropertyExtractor
    sample_input = {
        "class_info": class_info,
        "metamodel": {}
    }

    print("=== PropertyExtractor Test ===")
    print(f"Input: {len(class_info.get('classes', []))} classes loaded")
    print("\n" + "="*50 + "\n")

    try:
        # Execute PropertyExtractor
        result = extractor.invoke(sample_input)

        print("Output:")
        print(json.dumps(result, ensure_ascii=False, indent=2))

        print(f"\nSummary:")
        print(f"- Total properties extracted: {len(result['properties'])}")

        # Display each property relationship
        for i, prop in enumerate(result["properties"], 1):
            # Find source and destination class labels for display
            src_class = next((c for c in class_info["classes"] if c["id"] == prop["src_id"]), None)
            dest_class = next((c for c in class_info["classes"] if c["id"] == prop["dest_id"]), None)

            src_label = src_class["label"] if src_class else prop["src_id"]
            dest_label = dest_class["label"] if dest_class else prop["dest_id"]

            print(f"- Property {i}: {src_label} --[{prop['property_iri']}]--> {dest_label}")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
