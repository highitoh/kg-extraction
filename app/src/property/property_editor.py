from typing import Dict, Any, List
import asyncio
import json
import os

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_core.runnables import Runnable
from openai import APIConnectionError, RateLimitError, APIError

from src.utils.logger import Logger
from src.utils.llm_factory import LLMFactory


class PropertyEditor(Runnable):
    """
    REVIEW判定されたプロパティの強度を修正または除外するクラス

    処理フロー:
    1. 入力をACCEPT/REVIEW/REJECTに分離
    2. REVIEWプロパティのみをLLMで再評価（property_iriごとに並列実行）
    3. LLM出力:
       - revised_strength が 1-3 の場合: strengthを更新し judgment="REVIEW" のまま
       - revised_strength=0 の場合: judgment="REJECT" に変更
    4. ACCEPT/REJECT/修正済みREVIEWを統合して出力
    """

    def __init__(self,
                 llm: Any = None,
                 model: str = "gpt-5-mini",
                 temperature: float = 0.0,
                 progress: bool = True,
                 max_concurrency: int = 8,
                 max_retries: int = 5,
                 log_dir: str = "/workspace/app/log/property_editor"):
        self.llm = llm or LLMFactory.create_llm(
            model=model,
            temperature=temperature,
            reasoning_effort="low",
        )
        self.progress = progress
        self.max_concurrency = max_concurrency
        self.max_retries = max_retries
        self.logger = Logger(log_dir)

        # メタモデルを読み込み
        self.metamodel = self._load_metamodel()
        # プロンプトテンプレートを読み込み
        self.prompt_template = self._load_prompt_template()

    def _build_schema(self, n: int) -> dict:
        """LLMスキーマファイルを読み込み、要素数を設定

        Args:
            n: 判定対象のプロパティ数
        """
        # Go up 2 levels from src/XXX/ to /workspace/app/

        app_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

        schema_path = os.path.join(app_dir, "schemas",
                                   "property-editor", "llm.schema.json")
        with open(schema_path, "r", encoding="utf-8") as f:
            schema = json.load(f)

        # correctionsの配列要素数を設定
        schema["properties"]["corrections"]["minItems"] = n
        schema["properties"]["corrections"]["maxItems"] = n

        return schema

    def _load_metamodel(self) -> dict:
        """メタモデルファイルを読み込む"""
        # Go up 2 levels from src/XXX/ to /workspace/app/

        app_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

        metamodel_path = os.path.join(app_dir, "metamodel",
                                      "metamodel.json")
        with open(metamodel_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _build_property_rules_section(self, target_property_iri: str) -> str:
        """指定されたプロパティのルールセクションを動的に生成

        Args:
            target_property_iri: 対象のプロパティIRI
        """
        rules_lines = []

        for prop in self.metamodel.get("properties", []):
            if prop.get("iri", "") == target_property_iri:
                # property_rulesがあればそれを使用
                if "property_rules" in prop and prop.get("property_rules"):
                    for rule in prop["property_rules"]:
                        rules_lines.append(f"- {rule}")
                # prohibit_rulesも含める
                if "prohibit_rules" in prop and prop.get("prohibit_rules"):
                    rules_lines.append("\n【除外ルール】")
                    for rule in prop["prohibit_rules"]:
                        rules_lines.append(f"- {rule}")
                break

        result = "\n".join(rules_lines)
        return result if result else "(プロパティルールなし)"

    def _load_prompt_template(self) -> str:
        """プロンプトテンプレートファイルを読み込み"""
        # Go up 2 levels from src/XXX/ to /workspace/app/

        app_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

        prompt_path = os.path.join(app_dir, "prompts",
                                   "property_editor.txt")
        with open(prompt_path, "r", encoding="utf-8") as f:
            return f.read()

    def invoke(self, input: Dict[str, Any], config=None) -> Dict[str, Any]:
        """
        プロパティ強度修正処理のメインエントリーポイント

        Parameters
        ----------
        input: PropertyEditorInput (= PropertyReviewerOutput)
          - id: 抽出ID
          - properties: プロパティリスト（judgment/justificationを含む）

        Returns
        -------
        output: PropertyEditorOutput (= PropertyReviewerOutput)
        """
        properties = input.get("properties", [])

        # 判定ごとに分離
        accept_properties = [p for p in properties if p.get("judgment") == "ACCEPT"]
        review_properties = [p for p in properties if p.get("judgment") == "REVIEW"]
        reject_properties = [p for p in properties if p.get("judgment") == "REJECT"]

        if self.progress:
            print(f"[PropertyEditor] Total {len(properties)} properties "
                  f"[ACCEPT: {len(accept_properties)}, REVIEW: {len(review_properties)}, REJECT: {len(reject_properties)}]")

        if not review_properties:
            # REVIEWプロパティがない場合はそのまま返す
            if self.progress:
                print("[PropertyEditor] No REVIEW properties to process")
            return input

        # property_iriでグループ化
        review_groups = self._group_properties_by_property_iri(review_properties)

        if self.progress:
            print(f"[PropertyEditor] {len(review_groups)} property types found for review")

        # REVIEWプロパティごとに修正を並列実行
        corrected_properties = asyncio.run(self._process_properties_parallel(review_groups))

        # 出力を構築（ACCEPT + 修正済みREVIEW + REJECT）
        all_properties = accept_properties + corrected_properties + reject_properties

        output = input.copy()
        output["properties"] = all_properties

        if self.progress:
            # 修正結果の内訳を集計
            corrected_review_count = sum(1 for p in corrected_properties if p.get("judgment") == "REVIEW")
            corrected_reject_count = sum(1 for p in corrected_properties if p.get("judgment") == "REJECT")
            print(f"[PropertyEditor] Completed: Processed {len(review_properties)} REVIEW properties "
                  f"[REVIEW: {corrected_review_count}, REJECT: {corrected_reject_count}]")

        self.logger.save_log(output, filename_prefix="property_editor_output_")

        return output

    async def _process_properties_parallel(self, property_groups: Dict[str, list]) -> list:
        """プロパティグループを並列処理"""
        sem = asyncio.Semaphore(self.max_concurrency)
        total = len(property_groups)
        counter = 0
        lock = asyncio.Lock()

        if self.progress:
            print(f"[PropertyEditor] Starting parallel processing: {total} property types (concurrency={self.max_concurrency})")

        async def _wrapped(property_iri: str, property_instances: list) -> list:
            nonlocal counter
            async with sem:
                property_name = self._get_property_name_from_iri(property_iri)

                corrected = await self._correct_properties_async(property_instances, property_iri)

                async with lock:
                    counter += 1
                    if self.progress:
                        # 修正結果の内訳を集計
                        review_count = sum(1 for p in corrected if p.get("judgment") == "REVIEW")
                        reject_count = sum(1 for p in corrected if p.get("judgment") == "REJECT")
                        print(f"[{counter}/{total}] PropertyEditor: {property_name} processed "
                              f"({len(property_instances)} instances) "
                              f"[REVIEW: {review_count}, REJECT: {reject_count}]")

                return corrected

        results = await asyncio.gather(*[_wrapped(iri, instances) for iri, instances in property_groups.items()])

        # 結果をフラット化
        all_corrected = []
        for result in results:
            all_corrected.extend(result)

        return all_corrected

    def _group_properties_by_property_iri(self, properties: list) -> Dict[str, list]:
        """property_iriでグループ化"""
        property_groups = {}

        for p in properties:
            property_iri = p.get("property_iri", "")
            if property_iri not in property_groups:
                property_groups[property_iri] = []
            property_groups[property_iri].append(p)

        return property_groups

    def _get_property_name_from_iri(self, property_iri: str) -> str:
        """property_iriからプロパティ名を取得"""
        for prop in self.metamodel.get("properties", []):
            if prop.get("iri", "") == property_iri:
                return prop.get("name", "unknown")
        return property_iri.split("#")[-1] if "#" in property_iri else property_iri

    async def _correct_properties_async(self, properties: list, property_iri: str) -> list:
        """指定されたプロパティリストをLLMで強度修正（非同期版）

        Args:
            properties: 修正対象のプロパティインスタンスリスト（すべてREVIEW判定）
            property_iri: 対象プロパティのIRI（このプロパティのproperty_rulesのみを使用）
        """
        if not properties:
            return []

        schema = self._build_schema(len(properties))
        llm_json = self.llm.bind(
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "property_editor",
                    "schema": schema
                }
            })

        # LLMへの入力データを構築
        input_data = []
        for p in properties:
            # sourcesフィールドがあればそれを使用、なければ空文字列
            source_text = ""
            if "sources" in p and p["sources"]:
                source_text = p["sources"][0]

            input_data.append({
                "src_label": p.get("src_label", ""),
                "property_iri": p.get("property_iri", ""),
                "dest_label": p.get("dest_label", ""),
                "strength": p.get("strength", 0),
                "source": source_text,
                "justification": p.get("justification", "")
            })

        input_json = json.dumps(input_data, ensure_ascii=False, indent=2)

        # 対象プロパティのproperty_rulesのみを取得
        property_rules = self._build_property_rules_section(property_iri)

        # プロンプトを構築（対象プロパティ専用のルールを埋め込む）
        prompt_with_rules = self.prompt_template.replace("{PROPERTY_RULES}", property_rules)
        prompt = f"{prompt_with_rules}\n{input_json}"

        # LLMで強度修正実行（リトライ付き）
        response = await self._invoke_with_retry(llm_json, prompt)

        # レスポンスからテキストを抽出
        result_text = self._to_text(response.content)

        # JSON パース
        try:
            parsed = json.loads(result_text)
            corrections = parsed["corrections"]

            # 各プロパティに修正結果を適用
            corrected_properties = []
            for p, correction_obj in zip(properties, corrections):
                p_corrected = p.copy()

                revised_strength = correction_obj.get("revised_strength", 0)
                revision_reason = correction_obj.get("revision_reason", "")

                if revised_strength == 0:
                    # 強度0の場合はREJECTに変更
                    p_corrected["judgment"] = "REJECT"
                    p_corrected["strength"] = 0
                    p_corrected["justification"] = revision_reason
                else:
                    # 強度修正した場合はREVIEWのまま（PropertyReviewerで再評価される）
                    p_corrected["strength"] = revised_strength
                    # judgment = "REVIEW" のまま
                    p_corrected["justification"] = revision_reason

                corrected_properties.append(p_corrected)

        except (json.JSONDecodeError, KeyError) as e:
            if self.progress:
                print(f"[PropertyEditor] JSON parse error: {e}, response: {result_text}")
            # エラー時はすべてREJECT扱い
            corrected_properties = []
            for p in properties:
                p_rejected = p.copy()
                p_rejected["judgment"] = "REJECT"
                p_rejected["strength"] = 0
                p_rejected["justification"] = f"JSON parse error: {e}"
                corrected_properties.append(p_rejected)

        return corrected_properties

    async def _invoke_with_retry(self, llm_json, prompt: str):
        """リトライ機能付きのLLM呼び出し"""
        delay = 1.0
        max_delay = 60.0
        last_exception = None

        for attempt in range(self.max_retries + 1):
            try:
                return await llm_json.ainvoke([HumanMessage(content=prompt)])
            except (APIConnectionError, RateLimitError, APIError) as e:
                last_exception = e

                if attempt == self.max_retries:
                    raise

                wait_time = min(delay, max_delay)
                if self.progress:
                    print(f"[PropertyEditor Retry {attempt + 1}/{self.max_retries}] API error: {type(e).__name__}. Retrying in {wait_time:.1f}s...")
                await asyncio.sleep(wait_time)
                delay *= 2

        if last_exception:
            raise last_exception

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


if __name__ == "__main__":
    import glob

    # ログディレクトリから最新のproperty_reviewerログファイルを取得
    log_dir = "/workspace/app/log/property_reviewer"
    log_files = glob.glob(os.path.join(log_dir, "property_reviewer_output_*.json"))

    if not log_files:
        print("Error: No property_reviewer log files found")
        exit(1)

    # 最新のログファイルを取得
    latest_log = max(log_files, key=os.path.getmtime)
    print(f"Loading input from: {latest_log}")
    with open(latest_log, "r", encoding="utf-8") as f:
        test_input = json.load(f)

    print(f"\nInput data contains {len(test_input.get('properties', []))} properties")

    # Create and run the editor
    editor = PropertyEditor(progress=True, log_dir="/workspace/app/log/property_editor")
    result = editor.invoke(test_input)

    print(f"\nOutput data contains {len(result.get('properties', []))} properties")
