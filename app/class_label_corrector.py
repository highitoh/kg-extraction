from typing import Dict, Any, List
import asyncio
import json
import os

from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage
from langchain.schema.runnable import Runnable
from openai import APIConnectionError, RateLimitError, APIError

from logger import Logger


class ClassLabelCorrector(Runnable):
    """
    REVIEW判定されたクラスのラベルを修正または除外するクラス

    処理フロー:
    1. 入力をACCEPT/REVIEW/REJECTに分離
    2. REVIEWクラスのみをLLMで再評価（class_iriごとに並列実行）
    3. LLM出力:
       - revised_label が提供された場合: labelを更新し judgment="REVIEW" のまま
       - should_reject=true の場合: judgment="REJECT" に変更
    4. ACCEPT/REJECT/修正済みREVIEWを統合して出力
    """

    def __init__(self,
                 llm: Any = None,
                 model: str = "gpt-5-mini",
                 temperature: float = 0.0,
                 progress: bool = True,
                 max_concurrency: int = 8,
                 max_retries: int = 5,
                 log_dir: str = "log/class_label_corrector"):
        self.llm = llm or ChatOpenAI(
            model=model,
            temperature=temperature,
            reasoning={"effort": "low"},
            output_version="responses/v1",
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
            n: 判定対象のクラス数
        """
        schema_path = os.path.join(os.path.dirname(__file__), "schemas",
                                   "class-label-corrector", "llm.schema.json")
        with open(schema_path, "r", encoding="utf-8") as f:
            schema = json.load(f)

        # correctionsの配列要素数を設定
        schema["properties"]["corrections"]["minItems"] = n
        schema["properties"]["corrections"]["maxItems"] = n

        return schema

    def _load_metamodel(self) -> dict:
        """メタモデルファイルを読み込む"""
        metamodel_path = os.path.join(os.path.dirname(__file__), "metamodel",
                                      "metamodel.json")
        with open(metamodel_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _build_extract_rules_section(self, target_class_iri: str) -> str:
        """指定されたクラスの抽出ルールセクションを動的に生成

        Args:
            target_class_iri: 対象のクラスIRI
        """
        rules_lines = []

        for cls in self.metamodel.get("classes", []):
            if cls.get("iri", "") == target_class_iri:
                if "extract_rules" in cls and cls.get("extract_rules"):
                    for rule in cls["extract_rules"]:
                        rules_lines.append(f"- {rule}")
                break

        result = "\n".join(rules_lines)
        return result if result else "(抽出ルールなし)"

    def _load_prompt_template(self) -> str:
        """プロンプトテンプレートファイルを読み込み"""
        prompt_path = os.path.join(os.path.dirname(__file__), "prompts",
                                   "class_label_corrector.txt")
        with open(prompt_path, "r", encoding="utf-8") as f:
            return f.read()

    def invoke(self, input: Dict[str, Any], config=None) -> Dict[str, Any]:
        """
        クラスラベル修正処理のメインエントリーポイント

        Parameters
        ----------
        input: ClassLabelCorrectorInput (= ClassFilterOutput)
          - id: 抽出ID
          - classes: クラス個体リスト（judgment/justificationを含む）

        Returns
        -------
        output: ClassLabelCorrectorOutput (= ClassFilterOutput)
        """
        classes = input.get("classes", [])

        # 判定ごとに分離
        accept_classes = [c for c in classes if c.get("judgment") == "ACCEPT"]
        review_classes = [c for c in classes if c.get("judgment") == "REVIEW"]
        reject_classes = [c for c in classes if c.get("judgment") == "REJECT"]

        if self.progress:
            print(f"ClassLabelCorrector: Total {len(classes)} classes "
                  f"(ACCEPT: {len(accept_classes)}, REVIEW: {len(review_classes)}, REJECT: {len(reject_classes)})")

        if not review_classes:
            # REVIEWクラスがない場合はそのまま返す
            if self.progress:
                print("ClassLabelCorrector: No REVIEW classes to process")
            return input

        # class_iriでグループ化
        review_groups = self._group_classes_by_class_iri(review_classes)

        if self.progress:
            print(f"ClassLabelCorrector: {len(review_groups)} class types found for review")

        # REVIEWクラスごとに修正を並列実行
        corrected_classes = asyncio.run(self._process_classes_parallel(review_groups))

        # 出力を構築（ACCEPT + 修正済みREVIEW + REJECT）
        all_classes = accept_classes + corrected_classes + reject_classes

        output = input.copy()
        output["classes"] = all_classes

        if self.progress:
            # 修正結果の内訳を集計
            corrected_review_count = sum(1 for c in corrected_classes if c.get("judgment") == "REVIEW")
            corrected_reject_count = sum(1 for c in corrected_classes if c.get("judgment") == "REJECT")
            print(f"ClassLabelCorrector: Processed {len(review_classes)} REVIEW classes -> "
                  f"REVIEW: {corrected_review_count}, REJECT: {corrected_reject_count}")

        self.logger.save_log(output, filename_prefix="class_label_corrector_output_")

        return output

    async def _process_classes_parallel(self, class_groups: Dict[str, list]) -> list:
        """クラスグループを並列処理"""
        sem = asyncio.Semaphore(self.max_concurrency)
        total = len(class_groups)
        counter = 0
        lock = asyncio.Lock()

        async def _wrapped(class_iri: str, class_instances: list) -> list:
            nonlocal counter
            async with sem:
                class_name = self._get_class_name_from_iri(class_iri)

                async with lock:
                    if self.progress:
                        print(f"ClassLabelCorrector: processing class={class_name}, {len(class_instances)} instances")

                corrected = await self._correct_classes_async(class_instances, class_iri)

                async with lock:
                    counter += 1
                    if self.progress:
                        # 修正結果の内訳を集計
                        review_count = sum(1 for c in corrected if c.get("judgment") == "REVIEW")
                        reject_count = sum(1 for c in corrected if c.get("judgment") == "REJECT")
                        print(f"[{counter}/{total}] ClassLabelCorrector: class={class_name}, "
                              f"{len(class_instances)} instances -> "
                              f"REVIEW: {review_count}, REJECT: {reject_count}")

                return corrected

        results = await asyncio.gather(*[_wrapped(iri, instances) for iri, instances in class_groups.items()])

        # 結果をフラット化
        all_corrected = []
        for result in results:
            all_corrected.extend(result)

        return all_corrected

    def _group_classes_by_class_iri(self, classes: list) -> Dict[str, list]:
        """class_iriでグループ化"""
        class_groups = {}

        for c in classes:
            class_iri = c.get("class_iri", "")
            if class_iri not in class_groups:
                class_groups[class_iri] = []
            class_groups[class_iri].append(c)

        return class_groups

    def _get_class_name_from_iri(self, class_iri: str) -> str:
        """class_iriからクラス名を取得"""
        for cls in self.metamodel.get("classes", []):
            if cls.get("iri", "") == class_iri:
                return cls.get("name", "unknown")
        return class_iri.split("#")[-1] if "#" in class_iri else class_iri

    async def _correct_classes_async(self, classes: list, class_iri: str) -> list:
        """指定されたクラスリストをLLMでラベル修正（非同期版）

        Args:
            classes: 修正対象のクラスインスタンスリスト（すべてREVIEW判定）
            class_iri: 対象クラスのIRI（このクラスのextract_rulesのみを使用）
        """
        if not classes:
            return []

        schema = self._build_schema(len(classes))
        llm_json = self.llm.bind(
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "class_label_corrector",
                    "schema": schema
                }
            })

        # LLMへの入力データを構築
        input_data = []
        for c in classes:
            input_data.append({
                "label": c.get("label", ""),
                "source": c.get("sources", [""])[0] if c.get("sources") else "",
                "justification": c.get("justification", "")
            })

        input_json = json.dumps(input_data, ensure_ascii=False, indent=2)

        # 対象クラスのextract_rulesのみを取得
        extract_rules = self._build_extract_rules_section(class_iri)

        # プロンプトを構築（対象クラス専用の抽出ルールを埋め込む）
        prompt_with_rules = self.prompt_template.replace("{EXTRACT_RULES}", extract_rules)
        prompt = f"{prompt_with_rules}\n{input_json}"

        # LLMでラベル修正実行（リトライ付き）
        response = await self._invoke_with_retry(llm_json, prompt)

        # レスポンスからテキストを抽出
        result_text = self._to_text(response.content)

        # JSON パース
        try:
            parsed = json.loads(result_text)
            corrections = parsed["corrections"]

            # 各クラスに修正結果を適用
            corrected_classes = []
            for c, correction_obj in zip(classes, corrections):
                c_corrected = c.copy()

                should_reject = correction_obj.get("should_reject", False)
                revised_label = correction_obj.get("revised_label", "")
                revision_reason = correction_obj.get("revision_reason", "")

                if should_reject:
                    # 除外する場合はREJECTに変更
                    c_corrected["judgment"] = "REJECT"
                    c_corrected["justification"] = revision_reason
                else:
                    # ラベル修正した場合はREVIEWのまま（ClassFilterで再評価される）
                    if revised_label and revised_label != c.get("label", ""):
                        c_corrected["label"] = revised_label
                    # judgment = "REVIEW" のまま
                    c_corrected["justification"] = revision_reason

                corrected_classes.append(c_corrected)

        except (json.JSONDecodeError, KeyError) as e:
            if self.progress:
                print(f"[ClassLabelCorrector] JSON parse error: {e}, response: {result_text}")
            # エラー時はすべてREJECT扱い
            corrected_classes = []
            for c in classes:
                c_rejected = c.copy()
                c_rejected["judgment"] = "REJECT"
                c_rejected["justification"] = f"JSON parse error: {e}"
                corrected_classes.append(c_rejected)

        return corrected_classes

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
                    print(f"[ClassLabelCorrector Retry {attempt + 1}/{self.max_retries}] API error: {type(e).__name__}. Retrying in {wait_time:.1f}s...")
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

    # ログディレクトリから最新のclass_filterログファイルを取得
    log_dir = "/workspace/app/log/class_filter"
    log_files = glob.glob(os.path.join(log_dir, "class_filter_output_*.json"))

    if not log_files:
        print("Error: No class_filter log files found")
        exit(1)

    # 最新のログファイルを取得
    latest_log = max(log_files, key=os.path.getmtime)
    print(f"Loading input from: {latest_log}")
    with open(latest_log, "r", encoding="utf-8") as f:
        test_input = json.load(f)

    print(f"\nInput data contains {len(test_input.get('classes', []))} classes")

    # Create and run the corrector
    corrector = ClassLabelCorrector(progress=True, log_dir="log/class_label_corrector")
    result = corrector.invoke(test_input)

    print(f"\nOutput data contains {len(result.get('classes', []))} classes")
