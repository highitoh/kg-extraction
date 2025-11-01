from typing import Dict, Any
import asyncio
import json
import os
from functools import wraps

from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage
from langchain.schema.runnable import Runnable
from openai import APIConnectionError, RateLimitError, APIError

from logger import Logger


def async_retry_with_backoff(max_retries: int = 5, initial_delay: float = 1.0, max_delay: float = 60.0):
    """
    非同期関数用のリトライデコレータ（指数バックオフ付き）

    Args:
        max_retries: 最大リトライ回数
        initial_delay: 初回リトライ前の待機時間（秒）
        max_delay: 最大待機時間（秒）
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            delay = initial_delay
            last_exception = None

            for attempt in range(max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except (APIConnectionError, RateLimitError, APIError) as e:
                    last_exception = e

                    if attempt == max_retries:
                        raise

                    # リトライ前に待機
                    wait_time = min(delay, max_delay)
                    print(f"[Retry {attempt + 1}/{max_retries}] API error: {type(e).__name__}. Retrying in {wait_time:.1f}s...")
                    await asyncio.sleep(wait_time)

                    # 指数バックオフ
                    delay *= 2

            if last_exception:
                raise last_exception

        return wrapper
    return decorator


class ClassFilter(Runnable):
    """
    Filter and process extracted classes using LLM
    """

    def __init__(self,
                 llm: Any = None,
                 model: str = "gpt-5-mini",
                 temperature: float = 0.0,
                 progress: bool = True,
                 max_concurrency: int = 8,
                 max_retries: int = 5,
                 log_dir: str = "log/class_filter"):
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
                                   "class-filter", "llm.schema.json")
        with open(schema_path, "r", encoding="utf-8") as f:
            schema = json.load(f)

        # judgmentsの配列要素数を設定
        schema["properties"]["judgments"]["minItems"] = n
        schema["properties"]["judgments"]["maxItems"] = n

        return schema

    def _load_metamodel(self) -> dict:
        """メタモデルファイルを読み込む"""
        metamodel_path = os.path.join(os.path.dirname(__file__), "metamodel",
                                      "metamodel.json")
        with open(metamodel_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _build_prohibit_rules_section(self, target_class_iri: str) -> str:
        """指定されたクラスの禁止ルールセクションを動的に生成

        Args:
            target_class_iri: 対象のクラスIRI
        """
        rules_lines = []

        for cls in self.metamodel.get("classes", []):
            if cls.get("iri", "") == target_class_iri:
                if "prohibit_rules" in cls and cls.get("prohibit_rules"):
                    for rule in cls["prohibit_rules"]:
                        rules_lines.append(f"- {rule}")
                break

        result = "\n".join(rules_lines)
        return result if result else "(禁止ルールなし)"

    def _load_prompt_template(self) -> str:
        """プロンプトテンプレートファイルを読み込み、禁止ルールを動的生成"""
        prompt_path = os.path.join(os.path.dirname(__file__), "prompts",
                                   "class_filter.txt")
        with open(prompt_path, "r", encoding="utf-8") as f:
            template = f.read()

        return template

    def invoke(self, input: Dict[str, Any], config=None) -> Dict[str, Any]:
        classes = input.get("classes", [])

        if self.progress:
            print(f"ClassFilter: processing {len(classes)} classes")

        # class_iriでグループ化
        class_groups = self._group_classes_by_class_iri(classes)

        if self.progress:
            print(f"ClassFilter: {len(class_groups)} class types found")

        # クラスごとにフィルタリングを並列実行
        all_filtered_classes = asyncio.run(self._process_classes_parallel(class_groups))

        # 出力を構築
        output = input.copy()
        output["classes"] = all_filtered_classes

        if self.progress:
            print(
                f"ClassFilter: Total {len(classes)} -> {len(all_filtered_classes)} classes"
            )

        self.logger.save_log(output, filename_prefix="class_filter_output_")

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
                        print(f"ClassFilter: processing class={class_name}, {len(class_instances)} instances")

                filtered = await self._filter_classes_async(class_instances, class_iri)

                async with lock:
                    counter += 1
                    if self.progress:
                        # 判定結果の内訳を集計
                        accept_count = sum(1 for c in filtered if c.get("judgment") == "ACCEPT")
                        review_count = sum(1 for c in filtered if c.get("judgment") == "REVIEW")
                        reject_count = len(class_instances) - len(filtered)
                        print(f"[{counter}/{total}] ClassFilter: class={class_name}, "
                              f"{len(class_instances)} -> {len(filtered)} instances "
                              f"(ACCEPT: {accept_count}, REVIEW: {review_count}, REJECT: {reject_count})")

                return filtered

        results = await asyncio.gather(*[_wrapped(iri, instances) for iri, instances in class_groups.items()])

        # 結果をフラット化
        all_filtered = []
        for result in results:
            all_filtered.extend(result)

        return all_filtered

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

    async def _filter_classes_async(self, classes: list, class_iri: str) -> list:
        """指定されたクラスリストをLLMでフィルタリング（非同期版）

        Args:
            classes: フィルタリング対象のクラスインスタンスリスト
            class_iri: 対象クラスのIRI（このクラスのprohibit_rulesのみを使用）
        """
        if not classes:
            return []

        schema = self._build_schema(len(classes))
        llm_json = self.llm.bind(
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "class_filter",
                    "schema": schema
                }
            })

        # labelのみを抽出してプロンプトに渡す
        labels = [c.get("label", "") for c in classes]
        labels_json = json.dumps(labels, ensure_ascii=False, indent=2)

        # 対象クラスのprohibit_rulesのみを取得
        prohibit_rules = self._build_prohibit_rules_section(class_iri)

        # プロンプトを構築（対象クラス専用の禁止ルールを埋め込む）
        prompt_with_rules = self.prompt_template.replace("{PROHIBIT_RULES}", prohibit_rules)
        prompt = f"{prompt_with_rules}\n{labels_json}"

        # LLMでフィルタリング実行（リトライ付き）
        response = await self._invoke_with_retry(llm_json, prompt)

        # レスポンスからテキストを抽出
        result_text = self._to_text(response.content)

        # JSON パース
        try:
            parsed = json.loads(result_text)
            judgments = parsed["judgments"]

            # 各クラスに判定結果を追加
            filtered_classes = []
            for c, judgment_obj in zip(classes, judgments):
                judgment = judgment_obj.get("judgment", "REJECT")
                justification = judgment_obj.get("justification", "")

                # REJECT以外のクラスを通過させる（ACCEPT + REVIEW）
                if judgment != "REJECT":
                    # クラスに判定結果を追加
                    c_with_judgment = c.copy()
                    c_with_judgment["judgment"] = judgment
                    c_with_judgment["justification"] = justification
                    filtered_classes.append(c_with_judgment)

        except (json.JSONDecodeError, KeyError) as e:
            if self.progress:
                print(f"[ClassFilter] JSON parse error: {e}, response: {result_text}")
            # エラー時は全てREJECT扱いで通過させない
            filtered_classes = []

        return filtered_classes

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
                    print(f"[ClassFilter Retry {attempt + 1}/{self.max_retries}] API error: {type(e).__name__}. Retrying in {wait_time:.1f}s...")
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
    import json
    import glob
    import os

    # Find the latest ClassExtractor log file
    log_pattern = "log/class_extractor/class_extractor_output_*.json"
    log_files = glob.glob(log_pattern)

    if not log_files:
        print(f"No log files found matching pattern: {log_pattern}")
        exit(1)

    # Get the most recent file
    latest_log = max(log_files, key=os.path.getmtime)
    print(f"Loading test input from: {latest_log}")

    # Load the JSON data
    with open(latest_log, "r", encoding="utf-8") as f:
        test_input = json.load(f)

    print(f"\nInput data contains {len(test_input.get('classes', []))} classes")

    # Create and run the filter
    class_filter = ClassFilter(progress=True, log_dir="log/class_filter")
    result = class_filter.invoke(test_input)

    print(f"\nOutput data contains {len(result.get('classes', []))} classes")
