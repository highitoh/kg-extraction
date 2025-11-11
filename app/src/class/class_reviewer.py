from typing import Dict, Any
import asyncio
import json
import os
from functools import wraps

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_core.runnables import Runnable
from openai import APIConnectionError, RateLimitError, APIError

from src.utils.logger import Logger
from src.utils.llm_factory import LLMFactory


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


class ClassReviewer(Runnable):
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
                 log_dir: str = "/workspace/app/log/class_reviewer"):
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
            n: 判定対象のクラス数
        """
        # Go up 2 levels from src/class/ to /workspace/app/
        app_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        schema_path = os.path.join(app_dir, "schemas",
                                   "class-reviewer", "llm.schema.json")
        with open(schema_path, "r", encoding="utf-8") as f:
            schema = json.load(f)

        # judgmentsの配列要素数を設定
        schema["properties"]["judgments"]["minItems"] = n
        schema["properties"]["judgments"]["maxItems"] = n

        return schema

    def _build_duplicate_schema(self) -> dict:
        """重複検出用LLMスキーマファイルを読み込み

        Returns:
            dict: 重複検出用のJSONスキーマ
        """
        # Go up 2 levels from src/class/ to /workspace/app/
        app_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        schema_path = os.path.join(app_dir, "schemas",
                                   "class-reviewer", "llm-duplicate.schema.json")
        with open(schema_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _load_metamodel(self) -> dict:
        """メタモデルファイルを読み込む"""
        # Go up 2 levels from src/class/ to /workspace/app/
        app_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        metamodel_path = os.path.join(app_dir, "metamodel",
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
        # Go up 2 levels from src/class/ to /workspace/app/
        app_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        prompt_path = os.path.join(app_dir, "prompts",
                                   "class_reviewer.txt")
        with open(prompt_path, "r", encoding="utf-8") as f:
            template = f.read()

        return template

    def _load_duplicate_prompt_template(self) -> str:
        """重複検出用プロンプトテンプレートファイルを読み込み"""
        # Go up 2 levels from src/class/ to /workspace/app/
        app_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        prompt_path = os.path.join(app_dir, "prompts",
                                   "class_reviewer_duplicate.txt")
        with open(prompt_path, "r", encoding="utf-8") as f:
            return f.read()

    def _convert_duplicate_response_to_dict(self, parsed_response: dict) -> dict:
        """LLM応答を内部形式の辞書に変換

        Args:
            parsed_response: LLMからのJSON応答（duplicate_groupsを含む）

        Returns:
            dict: グループIDをキーとした辞書
                {
                    "dup_group_0": {
                        "class_ids": [...],
                        "labels": [...],
                        "reason": "..."
                    },
                    ...
                }
        """
        result = {}
        duplicate_groups = parsed_response.get("duplicate_groups", [])

        for i, group in enumerate(duplicate_groups):
            group_id = f"dup_group_{i}"
            result[group_id] = {
                "class_ids": group.get("class_ids", []),
                "labels": group.get("labels", []),
                "reason": group.get("reason", "")
            }

        return result

    async def _detect_duplicates_async(self, classes: list, class_iri: str) -> dict:
        """指定されたクラスリストで重複を検出（非同期版）

        Args:
            classes: 重複検出対象のクラスインスタンスリスト
            class_iri: 対象クラスのIRI

        Returns:
            dict: 重複グループの辞書
                {
                    "dup_group_0": {
                        "class_ids": ["cls_001", "cls_003"],
                        "labels": ["User", "System User"],
                        "reason": "..."
                    },
                    ...
                }
        """
        if not classes or len(classes) < 2:
            # 1つ以下の場合は重複なし
            return {}

        # スキーマを構築
        schema = self._build_duplicate_schema()
        llm_json = self.llm.bind(
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "class_reviewer_duplicate",
                    "schema": schema
                }
            }
        )

        # クラスIDとラベルを抽出
        class_data = [
            {"id": c.get("id", ""), "label": c.get("label", "")}
            for c in classes
        ]
        class_data_json = json.dumps(class_data, ensure_ascii=False, indent=2)

        # プロンプトテンプレートを読み込み
        prompt_template = self._load_duplicate_prompt_template()
        prompt = f"{prompt_template}\n{class_data_json}"

        # LLMで重複検出実行（リトライ付き）
        try:
            response = await self._invoke_with_retry(llm_json, prompt)

            # レスポンスからテキストを抽出
            result_text = self._to_text(response.content)

            # JSON パース
            parsed = json.loads(result_text)

            # 内部形式に変換
            duplicate_groups = self._convert_duplicate_response_to_dict(parsed)

            return duplicate_groups

        except (json.JSONDecodeError, KeyError) as e:
            if self.progress:
                print(f"[ClassReviewer] Duplicate detection JSON parse error: {e}")
            # エラー時は重複なしとして扱う
            return {}
        except Exception as e:
            if self.progress:
                print(f"[ClassReviewer] Duplicate detection error: {e}")
            return {}

    def invoke(self, input: Dict[str, Any], config=None) -> Dict[str, Any]:
        classes = input.get("classes", [])

        if self.progress:
            print(f"[ClassReviewer] Processing {len(classes)} classes")

        # class_iriでグループ化
        class_groups = self._group_classes_by_class_iri(classes)

        if self.progress:
            print(f"[ClassReviewer] {len(class_groups)} class types found")

        # クラスごとにフィルタリングを並列実行
        all_filtered_classes = asyncio.run(self._process_classes_parallel(class_groups))

        # 出力を構築
        output = input.copy()
        output["classes"] = all_filtered_classes

        if self.progress:
            print(
                f"[ClassReviewer] Completed: {len(classes)} -> {len(all_filtered_classes)} classes"
            )

        self.logger.save_log(output, filename_prefix="class_reviewer_output_")

        return output

    def _merge_review_results(self, classes: list, prohibit_judgments: list, duplicate_groups: dict) -> list:
        """禁止ルールレビューと重複検出の結果をマージ

        Args:
            classes: 元のクラスインスタンスリスト
            prohibit_judgments: 禁止ルールレビューの判定結果リスト
            duplicate_groups: 重複グループの辞書

        Returns:
            list: マージ後のクラスリスト（REJECT以外のみ）
        """
        results = []

        for i, cls in enumerate(classes):
            class_id = cls.get("id", "")

            # 禁止ルールレビューの結果を取得
            judgment = prohibit_judgments[i].get("judgment", "REJECT")
            justification = prohibit_judgments[i].get("justification", "")

            # 重複グループに含まれるかチェック
            found_in_duplicate = False
            for group_id, group_info in duplicate_groups.items():
                if class_id in group_info.get("class_ids", []):
                    found_in_duplicate = True

                    # 重複が見つかった場合の処理
                    if judgment != "REVIEW":
                        # 禁止ルールでREVIEWでない場合、重複理由でREVIEWに変更
                        judgment = "REVIEW"
                        justification = f"Duplicate detected: {group_info.get('reason', '')}"
                    # else: 禁止ルールでREVIEWの場合は、そのjustificationを優先

                    # 重複フィールドを追加
                    cls["duplicate_group_id"] = group_id
                    # 自分以外のラベルを抽出
                    cls["duplicate_labels"] = [
                        label for cid, label in zip(
                            group_info.get("class_ids", []),
                            group_info.get("labels", [])
                        )
                        if cid != class_id
                    ]
                    break

            # REJECT以外のクラスのみ結果に含める
            if judgment != "REJECT":
                cls["judgment"] = judgment
                cls["justification"] = justification
                results.append(cls)

        return results

    async def _process_classes_parallel(self, class_groups: Dict[str, list]) -> list:
        """クラスグループを並列処理（禁止ルールレビューと重複検出を並行実行）"""
        sem = asyncio.Semaphore(self.max_concurrency)
        total = len(class_groups)
        counter = 0
        lock = asyncio.Lock()

        if self.progress:
            print(f"[ClassReviewer] Starting parallel processing: {total} class types (concurrency={self.max_concurrency})")

        async def _wrapped(class_iri: str, class_instances: list) -> list:
            nonlocal counter
            async with sem:
                class_name = self._get_class_name_from_iri(class_iri)

                # 禁止ルールレビューと重複検出を並列実行
                prohibit_judgments, duplicate_groups = await asyncio.gather(
                    self._review_prohibit_rules_async(class_instances, class_iri),
                    self._detect_duplicates_async(class_instances, class_iri)
                )

                # 結果をマージ
                merged = self._merge_review_results(class_instances, prohibit_judgments, duplicate_groups)

                async with lock:
                    counter += 1
                    if self.progress:
                        # 判定結果の内訳を集計
                        accept_count = sum(1 for c in merged if c.get("judgment") == "ACCEPT")
                        review_count = sum(1 for c in merged if c.get("judgment") == "REVIEW")
                        reject_count = len(class_instances) - len(merged)
                        duplicate_count = len(duplicate_groups)

                        print(f"[{counter}/{total}] ClassReviewer: {class_name} processed "
                              f"({len(class_instances)} -> {len(merged)} instances) "
                              f"[ACCEPT: {accept_count}, REVIEW: {review_count}, REJECT: {reject_count}] "
                              f"[Duplicate groups: {duplicate_count}]")

                return merged

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

    async def _review_prohibit_rules_async(self, classes: list, class_iri: str) -> list:
        """指定されたクラスリストをLLMで禁止ルールレビュー（非同期版）

        Args:
            classes: レビュー対象のクラスインスタンスリスト
            class_iri: 対象クラスのIRI（このクラスのprohibit_rulesのみを使用）

        Returns:
            list: 判定結果のリスト [{"judgment": "ACCEPT", "justification": "..."}, ...]
        """
        if not classes:
            return []

        schema = self._build_schema(len(classes))
        llm_json = self.llm.bind(
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "class_reviewer",
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
            return judgments

        except (json.JSONDecodeError, KeyError) as e:
            if self.progress:
                print(f"[ClassReviewer] Prohibit rules review JSON parse error: {e}, response: {result_text}")
            # エラー時は全てREJECT扱い
            return [{"judgment": "REJECT", "justification": "Parse error"} for _ in classes]

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
                    print(f"[ClassReviewer Retry {attempt + 1}/{self.max_retries}] API error: {type(e).__name__}. Retrying in {wait_time:.1f}s...")
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
    log_pattern = "/workspace/app/log/class_extractor/class_extractor_output_*.json"
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
    class_reviewer = ClassReviewer(progress=True, log_dir="/workspace/app/log/class_reviewer")
    result = class_reviewer.invoke(test_input)

    print(f"\nOutput data contains {len(result.get('classes', []))} classes")
