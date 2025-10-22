from typing import Dict, Any, List
import asyncio
import json
import os

from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage
from langchain.schema.runnable import Runnable
from openai import APIConnectionError, RateLimitError, APIError

from logger import Logger


class ClassConsolidator(Runnable):
    """
    類似したクラスラベルを集約するクラス
    LLMで同一概念を判定し、正準ラベルを選択して統合する
    """

    def __init__(
        self,
        llm: Any = None,
        model: str = "gpt-5-mini",
        temperature: float = 0.0,
        progress: bool = True,
        max_concurrency: int = 8,
        max_retries: int = 5,
        log_dir: str = "log/class_consolidator"
    ):
        self.llm = llm or ChatOpenAI(
            model=model,
            temperature=temperature,
            reasoning={"effort": "minimal"},
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

    def _load_metamodel(self) -> dict:
        """メタモデルファイルを読み込む"""
        metamodel_path = os.path.join(os.path.dirname(__file__), "metamodel", "metamodel.json")
        with open(metamodel_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _load_prompt_template(self) -> str:
        """プロンプトテンプレートファイルを読み込み"""
        prompt_path = os.path.join(os.path.dirname(__file__), "prompts", "class_consolidator.txt")
        with open(prompt_path, "r", encoding="utf-8") as f:
            return f.read()

    def _build_schema(self) -> dict:
        """LLM用のJSONスキーマを読み込む"""
        schema_path = os.path.join(os.path.dirname(__file__), "schemas", "class-consolidator", "llm.schema.json")
        with open(schema_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def invoke(self, input: Dict[str, Any], config=None) -> Dict[str, Any]:
        """
        クラス集約処理のメインエントリーポイント

        Parameters
        ----------
        input: ClassConsolidatorInput (= ClassChainOutput)
          - id: 抽出ID
          - classes: クラス個体リスト

        Returns
        -------
        output: ClassConsolidatorOutput (= ClassChainOutput)
        """
        classes = input.get("classes", [])

        if self.progress:
            print(f"ClassConsolidator: processing {len(classes)} classes")

        # class_iriでグループ化
        class_groups = self._group_classes_by_class_iri(classes)

        if self.progress:
            print(f"ClassConsolidator: {len(class_groups)} class types found")

        # クラスごとに集約を並列実行
        all_consolidated_classes = asyncio.run(self._process_classes_parallel(class_groups))

        # 出力を構築
        output = input.copy()
        output["classes"] = all_consolidated_classes

        if self.progress:
            print(f"ClassConsolidator: Total {len(classes)} -> {len(all_consolidated_classes)} classes")

        self.logger.save_log(output, filename_prefix="class_consolidator_output_")

        return output

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
                        print(f"ClassConsolidator: processing class={class_name}, {len(class_instances)} instances")

                consolidated = await self._consolidate_classes_async(class_instances, class_iri)

                async with lock:
                    counter += 1
                    if self.progress:
                        print(f"[{counter}/{total}] ClassConsolidator: class={class_name}, {len(class_instances)} -> {len(consolidated)} instances")

                return consolidated

        results = await asyncio.gather(*[_wrapped(iri, instances) for iri, instances in class_groups.items()])

        # 結果をフラット化
        all_consolidated = []
        for result in results:
            all_consolidated.extend(result)

        return all_consolidated

    async def _consolidate_classes_async(self, classes: list, class_iri: str) -> list:
        """
        指定されたクラスリストをLLMで集約（非同期版）

        Args:
            classes: 集約対象のクラスインスタンスリスト
            class_iri: 対象クラスのIRI

        Returns:
            集約後のクラスインスタンスリスト
        """
        if not classes:
            return []

        # インスタンスが1つだけの場合は集約不要
        if len(classes) == 1:
            return classes

        # スキーマを構築
        schema = self._build_schema()
        llm_json = self.llm.bind(
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "class_consolidator",
                    "schema": schema
                }
            }
        )

        # ラベルリストを作成
        labels = [c.get("label", "") for c in classes]
        label_list_text = "\n".join([f"{i}: {label}" for i, label in enumerate(labels)])

        # クラス名と正規化ルールを取得
        class_name = self._get_class_name_from_iri(class_iri)
        consolidation_rules = self._build_consolidation_rules(class_iri)

        # プロンプトを構築
        prompt = self.prompt_template.format(
            class_name=class_name,
            consolidation_rules=consolidation_rules,
            label_list=label_list_text
        )

        # LLMで集約実行（リトライ付き）
        response = await self._invoke_with_retry(llm_json, prompt)

        # レスポンスからテキストを抽出
        result_text = self._to_text(response.content)

        # JSON パース
        try:
            parsed = json.loads(result_text)
            groups = parsed.get("groups", [])
        except json.JSONDecodeError:
            if self.progress:
                print(f"[ClassConsolidator] JSON parse error: {result_text}")
            return classes

        # グループ情報から統合後のクラスインスタンスを生成
        consolidated_classes = []
        for group in groups:
            indices = group.get("indices", [])
            canonical_label = group.get("canonical_label", "")
            reason = group.get("reason", "")

            if not indices or not canonical_label:
                continue

            # グループ内のクラスインスタンスを取得
            group_classes = [classes[i] for i in indices if i < len(classes)]
            if not group_classes:
                continue

            # 統合処理
            consolidated_class = self._merge_classes(group_classes, canonical_label, reason)
            consolidated_classes.append(consolidated_class)

        return consolidated_classes

    def _build_consolidation_rules(self, class_iri: str) -> str:
        """
        metamodelから該当クラスのextract_rulesを取得し、
        正規化に関連するルールのテキストを生成

        Parameters
        ----------
        class_iri : str
            対象クラスのIRI

        Returns
        -------
        str
            正規化ルールの文字列
        """
        consolidation_rules = []
        for cls in self.metamodel.get("classes", []):
            if cls.get("iri") == class_iri:
                consolidation_rules = cls.get("extract_rules", [])
                break

        return "\n".join(consolidation_rules) if consolidation_rules else []

    def _merge_classes(self, group_classes: List[Dict[str, Any]], canonical_label: str, reason: str = "") -> Dict[str, Any]:
        """
        グループ内のクラスインスタンスを1つに統合

        Parameters
        ----------
        group_classes : List[Dict[str, Any]]
            統合対象のクラスインスタンスリスト
        canonical_label : str
            正準ラベル
        reason : str
            選択理由（ログ用）

        Returns
        -------
        Dict[str, Any]
            統合後のクラスインスタンス
        """
        # 最初のインスタンスをベースにする
        merged = group_classes[0].copy()

        # labelを正準ラベルに更新
        merged["label"] = canonical_label

        # sourcesとfile_idsを統合
        all_sources = []
        all_file_ids = []

        for cls in group_classes:
            # sourcesフィールドを取得（配列または文字列）
            sources = cls.get("sources", [])
            if isinstance(sources, str):
                sources = [sources]
            all_sources.extend(sources)

            # file_idsフィールドを取得（配列または文字列）
            file_ids = cls.get("file_ids", [])
            if isinstance(file_ids, str):
                file_ids = [file_ids]
            all_file_ids.extend(file_ids)

        # 重複を除去しつつ順序を保持
        merged["sources"] = list(dict.fromkeys(all_sources))
        merged["file_ids"] = list(dict.fromkeys(all_file_ids))

        return merged

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
                    print(f"[ClassConsolidator Retry {attempt + 1}/{self.max_retries}] API error: {type(e).__name__}. Retrying in {wait_time:.1f}s...")
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

    # Create and run the consolidator
    consolidator = ClassConsolidator(progress=True, log_dir="log/class_consolidator")
    result = consolidator.invoke(test_input)

    print(f"\nOutput data contains {len(result.get('classes', []))} classes")
