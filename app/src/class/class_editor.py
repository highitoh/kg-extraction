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


class ClassEditor(Runnable):
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
                 log_dir: str = "/workspace/app/log/class_editor"):
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
                                   "class-editor", "llm.schema.json")
        with open(schema_path, "r", encoding="utf-8") as f:
            schema = json.load(f)

        # correctionsの配列要素数を設定
        schema["properties"]["corrections"]["minItems"] = n
        schema["properties"]["corrections"]["maxItems"] = n

        return schema

    def _load_metamodel(self) -> dict:
        """メタモデルファイルを読み込む"""
        # Go up 2 levels from src/class/ to /workspace/app/
        app_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        metamodel_path = os.path.join(app_dir, "metamodel",
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
        # Go up 2 levels from src/class/ to /workspace/app/
        app_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        prompt_path = os.path.join(app_dir, "prompts",
                                   "class_editor.txt")
        with open(prompt_path, "r", encoding="utf-8") as f:
            return f.read()

    def _build_duplicate_fix_schema(self) -> dict:
        """重複修正用LLMスキーマファイルを読み込み

        Returns:
            dict: 重複修正用のJSONスキーマ
        """
        # Go up 2 levels from src/class/ to /workspace/app/
        app_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        schema_path = os.path.join(app_dir, "schemas",
                                   "class-editor", "llm-duplicate.schema.json")
        with open(schema_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _load_duplicate_fix_prompt_template(self) -> str:
        """重複修正用プロンプトテンプレートファイルを読み込み"""
        # Go up 2 levels from src/class/ to /workspace/app/
        app_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        prompt_path = os.path.join(app_dir, "prompts",
                                   "class_editor_duplicate.txt")
        with open(prompt_path, "r", encoding="utf-8") as f:
            return f.read()

    def _merge_duplicate_classes(self, group_classes: list, canonical_label: str, merged_class_id: str) -> dict:
        """重複クラスを1つにマージ

        Args:
            group_classes: マージ対象のクラスインスタンスリスト
            canonical_label: 正準ラベル
            merged_class_id: 代表となるクラスのID

        Returns:
            dict: マージ後のクラスインスタンス (consolidated_ids と consolidated_labels を含む)
        """
        # 代表となるクラスを探す
        base_class = None
        for cls in group_classes:
            if cls.get("id") == merged_class_id:
                base_class = cls.copy()
                break

        if not base_class:
            # フォールバック: 最初のクラスを使用
            base_class = group_classes[0].copy()

        # ラベルを正準ラベルに更新
        base_class["label"] = canonical_label

        # Collect IDs and labels of merged classes (excluding canonical)
        consolidated_ids = []
        consolidated_labels = []

        for cls in group_classes:
            cls_id = cls.get("id")
            if cls_id != merged_class_id:
                consolidated_ids.append(cls_id)
                consolidated_labels.append(cls.get("label", ""))

        base_class["consolidated_ids"] = consolidated_ids
        base_class["consolidated_labels"] = consolidated_labels

        # sourcesとfile_idsをマージ（ClassConsolidatorパターン）
        all_pairs = []
        for cls in group_classes:
            sources = cls.get("sources", [])
            file_ids = cls.get("file_ids", [])

            # sourcesとfile_idsの長さを合わせる
            min_length = min(len(sources), len(file_ids))
            sources = sources[:min_length]
            file_ids = file_ids[:min_length]

            # (source, file_id)のペアとして収集
            for source, file_id in zip(sources, file_ids):
                all_pairs.append((source, file_id))

        # 重複を除去しつつ順序を保持
        unique_pairs = list(dict.fromkeys(all_pairs))

        # ペアを分解してsourcesとfile_idsに戻す
        if unique_pairs:
            base_class["sources"], base_class["file_ids"] = zip(*unique_pairs)
            base_class["sources"] = list(base_class["sources"])
            base_class["file_ids"] = list(base_class["file_ids"])
        else:
            base_class["sources"] = []
            base_class["file_ids"] = []

        return base_class

    def invoke(self, input: Dict[str, Any], config=None) -> Dict[str, Any]:
        """
        クラスラベル修正処理のメインエントリーポイント

        Parameters
        ----------
        input: ClassLabelCorrectorInput (= ClassReviewerOutput)
          - id: 抽出ID
          - classes: クラス個体リスト（judgment/justificationを含む）

        Returns
        -------
        output: ClassLabelCorrectorOutput (= ClassReviewerOutput)
        """
        classes = input.get("classes", [])

        # 判定ごとに分離
        accept_classes = [c for c in classes if c.get("judgment") == "ACCEPT"]
        review_classes = [c for c in classes if c.get("judgment") == "REVIEW"]
        reject_classes = [c for c in classes if c.get("judgment") == "REJECT"]

        if self.progress:
            print(f"[ClassEditor] Total {len(classes)} classes "
                  f"[ACCEPT: {len(accept_classes)}, REVIEW: {len(review_classes)}, REJECT: {len(reject_classes)}]")

        if not review_classes:
            # REVIEWクラスがない場合はそのまま返す
            if self.progress:
                print("[ClassEditor] No REVIEW classes to process")
            return input

        # class_iriでグループ化
        review_groups = self._group_classes_by_class_iri(review_classes)

        if self.progress:
            print(f"[ClassEditor] {len(review_groups)} class types found for review")

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
            print(f"[ClassEditor] Completed: Processed {len(review_classes)} REVIEW classes "
                  f"[REVIEW: {corrected_review_count}, REJECT: {corrected_reject_count}]")

        self.logger.save_log(output, filename_prefix="class_editor_output_")

        return output

    async def _fix_duplicates_async(self, classes: list, class_iri: str) -> dict:
        """重複クラスのマージ判定と修正（非同期版）

        Args:
            classes: 修正対象のクラスインスタンスリスト
            class_iri: 対象クラスのIRI

        Returns:
            dict: 重複修正結果の辞書
                {
                    "dup_group_0": {
                        "should_merge": true,
                        "canonical_label": "User",
                        "merged_class_id": "cls_001",
                        "reason": "...",
                        "merged_class": {...}
                    },
                    ...
                }
        """
        # duplicate_group_idを持つクラスのみ抽出
        duplicates = [c for c in classes if "duplicate_group_id" in c]

        if not duplicates:
            return {}

        # duplicate_group_idでグループ化
        groups = {}
        for cls in duplicates:
            group_id = cls.get("duplicate_group_id")
            if group_id not in groups:
                groups[group_id] = []
            groups[group_id].append(cls)

        if not groups:
            return {}

        # LLMへの入力データを構築
        duplicate_groups_input = []
        for group_id, group_classes in groups.items():
            candidates = []
            for cls in group_classes:
                candidates.append({
                    "id": cls.get("id", ""),
                    "label": cls.get("label", ""),
                    "sources": cls.get("sources", [])
                })

            duplicate_groups_input.append({
                "duplicate_group_id": group_id,
                "candidates": candidates
            })

        duplicate_groups_json = json.dumps(duplicate_groups_input, ensure_ascii=False, indent=2)

        # スキーマを構築
        schema = self._build_duplicate_fix_schema()
        llm_json = self.llm.bind(
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "class_editor_duplicate",
                    "schema": schema
                }
            }
        )

        # プロンプトを構築
        prompt_template = self._load_duplicate_fix_prompt_template()
        prompt = prompt_template.replace("{duplicate_groups_json}", duplicate_groups_json)

        # LLMで重複修正実行（リトライ付き）
        try:
            response = await self._invoke_with_retry(llm_json, prompt)

            # レスポンスからテキストを抽出
            result_text = self._to_text(response.content)

            # JSON パース
            parsed = json.loads(result_text)
            resolutions = parsed.get("resolutions", [])

            # 結果を辞書に変換
            result = {}
            for resolution in resolutions:
                group_id = resolution.get("duplicate_group_id")
                should_merge = resolution.get("should_merge", False)
                reason = resolution.get("reason", "")

                result[group_id] = {
                    "should_merge": should_merge,
                    "reason": reason
                }

                if should_merge:
                    canonical_label = resolution.get("canonical_label", "")
                    merged_class_id = resolution.get("merged_class_id", "")

                    # canonical_labelとmerged_class_idが必須
                    if not canonical_label or not merged_class_id:
                        # フォールバック: マージしない
                        result[group_id]["should_merge"] = False
                        result[group_id]["reason"] = f"Missing required fields: {reason}"
                    else:
                        result[group_id]["canonical_label"] = canonical_label
                        result[group_id]["merged_class_id"] = merged_class_id

                        # マージされたクラスを生成
                        group_classes = groups.get(group_id, [])
                        if group_classes:
                            merged_class = self._merge_duplicate_classes(
                                group_classes,
                                canonical_label,
                                merged_class_id
                            )
                            result[group_id]["merged_class"] = merged_class

            return result

        except (json.JSONDecodeError, KeyError) as e:
            if self.progress:
                print(f"[ClassEditor] Duplicate fix JSON parse error: {e}")
            # エラー時は空の辞書を返す（マージなし）
            return {}
        except Exception as e:
            if self.progress:
                print(f"[ClassEditor] Duplicate fix error: {e}")
            return {}

    def _merge_fix_results(self, classes: list, prohibit_fixed: list, duplicate_resolved: dict) -> list:
        """禁止ルール修正と重複修正の結果をマージ

        Args:
            classes: 元のクラスインスタンスリスト
            prohibit_fixed: 禁止ルール修正後のクラスリスト
            duplicate_resolved: 重複修正結果の辞書

        Returns:
            list: マージ後のクラスリスト
        """
        results = []
        processed_ids = set()

        # 禁止ルール修正結果とマッピング
        prohibit_map = {c.get("id"): c for c in prohibit_fixed}

        for cls in classes:
            class_id = cls.get("id", "")

            if class_id in processed_ids:
                continue

            # Step 1: 禁止ルール修正を適用
            corrected_cls = prohibit_map.get(class_id, cls)

            # REJECTされた場合はスキップ
            if corrected_cls.get("judgment") == "REJECT":
                results.append(corrected_cls)
                processed_ids.add(class_id)
                continue

            # Step 2: 重複修正をチェック
            duplicate_group_id = cls.get("duplicate_group_id")

            if duplicate_group_id and duplicate_group_id in duplicate_resolved:
                resolution = duplicate_resolved[duplicate_group_id]
                should_merge = resolution.get("should_merge", False)

                if should_merge:
                    merged_class_id = resolution.get("merged_class_id")

                    if class_id == merged_class_id:
                        # 代表クラス: マージされたクラスを使用
                        merged_class = resolution.get("merged_class")
                        if merged_class:
                            # 禁止ルール修正のjudgmentとjustificationを保持
                            merged_class["judgment"] = corrected_cls.get("judgment", "REVIEW")
                            merged_class["justification"] = corrected_cls.get("justification", "")
                            results.append(merged_class)
                            processed_ids.add(class_id)
                        else:
                            # マージ失敗時は修正後のクラスを使用
                            results.append(corrected_cls)
                            processed_ids.add(class_id)
                    else:
                        # 代表以外: マージされたのでスキップ
                        processed_ids.add(class_id)
                else:
                    # should_merge=false: そのまま保持
                    results.append(corrected_cls)
                    processed_ids.add(class_id)
            else:
                # 重複グループに含まれない: そのまま保持
                results.append(corrected_cls)
                processed_ids.add(class_id)

        # Clean up temporary duplicate_group_id field from all results
        for cls in results:
            if "duplicate_group_id" in cls:
                del cls["duplicate_group_id"]

        return results

    async def _process_classes_parallel(self, class_groups: Dict[str, list]) -> list:
        """クラスグループを並列処理（禁止ルール修正と重複修正を並行実行）"""
        sem = asyncio.Semaphore(self.max_concurrency)
        total = len(class_groups)
        counter = 0
        lock = asyncio.Lock()

        if self.progress:
            print(f"[ClassEditor] Starting parallel processing: {total} class types (concurrency={self.max_concurrency})")

        async def _wrapped(class_iri: str, class_instances: list) -> list:
            nonlocal counter
            async with sem:
                class_name = self._get_class_name_from_iri(class_iri)

                # 禁止ルール修正と重複修正を並列実行
                prohibit_fixed, duplicate_resolved = await asyncio.gather(
                    self._fix_prohibit_violations_async(class_instances, class_iri),
                    self._fix_duplicates_async(class_instances, class_iri)
                )

                # 結果をマージ
                merged = self._merge_fix_results(class_instances, prohibit_fixed, duplicate_resolved)

                async with lock:
                    counter += 1
                    if self.progress:
                        # 修正結果の内訳を集計
                        review_count = sum(1 for c in merged if c.get("judgment") == "REVIEW")
                        reject_count = sum(1 for c in merged if c.get("judgment") == "REJECT")

                        # マージ数をカウント
                        merge_count = sum(
                            1 for res in duplicate_resolved.values()
                            if res.get("should_merge", False)
                        )
                        total_dup_groups = len(duplicate_resolved)

                        print(f"[{counter}/{total}] ClassEditor: {class_name} processed "
                              f"({len(class_instances)} -> {len(merged)} instances) "
                              f"[REVIEW: {review_count}, REJECT: {reject_count}] "
                              f"[Duplicates merged: {merge_count}/{total_dup_groups}]")

                return merged

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

    async def _fix_prohibit_violations_async(self, classes: list, class_iri: str) -> list:
        """指定されたクラスリストをLLMで禁止ルール違反を修正（非同期版）

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
                    "name": "class_editor",
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
                    # ラベル修正した場合はREVIEWのまま（ClassReviewerで再評価される）
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

    # ログディレクトリから最新のclass_reviewerログファイルを取得
    log_dir = "/workspace/app/log/class_reviewer"
    log_files = glob.glob(os.path.join(log_dir, "class_reviewer_output_*.json"))

    if not log_files:
        print("Error: No class_reviewer log files found")
        exit(1)

    # 最新のログファイルを取得
    latest_log = max(log_files, key=os.path.getmtime)
    print(f"Loading input from: {latest_log}")
    with open(latest_log, "r", encoding="utf-8") as f:
        test_input = json.load(f)

    print(f"\nInput data contains {len(test_input.get('classes', []))} classes")

    # Create and run the corrector
    corrector = ClassEditor(progress=True, log_dir="/workspace/app/log/class_editor")
    result = corrector.invoke(test_input)

    print(f"\nOutput data contains {len(result.get('classes', []))} classes")
