from typing import Dict, Any, List, Union
import asyncio
import glob
import json
import os
import uuid

from langchain.schema.runnable import Runnable
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from openai import APIConnectionError, RateLimitError, APIError

from logger import Logger


class PropertyFilter(Runnable):

    def __init__(
        self,
        llm: ChatOpenAI | None = None,
        model: str = "gpt-5-mini",
        temperature: float = 0.0,
        max_concurrency: int = 8,
        max_retries: int = 5,
        progress: bool = True,
        log_dir: str = "log/property_filter",
        batch_size: int = 24,
        confidence_threshold: float = 0.5,
    ):
        self.llm = llm or ChatOpenAI(
            model=model,
            temperature=temperature,
            reasoning={"effort": "minimal"},
            output_version="responses/v1",
        )
        self.llm_schema = self._load_llm_schema()
        self.llm_json = self.llm.bind(response_format={"type": "json_schema", "json_schema": {"name": "property_filter", "schema": self.llm_schema}})
        self.max_concurrency = max_concurrency
        self.max_retries = max_retries
        self.progress = progress
        self.batch_size = batch_size
        self.confidence_threshold = confidence_threshold

        self.prompt = self._load_prompt()
        self.metamodel = self._load_metamodel()
        self.logger = Logger(log_dir)

    # ===== Utilities =====
    def _load_llm_schema(self) -> Dict[str, Any]:
        """LLM出力スキーマを読み込み"""
        schema_path = os.path.join(os.path.dirname(__file__), "schemas", "property-filter", "llm.schema.json")
        with open(schema_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _load_prompt(self) -> str:
        """プロンプトファイルを読み込み"""
        prompt_path = os.path.join(os.path.dirname(__file__), "prompts", "property_filter.txt")
        with open(prompt_path, "r", encoding="utf-8") as f:
            return f.read()

    def _load_metamodel(self) -> Dict[str, Any]:
        """メタモデルを読み込み"""
        metamodel_path = os.path.join(os.path.dirname(__file__), "metamodel", "metamodel.json")
        with open(metamodel_path, "r", encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def _to_text(c: Union[str, List[Any]]) -> str:
        """LLMの出力をテキスト形式に変換（PropertyExtractorと同様の実装）"""
        if isinstance(c, str):
            return c
        for part in c:
            if isinstance(part, dict) and part.get("type") in ("output_text", "text"):
                return part.get("text", "")
        return ""

    @staticmethod
    def _index_classes_by_id(class_info: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """class_infoのclasses配列をidで引けるようにインデックス化"""
        idx = {}
        for c in class_info.get("classes", []):
            cid = c.get("id")
            if cid:
                idx[cid] = c
        return idx

    def _group_by_property_iri(self, props: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """候補プロパティをproperty_iriごとにグルーピング"""
        groups: Dict[str, List[Dict[str, Any]]] = {}
        for p in props:
            prop_iri = p.get("property_iri", "")
            if prop_iri not in groups:
                groups[prop_iri] = []
            groups[prop_iri].append(p)
        return groups

    def _format_batch_message(self, batch: List[Dict[str, Any]], class_index: Dict[str, Dict[str, Any]]) -> str:
        """LLMへ渡すバッチ用のテキストを作成（labelとsourceを含める）"""
        lines = []
        lines.append("# 審査対象候補（src/destのlabel, class_iri, sourceを含む）")
        for i, p in enumerate(batch, 1):
            src = class_index.get(p.get("src_id", ""), {})
            dest = class_index.get(p.get("dest_id", ""), {})
            lines.append(f"候補{i}:")
            lines.append(f"  property_iri: {p.get('property_iri', '')}")
            lines.append(f"  src_id: {p.get('src_id','')}")
            lines.append(f"    - src_label: {src.get('label','')}")
            lines.append(f"    - src_class_iri: {src.get('class_iri','')}")
            lines.append(f"    - src_source: {src.get('source','')}")
            lines.append(f"  dest_id: {p.get('dest_id','')}")
            lines.append(f"    - dest_label: {dest.get('label','')}")
            lines.append(f"    - dest_class_iri: {dest.get('class_iri','')}")
            lines.append(f"    - dest_source: {dest.get('source','')}")
        return "\n".join(lines)

    def _get_prohibit_rules(self, property_iri: str) -> str:
        """メタモデルから該当プロパティの禁止ルールを取得"""
        for prop in self.metamodel.get("properties", []):
            if prop.get("iri") == property_iri:
                rules = prop.get("prohibit_rules", [])
                if rules:
                    return "\n".join(f"- {rule}" for rule in rules)
        return ""

    async def _judge_batch(self, batch: List[Dict[str, Any]], class_index: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
        """LLMで1バッチ分の候補を審査して結果(JSON)を返す"""
        batch_text = self._format_batch_message(batch, class_index)

        # バッチ内の最初のproperty_iriから禁止ルールを取得
        property_iri = batch[0].get("property_iri", "") if batch else ""
        prohibit_rules = self._get_prohibit_rules(property_iri)
        prompt_filled = self.prompt.replace("{PROHIBIT_RULES}", prohibit_rules)

        msg = HumanMessage(
            content=[
                {"type": "text", "text": prompt_filled},
                {"type": "text", "text": batch_text},
            ]
        )

        # リトライ機能付きでLLM呼び出し
        ai = await self._invoke_with_retry(msg)
        raw = self._to_text(ai.content)
        try:
            data = json.loads(raw)
        except Exception:
            data = {}

        results: List[Dict[str, Any]] = []
        for r in data.get("relations", []):
            if not isinstance(r, dict):
                continue
            results.append(
                {
                    "src_id": r.get("src_id", ""),
                    "property_iri": property_iri,  # バッチのproperty_iriを保持
                    "dest_id": r.get("dest_id", ""),
                    "prohibited": bool(r.get("prohibited", False)),
                    "justification": (r.get("justification") or "")[:500],
                    "confidence": float(r.get("confidence", 0.5)),
                }
            )
        return results

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
                    print(f"[PropertyFilter Retry {attempt + 1}/{self.max_retries}] API error: {type(e).__name__}. Retrying in {wait_time:.1f}s...")
                await asyncio.sleep(wait_time)
                delay *= 2

        if last_exception:
            raise last_exception

    async def _filter_properties(
        self,
        property_candidates: Dict[str, Any],
        class_info: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """候補プロパティをproperty_iriごとに分割してLLM審査でフィルタリング"""
        candidates = property_candidates.get("properties", [])
        if not candidates:
            return []

        class_index = self._index_classes_by_id(class_info)

        # property_iriごとにグルーピング
        groups = self._group_by_property_iri(candidates)

        results_all: List[Dict[str, Any]] = []
        failed_groups: List[str] = []  # エラーで失敗したproperty_iriを記録
        completed_count = 0
        total_groups = len(groups)

        async def worker(property_iri: str, group: List[Dict[str, Any]]):
            nonlocal completed_count
            try:
                res = await self._judge_batch(group, class_index)
                results_all.extend(res)
            except Exception as e:
                # グループ失敗時は記録
                failed_groups.append(property_iri)
                print(f"[PropertyFilter] error for {property_iri}: {e}")
            finally:
                completed_count += 1
                if self.progress:
                    print(f"[PropertyFilter] Progress: {completed_count}/{total_groups} properties completed ({property_iri}: {len(group)} candidates)")

        sem = asyncio.Semaphore(self.max_concurrency)

        async def sem_task(prop_iri: str, grp: List[Dict[str, Any]]):
            async with sem:
                await worker(prop_iri, grp)

        if self.progress and total_groups > 0:
            print(f"[PropertyFilter] Starting LLM inference: {total_groups} properties (concurrency={self.max_concurrency})")

        await asyncio.gather(*[sem_task(prop_iri, grp) for prop_iri, grp in groups.items()])

        # 審査結果をマッピング
        result_map = {(r["src_id"], r["property_iri"], r["dest_id"]): r for r in results_all}

        # 全候補を構築
        all_properties_with_judgment: List[Dict[str, Any]] = []
        for c in candidates:
            key = (c.get("src_id", ""), c.get("property_iri", ""), c.get("dest_id", ""))

            # クラスラベルを取得
            src_class = class_index.get(c.get("src_id", ""), {})
            dest_class = class_index.get(c.get("dest_id", ""), {})
            src_label = src_class.get("label", c.get("src_id", ""))
            dest_label = dest_class.get("label", c.get("dest_id", ""))

            if key in result_map:
                r = result_map[key]
                all_properties_with_judgment.append({
                    "id": str(uuid.uuid4()),
                    "src_id": r["src_id"],
                    "src_label": src_label,
                    "property_iri": r["property_iri"],
                    "dest_id": r["dest_id"],
                    "dest_label": dest_label,
                    "prohibited": r["prohibited"],
                    "confidence": r["confidence"],
                    "justification": r["justification"],
                })
            else:
                # LLM審査エラーまたは未審査
                all_properties_with_judgment.append({
                    "id": str(uuid.uuid4()),
                    "src_id": c.get("src_id", ""),
                    "src_label": src_label,
                    "property_iri": c.get("property_iri", ""),
                    "dest_id": c.get("dest_id", ""),
                    "dest_label": dest_label,
                    "prohibited": None,
                    "confidence": None,
                    "justification": "LLM審査エラー",
                })

        # フィルタリング（prohibited==False かつ confidence >= threshold のみ採用）
        filtered: List[Dict[str, Any]] = []
        for p in all_properties_with_judgment:
            if p["prohibited"] is False and p["confidence"] is not None and p["confidence"] >= self.confidence_threshold:
                filtered.append({
                    "id": p["id"],
                    "src_id": p["src_id"],
                    "src_label": p["src_label"],
                    "property_iri": p["property_iri"],
                    "dest_id": p["dest_id"],
                    "dest_label": p["dest_label"],
                    "justification": p["justification"],
                    "confidence": p["confidence"],
                })

        # ログ保存（全候補、ラベル・prohibited・confidence・justification付き）
        log_output: Dict[str, Any] = {
            "properties": all_properties_with_judgment,
        }
        self.logger.save_log(log_output, filename_prefix="property_filter_output_")

        return filtered

    def invoke(self, input: Dict[str, Any], config=None) -> Dict[str, Any]:
        """
        入力:
          - 推奨: {"property_candidates": {...}, "class_info": {...}}
          - 互換: {"properties":[...], "id":"...", "class_info": {...}}
        """
        # 柔軟に入力を解釈
        property_candidates = input.get("property_candidates") or {
            "id": input.get("id"),
            "properties": input.get("properties", []),
        }
        class_info = input.get("class_info") or {}

        if not class_info.get("classes"):
            # class_infoが無いとsourceが参照できないため、エラー
            raise ValueError("PropertyFilter requires class_info with classes for filtering. Cannot proceed without class information.")

        # LLMでフィルタリング
        filtered = asyncio.run(self._filter_properties(property_candidates, class_info))

        if self.progress:
            print(
                f"PropertyFilter: {len(property_candidates.get('properties', []))} candidates -> "
                f"{len(filtered)} accepted"
            )

        # 出力用（ラベルを削除してスキーマ準拠に）
        output_id = str(uuid.uuid4())
        properties = [
            {k: v for k, v in p.items() if k not in ["src_label", "dest_label"]}
            for p in filtered
        ]
        output: Dict[str, Any] = {
            "id": output_id,
            "properties": properties,
        }
        return output


if __name__ == "__main__":
    # テスト実行:
    # - 最新の PropertyExtractor 出力 (log/property_extractor/property_extractor_output_*.json)
    # - 最新の ClassConsolidator 出力 (log/class_consolidator/class_consolidator_output_*.json)
    # を読み込み、フィルタリング結果を表示
    pf = PropertyFilter()

    prop_files = sorted(glob.glob("log/property_extractor/property_extractor_output_*.json"), reverse=True)
    cls_files = sorted(glob.glob("log/class_consolidator/class_consolidator_output_*.json"), reverse=True)

    if not prop_files:
        print("Error: No property_extractor_output files found in log/property_extractor/")
        exit(1)
    if not cls_files:
        print("Error: No class_consolidator_output files found in log/class_consolidator/")
        exit(1)

    prop_path = prop_files[0]
    cls_path = cls_files[0]
    print(f"Loading latest PropertyExtractor output: {prop_path}")
    print(f"Loading latest ClassConsolidator output:    {cls_path}")

    with open(prop_path, "r", encoding="utf-8") as f:
        property_candidates = json.load(f)
    with open(cls_path, "r", encoding="utf-8") as f:
        class_info = json.load(f)

    input_data = {
        "property_candidates": property_candidates,
        "class_info": class_info,
    }

    print("\n=== PropertyFilter Test ===")
    print(f"- Candidates: {len(property_candidates.get('properties', []))}")
    print(f"- Classes:    {len(class_info.get('classes', []))}")
    print("\n" + "=" * 60 + "\n")

    try:
        result = pf.invoke(input_data)
        print("Output:")
        print(json.dumps(result, ensure_ascii=False, indent=2))

        print("\nSummary:")
        print(f"- Accepted properties: {len(result.get('properties', []))}")
        for i, p in enumerate(result.get("properties", []), 1):
            # 表示用にラベル解決
            idx = {c["id"]: c for c in class_info.get("classes", [])}
            s = idx.get(p["src_id"], {})
            d = idx.get(p["dest_id"], {})
            s_label = s.get("label", p["src_id"])
            d_label = d.get("label", p["dest_id"])
            print(f"  {i:02d}. {s_label} --[{p['property_iri']}]--> {d_label}")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()



