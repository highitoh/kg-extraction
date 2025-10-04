from typing import Dict, Any, List, Union
import asyncio
import glob
import json
import os
import uuid

from langchain.schema.runnable import Runnable
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

from logger import Logger


class PropertyFilter(Runnable):

    def __init__(
        self,
        llm: ChatOpenAI | None = None,
        model: str = "gpt-5-mini",
        temperature: float = 0.0,
        max_concurrency: int = 8,
        progress: bool = True,
        log_dir: str = "log/property_filter",
        batch_size: int = 24,
    ):
        self.llm = llm or ChatOpenAI(
            model=model,
            temperature=temperature,
            reasoning={"effort": "minimal"},
            output_version="responses/v1",
        )
        self.llm_json = self.llm.bind(response_format={"type": "json_object"})
        self.max_concurrency = max_concurrency
        self.progress = progress
        self.batch_size = batch_size

        self.prompt = self._load_prompt()
        self.metamodel = self._load_metamodel()
        self.logger = Logger(log_dir)

    # ===== Utilities =====
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
        ai = await self.llm_json.ainvoke([msg])
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
                }
            )
        return results

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
        completed_count = 0
        total_groups = len(groups)

        async def worker(property_iri: str, group: List[Dict[str, Any]]):
            nonlocal completed_count
            try:
                res = await self._judge_batch(group, class_index)
                results_all.extend(res)
            except Exception as e:
                # グループ失敗時はスキップ（ログのみ）
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

        # フィルタリング（prohibited==False のみ採用）
        filtered: List[Dict[str, Any]] = []
        ok_map = {(r["src_id"], r["property_iri"], r["dest_id"]): r for r in results_all if not r["prohibited"]}

        for c in candidates:
            key = (c.get("src_id", ""), c.get("property_iri", ""), c.get("dest_id", ""))
            if key in ok_map:
                r = ok_map[key]
                filtered.append(
                    {
                        "id": str(uuid.uuid4()),
                        "src_id": r["src_id"],
                        "property_iri": r["property_iri"],
                        "dest_id": r["dest_id"],
                        "justification": r["justification"],
                    }
                )

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
            # class_infoが無いとsourceが参照できないため、素通し
            if self.progress:
                print("PropertyFilter: class_info missing -> pass-through")
            output = {
                "id": str(uuid.uuid4()),
                "properties": property_candidates.get("properties", []),
            }
            self.logger.save_log(output, filename_prefix="property_filter_output_")
            return output

        # LLMでフィルタリング
        filtered = asyncio.run(self._filter_properties(property_candidates, class_info))

        output: Dict[str, Any] = {
            "id": str(uuid.uuid4()),
            "properties": filtered,
        }

        if self.progress:
            print(
                f"PropertyFilter: {len(property_candidates.get('properties', []))} candidates -> "
                f"{len(filtered)} accepted"
            )

        self.logger.save_log(output, filename_prefix="property_filter_output_")
        return output


if __name__ == "__main__":
    # テスト実行:
    # - 最新の PropertyExtractor 出力 (log/property_extractor/property_extractor_output_*.json)
    # - 最新の ClassFilter 出力 (log/class_filter/class_filter_output_*.json)
    # を読み込み、フィルタリング結果を表示
    pf = PropertyFilter()

    prop_files = sorted(glob.glob("log/property_extractor/property_extractor_output_*.json"), reverse=True)
    cls_files = sorted(glob.glob("log/class_filter/class_filter_output_*.json"), reverse=True)

    if not prop_files:
        print("Error: No property_extractor_output files found in log/property_extractor/")
        exit(1)
    if not cls_files:
        print("Error: No class_filter_output files found in log/class_filter/")
        exit(1)

    prop_path = prop_files[0]
    cls_path = cls_files[0]
    print(f"Loading latest PropertyExtractor output: {prop_path}")
    print(f"Loading latest ClassFilter output:    {cls_path}")

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



