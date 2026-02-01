from typing import Dict, Any, List, Union
import asyncio
import glob
import json
import os
import uuid

from langchain_core.runnables import Runnable
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from openai import APIConnectionError, RateLimitError, APIError

from src.utils.logger import Logger
from src.utils.llm_factory import LLMFactory
from src.utils.prompt_utils import fill_prompt_template
from src.utils.llm_retry import call_with_retry


class PropertyReviewer(Runnable):

    def __init__(
        self,
        llm: ChatOpenAI | None = None,
        model: str = "gpt-5-mini",
        temperature: float = 0.0,
        max_concurrency: int = 8,
        max_retries: int = 5,
        progress: bool = True,
        log_dir: str = "/workspace/app/log/property_reviewer",
        batch_size: int = 24,
    ):
        self.llm = llm or LLMFactory.create_llm(
            model=model,
            temperature=temperature,
            reasoning_effort="low",
        )
        self.llm_schema = self._load_llm_schema()
        self.llm_json = self.llm.bind(response_format={"type": "json_schema", "json_schema": {"name": "property_reviewer", "schema": self.llm_schema}})
        self.max_concurrency = max_concurrency
        self.max_retries = max_retries
        self.progress = progress
        self.batch_size = batch_size

        self.prompt = self._load_prompt()
        self.metamodel = self._load_metamodel()
        self.logger = Logger(log_dir)

    # ===== Utilities =====
    def _load_llm_schema(self) -> Dict[str, Any]:
        """Load LLM output schema"""
        # Go up 3 levels from src/extract/XXX/ to /workspace/app/

        app_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

        schema_path = os.path.join(app_dir, "schemas", "property-reviewer", "llm.schema.json")
        with open(schema_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _load_prompt(self) -> str:
        """Load prompt file"""
        # Go up 3 levels from src/extract/XXX/ to /workspace/app/

        app_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

        prompt_path = os.path.join(app_dir, "prompts", "property_reviewer.txt")
        with open(prompt_path, "r", encoding="utf-8") as f:
            return f.read()

    def _load_metamodel(self) -> Dict[str, Any]:
        """Load metamodel"""
        # Go up 3 levels from src/extract/XXX/ to /workspace/app/

        app_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

        metamodel_path = os.path.join(app_dir, "metamodel", "metamodel.json")
        with open(metamodel_path, "r", encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def _to_text(c: Union[str, List[Any]]) -> str:
        """Convert LLM output to text format (same implementation as PropertyExtractor)"""
        if isinstance(c, str):
            return c
        for part in c:
            if isinstance(part, dict) and part.get("type") in ("output_text", "text"):
                return part.get("text", "")
        return ""

    @staticmethod
    def _index_classes_by_id(class_info: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """Index class_info.classes array by id for lookup"""
        idx = {}
        for c in class_info.get("classes", []):
            cid = c.get("id")
            if cid:
                idx[cid] = c
        return idx

    def _group_by_property_iri(self, props: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """Group candidate properties by property_iri"""
        groups: Dict[str, List[Dict[str, Any]]] = {}
        for p in props:
            prop_iri = p.get("property_iri", "")
            if prop_iri not in groups:
                groups[prop_iri] = []
            groups[prop_iri].append(p)
        return groups

    def _format_batch_message(self, batch: List[Dict[str, Any]], class_index: Dict[str, Dict[str, Any]]) -> str:
        """Create batch text for LLM (including strength, ranker_justification)"""
        lines = []
        lines.append("# Review Candidates (includes src/dest label, class_iri, source, strength, ranker evaluation justification)")
        for i, p in enumerate(batch, 1):
            src = class_index.get(p.get("src_id", ""), {})
            dest = class_index.get(p.get("dest_id", ""), {})
            lines.append(f"Candidate {i}:")
            lines.append(f"  property_iri: {p.get('property_iri', '')}")
            lines.append(f"  src_id: {p.get('src_id','')}")
            lines.append(f"    - src_label: {src.get('label','')}")
            lines.append(f"    - src_class_iri: {src.get('class_iri','')}")
            lines.append(f"    - src_source: {src.get('source','')}")
            lines.append(f"  dest_id: {p.get('dest_id','')}")
            lines.append(f"    - dest_label: {dest.get('label','')}")
            lines.append(f"    - dest_class_iri: {dest.get('class_iri','')}")
            lines.append(f"    - dest_source: {dest.get('source','')}")
            lines.append(f"  strength: {p.get('strength', 0)} (evaluated by PropertyRanker)")
            ranker_justification = p.get('ranker_justification', '')
            if ranker_justification:
                lines.append(f"  ranker_justification: {ranker_justification}")
        return "\n".join(lines)

    def _get_prohibit_rules(self, property_iri: str) -> str:
        """Get prohibition rules for the property from metamodel"""
        for prop in self.metamodel.get("properties", []):
            if prop.get("iri") == property_iri:
                rules = prop.get("prohibit_rules", [])
                if rules:
                    return "\n".join(f"- {rule}" for rule in rules)
        return ""

    async def _judge_batch(self, batch: List[Dict[str, Any]], class_index: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Review one batch of candidates with LLM and return results (JSON)"""
        batch_text = self._format_batch_message(batch, class_index)

        # Get prohibition rules from the first property_iri in the batch
        property_iri = batch[0].get("property_iri", "") if batch else ""
        prohibit_rules = self._get_prohibit_rules(property_iri)

        # Fill in both template variables
        prompt_filled = fill_prompt_template(
            self.prompt,
            prohibit_rules=prohibit_rules,
            property_candidates=batch_text
        )

        msg = HumanMessage(
            content=[
                {"type": "text", "text": prompt_filled},
            ]
        )

        # Call LLM with retry functionality
        ai = await call_with_retry(
            lambda: self.llm_json.ainvoke([msg]),
            retries=self.max_retries,
            timeout=90.0,
            progress=self.progress,
            label="PropertyReviewer",
        )
        raw = self._to_text(ai.content)
        try:
            data = json.loads(raw)
        except Exception:
            data = {}

        results: List[Dict[str, Any]] = []
        for r in data.get("relations", []):
            if not isinstance(r, dict):
                continue

            # Get evaluated_strength
            evaluated_strength = r.get("evaluated_strength", 0)
            judgment = r.get("judgment", "REJECT")

            # Set evaluated_strength to 0 if REJECT
            if judgment == "REJECT":
                evaluated_strength = 0

            results.append(
                {
                    "src_id": r.get("src_id", ""),
                    "property_iri": property_iri,  # Keep the property_iri from the batch
                    "dest_id": r.get("dest_id", ""),
                    "judgment": judgment,
                    "evaluated_strength": evaluated_strength,
                    "justification": (r.get("justification") or "")[:500],
                }
            )
        return results

    async def _filter_properties(
        self,
        property_candidates: Dict[str, Any],
        class_info: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Split candidate properties by property_iri and filter with LLM review"""
        candidates = property_candidates.get("properties", [])
        if not candidates:
            return []

        class_index = self._index_classes_by_id(class_info)

        # Group by property_iri
        groups = self._group_by_property_iri(candidates)

        results_all: List[Dict[str, Any]] = []
        failed_groups: List[str] = []  # Record property_iris that failed with errors
        completed_count = 0
        total_groups = len(groups)

        async def worker(property_iri: str, group: List[Dict[str, Any]]):
            nonlocal completed_count
            try:
                res = await self._judge_batch(group, class_index)
                results_all.extend(res)
            except Exception as e:
                # Record on group failure
                failed_groups.append(property_iri)
                print(f"[PropertyReviewer] error for {property_iri}: {e}")
            finally:
                completed_count += 1
                if self.progress:
                    print(f"[{completed_count}/{total_groups}] PropertyReviewer: {property_iri} processed ({len(group)} candidates)")

        sem = asyncio.Semaphore(self.max_concurrency)

        async def sem_task(prop_iri: str, grp: List[Dict[str, Any]]):
            async with sem:
                await worker(prop_iri, grp)

        if self.progress and total_groups > 0:
            print(f"[PropertyReviewer] Starting parallel processing: {total_groups} property types (concurrency={self.max_concurrency})")

        await asyncio.gather(*[sem_task(prop_iri, grp) for prop_iri, grp in groups.items()])

        # Map review results
        result_map = {(r["src_id"], r["property_iri"], r["dest_id"]): r for r in results_all}

        # Build all candidates
        all_properties_with_judgment: List[Dict[str, Any]] = []
        for c in candidates:
            key = (c.get("src_id", ""), c.get("property_iri", ""), c.get("dest_id", ""))

            # Get class labels
            src_class = class_index.get(c.get("src_id", ""), {})
            dest_class = class_index.get(c.get("dest_id", ""), {})
            src_label = src_class.get("label", c.get("src_id", ""))
            dest_label = dest_class.get("label", c.get("dest_id", ""))

            if key in result_map:
                r = result_map[key]
                original_strength = c.get("strength", 0)
                evaluated_strength = r.get("evaluated_strength", 0)
                judgment = r["judgment"]

                # Set judgment to REVIEW if strength evaluation differs
                if judgment != "REJECT" and original_strength != evaluated_strength:
                    judgment = "REVIEW"

                # Final strength value (0 if REJECT, otherwise evaluated_strength)
                final_strength = 0 if judgment == "REJECT" else evaluated_strength

                all_properties_with_judgment.append({
                    "id": str(uuid.uuid4()),
                    "src_id": r["src_id"],
                    "src_label": src_label,
                    "property_iri": r["property_iri"],
                    "dest_id": r["dest_id"],
                    "dest_label": dest_label,
                    "original_strength": original_strength,  # For logging: original strength
                    "evaluated_strength": evaluated_strength,  # For logging: re-evaluated strength
                    "strength": final_strength,  # Final strength
                    "judgment": judgment,
                    "justification": r["justification"],
                })
            else:
                # LLM review error or not reviewed
                original_strength = c.get("strength", 0)
                all_properties_with_judgment.append({
                    "id": str(uuid.uuid4()),
                    "src_id": c.get("src_id", ""),
                    "src_label": src_label,
                    "property_iri": c.get("property_iri", ""),
                    "dest_id": c.get("dest_id", ""),
                    "dest_label": dest_label,
                    "original_strength": original_strength,
                    "evaluated_strength": 0,
                    "strength": 0,  # 0 because REJECT
                    "judgment": "REJECT",
                    "justification": "LLM review error",
                })

        # Filter (accept only ACCEPT/REVIEW, exclude REJECT)
        filtered: List[Dict[str, Any]] = []
        for p in all_properties_with_judgment:
            judgment = p["judgment"]
            # Pass through non-REJECT (ACCEPT, REVIEW)
            if judgment in ("ACCEPT", "REVIEW"):
                filtered.append({
                    "id": p["id"],
                    "src_id": p["src_id"],
                    "src_label": p["src_label"],
                    "property_iri": p["property_iri"],
                    "dest_id": p["dest_id"],
                    "dest_label": p["dest_label"],
                    "strength": p.get("strength", 0),  # Re-evaluated strength (0 if REJECT)
                    "judgment": p["judgment"],
                    "justification": p["justification"],
                })

        # Statistics of judgment results
        if self.progress:
            accept_count = sum(1 for p in all_properties_with_judgment if p.get("judgment") == "ACCEPT")
            review_count = sum(1 for p in all_properties_with_judgment if p.get("judgment") == "REVIEW")
            reject_count = sum(1 for p in all_properties_with_judgment if p.get("judgment") == "REJECT")

            # Statistics of strength changes
            strength_changed_count = sum(
                1 for p in all_properties_with_judgment
                if p.get("original_strength") != p.get("evaluated_strength")
            )

            print(f"[PropertyReviewer] Completed: {len(filtered)} properties [ACCEPT: {accept_count}, REVIEW: {review_count}, REJECT: {reject_count}]")
            print(f"[PropertyReviewer] Strength evaluation: {strength_changed_count} candidates had strength changed")

        # Save log (all candidates with labels, judgment, justification)
        log_output: Dict[str, Any] = {
            "properties": all_properties_with_judgment,
        }
        self.logger.save_log(log_output, filename_prefix="property_reviewer_output_")

        return filtered

    def invoke(self, input: Dict[str, Any], config=None) -> Dict[str, Any]:
        """
        Input:
          - Recommended: {"property_candidates": {...}, "class_info": {...}}
          - Compatible: {"properties":[...], "id":"...", "class_info": {...}}
        """
        # Flexibly interpret input
        property_candidates = input.get("property_candidates") or {
            "id": input.get("id"),
            "properties": input.get("properties", []),
        }
        class_info = input.get("class_info") or {}

        if not class_info.get("classes"):
            # Error: cannot reference source without class_info
            raise ValueError("PropertyReviewer requires class_info with classes for filtering. Cannot proceed without class information.")

        # Filter with LLM
        filtered = asyncio.run(self._filter_properties(property_candidates, class_info))

        if self.progress:
            print(
                f"PropertyReviewer: {len(property_candidates.get('properties', []))} candidates -> "
                f"{len(filtered)} accepted"
            )

        # For output (include judgment/justification for loop with PropertyEditor)
        output_id = str(uuid.uuid4())
        properties = [
            {
                "id": p["id"],
                "src_id": p["src_id"],
                "src_label": p["src_label"],
                "property_iri": p["property_iri"],
                "dest_id": p["dest_id"],
                "dest_label": p["dest_label"],
                "strength": p.get("strength", 0),  # Re-evaluated strength
                "judgment": p["judgment"],
                "justification": p["justification"],
            }
            for p in filtered
        ]
        output: Dict[str, Any] = {
            "id": output_id,
            "properties": properties,
        }
        return output


if __name__ == "__main__":
    # Test execution:
    # - Load latest PropertyExtractor output (log/property_extractor/property_extractor_output_*.json)
    # - Load latest ClassConsolidator output (log/class_consolidator/class_consolidator_output_*.json)
    # - Display filtering results
    pf = PropertyReviewer()

    prop_files = sorted(glob.glob("/workspace/app/log/property_extractor/property_extractor_output_*.json"), reverse=True)
    cls_files = sorted(glob.glob("/workspace/app/log/class_consolidator/class_consolidator_output_*.json"), reverse=True)

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

    print("\n=== PropertyReviewer Test ===")
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
            # Resolve labels for display
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


