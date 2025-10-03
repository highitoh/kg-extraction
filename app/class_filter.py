from typing import Dict, Any
import asyncio
import json
import os

from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage
from langchain.schema.runnable import Runnable

from logger import Logger


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
                 log_dir: str = "log/class_filter"):
        self.llm = llm or ChatOpenAI(
            model=model,
            temperature=temperature,
            reasoning={"effort": "minimal"},
            output_version="responses/v1",
        )
        self.progress = progress
        self.max_concurrency = max_concurrency
        self.logger = Logger(log_dir)

        # メタモデルを読み込み
        self.metamodel = self._load_metamodel()
        # プロンプトテンプレートを読み込み
        self.prompt_template = self._load_prompt_template()

    def _build_schema(self, n: int) -> dict:
        return {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "title": "ClassFilterFlags",
            "type": "object",
            "additionalProperties": False,
            "required": ["flags"],
            "properties": {
                "flags": {
                    "type": "array",
                    "items": {"type": "boolean"},
                    "minItems": n,
                    "maxItems": n
                }
            }
        }

    def _load_metamodel(self) -> dict:
        """メタモデルファイルを読み込む"""
        metamodel_path = os.path.join(os.path.dirname(__file__), "metamodel",
                                      "metamodel.json")
        with open(metamodel_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _build_prohibit_rules_section(self) -> str:
        """メタモデルから禁止ルールセクションを動的に生成"""
        rules_lines = []

        for cls in self.metamodel.get("classes", []):
            if "prohibit_rules" in cls and cls.get("prohibit_rules"):
                class_name = cls.get("name", "")
                class_iri = cls.get("iri", "").split("#")[-1]
                rules_lines.append(f"- {class_name}（{class_iri}）")
                for rule in cls["prohibit_rules"]:
                    rules_lines.append(f"  - {rule}")
                rules_lines.append("")

        return "\n".join(rules_lines).rstrip()

    def _load_prompt_template(self) -> str:
        """プロンプトテンプレートファイルを読み込み、禁止ルールを動的生成"""
        prompt_path = os.path.join(os.path.dirname(__file__), "prompts",
                                   "class_filter.txt")
        with open(prompt_path, "r", encoding="utf-8") as f:
            template = f.read()

        # 禁止ルールセクションを動的に生成
        prohibit_rules = self._build_prohibit_rules_section()

        # テンプレート内のプレースホルダーを置換
        replaced = template.replace("{PROHIBIT_RULES}", prohibit_rules)

        return replaced

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

                filtered = await self._filter_classes_async(class_instances)

                async with lock:
                    counter += 1
                    if self.progress:
                        print(f"[{counter}/{total}] ClassFilter: class={class_name}, {len(class_instances)} -> {len(filtered)} instances")

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

    async def _filter_classes_async(self, classes: list) -> list:
        """指定されたクラスリストをLLMでフィルタリング（非同期版）"""
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

        # プロンプトを構築
        prompt = f"{self.prompt_template}\n{json.dumps(classes, ensure_ascii=False, indent=2)}"

        # LLMでフィルタリング実行
        response = await llm_json.ainvoke([HumanMessage(content=prompt)])

        # レスポンスからテキストを抽出
        result_text = self._to_text(response.content)

        # JSON パース
        try:
            parsed = json.loads(result_text)
            flags = parsed["flags"]
            filtered_classes = [c for c, hit in zip(classes, flags) if not hit]
        except json.JSONDecodeError:
            if self.progress:
                print(f"[ClassFilter] JSON parse error: {result_text}")
            filtered_classes = classes

        return filtered_classes

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
