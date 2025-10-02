from typing import Dict, Any
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
                 log_dir: str = "log/class_filter"):
        self.llm = llm or ChatOpenAI(
            model=model,
            temperature=temperature,
            reasoning={"effort": "minimal"},
            output_version="responses/v1",
        )
        self.progress = progress
        self.logger = Logger(log_dir)

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

    def _load_prompt_template(self) -> str:
        """プロンプトテンプレートファイルを読み込む"""
        prompt_path = os.path.join(os.path.dirname(__file__), "prompts",
                                   "class_filter.txt")
        with open(prompt_path, "r", encoding="utf-8") as f:
            return f.read()

    def invoke(self, input: Dict[str, Any], config=None) -> Dict[str, Any]:
        classes = input.get("classes", [])

        if self.progress:
            print(f"ClassFilter: processing {len(classes)} classes")

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
        response = llm_json.invoke([HumanMessage(content=prompt)])

        # レスポンスからテキストを抽出
        result_text = self._to_text(response.content)

        # JSON パース
        try:
            parsed = json.loads(result_text)
            flags = parsed["flags"]

            filtered_classes = [c for c, hit in zip(classes, flags) if not hit]
            if self.progress:
                print(f"ClassFilter: {len(filtered_classes)} classes left")

        except json.JSONDecodeError:
            if self.progress:
                print(f"[ClassFilter] JSON parse error: {result_text}")
            filtered_classes = classes

        # 出力を構築
        output = input.copy()
        output["classes"] = filtered_classes

        if self.progress:
            print(
                f"ClassFilter: {len(classes)} -> {len(filtered_classes)} classes"
            )

        self.logger.save_log(output, filename_prefix="class_filter_output_")

        return output

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
