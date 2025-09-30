from typing import Dict, Any, List, Union
import asyncio
import json
import os
import uuid

from langchain.schema.runnable import Runnable
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from openai import APIConnectionError

from logger import Logger

class ClassExtractor(Runnable):
    """
    Class extractor from view information
    """

    def __init__(
        self,
        llm: ChatOpenAI | None = None,
        model: str = "gpt-5-nano",
        temperature: float = 0.0,
        max_concurrency: int = 8,
        progress: bool = True,
        log_dir: str = "log/class_extractor",
        max_retries: int = 5,
        retry_delay: float = 1.0,
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
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        self.prompt = self._load_prompt()

        self.views = [
            "value_analysis",
            "business_concept",
            "business_requirement",
            "system_requirement",
            "system_component",
            "data_analysis",
        ]

        self.logger = Logger(log_dir)

    def _load_prompt(self) -> str:
        prompt_path = os.path.join(os.path.dirname(__file__), "prompts", "ClassExtractor.txt")
        with open(prompt_path, "r", encoding="utf-8") as f:
            return f.read()

    @staticmethod
    def _to_text(c: Union[str, List[Any]]) -> str:
        if isinstance(c, str):
            return c
        for part in c:
            if isinstance(part, dict) and part.get("type") in ("output_text", "text"):
                return part.get("text", "")
        return ""

    async def _extract_classes_from_views(self, views: List[Dict[str, Any]]) -> Dict[str, List[str]]:
        view_texts = {view_type: [] for view_type in self.views}

        for view in views:
            view_type = view.get("type", "")
            if view_type not in self.views:
                continue
            texts = view.get("texts", [])
            for text_obj in texts:
                text = text_obj.get("text", "")
                if text.strip():
                    view_texts[view_type].append(text)

        formatted_views = []
        for view_type in self.views:
            texts = view_texts[view_type]
            if texts:
                formatted_views.append(f"[{view_type}]")
                for text in texts:
                    formatted_views.append(f"- {text}")
            else:
                formatted_views.append(f"[{view_type}]")
                formatted_views.append("- (none)")

        view_text_block = "\n".join(formatted_views)

        msg = HumanMessage(
            content=[
                {"type": "text", "text": self.prompt},
                {"type": "text", "text": f"# View Information\n{view_text_block}"},
            ]
        )

        ai = await self._invoke_with_retry([msg])
        raw = self._to_text(ai.content)
        try:
            data = json.loads(raw)
        except Exception:
            data = {}

        classes: Dict[str, List[str]] = {k: [] for k in self.views}
        for view_type in self.views:
            extracted_classes = data.get(view_type, [])
            if not isinstance(extracted_classes, list):
                continue

            seen = set()
            out = []
            for cls in extracted_classes:
                if not isinstance(cls, str):
                    continue
                cls = cls.strip()
                if not cls or cls in seen:
                    continue
                seen.add(cls)
                out.append(cls)
            classes[view_type] = out

        return classes

    async def _invoke_with_retry(self, messages: List) -> Any:
        """OpenAI API呼び出しをリトライロジック付きで実行"""
        last_exception = None

        for attempt in range(self.max_retries + 1):
            try:
                return await self.llm_json.ainvoke(messages)
            except APIConnectionError as e:
                last_exception = e
                if attempt < self.max_retries:
                    if self.progress:
                        print(f"ClassExtractor: API connection error (attempt {attempt + 1}/{self.max_retries + 1}), retrying in {self.retry_delay}s...")
                    await asyncio.sleep(self.retry_delay)
                    self.retry_delay *= 2  # Exponential backoff
                else:
                    if self.progress:
                        print(f"ClassExtractor: Failed after {self.max_retries + 1} attempts")
                    raise last_exception
            except Exception as e:
                if self.progress:
                    print(f"ClassExtractor: Unexpected error: {str(e)}")
                raise e

        raise last_exception

    def invoke(self, input: Dict[str, Any], config=None) -> Dict[str, Any]:
        view_info = input["view_info"]
        views = view_info.get("views", [])

        extracted_classes = asyncio.run(self._extract_classes_from_views(views))

        classes = []
        for view_type, class_labels in extracted_classes.items():
            for label in class_labels:
                file_id = "unknown"
                if views and views[0].get("texts"):
                    file_id = views[0]["texts"][0].get("file_id", "unknown")

                classes.append({
                    "id": str(uuid.uuid4()),
                    "class_iri": f"ex:{view_type.title().replace('_', '')}Class",
                    "label": label,
                    "file_id": file_id,
                })

        output: Dict[str, Any] = {
            "id": str(uuid.uuid4()),
            "classes": classes,
        }

        if self.progress:
            total_classes = len(classes)
            print(f"ClassExtractor: extracted {total_classes} classes")

        self.logger.save_log(output, filename_prefix="class_extractor_output_")

        return output


if __name__ == "__main__":
    import json

    # Create ClassExtractor instance
    extractor = ClassExtractor()

    # Sample input data for testing
    sample_input = {
        "view_info": {
            "id": "test-view-001",
            "views": [
                {
                    "type": "business_concept",
                    "texts": [
                        {
                            "file_id": "file-001",
                            "text": "顧客管理システムでは、ユーザー登録と認証を処理します"
                        },
                        {
                            "file_id": "file-001",
                            "text": "注文管理システムが商品の受注と配送を担当します"
                        }
                    ]
                },
                {
                    "type": "system_component",
                    "texts": [
                        {
                            "file_id": "file-002",
                            "text": "データベース接続モジュールとAPIゲートウェイの実装"
                        },
                        {
                            "file_id": "file-002",
                            "text": "認証サービスと権限管理サービスの構成"
                        }
                    ]
                },
                {
                    "type": "data_analysis",
                    "texts": [
                        {
                            "file_id": "file-003",
                            "text": "売上データの集計と顧客行動の分析機能"
                        }
                    ]
                }
            ]
        }
    }

    print("=== ClassExtractor Test ===")
    print("Input:")
    print(json.dumps(sample_input, ensure_ascii=False, indent=2))
    print("\n" + "="*50 + "\n")

    try:
        # Execute ClassExtractor
        result = extractor.invoke(sample_input)

        print("Output:")
        print(json.dumps(result, ensure_ascii=False, indent=2))

        print(f"\nSummary:")
        print(f"- Total classes extracted: {len(result['classes'])}")

        # Group by view type for display
        classes_by_view = {}
        for cls in result["classes"]:
            view_type = cls["class_iri"].replace("ex:", "").replace("Class", "")
            if view_type not in classes_by_view:
                classes_by_view[view_type] = []
            classes_by_view[view_type].append(cls["label"])

        for view_type, labels in classes_by_view.items():
            print(f"- {view_type}: {len(labels)} classes - {', '.join(labels)}")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
