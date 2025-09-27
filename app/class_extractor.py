from typing import Dict, Any, List, Union
import asyncio
import json
import os
import uuid

from langchain.schema.runnable import Runnable
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

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

        ai = await self.llm_json.ainvoke([msg])
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

        self.logger.save_log(output)

        return output
