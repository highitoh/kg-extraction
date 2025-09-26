from typing import Any, Dict, List
import asyncio
import json
import uuid

from langchain.schema.runnable import Runnable
from langchain_core.runnables.config import RunnableConfig
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

class TextFilter(Runnable):
    """抽出テキストをフィルタリングするタスク"""

    def __init__(self):
        with open("prompts/sentence_filter.txt", "r", encoding="utf-8") as f:
            self.classification_prompt = f.read()

        self._llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.0
        )
        self._llm_json = self._llm.bind(response_format={"type": "json_object"})

    async def _classify_text(self, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """LLMを使用してテキストを分類し、文章のみを抽出"""
        # 行番号付きテキストを作成
        numbered_text = ""
        for item in items:
            numbered_text += f"{item['line']}: {item['text']}\n"

        msg = HumanMessage(
            content=self.classification_prompt + f"\n\n# 対象テキスト\n{numbered_text}"
        )

        try:
            ai = await self._llm_json.ainvoke([msg])
            raw_content = ai.content if isinstance(ai.content, str) else str(ai.content)

            classifications = json.loads(raw_content)
            if not isinstance(classifications, list):
                return []

            # "文章"に分類された行のみを抽出
            sentence_items = []
            for classification in classifications:
                if classification.get("content") == "文章":
                    start_text = classification.get("startLine", "")
                    end_text = classification.get("endLine", "")

                    # 該当する行を特定して追加
                    for item in items:
                        if start_text in item["text"] or end_text in item["text"] or item["text"] in start_text:
                            sentence_items.append(item)
                            break

            return sentence_items

        except Exception as e:
            print(f"テキスト分類中にエラーが発生しました: {e}")
            return []

    def invoke(self, input: Dict[str, Any], config: RunnableConfig = None) -> Dict[str, Any]:
        # input: TextFilterInput
        # output: TextFilterOutput

        source = input["source"]

        # TextExtractorOutput/TextFilterOutputの両方から"items"を取得
        items = source.get("items", [])

        # LLMを使用して文章のみを抽出
        filtered_items = asyncio.run(self._classify_text(items))

        return {
            "id": str(uuid.uuid4()),
            "file_id": source["file_id"],
            "file_name": source["file_name"],
            "items": filtered_items
        }