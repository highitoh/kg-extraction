from typing import Any, Dict, List
import asyncio
import json
import uuid

from langchain.schema.runnable import Runnable
from langchain_core.runnables.config import RunnableConfig
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from logger import Logger

class TextFilter(Runnable):
    """抽出テキストをフィルタリングするタスク"""

    def __init__(self):
        with open("./prompts/sentence_filter.txt", "r", encoding="utf-8") as f:
            self.classification_prompt = f.read()

        self._llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.0
        )
        self._llm_json = self._llm.bind(response_format={"type": "json_object"})
        self.logger = Logger("./log/text_filter")

    async def _classify_text(self, sentences: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """LLMを使用してテキストを分類し、文章のみを抽出"""
        # 行番号付きテキストを作成
        numbered_text = ""
        for sentence in sentences:
            numbered_text += f"{sentence['line']}: {sentence['text']}\n"

        msg = HumanMessage(
            content=self.classification_prompt + f"\n\n# 対象テキスト\n{numbered_text}"
        )

        try:
            ai = await self._llm_json.ainvoke([msg])
            raw_content = ai.content if isinstance(ai.content, str) else str(ai.content)

            classifications = json.loads(raw_content)
            if not isinstance(classifications, list):
                classifications = [classifications]

            # "文章"に分類された行のみを抽出
            filtered_sentences = []
            for classification in classifications:
                if classification.get("content") == "文章":
                    start_text = classification.get("startLine", "")
                    end_text = classification.get("endLine", "")

                    # startLineとendLineの範囲内のすべての行を抽出
                    start_index = None
                    end_index = None

                    # startLineとendLineのインデックスを特定
                    for i, sentence in enumerate(sentences):
                        if start_text in sentence["text"] or sentence["text"] in start_text:
                            start_index = i
                        if end_text in sentence["text"] or sentence["text"] in end_text:
                            end_index = i

                    # 範囲が特定できた場合、その範囲のすべての行を追加
                    if start_index is not None and end_index is not None:
                        for i in range(start_index, end_index + 1):
                            filtered_sentences.append(sentences[i])

            return filtered_sentences

        except Exception as e:
            print(f"テキスト分類中にエラーが発生しました: {e}")
            return []

    def invoke(self, input: Dict[str, Any], config: RunnableConfig = None) -> Dict[str, Any]:
        # input: TextFilterInput
        # output: TextFilterOutput

        source = input["source"]

        # TextExtractorOutput/TextFilterOutputの両方から"sentences"を取得
        sentences = source.get("sentences", [])

        # LLMを使用して文章のみを抽出
        filtered_sentences = asyncio.run(self._classify_text(sentences))

        output = {
            "id": str(uuid.uuid4()),
            "file_name": source["file_name"],
            "sentences": filtered_sentences
        }

        # 出力JSONをログに保存
        self.logger.save_log(output, "text_filter_output_")

        return output

if __name__ == "__main__":
    # テスト用の入力データ
    test_input = {
        "source": {
            "file_name": "test.pdf",
            "sentences": [
                {"line": 1, "text": "これは通常の文章です。"},
                {"line": 2, "text": "株式会社テスト"},
                {"line": 3, "text": "もう一つの文章example。"},
                {"line": 4, "text": "図1: グラフの説明"},
                {"line": 5, "text": "最後の文章です。"}
            ]
        }
    }

    # TextFilterのインスタンス作成とテスト実行
    text_filter = TextFilter()
    result = text_filter.invoke(test_input)

    print("TextFilter Test Result:")
    print(f"File name: {result['file_name']}")
    print(f"Filtered sentences count: {len(result['sentences'])}")
    for sentence in result["sentences"]:
        print(f"  Line {sentence['line']}: {sentence['text']}")
    print(f"Output ID: {result['id']}")