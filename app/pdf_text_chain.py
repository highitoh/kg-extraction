from typing import Any, Dict, List, Optional
import uuid
import os
import json
import asyncio
from langchain.schema.runnable import Runnable
from langchain.schema import RunnableConfig
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
import pdfplumber
from .logger import Logger

############################################################
# Text Extraction Task
############################################################
class TextExtractor(Runnable):
    """ファイルからテキストを抽出するタスク"""

    def _extract_text(self, pdf_path: str) -> str:
        """
        pdfplumberを使用してPDFからテキストを抽出
        """
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDFファイルが見つかりません: {pdf_path}")

        if not pdf_path.lower().endswith('.pdf'):
            raise ValueError("PDFファイルではありません")

        text = ""
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
        except Exception as e:
            print(f"pdfplumberでのテキスト抽出中にエラーが発生しました: {e}")

        return text.strip()

    def invoke(self, input: Dict[str, Any], config: RunnableConfig = None) -> Dict[str, Any]:
        # input: TextExtractorInput
        # output: TextExtractorOutput

        file_path = input["files"][0]["path"]
        file_id = str(uuid.uuid4())

        # PDFからテキストを抽出
        extracted_text = self._extract_text(file_path)

        # テキストを行ごとに分割
        items = []
        for line_num, line_text in enumerate(extracted_text.split('\n'), 1):
            if line_text.strip():  # 空行は除外
                items.append({
                    "line": line_num,
                    "text": line_text.strip()
                })

        return {
            "id": str(uuid.uuid4()),
            "file_id": file_id,
            "file_name": os.path.basename(file_path),
            "items": items
        }

############################################################
# Text Filtering Task
############################################################
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

############################################################
# PDFTextChainOutput Formatter
############################################################
class PDFTextChainPostProcess(Runnable):
    """PDFテキスト抽出チェインの後処理"""

    def __init__(self):
        self.logger = Logger("./app/log/pdf_text_chain")

    def invoke(self, input: Dict[str, Any], config: RunnableConfig = None) -> Dict[str, Any]:
        # input: TextExtractorOutput または TextFilterOutput
        # output: PDFTextChainOutput

        # 入力の形式を判定し、適切にデータを抽出
        if "items" in input:
            # TextExtractorOutput または TextFilterOutput
            sentences = [
                {"line": item["line"], "text": item["text"]}
                for item in input["items"]
            ]
        else:
            # 予期しない形式の場合は空配列
            sentences = []

        output = {
            "id": input.get("id", str(uuid.uuid4())),
            "file_name": input.get("file_name", "unknown"),
            "sentences": sentences
        }

        # ログ出力
        self.logger.save_log(output)

        return output

############################################################
# PDFTextChain (Runnable chain)
############################################################
class PDFTextChain(Runnable):
    """PDFテキスト抽出チェイン"""

    def __init__(self):
        self.extractor = TextExtractor()
        self.filter = TextFilter()
        self.post_process = PDFTextChainPostProcess()

    def invoke(self, input: Dict[str, Any], config: RunnableConfig = None) -> Dict[str, Any]:
        # Step 1: テキスト抽出
        extracted = self.extractor.invoke(input, config)
        # Step 2: フィルタリング
        filtered = self.filter.invoke({"source": extracted, "filter": {"name": "dummy"}}, config)
        # Step 3: 後処理
        output = self.post_process.invoke(filtered, config)
        return output


if __name__ == "__main__":
    chain = PDFTextChain()
    result = chain.invoke({
        "files": [{"path": "sample.pdf"}]
    })
    print(result)
