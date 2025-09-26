from typing import Any, Dict
import os
import uuid

from langchain.schema.runnable import Runnable
from langchain_core.runnables.config import RunnableConfig
import pdfplumber

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

        # PDFからテキストを抽出
        extracted_text = self._extract_text(file_path)

        # テキストを行ごとに分割
        sentences = []
        for line_num, line_text in enumerate(extracted_text.split('\n'), 1):
            if line_text.strip():  # 空行は除外
                sentences.append({
                    "line": line_num,
                    "text": line_text.strip()
                })

        return {
            "id": str(uuid.uuid4()),
            "file_name": os.path.basename(file_path),
            "sentences": sentences
        }
