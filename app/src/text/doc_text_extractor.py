from typing import Any, Dict
import os
import uuid

from langchain_core.runnables import Runnable
from langchain_core.runnables.config import RunnableConfig
import pdfplumber
from src.utils.logger import Logger

class DocTextExtractor(Runnable):
    """Task to extract text from document files"""

    def __init__(self):
        self.logger = Logger("/workspace/app/log/doc_text_extractor")

    def _extract_text(self, pdf_path: str) -> str:
        """
        Extract text from PDF using pdfplumber
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
        # input: DocTextExtractorInput
        # output: DocTextExtractorOutput

        file_path = input["files"][0]["path"]

        # Extract text from PDF
        extracted_text = self._extract_text(file_path)

        # Split text by line
        sentences = []
        for line_text in extracted_text.split('\n'):
            if line_text.strip():  # Exclude empty lines
                sentences.append({
                    "text": line_text.strip()
                })

        output = {
            "id": str(uuid.uuid4()),
            "file_name": os.path.basename(file_path),
            "sentences": sentences
        }

        # Save output JSON to log
        self.logger.save_log(output, "doc_text_extractor_output_")

        return output


if __name__ == "__main__":
    extractor = DocTextExtractor()
    result = extractor.invoke({"files": [{"path": "../doc/sample.pdf"}]})
    print(f"処理結果: {result}")
