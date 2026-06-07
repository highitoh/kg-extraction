import os
from typing import Any, Dict

import pdfplumber
from langchain_core.runnables.config import RunnableConfig

from .text_extractor import TextExtractor


class DocTextExtractor(TextExtractor):
    """Task to extract text from document files"""

    def __init__(self):
        super().__init__("/workspace/app/log/doc_text_extractor")

    def _extract_text(self, pdf_path: str) -> str:
        """
        Extract text from PDF using pdfplumber
        """
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        if not pdf_path.lower().endswith(".pdf"):
            raise ValueError("Not a PDF file")

        text = ""
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
        except Exception as e:
            print(f"Error occurred during text extraction with pdfplumber: {e}")

        return text.strip()

    def invoke(self, input: Dict[str, Any], config: RunnableConfig = None) -> Dict[str, Any]:
        # input: DocTextExtractorInput
        # output: DocTextExtractorOutput

        file_path = input["files"][0]["path"]

        # Extract text from PDF
        extracted_text = self._extract_text(file_path)

        # Split text by line
        sentences = []
        for line_text in extracted_text.split("\n"):
            if line_text.strip():  # Exclude empty lines
                sentences.append({"text": line_text.strip()})

        output = self.build_output(file_path, sentences)

        # Save output JSON to log
        self.logger.save_log(output, "doc_text_extractor_output_")

        return output


if __name__ == "__main__":
    extractor = DocTextExtractor()
    result = extractor.invoke({"files": [{"path": "../doc/sample.pdf"}]})
    print(f"Processing result: {result}")
