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

    def _extract_table_blocks(self, pdf_path: str) -> list:
        """
        Extract structured table blocks from PDF.

        Args:
            pdf_path: Path to PDF file

        Returns:
            List of structured blocks (see KnowledgeExtractionState.structured_blocks)

        Examples:
            >>> extractor = DocTextExtractor()
            >>> blocks = extractor._extract_table_blocks("../doc/sample.pdf")
            >>> len(blocks) > 0  # If PDF has tables
            True
        """
        from src.utils.structured_block_builder import build_table_blocks
        from src.utils.table_extractor import extract_tables_from_pdf

        all_blocks = []
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page_number, page in enumerate(pdf.pages, start=1):
                    # Extract and normalize tables from this page
                    tables = extract_tables_from_pdf(page)
                    if not tables:
                        continue

                    # Build metadata for this page
                    filename = os.path.basename(pdf_path)
                    file_id = f"pdf:{filename}:page_{page_number}"
                    scope_ref = f"page_{page_number}"
                    section_path = f"{filename} - Page {page_number}"

                    # Extract context (first 200 chars of page text)
                    page_text = page.extract_text() or ""
                    context_window = " ".join(page_text.split())[:200]

                    # Build structured blocks using shared utility
                    page_blocks = build_table_blocks(tables, file_id, scope_ref, section_path, context_window)
                    all_blocks.extend(page_blocks)
        except Exception as e:
            print(f"Error extracting tables from PDF: {e}")
            # Continue with text-only extraction

        return all_blocks

    def invoke(self, input: Dict[str, Any], config: RunnableConfig = None) -> Dict[str, Any]:
        # input: DocTextExtractorInput
        # output: DocTextExtractorOutput

        file_path = input["files"][0]["path"]

        # Extract text from PDF
        extracted_text = self._extract_text(file_path)

        # Extract structured table blocks
        all_blocks = self._extract_table_blocks(file_path)

        # Split text by line
        sentences = []
        for line_text in extracted_text.split("\n"):
            if line_text.strip():  # Exclude empty lines
                sentences.append({"text": line_text.strip()})

        output = self.build_output(file_path, sentences)

        # Attach structured blocks at output level (cleaner than attaching to first sentence)
        if all_blocks:
            output["_pdf_structured_blocks"] = all_blocks

        # Save output JSON to log
        self.logger.save_log(output, "doc_text_extractor_output_")

        return output


if __name__ == "__main__":
    extractor = DocTextExtractor()
    result = extractor.invoke({"files": [{"path": "../doc/sample.pdf"}]})
    print(f"Processing result: {result}")
