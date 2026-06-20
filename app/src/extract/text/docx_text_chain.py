from typing import Any, Dict

from langchain_core.runnables import Runnable

from .docx_text_extractor import DocxTextExtractor


class DocxTextChain(Runnable):
    """
    DOCX text extraction chain

    Pipeline: DocxTextExtractor → (optional filter in future)
    """

    def __init__(self):
        self.extractor = DocxTextExtractor()

    def invoke(self, input: Dict[str, Any], config=None) -> Dict[str, Any]:
        """
        Execute DOCX text extraction

        Input: {"files": [{"path": "..."}]}
        Output: Extractor output (text + structured blocks)
        """
        result = self.extractor.invoke(input)

        # Future: Could add DocxTextFilter here (similar to PdfTextFilter)
        # For now, DOCX text is clean enough without filtering

        return result


if __name__ == "__main__":
    # Test the chain
    chain = DocxTextChain()
    result = chain.invoke({"files": [{"path": "../doc/sample.docx"}]})
    print(f"Extracted {len(result['sentences'])} paragraphs")
    if "_docx_structured_blocks" in result:
        print(f"Found {len(result['_docx_structured_blocks'])} tables")
