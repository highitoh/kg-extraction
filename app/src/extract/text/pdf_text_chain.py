from typing import Any, Dict

from langchain_core.runnables import Runnable
from langchain_core.runnables.config import RunnableConfig

from .pdf_text_extractor import PdfTextExtractor
from .pdf_text_filter import PdfTextFilter
from .text_transformer import TextTransformer


class PdfTextChain(Runnable):
    """PDF text extraction chain"""

    def __init__(self, model: str = None, progress: bool = True):
        """
        Args:
            model: LLM model name (currently unused, for future extension)
            progress: Progress display flag
        """
        self.extractor = PdfTextExtractor()
        self.transformer = TextTransformer()
        self.filter = PdfTextFilter()
        self.model = model
        self.progress = progress

    def invoke(self, input: Dict[str, Any], config: RunnableConfig = None) -> Dict[str, Any]:
        # Step 1: Text extraction
        if self.progress:
            print("\n=== PdfTextExtractor ===")
        extracted = self.extractor.invoke(input, config)

        # Step 2: Text concatenation
        # transformed = self.transformer.invoke(extracted)

        # Step 3: Filtering
        if self.progress:
            print("\n=== PdfTextFilter ===")
        filtered = self.filter.invoke({"source": extracted, "filter": {"name": "dummy"}}, config)
        return filtered


if __name__ == "__main__":
    chain = PdfTextChain()
    result = chain.invoke({"files": [{"path": "../doc/sample.pdf"}]})
    print(result)
