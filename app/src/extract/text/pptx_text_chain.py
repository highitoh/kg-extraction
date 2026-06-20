from typing import Any, Dict

from langchain_core.runnables import Runnable

from .pptx_text_extractor import PptxTextExtractor


class PptxTextChain(Runnable):
    """
    PPTX text extraction chain

    Pipeline: PptxTextExtractor → (optional filter in future)
    """

    def __init__(self):
        self.extractor = PptxTextExtractor()

    def invoke(self, input: Dict[str, Any], config=None) -> Dict[str, Any]:
        """
        Execute PPTX text extraction

        Input: {"files": [{"path": "..."}]}
        Output: Extractor output (text + structured blocks)
        """
        result = self.extractor.invoke(input)

        # Future: Could add PptxTextFilter here if needed
        # For now, PPTX text is clean enough without filtering

        return result


if __name__ == "__main__":
    # Test the chain
    chain = PptxTextChain()
    result = chain.invoke({"files": [{"path": "../doc/sample.pptx"}]})
    print(f"Extracted {len(result['sentences'])} lines")
    if "_pptx_structured_blocks" in result:
        print(f"Found {len(result['_pptx_structured_blocks'])} tables")
