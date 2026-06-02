from typing import Any, Dict

from langchain_core.runnables import Runnable
from langchain_core.runnables.config import RunnableConfig

from .presentation_text_extractor import PresentationTextExtractor
from .text_filter import TextFilter
from .text_transformer import TextTransformer


class PresentationTextChain(Runnable):
    """Presentation text extraction chain"""

    def __init__(self, model: str = None, progress: bool = True):
        """
        Args:
            model: LLM model name (currently unused, for future extension)
            progress: Progress display flag
        """
        self.extractor = PresentationTextExtractor()
        # self.filter = TextFilter()
        self.model = model
        self.progress = progress

    def invoke(self, input: Dict[str, Any], config: RunnableConfig = None) -> Dict[str, Any]:
        # Step 1: Text extraction
        if self.progress:
            print("\n=== PresentationTextExtractor ===")
        extracted = self.extractor.invoke(input, config)

        # Step 2: Filtering
        # filtered = self.filter.invoke({"source": extracted, "filter": {"name": "dummy"}}, config)
        return extracted


if __name__ == "__main__":
    chain = PresentationTextChain()
    result = chain.invoke({"files": [{"path": "../doc/presentation_sample.pdf"}]})
    print(result)
