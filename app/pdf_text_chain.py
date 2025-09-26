from typing import Any, Dict

from langchain.schema.runnable import Runnable
from langchain_core.runnables.config import RunnableConfig

from text_extractor import TextExtractor
from text_filter import TextFilter

class PDFTextChain(Runnable):
    """PDFテキスト抽出チェイン"""

    def __init__(self):
        self.extractor = TextExtractor()
        self.filter = TextFilter()

    def invoke(self, input: Dict[str, Any], config: RunnableConfig = None) -> Dict[str, Any]:
        # Step 1: テキスト抽出
        extracted = self.extractor.invoke(input, config)
        # Step 2: フィルタリング
        filtered = self.filter.invoke({"source": extracted, "filter": {"name": "dummy"}}, config)
        return filtered


if __name__ == "__main__":
    chain = PDFTextChain()
    result = chain.invoke({
        "files": [{"path": "../doc/sample.pdf"}]
    })
    print(result)
