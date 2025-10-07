from typing import Any, Dict

from langchain.schema.runnable import Runnable
from langchain_core.runnables.config import RunnableConfig

from presentation_text_extractor import PresentationTextExtractor
from text_filter import TextFilter
from text_transformer import TextTransformer

class PresentationTextChain(Runnable):
    """プレゼンテーションテキスト抽出チェイン"""

    def __init__(self):
        self.extractor = PresentationTextExtractor()
        self.transformer = TextTransformer()
        self.filter = TextFilter()

    def invoke(self, input: Dict[str, Any], config: RunnableConfig = None) -> Dict[str, Any]:
        # Step 1: テキスト抽出
        extracted = self.extractor.invoke(input, config)
        # Step 2: テキスト連結
        # transformed = self.transformer.invoke(extracted)
        # Step 3: フィルタリング
        filtered = self.filter.invoke({"source": extracted, "filter": {"name": "dummy"}}, config)
        return filtered


if __name__ == "__main__":
    chain = PresentationTextChain()
    result = chain.invoke({
        "files": [{"path": "../doc/sample.pdf"}]
    })
    print(result)
