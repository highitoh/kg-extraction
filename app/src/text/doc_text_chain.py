from typing import Any, Dict

from langchain_core.runnables import Runnable
from langchain_core.runnables.config import RunnableConfig

from .doc_text_extractor import DocTextExtractor
from .doc_text_filter import DocTextFilter
from .text_transformer import TextTransformer

class DocTextChain(Runnable):
    """ドキュメントテキスト抽出チェイン"""

    def __init__(self, model: str = None, progress: bool = True):
        """
        Args:
            model: LLMモデル名（現在未使用、将来の拡張用）
            progress: 進捗表示フラグ
        """
        self.extractor = DocTextExtractor()
        self.transformer = TextTransformer()
        self.filter = DocTextFilter()
        self.model = model
        self.progress = progress

    def invoke(self, input: Dict[str, Any], config: RunnableConfig = None) -> Dict[str, Any]:
        # Step 1: テキスト抽出
        if self.progress:
            print("\n=== DocTextExtractor ===")
        extracted = self.extractor.invoke(input, config)

        # Step 2: テキスト連結
        # transformed = self.transformer.invoke(extracted)

        # Step 3: フィルタリング
        if self.progress:
            print("\n=== DocTextFilter ===")
        filtered = self.filter.invoke({"source": extracted, "filter": {"name": "dummy"}}, config)
        return filtered


if __name__ == "__main__":
    chain = DocTextChain()
    result = chain.invoke({
        "files": [{"path": "../doc/sample.pdf"}]
    })
    print(result)
