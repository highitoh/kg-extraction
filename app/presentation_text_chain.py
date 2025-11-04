from typing import Any, Dict

from langchain.schema.runnable import Runnable
from langchain_core.runnables.config import RunnableConfig

from presentation_text_extractor import PresentationTextExtractor
from text_filter import TextFilter
from text_transformer import TextTransformer

class PresentationTextChain(Runnable):
    """プレゼンテーションテキスト抽出チェイン"""

    def __init__(self, model: str = None, progress: bool = True):
        """
        Args:
            model: LLMモデル名（現在未使用、将来の拡張用）
            progress: 進捗表示フラグ
        """
        self.extractor = PresentationTextExtractor()
        #self.filter = TextFilter()
        self.model = model
        self.progress = progress

    def invoke(self, input: Dict[str, Any], config: RunnableConfig = None) -> Dict[str, Any]:
        # Step 1: テキスト抽出
        if self.progress:
            print("\n=== PresentationTextExtractor ===")
        extracted = self.extractor.invoke(input, config)

        # Step 2: フィルタリング
        #filtered = self.filter.invoke({"source": extracted, "filter": {"name": "dummy"}}, config)
        return extracted


if __name__ == "__main__":
    chain = PresentationTextChain()
    result = chain.invoke({
        "files": [{"path": "../doc/presentation_sample.pdf"}]
    })
    print(result)
