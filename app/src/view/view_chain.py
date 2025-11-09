from typing import Dict, Any
from langchain_core.runnables import Runnable

from .view_extractor import ViewExtractor
from .view_filter import ViewFilter


class ViewChain(Runnable):
    """
    ビュー記述抽出チェイン:
      ViewExtractor -> ViewFilter
    """

    def __init__(self, model: str = None, progress: bool = True):
        """
        Args:
            model: LLMモデル名（Noneの場合はViewExtractorのデフォルトを使用）
            progress: 進捗表示フラグ
        """
        if model is not None:
            self.extractor = ViewExtractor(model=model, progress=progress)
        else:
            self.extractor = ViewExtractor(progress=progress)
        self.view_filter = ViewFilter(progress=progress)
        self.progress = progress

    def invoke(self, input: Dict[str, Any], config=None) -> Dict[str, Any]:
        # ViewExtractor実行
        if self.progress:
            print("\n=== ViewExtractor ===")
        result = self.extractor.invoke(input, config)

        # ViewFilter実行
        if self.progress:
            print("\n=== ViewFilter ===")
        result = self.view_filter.invoke(result, config)

        return result


def create_view_chain(model: str = None, progress: bool = True) -> Runnable:
    """
    ビュー記述抽出チェイン:
      ViewExtractor -> ViewFilter

    Args:
        model: LLMモデル名（Noneの場合はViewExtractorのデフォルトを使用）
        progress: 進捗表示フラグ
    """
    return ViewChain(model=model, progress=progress)

if __name__ == "__main__":
    chain = create_view_chain()
    sample_input = {
        "target": {
            "id": "pdf-text-id-001",
            "file_id": "file-abc",
            "file_name": "sample.pdf",
            "chunks": [
                {"text": "本書はシステムのビジョンを述べる。利用者は注文をアプリから行える。"},
            ],
        },
        "metamodel": {
            # 実装時: 参照するメタモデル（JSON Schema / OWL等）の情報をここに配置
        },
    }

    result = chain.invoke(sample_input)
    print(result)
