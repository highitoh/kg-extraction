import logging

from langchain.schema.runnable import Runnable

from view_extractor import ViewExtractor
from view_filter import ViewFilter

def create_view_chain(model: str = None) -> Runnable:
    """
    ビュー記述抽出チェイン:
      ViewExtractor -> ViewFilter -> ViewChainLogger
    いずれもテンプレート（抽出/フィルタの中身は未実装）

    Args:
        model: LLMモデル名（Noneの場合はViewExtractorのデフォルトを使用）
    """
    if model is not None:
        extractor = ViewExtractor(model=model)
    else:
        extractor = ViewExtractor()
    view_filter = ViewFilter()
    return extractor | view_filter

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

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
