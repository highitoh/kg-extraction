from langchain.schema.runnable import RunnableSequence, Runnable

from class_extractor import ClassExtractor
from class_filter import ClassFilter
from class_consolidator import ClassConsolidator

def create_class_chain(model: str = None) -> Runnable:
    """
    Create class extraction chain:
      ClassExtractor -> ClassFilter -> ClassConsolidator

    Args:
        model: LLMモデル名（Noneの場合は各コンポーネントのデフォルトを使用）
    """
    if model is not None:
        extractor = ClassExtractor(model=model)
        class_filter = ClassFilter(model=model)
        consolidator = ClassConsolidator(model=model)
    else:
        extractor = ClassExtractor()
        class_filter = ClassFilter()
        consolidator = ClassConsolidator()
    return RunnableSequence(extractor, class_filter, consolidator)

if __name__ == "__main__":
    import json

    chain = create_class_chain()
    sample_input = {
        "view_info": {
            "id": "view-chain-001",
            "views": [
                {
                    "type": "business_concept",
                    "texts": [
                        {
                            "file_id": "file-abc",
                            "line": 1,
                            "text": "顧客管理システムでは、ユーザー登録と認証を処理します"
                        }
                    ]
                },
                {
                    "type": "system_component",
                    "texts": [
                        {
                            "file_id": "file-abc",
                            "line": 2,
                            "text": "データベース接続モジュールの実装"
                        }
                    ]
                }
            ]
        },
        "metamodel": {
        },
    }

    result = chain.invoke(sample_input)
    print(json.dumps(result, ensure_ascii=False, indent=2))