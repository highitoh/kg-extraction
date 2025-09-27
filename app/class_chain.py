from langchain.schema.runnable import RunnableSequence, Runnable

from class_extractor import ClassExtractor
#from class_filter import ClassFilter

def create_class_chain() -> Runnable:
    """
    Create class extraction chain:
      ClassExtractor -> ClassFilter
    """
    extractor = ClassExtractor()
    #class_filter = ClassFilter()
    return extractor

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