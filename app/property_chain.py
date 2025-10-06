from langchain.schema.runnable import Runnable, RunnableSequence

from property_extractor import PropertyExtractor
from property_filter import PropertyFilter
from property_validator import PropertyValidator

def create_property_chain() -> Runnable:
    """
    プロパティ抽出チェーン:
      PropertyExtractor -> PropertyFilter -> PropertyValidator
    """
    extractor = PropertyExtractor()
    property_filter = PropertyFilter()
    validator = PropertyValidator()
    return RunnableSequence(extractor, property_filter, validator)


if __name__ == "__main__":
    import json

    chain = create_property_chain()
    sample_input = {
        "class_info": {
            "id": "class-chain-001",
            "classes": [
                {
                    "id": "class-001",
                    "class_iri": "ex:BusinessConceptClass",
                    "label": "顧客管理システム",
                    "file_id": "file-abc",
                },
                {
                    "id": "class-002",
                    "class_iri": "ex:SystemComponentClass",
                    "label": "データベース",
                    "file_id": "file-abc",
                }
            ]
        },
        "metamodel": {
            # メタモデル情報はここに格納
        },
    }

    result = chain.invoke(sample_input)
    print(json.dumps(result, ensure_ascii=False, indent=2))