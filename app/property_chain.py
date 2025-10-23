from typing import Dict, Any
from langchain.schema.runnable import Runnable, RunnableSequence

from property_extractor import PropertyExtractor
from property_filter import PropertyFilter
from property_validator import PropertyValidator


class ExtractorToFilterAdapter(Runnable):
    """
    PropertyExtractorの出力とオリジナルの入力を組み合わせて
    PropertyFilterの入力形式に変換するアダプター
    """

    def __init__(self, original_input: Dict[str, Any]):
        self.original_input = original_input

    def invoke(self, input: Dict[str, Any], config=None) -> Dict[str, Any]:
        """
        入力: PropertyExtractorの出力 {id, properties}
        出力: PropertyFilterの入力 {property_candidates: {...}, class_info: {...}}
        """
        return {
            "property_candidates": {
                "id": input.get("id"),
                "properties": input.get("properties", [])
            },
            "class_info": self.original_input.get("class_info", {})
        }


class PropertyChainWithAdapter(Runnable):
    """
    PropertyExtractor -> Adapter -> PropertyFilter -> PropertyValidator
    をRunnableSequenceで連結するラッパー
    """

    def __init__(self, model: str = None):
        self.model = model

    def invoke(self, input: Dict[str, Any], config=None) -> Dict[str, Any]:
        # コンポーネントを作成
        if self.model is not None:
            extractor = PropertyExtractor(model=self.model)
            property_filter = PropertyFilter(model=self.model)
        else:
            extractor = PropertyExtractor()
            property_filter = PropertyFilter()
        validator = PropertyValidator()

        # アダプターを作成（元の入力を保持）
        adapter = ExtractorToFilterAdapter(input)

        # RunnableSequenceで連結
        chain = RunnableSequence(extractor, adapter, property_filter, validator)

        return chain.invoke(input, config)


def create_property_chain(model: str = None) -> Runnable:
    """
    プロパティ抽出チェーン:
      PropertyExtractor -> PropertyFilter -> PropertyValidator

    Args:
        model: LLMモデル名（Noneの場合は各コンポーネントのデフォルトを使用）
    """
    return PropertyChainWithAdapter(model=model)


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