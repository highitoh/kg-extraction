from typing import Dict, Any
from langchain_core.runnables import Runnable, RunnableSequence

from property_extractor import PropertyExtractor
from property_consolidator import PropertyConsolidator
from property_ranker import PropertyRanker
from property_reviewer import PropertyReviewer
from property_editor import PropertyEditor
from property_validator import PropertyValidator


class ExtractorToRankerAdapter(Runnable):
    """
    PropertyExtractorの出力とオリジナルの入力を組み合わせて
    PropertyRankerの入力形式に変換するアダプター
    """

    def __init__(self, original_input: Dict[str, Any]):
        self.original_input = original_input

    def invoke(self, input: Dict[str, Any], config=None) -> Dict[str, Any]:
        """
        入力: PropertyExtractorの出力 {id, properties}
        出力: PropertyRankerの入力 {property_candidates: {...}, class_info: {...}}
        """
        return {
            "property_candidates": {
                "id": input.get("id"),
                "properties": input.get("properties", [])
            },
            "class_info": self.original_input.get("class_info", {})
        }


class RankerToFilterAdapter(Runnable):
    """
    PropertyRankerの出力とオリジナルの入力を組み合わせて
    PropertyReviewerの入力形式に変換するアダプター
    """

    def __init__(self, original_input: Dict[str, Any]):
        self.original_input = original_input

    def invoke(self, input: Dict[str, Any], config=None) -> Dict[str, Any]:
        """
        入力: PropertyRankerの出力 {id, properties}
        出力: PropertyReviewerの入力 {property_candidates: {...}, class_info: {...}}
        """
        return {
            "property_candidates": {
                "id": input.get("id"),
                "properties": input.get("properties", [])
            },
            "class_info": self.original_input.get("class_info", {})
        }


class FilterToEditorAdapter(Runnable):
    """
    PropertyReviewerの出力とオリジナルの入力を組み合わせて
    PropertyEditorの入力形式に変換するアダプター
    """

    def __init__(self, original_input: Dict[str, Any]):
        self.original_input = original_input

    def invoke(self, input: Dict[str, Any], config=None) -> Dict[str, Any]:
        """
        入力: PropertyReviewerの出力 {id, properties}
        出力: PropertyEditorの入力 {id, properties, class_info}
        """
        return {
            "id": input.get("id"),
            "properties": input.get("properties", []),
            "class_info": self.original_input.get("class_info", {})
        }


class EditorToFilterAdapter(Runnable):
    """
    PropertyEditorの出力をPropertyReviewerの入力形式に変換するアダプター
    """

    def __init__(self, original_input: Dict[str, Any]):
        self.original_input = original_input

    def invoke(self, input: Dict[str, Any], config=None) -> Dict[str, Any]:
        """
        入力: PropertyEditorの出力 {id, properties}
        出力: PropertyReviewerの入力 {property_candidates: {...}, class_info: {...}}
        """
        return {
            "property_candidates": {
                "id": input.get("id"),
                "properties": input.get("properties", [])
            },
            "class_info": self.original_input.get("class_info", {})
        }


class PropertyChainWithReview(Runnable):
    """
    Property extraction chain with iterative review loop:
      PropertyExtractor -> PropertyConsolidator -> PropertyRanker -> PropertyReviewer -> (PropertyEditor -> PropertyReviewer)* -> PropertyValidator

    ループロジック:
    - PropertyReviewerでREVIEW判定されたプロパティをPropertyEditorで修正
    - 修正後、再度PropertyReviewerで評価
    - すべてACCEPT/REJECTになるか、最大反復回数に達するまで繰り返す
    - 最大回数到達後もREVIEWが残る場合はACCEPTに変換
    """

    def __init__(self,
                 extractor: PropertyExtractor,
                 consolidator: PropertyConsolidator,
                 ranker: PropertyRanker,
                 property_reviewer: PropertyReviewer,
                 editor: PropertyEditor,
                 validator: PropertyValidator,
                 original_input: Dict[str, Any],
                 existing_properties: list = None,
                 max_review_iterations: int = 3,
                 progress: bool = True):
        """
        Args:
            extractor: PropertyExtractorインスタンス
            consolidator: PropertyConsolidatorインスタンス
            ranker: PropertyRankerインスタンス
            property_reviewer: PropertyReviewerインスタンス
            editor: PropertyEditorインスタンス
            validator: PropertyValidatorインスタンス
            original_input: 元の入力（アダプター用）
            existing_properties: 既存プロパティリスト（オプション）
            max_review_iterations: REVIEW判定の最大反復回数（デフォルト: 3）
            progress: 進捗表示フラグ
        """
        self.extractor = extractor
        self.consolidator = consolidator
        self.ranker = ranker
        self.property_reviewer = property_reviewer
        self.editor = editor
        self.validator = validator
        self.original_input = original_input
        self.existing_properties = existing_properties or []
        self.max_review_iterations = max_review_iterations
        self.progress = progress

    def invoke(self, input: Dict[str, Any], config=None) -> Dict[str, Any]:
        """
        プロパティ抽出チェインのメイン実行

        Parameters
        ----------
        input: PropertyChainInput
          - class_info: クラス情報
          - metamodel: メタモデル情報

        Returns
        -------
        output: PropertyValidatorOutput
          - id: 抽出ID
          - properties: 検証済みプロパティリスト
        """
        # 1. PropertyExtractor実行
        if self.progress:
            print("\n=== PropertyExtractor ===")
        result = self.extractor.invoke(input)

        # 2. PropertyConsolidator実行（既存プロパティとマージ）
        if self.progress:
            print("\n=== PropertyConsolidator ===")
        consolidator_input = {
            "id": result.get("id"),
            "properties": result.get("properties", []),
            "existing_properties": self.existing_properties,
            "class_info": self.original_input.get("class_info", {})
        }
        result = self.consolidator.invoke(consolidator_input)

        # 3. PropertyRanker実行
        if self.progress:
            print("\n=== PropertyRanker ===")
        adapter1 = ExtractorToRankerAdapter(self.original_input)
        ranker_input = adapter1.invoke(result)
        result = self.ranker.invoke(ranker_input)

        # 4. PropertyReviewer実行（初回）
        if self.progress:
            print("\n=== PropertyReviewer (Initial) ===")
        adapter2 = RankerToFilterAdapter(self.original_input)
        filter_input = adapter2.invoke(result)
        result = self.property_reviewer.invoke(filter_input)

        # 5. ループ: PropertyEditor -> PropertyReviewer
        for iteration in range(self.max_review_iterations):
            # REVIEWが残っているか確認
            review_count = sum(1 for p in result.get("properties", []) if p.get("judgment") == "REVIEW")

            if review_count == 0:
                # すべてACCEPT/REJECT → ループ終了
                if self.progress:
                    print(f"\n=== Review Loop Complete (iteration {iteration + 1}) ===")
                    print("No REVIEW properties remaining")
                break

            if self.progress:
                print(f"\n=== Review Iteration {iteration + 1}/{self.max_review_iterations} ===")
                print(f"REVIEW properties remaining: {review_count}")

            # PropertyEditor実行
            if self.progress:
                print(f"\n--- PropertyEditor (iteration {iteration + 1}) ---")
            result = self.editor.invoke(result)

            # PropertyReviewer再実行
            if self.progress:
                print(f"\n--- PropertyReviewer (iteration {iteration + 1}) ---")
            adapter_loop = EditorToFilterAdapter(self.original_input)
            filter_input = adapter_loop.invoke(result)
            result = self.property_reviewer.invoke(filter_input)

        # 6. 最大回数到達後もREVIEWが残っている場合はACCEPTに変換
        final_properties = []
        accepted_review_count = 0
        for p in result.get("properties", []):
            if p.get("judgment") == "REVIEW":
                p_copy = p.copy()
                p_copy["judgment"] = "ACCEPT"
                p_copy["justification"] = f"Maximum review iterations ({self.max_review_iterations}) reached. Auto-accepted. {p.get('justification', '')}"
                final_properties.append(p_copy)
                accepted_review_count += 1
            else:
                final_properties.append(p)

        if accepted_review_count > 0 and self.progress:
            print(f"\n=== Converting {accepted_review_count} remaining REVIEW properties to ACCEPT ===")

        result["properties"] = final_properties

        # 7. PropertyValidator実行
        if self.progress:
            print("\n=== PropertyValidator ===")
        result = self.validator.invoke(result)

        return result


class PropertyChainWithAdapter(Runnable):
    """
    PropertyExtractor -> Adapter -> Ranker -> Adapter -> PropertyReviewer -> PropertyValidator
    をRunnableSequenceで連結するラッパー
    """

    def __init__(self, model: str = None):
        self.model = model

    def invoke(self, input: Dict[str, Any], config=None) -> Dict[str, Any]:
        # コンポーネントを作成
        if self.model is not None:
            extractor = PropertyExtractor(model=self.model)
            ranker = PropertyRanker(model=self.model)
            property_reviewer = PropertyReviewer(model=self.model)
        else:
            extractor = PropertyExtractor()
            ranker = PropertyRanker()
            property_reviewer = PropertyReviewer()
        validator = PropertyValidator()

        # アダプターを作成（元の入力を保持）
        adapter1 = ExtractorToRankerAdapter(input)
        adapter2 = RankerToFilterAdapter(input)

        # RunnableSequenceで連結
        # extractor -> adapter1 -> ranker -> adapter2 -> property_reviewer -> validator
        chain = RunnableSequence(extractor, adapter1, ranker, adapter2, property_reviewer, validator)

        return chain.invoke(input, config)


def create_property_chain(model: str = None, max_review_iterations: int = 3, enable_review: bool = True, existing_properties: list = None) -> Runnable:
    """
    プロパティ抽出チェーン:
      - enable_review=True: PropertyExtractor -> PropertyConsolidator -> PropertyRanker -> PropertyReviewer -> (PropertyEditor -> PropertyReviewer)* -> PropertyValidator
      - enable_review=False: PropertyExtractor -> PropertyConsolidator -> PropertyRanker -> PropertyReviewer -> PropertyValidator

    Args:
        model: LLMモデル名（Noneの場合は各コンポーネントのデフォルトを使用）
        max_review_iterations: REVIEW判定の最大反復回数（デフォルト: 3）
        enable_review: Reviewループを有効にするか（デフォルト: True）
        existing_properties: 既存プロパティリスト（統合用、デフォルト: None）
    """
    if not enable_review:
        return PropertyChainWithAdapter(model=model)

    # PropertyChainWithReviewを使用する場合、invoke時に入力が必要なのでラッパーを返す
    class PropertyChainWithReviewWrapper(Runnable):
        def __init__(self, model: str = None, max_review_iterations: int = 3, existing_properties: list = None):
            self.model = model
            self.max_review_iterations = max_review_iterations
            self.existing_properties = existing_properties or []

        def invoke(self, input: Dict[str, Any], config=None) -> Dict[str, Any]:
            # コンポーネントを作成
            if self.model is not None:
                extractor = PropertyExtractor(model=self.model)
                consolidator = PropertyConsolidator(model=self.model)
                ranker = PropertyRanker(model=self.model)
                property_reviewer = PropertyReviewer(model=self.model)
                editor = PropertyEditor(model=self.model)
            else:
                extractor = PropertyExtractor()
                consolidator = PropertyConsolidator()
                ranker = PropertyRanker()
                property_reviewer = PropertyReviewer()
                editor = PropertyEditor()
            validator = PropertyValidator()

            # PropertyChainWithReviewを作成
            chain = PropertyChainWithReview(
                extractor=extractor,
                consolidator=consolidator,
                ranker=ranker,
                property_reviewer=property_reviewer,
                editor=editor,
                validator=validator,
                original_input=input,
                existing_properties=self.existing_properties,
                max_review_iterations=self.max_review_iterations,
                progress=True
            )

            return chain.invoke(input, config)

    return PropertyChainWithReviewWrapper(model=model, max_review_iterations=max_review_iterations, existing_properties=existing_properties)


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