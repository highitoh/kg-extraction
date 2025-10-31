from typing import Dict, Any
from langchain.schema.runnable import RunnableSequence, Runnable

from class_extractor import ClassExtractor
from class_filter import ClassFilter
from class_label_corrector import ClassLabelCorrector
from class_consolidator import ClassConsolidator


class ClassChainWithReview(Runnable):
    """
    Class extraction chain with iterative review loop:
      ClassExtractor -> ClassFilter -> (ClassLabelCorrector -> ClassFilter)* -> ClassConsolidator

    ループロジック:
    - ClassFilterでREVIEW判定されたクラスをClassLabelCorrectorで修正
    - 修正後、再度ClassFilterで評価
    - すべてACCEPT/REJECTになるか、最大反復回数に達するまで繰り返す
    - 最大回数到達後もREVIEWが残る場合はREJECTに変換
    """

    def __init__(self,
                 extractor: ClassExtractor,
                 class_filter: ClassFilter,
                 corrector: ClassLabelCorrector,
                 consolidator: ClassConsolidator,
                 max_review_iterations: int = 3,
                 progress: bool = True):
        """
        Args:
            extractor: ClassExtractorインスタンス
            class_filter: ClassFilterインスタンス
            corrector: ClassLabelCorrectorインスタンス
            consolidator: ClassConsolidatorインスタンス
            max_review_iterations: REVIEW判定の最大反復回数（デフォルト: 3）
            progress: 進捗表示フラグ
        """
        self.extractor = extractor
        self.class_filter = class_filter
        self.corrector = corrector
        self.consolidator = consolidator
        self.max_review_iterations = max_review_iterations
        self.progress = progress

    def invoke(self, input: Dict[str, Any], config=None) -> Dict[str, Any]:
        """
        クラス抽出チェインのメイン実行

        Parameters
        ----------
        input: ViewChainOutput
          - view_info: ビュー抽出結果
          - metamodel: メタモデル情報

        Returns
        -------
        output: ClassConsolidatorOutput
          - id: 抽出ID
          - classes: 統合されたクラス個体リスト
        """
        # 1. ClassExtractor実行
        if self.progress:
            print("\n=== ClassExtractor ===")
        result = self.extractor.invoke(input)

        # 2. ClassFilter実行（初回）
        if self.progress:
            print("\n=== ClassFilter (Initial) ===")
        result = self.class_filter.invoke(result)

        # 3. ループ: ClassLabelCorrector -> ClassFilter
        for iteration in range(self.max_review_iterations):
            # REVIEWが残っているか確認
            review_count = sum(1 for c in result.get("classes", []) if c.get("judgment") == "REVIEW")

            if review_count == 0:
                # すべてACCEPT/REJECT → ループ終了
                if self.progress:
                    print(f"\n=== Review Loop Complete (iteration {iteration + 1}) ===")
                    print("No REVIEW classes remaining")
                break

            if self.progress:
                print(f"\n=== Review Iteration {iteration + 1}/{self.max_review_iterations} ===")
                print(f"REVIEW classes remaining: {review_count}")

            # ClassLabelCorrector実行
            if self.progress:
                print(f"\n--- ClassLabelCorrector (iteration {iteration + 1}) ---")
            result = self.corrector.invoke(result)

            # ClassFilter再実行
            if self.progress:
                print(f"\n--- ClassFilter (iteration {iteration + 1}) ---")
            result = self.class_filter.invoke(result)

        # 4. 最大回数到達後もREVIEWが残っている場合はREJECTに変換
        final_classes = []
        rejected_review_count = 0
        for c in result.get("classes", []):
            if c.get("judgment") == "REVIEW":
                c_copy = c.copy()
                c_copy["judgment"] = "REJECT"
                c_copy["justification"] = f"Maximum review iterations ({self.max_review_iterations}) reached. {c.get('justification', '')}"
                final_classes.append(c_copy)
                rejected_review_count += 1
            else:
                final_classes.append(c)

        if rejected_review_count > 0 and self.progress:
            print(f"\n=== Converting {rejected_review_count} remaining REVIEW classes to REJECT ===")

        result["classes"] = final_classes

        # 5. ClassConsolidator実行
        # ClassFilterでREJECTは除外されるため、ACCEPTのみが渡される
        if self.progress:
            print("\n=== ClassConsolidator ===")
        result = self.consolidator.invoke(result)

        return result


def create_class_chain(model: str = None, max_review_iterations: int = 3, progress: bool = True) -> Runnable:
    """
    Create class extraction chain with review loop:
      ClassExtractor -> ClassFilter -> (ClassLabelCorrector -> ClassFilter)* -> ClassConsolidator

    Args:
        model: LLMモデル名（Noneの場合は各コンポーネントのデフォルトを使用）
        max_review_iterations: REVIEW判定の最大反復回数（デフォルト: 3）
        progress: 進捗表示フラグ（デフォルト: True）
    """
    if model is not None:
        extractor = ClassExtractor(model=model)
        class_filter = ClassFilter(model=model)
        corrector = ClassLabelCorrector(model=model)
        consolidator = ClassConsolidator(model=model)
    else:
        extractor = ClassExtractor()
        class_filter = ClassFilter()
        corrector = ClassLabelCorrector()
        consolidator = ClassConsolidator()

    return ClassChainWithReview(
        extractor=extractor,
        class_filter=class_filter,
        corrector=corrector,
        consolidator=consolidator,
        max_review_iterations=max_review_iterations,
        progress=progress
    )

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