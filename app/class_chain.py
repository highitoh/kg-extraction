from typing import Dict, Any
from langchain_core.runnables import RunnableSequence, Runnable

from class_extractor import ClassExtractor
from class_reviewer import ClassReviewer
from class_editor import ClassEditor
from class_consolidator import ClassConsolidator


class ClassChainWithReview(Runnable):
    """
    Class extraction chain with iterative review loop:
      ClassExtractor -> ClassReviewer -> (ClassLabelCorrector -> ClassReviewer)* -> ClassConsolidator

    ループロジック:
    - ClassReviewerでREVIEW判定されたクラスをClassLabelCorrectorで修正
    - 修正後、再度ClassReviewerで評価
    - すべてACCEPT/REJECTになるか、最大反復回数に達するまで繰り返す
    - 最大回数到達後もREVIEWが残る場合はACCEPTに変換
    """

    def __init__(self,
                 extractor: ClassExtractor,
                 class_reviewer: ClassReviewer,
                 corrector: ClassEditor,
                 consolidator: ClassConsolidator,
                 max_review_iterations: int = 3,
                 progress: bool = True):
        """
        Args:
            extractor: ClassExtractorインスタンス
            class_reviewer: ClassReviewerインスタンス
            corrector: ClassLabelCorrectorインスタンス
            consolidator: ClassConsolidatorインスタンス
            max_review_iterations: REVIEW判定の最大反復回数（デフォルト: 3）
            progress: 進捗表示フラグ
        """
        self.extractor = extractor
        self.class_reviewer = class_reviewer
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
          - existing_classes: (オプション) 既存の抽出済みクラス

        Returns
        -------
        output: ClassConsolidatorOutput
          - id: 抽出ID
          - classes: 統合されたクラス個体リスト
        """
        # 既存クラスを取得（あれば）
        existing_classes = input.get("existing_classes", [])

        # 1. ClassExtractor実行
        if self.progress:
            print("\n=== ClassExtractor ===")
        result = self.extractor.invoke(input)

        # 2. ClassReviewer実行（初回）
        if self.progress:
            print("\n=== ClassReviewer (Initial) ===")
        result = self.class_reviewer.invoke(result)

        # 3. ループ: ClassLabelCorrector -> ClassReviewer
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

            # ClassReviewer再実行
            if self.progress:
                print(f"\n--- ClassReviewer (iteration {iteration + 1}) ---")
            result = self.class_reviewer.invoke(result)

        # 4. 最大回数到達後もREVIEWが残っている場合はACCEPTに変換
        final_classes = []
        accepted_review_count = 0
        for c in result.get("classes", []):
            if c.get("judgment") == "REVIEW":
                c_copy = c.copy()
                c_copy["judgment"] = "ACCEPT"
                c_copy["justification"] = f"Maximum review iterations ({self.max_review_iterations}) reached. Auto-accepted. {c.get('justification', '')}"
                final_classes.append(c_copy)
                accepted_review_count += 1
            else:
                final_classes.append(c)

        if accepted_review_count > 0 and self.progress:
            print(f"\n=== Converting {accepted_review_count} remaining REVIEW classes to ACCEPT ===")

        result["classes"] = final_classes

        # 4.5. 既存クラスを統合対象に追加
        # ClassConsolidatorに渡す前に、新規抽出クラスと既存クラスをマージ
        if existing_classes:
            if self.progress:
                print(f"\n=== Merging {len(existing_classes)} existing classes with {len(final_classes)} new classes ===")
            result["classes"] = final_classes + existing_classes

        # 5. ClassConsolidator実行
        # ClassReviewerでREJECTは除外されるため、ACCEPTのみが渡される
        # ClassConsolidatorのLLMが新規クラスと既存クラスの重複を判定して統合する
        if self.progress:
            print("\n=== ClassConsolidator ===")
        result = self.consolidator.invoke(result)

        return result


def create_class_chain(model: str = None, max_review_iterations: int = 3, progress: bool = True) -> Runnable:
    """
    Create class extraction chain with review loop:
      ClassExtractor -> ClassReviewer -> (ClassLabelCorrector -> ClassReviewer)* -> ClassConsolidator

    Args:
        model: LLMモデル名（Noneの場合は各コンポーネントのデフォルトを使用）
        max_review_iterations: REVIEW判定の最大反復回数（デフォルト: 3）
        progress: 進捗表示フラグ（デフォルト: True）
    """
    if model is not None:
        extractor = ClassExtractor(model=model)
        class_reviewer = ClassReviewer(model=model)
        corrector = ClassEditor(model=model)
        consolidator = ClassConsolidator(model=model)
    else:
        extractor = ClassExtractor()
        class_reviewer = ClassReviewer()
        corrector = ClassEditor()
        consolidator = ClassConsolidator()

    return ClassChainWithReview(
        extractor=extractor,
        class_reviewer=class_reviewer,
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