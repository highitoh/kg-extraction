from typing import Dict, Any
from langchain_core.runnables import RunnableSequence, Runnable

from .class_extractor import ClassExtractor
from .class_reviewer import ClassReviewer
from .class_editor import ClassEditor
from .class_consolidator import ClassConsolidator


class ClassChainWithReview(Runnable):
    """
    Class extraction chain with iterative review loop:
      ClassExtractor -> ClassConsolidator -> ClassReviewer -> (ClassLabelCorrector -> ClassReviewer)*

    Loop logic:
    - ClassConsolidator first consolidates similar labels
    - ClassEditor corrects classes judged as REVIEW by ClassReviewer
    - After correction, ClassReviewer re-evaluates
    - Repeats until all become ACCEPT/REJECT or maximum iterations reached
    - If REVIEW remains after maximum iterations, converts to ACCEPT
    """

    def __init__(self,
                 extractor: ClassExtractor,
                 class_reviewer: ClassReviewer,
                 corrector: ClassEditor,
                 consolidator: ClassConsolidator,
                 max_review_iterations: int = 2,
                 progress: bool = True):
        """
        Args:
            extractor: ClassExtractor instance
            class_reviewer: ClassReviewer instance
            corrector: ClassLabelCorrector instance
            consolidator: ClassConsolidator instance
            max_review_iterations: Maximum iterations for REVIEW judgments (default: 2)
            progress: Progress display flag
        """
        self.extractor = extractor
        self.class_reviewer = class_reviewer
        self.corrector = corrector
        self.consolidator = consolidator
        self.max_review_iterations = max_review_iterations
        self.progress = progress

    def invoke(self, input: Dict[str, Any], config=None) -> Dict[str, Any]:
        """
        Main execution of class extraction chain

        Parameters
        ----------
        input: ViewChainOutput
          - view_info: View extraction results
          - metamodel: Metamodel information
          - existing_classes: (optional) Previously extracted classes

        Returns
        -------
        output: ClassConsolidatorOutput
          - id: Extraction ID
          - classes: Consolidated list of class instances
        """
        # Get existing classes if available
        existing_classes = input.get("existing_classes", [])

        # 1. Execute ClassExtractor
        if self.progress:
            print("\n=== ClassExtractor ===")
        result = self.extractor.invoke(input)

        # 2. Add existing classes to consolidation targets
        # Merge newly extracted classes with existing classes before passing to ClassConsolidator
        if existing_classes:
            if self.progress:
                print(f"\n=== Merging {len(existing_classes)} existing classes with {len(result.get('classes', []))} new classes ===")
            result["classes"] = result.get("classes", []) + existing_classes

        # 3. Execute ClassConsolidator
        # First execute label consolidation of extracted classes
        if self.progress:
            print("\n=== ClassConsolidator ===")
        result = self.consolidator.invoke(result)

        # 4. Execute ClassReviewer (initial)
        if self.progress:
            print("\n=== ClassReviewer (Initial) ===")
        result = self.class_reviewer.invoke(result)

        # 5. Loop: ClassLabelCorrector -> ClassReviewer
        for iteration in range(self.max_review_iterations):
            # Check if REVIEW classes remain
            review_count = sum(1 for c in result.get("classes", []) if c.get("judgment") == "REVIEW")

            if review_count == 0:
                # All ACCEPT/REJECT -> End loop
                if self.progress:
                    print(f"\n=== Review Loop Complete (iteration {iteration + 1}) ===")
                    print("No REVIEW classes remaining")
                break

            if self.progress:
                print(f"\n=== Review Iteration {iteration + 1}/{self.max_review_iterations} ===")
                print(f"REVIEW classes remaining: {review_count}")

            # Execute ClassLabelCorrector
            if self.progress:
                print(f"\n--- ClassLabelCorrector (iteration {iteration + 1}) ---")
            result = self.corrector.invoke(result)

            # Re-execute ClassReviewer
            if self.progress:
                print(f"\n--- ClassReviewer (iteration {iteration + 1}) ---")
            result = self.class_reviewer.invoke(result)

        # 6. Convert remaining REVIEW to ACCEPT after reaching maximum iterations
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

        return result


def create_class_chain(model: str = None, max_review_iterations: int = 2, progress: bool = True) -> Runnable:
    """
    Create class extraction chain with review loop:
      ClassExtractor -> ClassConsolidator -> ClassReviewer -> (ClassLabelCorrector -> ClassReviewer)*

    Args:
        model: LLM model name (if None, use default for each component)
        max_review_iterations: Maximum iterations for REVIEW judgments (default: 2)
        progress: Progress display flag (default: True)
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