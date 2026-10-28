from typing import Any, Dict

from langchain_core.runnables import Runnable

from .class_consolidator import ClassConsolidator
from .class_editor import ClassEditor
from .class_extractor import ClassExtractor
from .class_reviewer import ClassReviewer


class ClassChainWithReview(Runnable):
    """
    Class extraction chain with iterative review loop:
      ClassExtractor -> ClassConsolidator -> ClassReviewer -> (ClassLabelCorrector -> ClassReviewer)*

    Loop logic:
    - ClassConsolidator first consolidates similar labels within each class, then resolves
      same-label classes extracted as different classes into one class
    - ClassEditor corrects classes judged as REVIEW by ClassReviewer
    - After correction, ClassReviewer re-evaluates
    - Repeats until all become ACCEPT/REJECT or maximum iterations reached
    - If REVIEW remains after maximum iterations, converts to ACCEPT
    - REJECT classes are excluded from the output
    """

    def __init__(
        self,
        extractor: ClassExtractor,
        class_reviewer: ClassReviewer,
        corrector: ClassEditor,
        consolidator: ClassConsolidator,
        max_review_iterations: int = 2,
        progress: bool = True,
    ):
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

    @staticmethod
    def _is_frozen_unchanged(cls: Dict[str, Any]) -> bool:
        """Return True for carry-over ACCEPT classes with unchanged label and class."""
        return (
            bool(cls.get("_frozen_from_existing_accept"))
            and cls.get("label") == cls.get("_frozen_label")
            and cls.get("class_iri") == cls.get("_frozen_class_iri")
        )

    @staticmethod
    def _merge_reviewed_classes(
        all_classes: list,
        reviewed_classes: list,
        should_replace,
    ) -> list:
        """Merge reviewer outputs back into the original class order.

        Classes matched by should_replace are replaced from reviewed_classes.
        If no reviewed match is found, the class is treated as rejected and dropped.
        """
        reviewed_by_id = {c.get("id"): c for c in reviewed_classes if c.get("id")}
        reviewed_by_key = {}
        for c in reviewed_classes:
            if c.get("id"):
                continue
            key = (c.get("class_iri"), c.get("label"))
            reviewed_by_key.setdefault(key, []).append(c)

        merged_classes = []
        for c in all_classes:
            if not should_replace(c):
                merged_classes.append(c)
                continue

            cid = c.get("id")
            if cid and cid in reviewed_by_id:
                merged_classes.append(reviewed_by_id.pop(cid))
                continue

            key = (c.get("class_iri"), c.get("label"))
            bucket = reviewed_by_key.get(key, [])
            if bucket:
                merged_classes.append(bucket.pop(0))
            # else: rejected -> drop

        return merged_classes

    @staticmethod
    def _clear_internal_flags(classes: list) -> list:
        """Remove internal control flags from output classes."""
        for c in classes:
            c.pop("_frozen_from_existing_accept", None)
            c.pop("_frozen_label", None)
            c.pop("_frozen_class_iri", None)
        return classes

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
          - classes: Consolidated list of class instances
        """
        # Get existing classes if available
        existing_classes = input.get("existing_classes", [])

        # 1. Execute ClassExtractor
        if self.progress:
            print("\n=== ClassExtractor ===")
        result = self.extractor.invoke(input)

        # 2. Consolidate new classes among themselves
        if self.progress:
            print("\n=== ClassConsolidator: New Classes ===")
            print(f"Consolidating {len(result.get('classes', []))} newly extracted classes")
        result = self.consolidator.invoke(result)

        # 3. Match consolidated new against existing (if any)
        if existing_classes:
            if self.progress:
                print("\n=== ClassConsolidator: Match Against Existing ===")
                print(
                    f"Matching {len(result.get('classes', []))} consolidated new classes against "
                    f"{len(existing_classes)} existing classes"
                )

            result = self.consolidator.match_against_existing(
                new_classes=result.get("classes", []), frozen_classes=existing_classes
            )
        # else: no existing classes, the consolidated result is final

        # 3b. Resolve same-label classes extracted as different classes
        if self.progress:
            print("\n=== ClassConsolidator: Cross-Class ===")
        result = self.consolidator.resolve_cross_class(result.get("classes", []))

        # 4. Execute ClassReviewer (initial)
        classes_before_review = result.get("classes", [])
        frozen_classes = [c for c in classes_before_review if self._is_frozen_unchanged(c)]
        review_targets = [c for c in classes_before_review if not self._is_frozen_unchanged(c)]

        if self.progress:
            print("\n=== ClassReviewer (Initial) ===")
            if frozen_classes:
                print(f"Skipping {len(frozen_classes)} frozen carry-over ACCEPT classes")
            print(f"Review targets: {len(review_targets)}")

        reviewed_classes = []
        if review_targets:
            review_input = result.copy()
            review_input["classes"] = review_targets
            reviewed_result = self.class_reviewer.invoke(review_input)
            reviewed_classes = reviewed_result.get("classes", [])

        result["classes"] = self._merge_reviewed_classes(
            classes_before_review,
            reviewed_classes,
            should_replace=lambda c: not self._is_frozen_unchanged(c),
        )

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

            # Re-execute ClassReviewer only for classes still marked REVIEW.
            # ACCEPT/REJECT classes are kept as-is to avoid repeated review cost.
            all_classes = result.get("classes", [])
            review_classes = [c for c in all_classes if c.get("judgment") == "REVIEW"]
            non_review_classes = [c for c in all_classes if c.get("judgment") != "REVIEW"]

            if not review_classes:
                if self.progress:
                    print(f"\n--- ClassReviewer (iteration {iteration + 1}) skipped: no REVIEW classes ---")
                result["classes"] = non_review_classes
                break

            if self.progress:
                print(f"\n--- ClassReviewer (iteration {iteration + 1}, REVIEW-only) ---")

            review_input = result.copy()
            review_input["classes"] = review_classes
            reviewed_result = self.class_reviewer.invoke(review_input)
            reviewed_classes = reviewed_result.get("classes", [])

            result["classes"] = self._merge_reviewed_classes(
                all_classes,
                reviewed_classes,
                should_replace=lambda c: c.get("judgment") == "REVIEW",
            )

        # 6. Convert remaining REVIEW to ACCEPT after reaching maximum iterations
        final_classes = []
        accepted_review_count = 0
        for c in result.get("classes", []):
            if c.get("judgment") == "REVIEW":
                c_copy = c.copy()
                c_copy["judgment"] = "ACCEPT"
                c_copy["justification"] = (
                    f"Maximum review iterations ({self.max_review_iterations}) reached. Auto-accepted. "
                    f"{c.get('justification', '')}"
                )
                final_classes.append(c_copy)
                accepted_review_count += 1
            else:
                final_classes.append(c)

        if accepted_review_count > 0 and self.progress:
            print(f"\n=== Converting {accepted_review_count} remaining REVIEW classes to ACCEPT ===")

        # 7. Drop REJECT classes so the chain output carries accepted classes only.
        accepted_classes = [c for c in final_classes if c.get("judgment") != "REJECT"]

        result["classes"] = self._clear_internal_flags(accepted_classes)

        return result


def create_class_chain(
    model: str = None, max_review_iterations: int = 2, progress: bool = True, metamodel_path: str = None
) -> Runnable:
    """
    Create class extraction chain with review loop:
      ClassExtractor -> ClassConsolidator -> ClassReviewer -> (ClassLabelCorrector -> ClassReviewer)*

    Args:
        model: LLM model name (if None, use default for each component)
        max_review_iterations: Maximum iterations for REVIEW judgments (default: 2)
        progress: Progress display flag (default: True)
        metamodel_path: Metamodel file path override (default: metamodel/metamodel.json)
    """
    if model is not None:
        extractor = ClassExtractor(model=model)
        class_reviewer = ClassReviewer(model=model, metamodel_path=metamodel_path)
        corrector = ClassEditor(model=model, metamodel_path=metamodel_path)
        consolidator = ClassConsolidator(model=model, metamodel_path=metamodel_path)
    else:
        extractor = ClassExtractor()
        class_reviewer = ClassReviewer(metamodel_path=metamodel_path)
        corrector = ClassEditor(metamodel_path=metamodel_path)
        consolidator = ClassConsolidator(metamodel_path=metamodel_path)

    return ClassChainWithReview(
        extractor=extractor,
        class_reviewer=class_reviewer,
        corrector=corrector,
        consolidator=consolidator,
        max_review_iterations=max_review_iterations,
        progress=progress,
    )


if __name__ == "__main__":
    import json

    chain = create_class_chain()
    sample_input = {
        "view_info": {
            "views": [
                {
                    "type": "service",
                    "texts": [
                        {
                            "file_id": "file-abc",
                            "line": 1,
                            "text": "The customer management system handles user registration and authentication",
                        }
                    ],
                },
                {
                    "type": "resource",
                    "texts": [
                        {"file_id": "file-abc", "line": 2, "text": "Implementation of database connection module"}
                    ],
                },
            ]
        },
        "metamodel": {},
    }

    result = chain.invoke(sample_input)
    print(json.dumps(result, ensure_ascii=False, indent=2))
