from typing import Dict, Any
from langchain_core.runnables import Runnable, RunnableSequence

from .property_extractor import PropertyExtractor
from .property_consolidator import PropertyConsolidator
from .property_ranker import PropertyRanker
from .property_reviewer import PropertyReviewer
from .property_editor import PropertyEditor
from .property_validator import PropertyValidator


class ExtractorToRankerAdapter(Runnable):
    """
    Adapter that combines PropertyExtractor output with the original input
    to convert into PropertyRanker input format
    """

    def __init__(self, original_input: Dict[str, Any]):
        self.original_input = original_input

    def invoke(self, input: Dict[str, Any], config=None) -> Dict[str, Any]:
        """
        Input: PropertyExtractor output {properties}
        Output: PropertyRanker input {property_candidates: {...}, class_info: {...}}
        """
        return {
            "property_candidates": {
                "properties": input.get("properties", [])
            },
            "class_info": self.original_input.get("class_info", {})
        }


class RankerToFilterAdapter(Runnable):
    """
    Adapter that combines PropertyRanker output with the original input
    to convert into PropertyReviewer input format
    """

    def __init__(self, original_input: Dict[str, Any]):
        self.original_input = original_input

    def invoke(self, input: Dict[str, Any], config=None) -> Dict[str, Any]:
        """
        Input: PropertyRanker output {properties}
        Output: PropertyReviewer input {property_candidates: {...}, class_info: {...}}
        """
        return {
            "property_candidates": {
                "properties": input.get("properties", [])
            },
            "class_info": self.original_input.get("class_info", {})
        }


class FilterToEditorAdapter(Runnable):
    """
    Adapter that combines PropertyReviewer output with the original input
    to convert into PropertyEditor input format
    """

    def __init__(self, original_input: Dict[str, Any]):
        self.original_input = original_input

    def invoke(self, input: Dict[str, Any], config=None) -> Dict[str, Any]:
        """
        Input: PropertyReviewer output {properties}
        Output: PropertyEditor input {properties, class_info}
        """
        return {
            "properties": input.get("properties", []),
            "class_info": self.original_input.get("class_info", {})
        }


class EditorToFilterAdapter(Runnable):
    """
    Adapter that converts PropertyEditor output to PropertyReviewer input format
    """

    def __init__(self, original_input: Dict[str, Any]):
        self.original_input = original_input

    def invoke(self, input: Dict[str, Any], config=None) -> Dict[str, Any]:
        """
        Input: PropertyEditor output {properties}
        Output: PropertyReviewer input {property_candidates: {...}, class_info: {...}}
        """
        return {
            "property_candidates": {
                "properties": input.get("properties", [])
            },
            "class_info": self.original_input.get("class_info", {})
        }


class PropertyChainWithReview(Runnable):
    """
    Property extraction chain with iterative review loop:
      PropertyExtractor -> PropertyConsolidator -> PropertyRanker -> PropertyReviewer -> (PropertyEditor -> PropertyReviewer)* -> PropertyValidator

    Loop logic:
    - Properties judged as REVIEW by PropertyReviewer are corrected by PropertyEditor
    - After correction, they are re-evaluated by PropertyReviewer
    - Repeats until all properties become ACCEPT/REJECT or until max iterations is reached
    - If REVIEW properties remain after reaching max iterations, they are converted to ACCEPT
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
                 max_review_iterations: int = 2,
                 progress: bool = True):
        """
        Args:
            extractor: PropertyExtractor instance
            consolidator: PropertyConsolidator instance
            ranker: PropertyRanker instance
            property_reviewer: PropertyReviewer instance
            editor: PropertyEditor instance
            validator: PropertyValidator instance
            original_input: Original input (for adapters)
            existing_properties: Existing property list (optional)
            max_review_iterations: Maximum number of REVIEW iterations (default: 2)
            progress: Progress display flag
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

    @staticmethod
    def _is_frozen_unchanged(prop: Dict[str, Any]) -> bool:
        """Return True for carry-over properties with unchanged ranked strength."""
        if not prop.get("_frozen_from_existing_accept"):
            return False
        frozen_strength = prop.get("_frozen_strength")
        return frozen_strength is not None and prop.get("strength") == frozen_strength

    @staticmethod
    def _merge_reviewed_properties(
        all_properties: list,
        reviewed_properties: list,
        should_replace,
    ) -> list:
        """Merge reviewer outputs back into original order.

        Items matched by should_replace are replaced from reviewed_properties.
        If no match is found, the item is treated as rejected and dropped.
        """
        reviewed_by_id = {
            p.get("id"): p
            for p in reviewed_properties
            if p.get("id")
        }
        reviewed_by_key = {}
        for p in reviewed_properties:
            if p.get("id"):
                continue
            key = (p.get("src_id"), p.get("property_iri"), p.get("dest_id"))
            reviewed_by_key.setdefault(key, []).append(p)

        merged = []
        for p in all_properties:
            if not should_replace(p):
                merged.append(p)
                continue

            pid = p.get("id")
            if pid and pid in reviewed_by_id:
                merged.append(reviewed_by_id.pop(pid))
                continue

            key = (p.get("src_id"), p.get("property_iri"), p.get("dest_id"))
            bucket = reviewed_by_key.get(key, [])
            if bucket:
                merged.append(bucket.pop(0))
            # else: rejected -> drop

        return merged

    @staticmethod
    def _clear_internal_flags(properties: list) -> list:
        """Remove internal control flags from output properties."""
        for p in properties:
            p.pop("_frozen_from_existing_accept", None)
            p.pop("_frozen_strength", None)
        return properties

    def invoke(self, input: Dict[str, Any], config=None) -> Dict[str, Any]:
        """
        Main execution of property extraction chain

        Parameters
        ----------
        input: PropertyChainInput
          - class_info: Class information
          - metamodel: Metamodel information

        Returns
        -------
        output: PropertyValidatorOutput
          - properties: Validated property list
        """
        # 1. Execute PropertyExtractor
        if self.progress:
            print("\n=== PropertyExtractor ===")
        result = self.extractor.invoke(input)

        # 2. Execute PropertyConsolidator (merge with existing properties)
        if self.progress:
            print("\n=== PropertyConsolidator ===")
        consolidator_input = {
            "properties": result.get("properties", []),
            "existing_properties": self.existing_properties,
            "class_info": self.original_input.get("class_info", {})
        }
        result = self.consolidator.invoke(consolidator_input)

        # 3. Execute PropertyRanker
        if self.progress:
            print("\n=== PropertyRanker ===")
        adapter1 = ExtractorToRankerAdapter(self.original_input)
        ranker_input = adapter1.invoke(result)
        result = self.ranker.invoke(ranker_input)

        # 4. Execute PropertyReviewer (initial)
        if self.progress:
            print("\n=== PropertyReviewer (Initial) ===")
        adapter2 = RankerToFilterAdapter(self.original_input)
        all_properties = result.get("properties", [])
        frozen_properties = [p for p in all_properties if self._is_frozen_unchanged(p)]
        review_targets = [p for p in all_properties if not self._is_frozen_unchanged(p)]

        if self.progress:
            if frozen_properties:
                print(f"Skipping {len(frozen_properties)} frozen carry-over properties")
            print(f"Review targets: {len(review_targets)}")

        reviewed_properties = []
        if review_targets:
            filter_input = adapter2.invoke({"properties": review_targets})
            reviewed_result = self.property_reviewer.invoke(filter_input)
            reviewed_properties = reviewed_result.get("properties", [])

        # Preserve original order. For frozen unchanged properties, set ACCEPT directly.
        merged_initial = self._merge_reviewed_properties(
            all_properties,
            reviewed_properties,
            should_replace=lambda p: not self._is_frozen_unchanged(p),
        )
        result["properties"] = [
            {
                **p,
                "judgment": "ACCEPT",
                "justification": p.get("justification", ""),
            } if self._is_frozen_unchanged(p) else p
            for p in merged_initial
        ]

        # 5. Loop: PropertyEditor -> PropertyReviewer
        accepted_properties = []  # Accumulate ACCEPT properties across iterations

        for iteration in range(self.max_review_iterations):
            # Check if REVIEW properties remain
            review_count = sum(1 for p in result.get("properties", []) if p.get("judgment") == "REVIEW")

            if review_count == 0:
                # All ACCEPT/REJECT -> loop complete
                if self.progress:
                    print(f"\n=== Review Loop Complete (iteration {iteration + 1}) ===")
                    print("No REVIEW properties remaining")
                break

            if self.progress:
                print(f"\n=== Review Iteration {iteration + 1}/{self.max_review_iterations} ===")
                print(f"REVIEW properties remaining: {review_count}")

            # Extract and accumulate ACCEPT properties before editing
            current_accept = [p for p in result.get("properties", []) if p.get("judgment") == "ACCEPT"]
            accepted_properties.extend(current_accept)

            # Execute PropertyEditor
            if self.progress:
                print(f"\n--- PropertyEditor (iteration {iteration + 1}) ---")
            result = self.editor.invoke(result)

            # Extract only REVIEW properties for re-review
            review_only = [p for p in result.get("properties", []) if p.get("judgment") == "REVIEW"]

            if not review_only:
                # All became ACCEPT or REJECT after editing
                new_accept = [p for p in result.get("properties", []) if p.get("judgment") == "ACCEPT"]
                accepted_properties.extend(new_accept)
                result["properties"] = []
                if self.progress:
                    print("No REVIEW properties remaining after editing")
                break

            # Re-execute PropertyReviewer
            if self.progress:
                print(f"\n--- PropertyReviewer (iteration {iteration + 1}) ---")
                print(f"Re-reviewing {len(review_only)} REVIEW properties (skipping {len(accepted_properties)} already accepted)")
            adapter_loop = EditorToFilterAdapter(self.original_input)
            filter_input = adapter_loop.invoke({"properties": review_only})
            result = self.property_reviewer.invoke(filter_input)

        # Merge accumulated ACCEPT properties with final result
        result["properties"] = accepted_properties + result.get("properties", [])

        # 6. If REVIEW properties remain after reaching max iterations, convert to ACCEPT
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

        result["properties"] = self._clear_internal_flags(final_properties)

        # 7. Execute PropertyValidator
        if self.progress:
            print("\n=== PropertyValidator ===")
        validator_input = {
            "property_candidates": {
                "properties": result.get("properties", [])
            },
            "class_info": self.original_input.get("class_info", {}),
            "metamodel": self.original_input.get("metamodel", {})
        }
        result = self.validator.invoke(validator_input)

        return result


class PropertyChainWithAdapter(Runnable):
    """
    Wrapper that connects PropertyExtractor -> Adapter -> Ranker -> Adapter -> PropertyReviewer -> PropertyValidator
    using RunnableSequence
    """

    def __init__(self, model: str = None):
        self.model = model

    def invoke(self, input: Dict[str, Any], config=None) -> Dict[str, Any]:
        # Create components
        if self.model is not None:
            extractor = PropertyExtractor(model=self.model)
            ranker = PropertyRanker(model=self.model)
            property_reviewer = PropertyReviewer(model=self.model)
        else:
            extractor = PropertyExtractor()
            ranker = PropertyRanker()
            property_reviewer = PropertyReviewer()
        validator = PropertyValidator()

        # Create adapters (preserve original input)
        adapter1 = ExtractorToRankerAdapter(input)
        adapter2 = RankerToFilterAdapter(input)

        # Connect with RunnableSequence
        # extractor -> adapter1 -> ranker -> adapter2 -> property_reviewer -> validator
        chain = RunnableSequence(extractor, adapter1, ranker, adapter2, property_reviewer, validator)

        return chain.invoke(input, config)


def create_property_chain(model: str = None, max_review_iterations: int = 2, enable_review: bool = True, existing_properties: list = None) -> Runnable:
    """
    Property extraction chain:
      - enable_review=True: PropertyExtractor -> PropertyConsolidator -> PropertyRanker -> PropertyReviewer -> (PropertyEditor -> PropertyReviewer)* -> PropertyValidator
      - enable_review=False: PropertyExtractor -> PropertyConsolidator -> PropertyRanker -> PropertyReviewer -> PropertyValidator

    Args:
        model: LLM model name (if None, use default for each component)
        max_review_iterations: Maximum number of REVIEW iterations (default: 2)
        enable_review: Enable review loop (default: True)
        existing_properties: Existing property list (for consolidation, default: None)
    """
    if not enable_review:
        return PropertyChainWithAdapter(model=model)

    # When using PropertyChainWithReview, return a wrapper since input is needed at invoke time
    class PropertyChainWithReviewWrapper(Runnable):
        def __init__(self, model: str = None, max_review_iterations: int = 2, existing_properties: list = None):
            self.model = model
            self.max_review_iterations = max_review_iterations
            self.existing_properties = existing_properties or []

        def invoke(self, input: Dict[str, Any], config=None) -> Dict[str, Any]:
            # Create components
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

            # Create PropertyChainWithReview
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
            )

            return chain.invoke(input, config)

    return PropertyChainWithReviewWrapper(model=model, max_review_iterations=max_review_iterations, existing_properties=existing_properties)


if __name__ == "__main__":
    import json

    chain = create_property_chain()
    sample_input = {
        "class_info": {
            "classes": [
                {
                    "id": "class-001",
                    "class_iri": "ex:BusinessConceptClass",
                    "label": "Customer Management System",
                    "file_id": "file-abc",
                },
                {
                    "id": "class-002",
                    "class_iri": "ex:SystemComponentClass",
                    "label": "Database",
                    "file_id": "file-abc",
                }
            ]
        },
        "metamodel": {
            # Metamodel information is stored here
        },
    }

    result = chain.invoke(sample_input)
    print(json.dumps(result, ensure_ascii=False, indent=2))
