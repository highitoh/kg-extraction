from typing import Dict, Any, List, Tuple

from langchain_core.runnables import Runnable

from src.utils.logger import Logger


class ViewFilter(Runnable):
    """
    Task to apply filter processing to view description extraction results
    Input: ViewChainOutput
    Output: ViewChainOutput (same schema)

    Apply DuplicateSentenceRemovalRule to remove duplicate sentences
    """

    def __init__(self, progress: bool = True, log_dir: str = "/workspace/app/log/view_filter"):
        self.progress = progress
        # Initialize Logger
        self.logger = Logger(log_dir)

    def _remove_duplicate_sentences(self, views: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Remove duplicate sentences across views
        - Determine as duplicate if exact match or inclusion relationship exists
        - For exact match, keep the earlier sentence; for inclusion, keep the longer one
        """
        if not views:
            return views

        # Group texts by view
        view_text_map: Dict[str, List[Tuple[int, int, Dict[str, Any]]]] = {}

        for view_idx, view in enumerate(views):
            view_type = view.get("type", "unknown")
            texts = view.get("texts", [])

            if view_type not in view_text_map:
                view_text_map[view_type] = []

            for text_idx, text_obj in enumerate(texts):
                view_text_map[view_type].append((view_idx, text_idx, text_obj))

        # Detect and remove duplicates within each view
        to_remove: List[Tuple[int, int]] = []  # (view_index, text_index)

        for view_type, text_list in view_text_map.items():
            if len(text_list) < 2:
                continue

            i = 0
            while i < len(text_list):
                view_idx1, text_idx1, text_obj1 = text_list[i]
                text1 = text_obj1.get("text", "")

                j = i + 1
                while j < len(text_list):
                    view_idx2, text_idx2, text_obj2 = text_list[j]
                    text2 = text_obj2.get("text", "")

                    # Duplicate detection: exact match or inclusion relationship
                    is_duplicate = False
                    text_to_remove = None

                    if text1 == text2:
                        # For exact match, remove the later sentence
                        is_duplicate = True
                        text_to_remove = (view_idx2, text_idx2)
                    elif text1 in text2:
                        # If text1 is included in text2, remove the shorter one (text1)
                        is_duplicate = True
                        text_to_remove = (view_idx1, text_idx1)
                    elif text2 in text1:
                        # If text2 is included in text1, remove the shorter one (text2)
                        is_duplicate = True
                        text_to_remove = (view_idx2, text_idx2)

                    if is_duplicate and text_to_remove:
                        to_remove.append(text_to_remove)
                        if self.progress:
                            print(f"Removing duplicate text from {view_type}: {text_to_remove[0]}.{text_to_remove[1]}")

                        # Remove the target record from the list
                        if text_to_remove == (view_idx1, text_idx1):
                            text_list.pop(i)
                            j = i  # Reset j since i's position changed
                            break
                        else:
                            text_list.pop(j)
                            continue

                    j += 1
                i += 1

        # Sort to_remove and delete from back (prevent index shift)
        to_remove = sorted(set(to_remove), key=lambda x: (x[0], x[1]), reverse=True)

        # Create new views list (with duplicate texts removed)
        filtered_views = []
        for view_idx, view in enumerate(views):
            texts = view.get("texts", [])
            filtered_texts = []

            for text_idx, text_obj in enumerate(texts):
                if (view_idx, text_idx) not in to_remove:
                    filtered_texts.append(text_obj)

            # Add view only if texts exist
            if filtered_texts:
                filtered_view = view.copy()
                filtered_view["texts"] = filtered_texts
                filtered_views.append(filtered_view)

        return filtered_views

    def invoke(self, input: Dict[str, Any], config=None) -> Dict[str, Any]:
        """Filter ViewChainOutput by removing duplicate sentences"""
        views = input.get("views", [])

        # Remove duplicate sentences
        filtered_views = self._remove_duplicate_sentences(views)

        # Return result
        output = input.copy()
        output["views"] = filtered_views

        if self.progress:
            original_count = sum(len(v.get("texts", [])) for v in views)
            filtered_count = sum(len(v.get("texts", [])) for v in filtered_views)
            print(
                f"ViewFilter: {original_count} -> {filtered_count} texts (removed {original_count - filtered_count} duplicates)"
            )

        # Log ViewChainOutput using Logger
        self.logger.save_log(output, filename_prefix="view_filter_output_")

        return output


if __name__ == "__main__":
    # Sample data for testing (includes duplicates)
    sample_input = {
        "views": [
            {
                "type": "business_concept",
                "texts": [
                    {"file_id": "test_file", "text": "Customer Management System"},
                    {
                        "file_id": "test_file",
                        "text": "Customer Management System Development",  # Inclusion relationship (longer)
                    },
                    {
                        "file_id": "test_file",
                        "text": "Customer Management System",  # Exact match (duplicate)
                    },
                ],
            },
            {
                "type": "system_requirement",
                "texts": [
                    {"file_id": "test_file", "text": "Database Integration Feature"},
                    {
                        "file_id": "test_file",
                        "text": "Database Integration",  # Inclusion relationship (shorter)
                    },
                    {"file_id": "test_file", "text": "API Design Requirements"},
                ],
            },
            {"type": "data_analysis", "texts": [{"file_id": "test_file", "text": "Sales Data Analysis"}]},
        ]
    }

    # Create ViewFilter instance
    print("Starting ViewFilter test...")
    filter_instance = ViewFilter(progress=True)

    print("\n=== Original Data ===")
    for view in sample_input["views"]:
        print(f"View Type: {view['type']}")
        for i, text in enumerate(view["texts"]):
            print(f"  [{i + 1}] {text['text']}")

    try:
        # Run test
        result = filter_instance.invoke(sample_input)

        print("\n=== Filtering Results ===")
        print(f"Views After Filtering: {len(result['views'])}")

        for view in result["views"]:
            print(f"\nView Type: {view['type']}")
            print(f"Text Count: {len(view['texts'])}")
            for i, text in enumerate(view["texts"]):
                print(f"  [{i + 1}] {text['text']}")

        # Verify duplicate removal
        original_text_count = sum(len(v.get("texts", [])) for v in sample_input["views"])
        filtered_text_count = sum(len(v.get("texts", [])) for v in result["views"])

        print("\n=== Statistics ===")
        print(f"Original Text Count: {original_text_count}")
        print(f"Filtered Text Count: {filtered_text_count}")
        print(f"Removed Text Count: {original_text_count - filtered_text_count}")

        print("\n✅ ViewFilter test completed")

    except Exception as e:
        print(f"\n❌ Test execution error: {e}")
        import traceback

        traceback.print_exc()
