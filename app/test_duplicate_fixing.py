"""Test script for duplicate fixing functionality in ClassEditor"""
import sys
import importlib
sys.path.insert(0, '/workspace/app')

# Import using importlib to avoid "class" keyword issue
class_editor_module = importlib.import_module('src.class.class_editor')
ClassEditor = class_editor_module.ClassEditor

def test_merge_duplicate_classes():
    """Test the merge logic without LLM calls"""

    editor = ClassEditor.__new__(ClassEditor)

    # Test data: 3 duplicate classes
    group_classes = [
        {"id": "cls_001", "label": "User", "sources": ["source1", "source2"], "file_ids": ["f1", "f2"]},
        {"id": "cls_002", "label": "System User", "sources": ["source3"], "file_ids": ["f1"]},
        {"id": "cls_003", "label": "Account User", "sources": ["source4", "source5"], "file_ids": ["f2", "f3"]},
    ]

    canonical_label = "User"
    merged_class_id = "cls_001"

    merged = editor._merge_duplicate_classes(group_classes, canonical_label, merged_class_id)

    print("\n=== Test: Merge Duplicate Classes ===\n")
    print(f"Input classes: {len(group_classes)}")
    print(f"Merged class ID: {merged['id']}")
    print(f"Merged label: {merged['label']}")
    print(f"Merged sources: {len(merged['sources'])}")
    print(f"Merged file_ids: {len(merged['file_ids'])}")
    print(f"Sources: {merged['sources']}")
    print(f"File IDs: {merged['file_ids']}")

    # Assertions
    assert merged['id'] == "cls_001", "Should use specified merged_class_id"
    assert merged['label'] == "User", "Should use canonical label"
    assert len(merged['sources']) == 5, "Should merge all sources"
    assert len(merged['file_ids']) == 5, "Should merge all file_ids"

    # Check sources and file_ids are paired correctly
    expected_pairs = [
        ("source1", "f1"), ("source2", "f2"), ("source3", "f1"),
        ("source4", "f2"), ("source5", "f3")
    ]
    actual_pairs = list(zip(merged['sources'], merged['file_ids']))

    # Note: dict.fromkeys() removes duplicates but may reorder
    # Just check that all expected pairs are present
    for pair in expected_pairs:
        assert pair in actual_pairs, f"Expected pair {pair} not found"

    print("\n✅ Merge duplicate classes test passed!")

def test_merge_fix_results():
    """Test merging of prohibit fixes and duplicate resolutions"""

    editor = ClassEditor.__new__(ClassEditor)

    # Original classes with duplicate_group_id
    classes = [
        {"id": "cls_001", "label": "User", "duplicate_group_id": "dup_group_0", "sources": ["s1"], "file_ids": ["f1"]},
        {"id": "cls_002", "label": "System User", "duplicate_group_id": "dup_group_0", "sources": ["s2"], "file_ids": ["f1"]},
        {"id": "cls_003", "label": "Admin", "sources": ["s3"], "file_ids": ["f1"]},  # No duplicate
    ]

    # Prohibit fixes (all passed as REVIEW)
    prohibit_fixed = [
        {"id": "cls_001", "label": "User", "judgment": "REVIEW", "justification": "OK"},
        {"id": "cls_002", "label": "System User", "judgment": "REVIEW", "justification": "OK"},
        {"id": "cls_003", "label": "Admin", "judgment": "REVIEW", "justification": "OK"},
    ]

    # Duplicate resolution: should_merge=true
    duplicate_resolved = {
        "dup_group_0": {
            "should_merge": True,
            "canonical_label": "User",
            "merged_class_id": "cls_001",
            "reason": "Same concept",
            "merged_class": {
                "id": "cls_001",
                "label": "User",
                "sources": ["s1", "s2"],
                "file_ids": ["f1", "f1"]
            }
        }
    }

    results = editor._merge_fix_results(classes, prohibit_fixed, duplicate_resolved)

    print("\n=== Test: Merge Fix Results ===\n")
    print(f"Input classes: {len(classes)}")
    print(f"Output results: {len(results)}")

    # Should have 2 classes: merged cls_001 + cls_003
    assert len(results) == 2, f"Expected 2 classes, got {len(results)}"

    # Check cls_001 (merged representative)
    cls_001_result = next((c for c in results if c.get("id") == "cls_001"), None)
    assert cls_001_result is not None, "cls_001 should be in results"
    assert cls_001_result["label"] == "User", "Should have canonical label"
    assert len(cls_001_result["sources"]) == 2, "Should have merged sources"

    # Check cls_002 NOT in results (merged away)
    cls_002_result = next((c for c in results if c.get("id") == "cls_002"), None)
    assert cls_002_result is None, "cls_002 should be merged away"

    # Check cls_003 (no duplicate)
    cls_003_result = next((c for c in results if c.get("id") == "cls_003"), None)
    assert cls_003_result is not None, "cls_003 should be in results"
    assert cls_003_result["label"] == "Admin", "Should keep original label"

    print(f"\nResults:")
    for r in results:
        print(f"  - {r['id']}: {r['label']} (sources: {len(r.get('sources', []))})")

    print("\n✅ Merge fix results test passed!")

def test_merge_fix_results_no_merge():
    """Test when should_merge=false"""

    editor = ClassEditor.__new__(ClassEditor)

    classes = [
        {"id": "cls_001", "label": "Admin", "duplicate_group_id": "dup_group_0", "sources": ["s1"], "file_ids": ["f1"]},
        {"id": "cls_002", "label": "Administrator", "duplicate_group_id": "dup_group_0", "sources": ["s2"], "file_ids": ["f1"]},
    ]

    prohibit_fixed = [
        {"id": "cls_001", "label": "Admin", "judgment": "REVIEW", "justification": "OK"},
        {"id": "cls_002", "label": "Administrator", "judgment": "REVIEW", "justification": "OK"},
    ]

    # should_merge=false - keep both
    duplicate_resolved = {
        "dup_group_0": {
            "should_merge": False,
            "reason": "Different roles"
        }
    }

    results = editor._merge_fix_results(classes, prohibit_fixed, duplicate_resolved)

    print("\n=== Test: Merge Fix Results (No Merge) ===\n")
    print(f"Input classes: {len(classes)}")
    print(f"Output results: {len(results)}")

    # Both should remain
    assert len(results) == 2, f"Expected 2 classes, got {len(results)}"

    print(f"\nResults:")
    for r in results:
        print(f"  - {r['id']}: {r['label']}")

    print("\n✅ No merge test passed!")

if __name__ == "__main__":
    test_merge_duplicate_classes()
    test_merge_fix_results()
    test_merge_fix_results_no_merge()
    print("\n🎉 All tests passed successfully!")
