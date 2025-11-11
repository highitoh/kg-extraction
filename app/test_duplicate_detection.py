"""Test script for duplicate detection functionality in ClassReviewer"""
import sys
sys.path.insert(0, '/workspace/app')

from src.class.class_reviewer import ClassReviewer

def test_merge_review_results():
    """Test the merge logic without LLM calls"""

    # Create a ClassReviewer instance (no LLM needed for this test)
    reviewer = ClassReviewer.__new__(ClassReviewer)

    # Test data: 5 classes with 2 duplicate groups
    classes = [
        {"id": "cls_001", "class_iri": "http://example.com/User", "label": "User", "sources": ["p1"], "file_ids": ["f1"]},
        {"id": "cls_002", "class_iri": "http://example.com/User", "label": "System User", "sources": ["p2"], "file_ids": ["f1"]},
        {"id": "cls_003", "class_iri": "http://example.com/User", "label": "Account User", "sources": ["p3"], "file_ids": ["f1"]},
        {"id": "cls_004", "class_iri": "http://example.com/User", "label": "Administrator", "sources": ["p4"], "file_ids": ["f1"]},
        {"id": "cls_005", "class_iri": "http://example.com/User", "label": "Admin", "sources": ["p5"], "file_ids": ["f1"]},
    ]

    # Prohibit rules judgments: cls_001 is REVIEW, others are ACCEPT
    prohibit_judgments = [
        {"judgment": "REVIEW", "justification": "Violates prohibit rule X"},
        {"judgment": "ACCEPT", "justification": "No violations"},
        {"judgment": "ACCEPT", "justification": "No violations"},
        {"judgment": "ACCEPT", "justification": "No violations"},
        {"judgment": "ACCEPT", "justification": "No violations"},
    ]

    # Duplicate groups: [cls_001, cls_002, cls_003] and [cls_004, cls_005]
    duplicate_groups = {
        "dup_group_0": {
            "class_ids": ["cls_001", "cls_002", "cls_003"],
            "labels": ["User", "System User", "Account User"],
            "reason": "All refer to the same user entity concept"
        },
        "dup_group_1": {
            "class_ids": ["cls_004", "cls_005"],
            "labels": ["Administrator", "Admin"],
            "reason": "Admin is abbreviation of Administrator"
        }
    }

    # Test the merge
    results = reviewer._merge_review_results(classes, prohibit_judgments, duplicate_groups)

    print("\n=== Test Results ===\n")
    print(f"Total classes: {len(classes)}")
    print(f"Results after merge: {len(results)}")

    for r in results:
        print(f"\n[{r['id']}] {r['label']}")
        print(f"  Judgment: {r['judgment']}")
        print(f"  Justification: {r['justification']}")
        if 'duplicate_group_id' in r:
            print(f"  Duplicate Group: {r['duplicate_group_id']}")
            print(f"  Duplicate Labels: {r['duplicate_labels']}")

    # Assertions
    assert len(results) == 5, "All classes should pass (no REJECTs)"

    # cls_001: REVIEW from prohibit rules (takes priority over duplicate)
    assert results[0]["id"] == "cls_001"
    assert results[0]["judgment"] == "REVIEW"
    assert "prohibit rule" in results[0]["justification"].lower()
    assert results[0].get("duplicate_group_id") == "dup_group_0"
    assert set(results[0].get("duplicate_labels", [])) == {"System User", "Account User"}

    # cls_002: REVIEW from duplicate detection
    assert results[1]["id"] == "cls_002"
    assert results[1]["judgment"] == "REVIEW"
    assert "duplicate" in results[1]["justification"].lower()
    assert results[1].get("duplicate_group_id") == "dup_group_0"
    assert set(results[1].get("duplicate_labels", [])) == {"User", "Account User"}

    # cls_003: REVIEW from duplicate detection
    assert results[2]["id"] == "cls_003"
    assert results[2]["judgment"] == "REVIEW"
    assert "duplicate" in results[2]["justification"].lower()
    assert results[2].get("duplicate_group_id") == "dup_group_0"
    assert set(results[2].get("duplicate_labels", [])) == {"User", "System User"}

    # cls_004: REVIEW from duplicate detection
    assert results[3]["id"] == "cls_004"
    assert results[3]["judgment"] == "REVIEW"
    assert "duplicate" in results[3]["justification"].lower()
    assert results[3].get("duplicate_group_id") == "dup_group_1"
    assert results[3].get("duplicate_labels") == ["Admin"]

    # cls_005: REVIEW from duplicate detection
    assert results[4]["id"] == "cls_005"
    assert results[4]["judgment"] == "REVIEW"
    assert "duplicate" in results[4]["justification"].lower()
    assert results[4].get("duplicate_group_id") == "dup_group_1"
    assert results[4].get("duplicate_labels") == ["Administrator"]

    print("\n✅ All assertions passed!")

def test_convert_duplicate_response():
    """Test the conversion of LLM response to internal format"""

    reviewer = ClassReviewer.__new__(ClassReviewer)

    llm_response = {
        "duplicate_groups": [
            {
                "class_ids": ["cls_001", "cls_002"],
                "labels": ["User", "System User"],
                "reason": "Same concept"
            },
            {
                "class_ids": ["cls_003", "cls_004", "cls_005"],
                "labels": ["Admin", "Administrator", "System Admin"],
                "reason": "Administrator variants"
            }
        ]
    }

    result = reviewer._convert_duplicate_response_to_dict(llm_response)

    print("\n=== Convert Response Test ===\n")
    print(f"Groups found: {len(result)}")

    assert len(result) == 2
    assert "dup_group_0" in result
    assert "dup_group_1" in result

    assert result["dup_group_0"]["class_ids"] == ["cls_001", "cls_002"]
    assert result["dup_group_0"]["labels"] == ["User", "System User"]
    assert result["dup_group_0"]["reason"] == "Same concept"

    assert result["dup_group_1"]["class_ids"] == ["cls_003", "cls_004", "cls_005"]

    print("✅ Conversion test passed!")

if __name__ == "__main__":
    test_convert_duplicate_response()
    test_merge_review_results()
    print("\n🎉 All tests passed successfully!")
