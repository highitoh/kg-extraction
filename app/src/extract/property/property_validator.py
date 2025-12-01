from typing import Dict, Any, List
import json
import os
import uuid

from langchain_core.runnables import Runnable

from src.utils.logger import Logger


class PropertyValidator(Runnable):
    """
    Validator to verify property validity

    Validation checks:
    0. strength is not 0 (0 = no relationship, excluded)
    1. src_id, dest_id exist in class_info.classes
    2. property_iri is defined in metamodel.properties
    3. src/dest class_iri matches src_class/dest_class in property definition
    """

    def __init__(
        self,
        progress: bool = True,
        log_dir: str = "/workspace/app/log/property_validator",
    ):
        self.progress = progress
        self.logger = Logger(log_dir)

    def _build_class_index(self, classes: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """Create index: class_id -> class information"""
        return {c["id"]: c for c in classes}

    def _build_property_index(self, metamodel: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """Create index: property_iri -> property definition"""
        return {p["iri"]: p for p in metamodel.get("properties", [])}

    def _validate_property(
        self,
        prop: Dict[str, Any],
        class_index: Dict[str, Dict[str, Any]],
        property_index: Dict[str, Dict[str, Any]],
    ) -> tuple[bool, str]:
        """
        Validate property validity

        Returns:
            (is_valid, reason): Validation result and reason
        """
        # 0. Check strength (0 = no relationship, excluded)
        strength = prop.get("strength", 0)
        if strength == 0:
            return False, "strength is 0 (no relationship)"

        src_id = prop.get("src_id", "")
        dest_id = prop.get("dest_id", "")
        property_iri = prop.get("property_iri", "")

        # 1. Check src_id existence
        if src_id not in class_index:
            return False, f"src_id '{src_id}' not found in classes"

        # 2. Check dest_id existence
        if dest_id not in class_index:
            return False, f"dest_id '{dest_id}' not found in classes"

        # 3. Check property_iri existence
        if property_iri not in property_index:
            return False, f"property_iri '{property_iri}' not found in metamodel"

        # 4. Verify class_iri consistency
        src_class = class_index[src_id]
        dest_class = class_index[dest_id]
        property_def = property_index[property_iri]

        src_class_iri = src_class.get("class_iri", "")
        dest_class_iri = dest_class.get("class_iri", "")
        expected_src_class = property_def.get("src_class", "")
        expected_dest_class = property_def.get("dest_class", "")

        if src_class_iri != expected_src_class:
            return False, (
                f"src class_iri mismatch: expected '{expected_src_class}', "
                f"got '{src_class_iri}' (src_id: {src_id})"
            )

        if dest_class_iri != expected_dest_class:
            return False, (
                f"dest class_iri mismatch: expected '{expected_dest_class}', "
                f"got '{dest_class_iri}' (dest_id: {dest_id})"
            )

        return True, "valid"

    def invoke(self, input: Dict[str, Any], config=None) -> Dict[str, Any]:
        """
        Input:
          - Recommended: {"property_candidates": {...}, "class_info": {...}, "metamodel": {...}}
          - Compatible: {"properties":[...], "id":"...", "class_info": {...}, "metamodel": {...}}
        """
        # Flexibly interpret input
        property_candidates = input.get("property_candidates") or {
            "id": input.get("id"),
            "properties": input.get("properties", []),
        }
        class_info = input.get("class_info") or {}
        metamodel = input.get("metamodel") or {}

        properties = property_candidates.get("properties", [])
        classes = class_info.get("classes", [])

        if not classes:
            # Skip if class_info is missing
            if self.progress:
                print("PropertyValidator: class_info missing -> pass-through")
            output = {
                "id": str(uuid.uuid4()),
                "properties": properties,
            }
            return output

        if not metamodel:
            # Skip if metamodel is missing
            if self.progress:
                print("PropertyValidator: metamodel missing -> pass-through")
            output = {
                "id": str(uuid.uuid4()),
                "properties": properties,
            }
            return output

        # Build indexes
        class_index = self._build_class_index(classes)
        property_index = self._build_property_index(metamodel)

        # Validate properties
        valid_properties = []
        invalid_properties = []

        for prop in properties:
            is_valid, reason = self._validate_property(prop, class_index, property_index)

            if is_valid:
                valid_properties.append(prop)
            else:
                invalid_properties.append({
                    "property": prop,
                    "reason": reason
                })

        # Build output
        output: Dict[str, Any] = {
            "id": str(uuid.uuid4()),
            "properties": valid_properties,
        }

        if self.progress:
            print(
                f"PropertyValidator: {len(properties)} candidates -> "
                f"{len(valid_properties)} valid, {len(invalid_properties)} invalid"
            )

        # Save log (invalid only)
        if invalid_properties:
            log_data = {
                "total": len(properties),
                "valid": len(valid_properties),
                "invalid": len(invalid_properties),
                "invalid_properties": invalid_properties
            }
            self.logger.save_log(log_data, filename_prefix="property_validator_output_")

        return output


if __name__ == "__main__":
    import glob

    # Test execution:
    # - Load latest PropertyReviewer output (log/property_reviewer/property_reviewer_output_*.json)
    # - Load latest ClassReviewer output (log/class_reviewer/class_reviewer_output_*.json)
    # - Display validation results
    pv = PropertyValidator()

    prop_files = sorted(glob.glob("/workspace/app/log/property_reviewer/property_reviewer_output_*.json"), reverse=True)
    cls_files = sorted(glob.glob("/workspace/app/log/class_reviewer/class_reviewer_output_*.json"), reverse=True)

    if not prop_files:
        print("Error: No property_reviewer_output files found in log/property_reviewer/")
        exit(1)
    if not cls_files:
        print("Error: No class_reviewer_output files found in log/class_reviewer/")
        exit(1)

    prop_path = prop_files[0]
    cls_path = cls_files[0]
    print(f"Loading latest PropertyReviewer output: {prop_path}")
    print(f"Loading latest ClassReviewer output:    {cls_path}")

    with open(prop_path, "r", encoding="utf-8") as f:
        property_candidates = json.load(f)
    with open(cls_path, "r", encoding="utf-8") as f:
        class_info = json.load(f)

    # Load metamodel
    # Go up 3 levels from src/extract/XXX/ to /workspace/app/

    app_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

    metamodel_path = os.path.join(app_dir, "metamodel", "metamodel.json")
    with open(metamodel_path, "r", encoding="utf-8") as f:
        metamodel = json.load(f)

    input_data = {
        "property_candidates": property_candidates,
        "class_info": class_info,
        "metamodel": metamodel,
    }

    print("\n=== PropertyValidator Test ===")
    print(f"- Candidates: {len(property_candidates.get('properties', []))}")
    print(f"- Classes:    {len(class_info.get('classes', []))}")
    print("\n" + "=" * 60 + "\n")

    try:
        result = pv.invoke(input_data)

        print("\nSummary:")
        print(f"- Total candidates: {len(property_candidates.get('properties', []))}")
        print(f"- Valid properties: {len(result.get('properties', []))}")

        # Display valid properties
        if result.get('properties'):
            print("\nValid properties:")
            idx = {c["id"]: c for c in class_info.get("classes", [])}
            for i, p in enumerate(result.get("properties", []), 1):
                s = idx.get(p["src_id"], {})
                d = idx.get(p["dest_id"], {})
                s_label = s.get("label", p["src_id"])
                d_label = d.get("label", p["dest_id"])
                print(f"  {i:02d}. {s_label} --[{p['property_iri']}]--> {d_label}")

        # Load and display log file if available
        log_files = sorted(glob.glob("/workspace/app/log/property_validator/property_validator_output_*.json"), reverse=True)
        if log_files:
            with open(log_files[0], "r", encoding="utf-8") as f:
                log_data = json.load(f)

            print(f"\n- Invalid properties: {log_data.get('invalid', 0)}")
            if log_data.get("invalid_properties"):
                print("\nInvalid details:")
                for i, invalid in enumerate(log_data.get("invalid_properties", []), 1):
                    prop = invalid["property"]
                    reason = invalid["reason"]
                    print(f"  {i:02d}. {prop.get('src_id')} --[{prop.get('property_iri')}]--> {prop.get('dest_id')}")
                    print(f"      Reason: {reason}")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
