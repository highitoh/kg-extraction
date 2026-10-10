# turtle_generator.py
# Turtle RDF/OWL file generator based on class_extractor.py logic

import os
import re
import uuid
from datetime import datetime
from typing import Any, Dict, Optional, Tuple


# ---------- Utilities ----------
def _normalize_ws(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip()


def _uuid8() -> str:
    return str(uuid.uuid4())[:8]


def _slug(s: str) -> str:
    s = _normalize_ws(s)
    s = re.sub(r"[^0-9A-Za-z\u00C0-\u024F\u3040-\u30FF\u4E00-\u9FFF\-_.]", "-", s)
    s = re.sub(r"-{2,}", "-", s).strip("-")
    return s[:48] or _uuid8()


class TurtleGenerator:
    """
    Receives class extraction information and property extraction information, and generates RDF/OWL (Turtle) files.
    """

    def __init__(
        self,
        *,
        base_prefix: str = "https://github.com/highitoh/kg-extraction#",
        output_dir: str = "./output",
    ):
        self.base_prefix = base_prefix
        self.output_dir = output_dir

        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)

    def _iri(self, path: str) -> str:
        """
        Returns the full IRI <https://...> instead of the prefix notation ex:.
        Example) path="Extraction/0-aaaa" -> "<https://github.com/highitoh/kg-extraction#Extraction/0-aaaa>"
        """
        base = self.base_prefix.rstrip("#/") + "#"
        return f"<{base}{path}>"

    def _term(self, iri: str) -> str:
        """
        Returns an IRI as a Turtle term: absolute IRIs are wrapped in <>, prefixed names are kept as-is.
        """
        if iri.startswith("<") or not re.match(r"^[A-Za-z][A-Za-z0-9+.-]*://", iri):
            return iri
        return f"<{iri}>"

    def _escape_string(self, s: str) -> str:
        """
        Escape processing for Turtle string literals
        """
        return s.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n").replace("\r", "\\r")

    def _escape_turtle_text(self, s: str) -> str:
        """
        Escape processing for Turtle long string literals
        """
        return s.replace('"""', '\\"""')

    def _generate_turtle_header(self) -> str:
        """
        Generate the header section of the Turtle file
        """
        header_lines = [
            f"# Generated on {datetime.now().isoformat()}",
            "",
            "@prefix ex: <https://github.com/highitoh/kg-extraction#> .",
            "@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .",
            "@prefix owl: <http://www.w3.org/2002/07/owl#> .",
            "@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .",
            "@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .",
            "",
            "# Ontology Declaration",
            "ex: a owl:Ontology .",
            "",
        ]
        return "\n".join(header_lines)

    def _generate_class_declarations(self, class_iris: set) -> str:
        """
        Generate the class declarations section
        """
        lines = ["# Class Declarations"]
        for class_iri in sorted(class_iris):
            lines.append(f"{class_iri} a owl:Class .")
        lines.append("")
        return "\n".join(lines)

    def _generate_property_declarations(self, property_iris: set) -> str:
        """
        Generate the property declarations section
        """
        lines = ["# Property Declarations"]
        for prop_iri in sorted(property_iris):
            lines.append(f"{prop_iri} a owl:ObjectProperty .")
        lines.append("")
        return "\n".join(lines)

    def _generate_instances(self, class_info: Dict[str, Any], property_info: Dict[str, Any]) -> Tuple[str, set, set]:
        """
        Generate the instances section and return the set of class IRIs and property IRIs being used
        """
        lines = ["# Instances"]
        class_iris = set()
        property_iris = set()

        # Instance IRI per class instance ID, shared by declarations and relationships
        instance_iris = {}
        for class_item in class_info.get("classes", []):
            instance_iris.setdefault(class_item["id"], self._iri(f"instance/{_slug(class_item['label'])}-{_uuid8()}"))

        # Output class instances
        for class_item in class_info.get("classes", []):
            class_id = class_item["id"]
            class_iri = self._term(class_item["class_iri"])
            label = class_item["label"]

            # Get file_ids field (array or string)
            file_ids = class_item.get("file_ids", [])
            if isinstance(file_ids, str):
                file_ids = [file_ids]
            # Concatenate with comma separation (retain multiple sources)
            file_id = ",".join(file_ids) if file_ids else ""

            instance_iri = instance_iris[class_id]

            # Record class IRI
            class_iris.add(class_iri)

            # Instance definition
            lines.append(f"{instance_iri} a {class_iri} ;")
            lines.append(f'  rdfs:label "{self._escape_string(label)}"@ja ;')
            if file_id:
                lines.append(f'  ex:fileId "{self._escape_string(file_id)}" ;')
            lines.append(f'  ex:classInstanceId "{self._escape_string(class_id)}" .')
            lines.append("")

        # Output property relationships
        property_map = {}
        for prop_item in property_info.get("properties", []):
            src_id = prop_item["src_id"]
            dest_id = prop_item["dest_id"]
            property_iri = self._term(prop_item["property_iri"])

            # Record property IRI
            property_iris.add(property_iri)

            # Mapping from class instance ID to instance IRI
            if src_id not in property_map:
                property_map[src_id] = []
            property_map[src_id].append((property_iri, dest_id))

        # Add property relationships
        for src_id, relations in property_map.items():
            if src_id in instance_iris:
                for property_iri, dest_id in relations:
                    if dest_id in instance_iris:
                        lines.append(f"{instance_iris[src_id]} {property_iri} {instance_iris[dest_id]} .")

        if len(lines) > 1:  # If there are lines other than "# Instances"
            lines.append("")

        return "\n".join(lines), class_iris, property_iris

    def generate(self, input_data: Dict[str, Any]) -> Dict[str, str]:
        """
        Generate a Turtle file from input data

        Args:
            input_data: Input data following the schema

        Returns:
            Dictionary containing the generated Turtle string
        """
        class_info = input_data.get("class_info", {})
        property_info = input_data.get("property_info", {})

        # Generate each section of the Turtle file
        header = self._generate_turtle_header()
        instances_content, class_iris, property_iris = self._generate_instances(class_info, property_info)
        class_declarations = self._generate_class_declarations(class_iris)
        property_declarations = self._generate_property_declarations(property_iris)

        # Combine everything
        turtle_content = header + class_declarations + property_declarations + instances_content

        return {"turtle": turtle_content}

    def generate_and_save(
        self, input_data: Dict[str, Any], filename: Optional[str] = None
    ) -> Tuple[str, Dict[str, str]]:
        """
        Generate a Turtle file and save it to disk

        Args:
            input_data: Input data following the schema
            filename: Output filename (auto-generated if omitted)

        Returns:
            (saved file path, generated result dictionary)
        """
        result = self.generate(input_data)

        if filename is None:
            filename = "knowledge_graph.ttl"

        if not filename.endswith(".ttl"):
            filename += ".ttl"

        filepath = os.path.join(self.output_dir, filename)

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(result["turtle"])

        return filepath, result


# -------------- Usage Example --------------
if __name__ == "__main__":
    # Sample data for testing
    sample_input = {
        "class_info": {
            "classes": [
                {"id": "cls-001", "class_iri": "ex:Stakeholder", "label": "Store Staff", "file_id": "doc-001"},
                {"id": "cls-002", "class_iri": "ex:Value", "label": "Reduced Wait Time", "file_id": "doc-001"},
                {"id": "cls-003", "class_iri": "ex:Vision", "label": "No-Wait Store Experience", "file_id": "doc-001"},
            ]
        },
        "property_info": {
            "properties": [
                {"id": "prop-001", "src_id": "cls-001", "property_iri": "ex:hasValue", "dest_id": "cls-002"},
                {"id": "prop-002", "src_id": "cls-003", "property_iri": "ex:realizesValue", "dest_id": "cls-002"},
            ]
        },
    }

    generator = TurtleGenerator()
    filepath, result = generator.generate_and_save(sample_input, "test_output.ttl")

    print(f"Generated Turtle file: {filepath}")
    print("\nTurtle content preview:")
    print(result["turtle"][:500] + "..." if len(result["turtle"]) > 500 else result["turtle"])
