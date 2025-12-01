"""
Neo4j Bulk Import CSV Generator

Generates CSV files for Neo4j Bulk Import from JSON input.
Retrieves information from JSON schema and generates CSV, based on the logic in bk/rdf_to_property_graph.py.
"""

import os
import csv
import json
from typing import Dict, Set, Tuple, Any


class Neo4jCSVGenerator:
    """
    Generates CSV files (nodes.csv, relationships.csv) for Neo4j Bulk Import from JSON input.

    Input: class_info (ClassChainOutput) and property_info (PropertyChainOutput)
    Output: Generates nodes.csv and relationships.csv in app/output
    """

    NODE_HEADERS = [
        "id:ID",             # Class instance ID
        "iri",               # class_iri (for display/verification)
        ":LABEL",            # Class local name
        "label",             # Extracted class label
        "file_id",           # Source file ID
    ]

    REL_HEADERS = [
        ":START_ID",         # src_id
        ":END_ID",           # dest_id
        ":TYPE",             # Property local name
        "property_iri",      # Property IRI
        "id",                # Property ID
        "strength:int",      # Relationship strength
    ]

    def __init__(self, output_dir: str = "./output"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        # Node information
        self.nodes: Dict[str, Dict[str, str]] = {}
        # Relationship information
        self.relationships: Set[Tuple[str, str, str, str, str, int]] = set()

    @staticmethod
    def _local_name(iri: str) -> str:
        """Extract local name from IRI"""
        if "#" in iri:
            return iri.rsplit("#", 1)[-1]
        if "/" in iri:
            return iri.rstrip("/").rsplit("/", 1)[-1]
        if ":" in iri:
            return iri.rsplit(":", 1)[-1]
        return iri

    @staticmethod
    def _to_snake_case(name: str) -> str:
        """Convert chain case (hyphen-separated) to snake case"""
        return name.replace("-", "_")

    def _process_classes(self, class_info: Dict[str, Any]):
        """Generate nodes from class information"""
        for class_item in class_info.get("classes", []):
            node_id = class_item["id"]
            class_iri = class_item["class_iri"]
            label = class_item["label"]

            # Get file_ids field (array or string)
            file_ids = class_item.get("file_ids", [])
            if isinstance(file_ids, str):
                file_ids = [file_ids]
            # Concatenate with comma separation (retain multiple sources)
            file_id = ",".join(file_ids) if file_ids else ""

            # Node label (convert class local name to snake case)
            class_label = self._to_snake_case(self._local_name(class_iri))

            self.nodes[node_id] = {
                "id:ID": node_id,
                "iri": class_iri,
                ":LABEL": class_label,
                "label": label,
                "file_id": file_id,
            }

    def _process_properties(self, property_info: Dict[str, Any]):
        """Generate relationships from property information"""
        for prop_item in property_info.get("properties", []):
            prop_id = prop_item["id"]
            src_id = prop_item["src_id"]
            dest_id = prop_item["dest_id"]
            property_iri = prop_item["property_iri"]
            strength = int(prop_item.get("strength", 0))

            # Relationship type (convert property local name to snake case)
            rel_type = self._to_snake_case(self._local_name(property_iri))

            self.relationships.add((
                src_id,         # :START_ID
                dest_id,        # :END_ID
                rel_type,       # :TYPE
                property_iri,   # property_iri
                prop_id,        # id
                strength,       # strength
            ))

    def generate(self, input_data: Dict[str, Any]) -> Dict[str, str]:
        """
        Generate CSV files from JSON input

        Args:
            input_data: {"class_info": {...}, "property_info": {...}}

        Returns:
            {"nodes_path": str, "properties_path": str}
        """
        # Process input data
        class_info = input_data.get("class_info", {})
        property_info = input_data.get("property_info", {})

        # Process nodes and relationships
        self._process_classes(class_info)
        self._process_properties(property_info)

        # Generate CSV files
        nodes_path = self._write_nodes_csv()
        relationships_path = self._write_relationships_csv()

        return {
            "nodes_path": nodes_path,
            "properties_path": relationships_path
        }

    def _write_nodes_csv(self) -> str:
        """Write nodes CSV"""
        nodes_path = os.path.join(self.output_dir, "nodes.csv")

        with open(nodes_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.NODE_HEADERS, quoting=csv.QUOTE_MINIMAL)
            writer.writeheader()

            for node_id in sorted(self.nodes.keys()):
                row = self.nodes[node_id]
                # Fill missing columns with empty strings
                for header in self.NODE_HEADERS:
                    row.setdefault(header, "")
                writer.writerow(row)

        return nodes_path

    def _write_relationships_csv(self) -> str:
        """Write relationships CSV"""
        rels_path = os.path.join(self.output_dir, "relationships.csv")

        with open(rels_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.REL_HEADERS, quoting=csv.QUOTE_MINIMAL)
            writer.writeheader()

            for (start_id, end_id, rel_type, prop_iri, prop_id, strength) in sorted(self.relationships):
                writer.writerow({
                    ":START_ID": start_id,
                    ":END_ID": end_id,
                    ":TYPE": rel_type,
                    "property_iri": prop_iri,
                    "id": prop_id,
                    "strength:int": strength,
                })

        return rels_path


def load_input_from_file(file_path: str) -> Dict[str, Any]:
    """Load input data from JSON file"""
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    """Sample execution"""
    # Sample data
    sample_input = {
        "class_info": {
            "id": "extraction_001",
            "classes": [
                {
                    "id": "class_001",
                    "class_iri": "https://github.com/highitoh/kg-extraction#Stakeholder",
                    "label": "Store Staff",
                    "file_id": "file_001"
                },
                {
                    "id": "class_002",
                    "class_iri": "https://github.com/highitoh/kg-extraction#Value",
                    "label": "Reduced Wait Time",
                    "file_id": "file_001"
                }
            ]
        },
        "property_info": {
            "id": "extraction_001",
            "properties": [
                {
                    "id": "prop_001",
                    "src_id": "class_001",
                    "property_iri": "https://github.com/highitoh/kg-extraction#hasValue",
                    "dest_id": "class_002"
                }
            ]
        }
    }

    generator = Neo4jCSVGenerator()
    result = generator.generate(sample_input)

    print(f"Generated nodes CSV: {result['nodes_path']}")
    print(f"Generated relationships CSV: {result['properties_path']}")


if __name__ == "__main__":
    main()
