import os
import re
from typing import Any, Dict, List, Tuple

from langchain_core.runnables.config import RunnableConfig

from src.utils.text_utils import detect_file_encoding

from .text_extractor import TextExtractor


class MdTextExtractor(TextExtractor):
    """Extract text, section metadata, and tables from Markdown files"""

    def __init__(self):
        super().__init__("/workspace/app/log/md_text_extractor")

    def _parse_header_level(self, line: str) -> Tuple[int, str]:
        """
        Parse Markdown header line

        Returns:
            (level, header_text) or (0, "") if not a header
        """
        match = re.match(r"^(#{1,6})\s+(.+)$", line.strip())
        if match:
            level = len(match.group(1))
            text = match.group(2).strip()
            return (level, text)
        return (0, "")

    def _build_section_path(self, header_stack: List[Tuple[int, str]]) -> str:
        """
        Build section path from header stack

        Example: [("Installation Guide", 1), ("Prerequisites", 2)]
        -> "Installation Guide > Prerequisites"
        """
        if not header_stack:
            return ""
        return " > ".join([h[1] for h in header_stack])

    def _extract_text_with_sections(self, md_path: str) -> List[Dict[str, str]]:
        """
        Extract text from Markdown with section metadata

        Returns:
            List of sentences with optional section metadata
            [{"text": "...", "section": "Header > Subheader"}, ...]
        """
        if not os.path.exists(md_path):
            raise FileNotFoundError(f"Markdown file not found: {md_path}")

        if not md_path.lower().endswith((".md", ".markdown")):
            raise ValueError("Not a Markdown file")

        encoding = detect_file_encoding(md_path)
        with open(md_path, "r", encoding=encoding) as f:
            content = f.read()

        sentences = []
        header_stack = []  # Stack of (level, text) tuples
        current_section = ""

        for line in content.split("\n"):
            stripped = line.strip()

            # Skip empty lines
            if not stripped:
                continue

            # Check if line is a header
            level, header_text = self._parse_header_level(line)
            if level > 0:
                # Update header stack
                # Remove headers at same or deeper level
                header_stack = [(lvl, txt) for lvl, txt in header_stack if lvl < level]
                # Add current header
                header_stack.append((level, header_text))
                # Update current section path
                current_section = self._build_section_path(header_stack)
                continue

            # Regular text line
            if stripped:
                sentence = {"text": stripped}
                if current_section:
                    sentence["section"] = current_section
                sentences.append(sentence)

        return sentences

    def _extract_table_blocks(self, md_path: str) -> list:
        """
        Extract structured table blocks from Markdown

        Returns:
            List of structured blocks (same format as DOCX/PPTX tables)
        """
        from src.utils.structured_block_builder import build_table_blocks
        from src.utils.table_extractor import extract_tables_from_markdown

        encoding = detect_file_encoding(md_path)
        with open(md_path, "r", encoding=encoding) as f:
            content = f.read()

        # Extract tables from markdown text
        tables = extract_tables_from_markdown(content)
        if not tables:
            return []

        all_blocks = []
        filename = os.path.basename(md_path)

        for table_idx, table_data in enumerate(tables, start=1):
            # Build metadata
            file_id = f"md:{filename}:table_{table_idx}"
            scope_ref = f"table_{table_idx}"
            section_path = f"{filename} - Table {table_idx}"

            # Context window (text before table, if available)
            context_window = table_data.get("context", "")

            # Build structured blocks
            table_blocks = build_table_blocks([table_data], file_id, scope_ref, section_path, context_window)
            all_blocks.extend(table_blocks)

        return all_blocks

    def invoke(self, input: Dict[str, Any], config: RunnableConfig = None) -> Dict[str, Any]:
        """
        Main extraction entry point

        Input: {"files": [{"path": "/path/to/file.md"}]}
        Output: {
            "file_id": "md:filename.md",
            "file_name": "filename.md",
            "sentences": [{"text": "...", "section": "..."}, ...],
            "_md_structured_blocks": [...]  # Optional
        }
        """
        file_path = input["files"][0]["path"]

        # Extract text with section metadata
        sentences = self._extract_text_with_sections(file_path)

        # Extract structured table blocks
        all_blocks = self._extract_table_blocks(file_path)

        # Build output using base class helper
        output = self.build_output(file_path, sentences)

        # Attach structured blocks if any tables exist
        if all_blocks:
            output["_md_structured_blocks"] = all_blocks

        # Save log
        self.logger.save_log(output, "md_text_extractor_output_")

        return output


if __name__ == "__main__":
    extractor = MdTextExtractor()
    result = extractor.invoke({"files": [{"path": "../doc/README.md"}]})
    print(f"Processing result: {result}")
