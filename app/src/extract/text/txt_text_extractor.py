import os
from typing import Any, Dict

from langchain_core.runnables.config import RunnableConfig

from src.utils.text_utils import detect_file_encoding

from .text_extractor import TextExtractor


class TxtTextExtractor(TextExtractor):
    """Extract text from plain text files"""

    def __init__(self):
        super().__init__("/workspace/app/log/txt_text_extractor")

    def _extract_text(self, txt_path: str) -> str:
        """
        Extract text from TXT file

        Returns:
            File content as string
        """
        if not os.path.exists(txt_path):
            raise FileNotFoundError(f"TXT file not found: {txt_path}")

        if not txt_path.lower().endswith(".txt"):
            raise ValueError("Not a TXT file")

        encoding = detect_file_encoding(txt_path)
        with open(txt_path, "r", encoding=encoding) as f:
            return f.read()

    def invoke(self, input: Dict[str, Any], config: RunnableConfig = None) -> Dict[str, Any]:
        """
        Main extraction entry point

        Input: {"files": [{"path": "/path/to/file.txt"}]}
        Output: {
            "file_id": "txt:filename.txt",
            "file_name": "filename.txt",
            "sentences": [{"text": "line 1"}, ...]
        }
        """
        file_path = input["files"][0]["path"]

        # Extract text
        extracted_text = self._extract_text(file_path)

        # Split text into sentences (one line = one sentence, preserve empty lines as separators)
        sentences = []
        for line_text in extracted_text.split("\n"):
            if line_text.strip():
                sentences.append({"text": line_text.strip()})

        # Build output using base class helper
        output = self.build_output(file_path, sentences)

        # Save log
        self.logger.save_log(output, "txt_text_extractor_output_")

        return output


if __name__ == "__main__":
    extractor = TxtTextExtractor()
    result = extractor.invoke({"files": [{"path": "../doc/sample.txt"}]})
    print(f"Processing result: {result}")
