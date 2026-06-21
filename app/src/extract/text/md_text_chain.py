from typing import Any, Dict

from langchain_core.runnables import Runnable

from .md_text_extractor import MdTextExtractor


class MdTextChain(Runnable):
    """Chain for extracting text and tables from Markdown files"""

    def __init__(self):
        self.extractor = MdTextExtractor()

    def invoke(self, input: Dict[str, Any], config=None) -> Dict[str, Any]:
        """
        Invoke Markdown text extraction

        Args:
            input: {"files": [{"path": "/path/to/file.md"}]}

        Returns:
            Extraction result from MdTextExtractor
        """
        return self.extractor.invoke(input)
