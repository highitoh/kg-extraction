from typing import Any, Dict

from langchain_core.runnables import Runnable

from .txt_text_extractor import TxtTextExtractor


class TxtTextChain(Runnable):
    """Chain for extracting text from TXT files"""

    def __init__(self):
        self.extractor = TxtTextExtractor()

    def invoke(self, input: Dict[str, Any], config=None) -> Dict[str, Any]:
        """
        Invoke TXT text extraction

        Args:
            input: {"files": [{"path": "/path/to/file.txt"}]}

        Returns:
            Extraction result from TxtTextExtractor
        """
        return self.extractor.invoke(input)
