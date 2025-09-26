import re
from typing import List


class ChunkCreator:
    """
    テキストをチャンクに分割するユーティリティクラス
    ViewExtractorで再利用可能
    """

    def __init__(self, max_chars: int = 1600, overlap_chars: int = 200):
        self.max_chars = max_chars
        self.overlap_chars = overlap_chars

    def _split_into_sentences(self, text: str) -> List[str]:
        """テキストを文に分割"""
        text = text.replace("\r\n", "\n")
        parts = re.split(r'(?<=[。！？!?])\s*|\n{2,}', text)
        return [p.strip() for p in parts if p and p.strip()]

    def create(self, text: str) -> List[str]:
        """テキストをチャンクに分割（重複あり）"""
        sents = self._split_into_sentences(text)
        chunks, cur = [], ""
        for s in sents:
            if len(cur) + len(s) + 1 <= self.max_chars:
                cur = (cur + " " + s).strip()
            else:
                if cur:
                    chunks.append(cur)
                if self.overlap_chars > 0 and cur:
                    tail = cur[-self.overlap_chars:]
                    cur = (tail + " " + s).strip()
                else:
                    cur = s
        if cur:
            chunks.append(cur)
        return chunks