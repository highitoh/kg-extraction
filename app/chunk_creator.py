import re
from typing import List, Literal


class ChunkCreator:
    """
    テキストをチャンクに分割するユーティリティクラス
    ViewExtractorで再利用可能
    """

    def __init__(
        self,
        chunk_mode: Literal["char", "page"] = "char",
        max_chars: int = 1600,
        overlap_chars: int = 200
    ):
        """
        Args:
            chunk_mode: チャンク分割モード
                - "char": 文字数ベースで分割（デフォルト）
                - "page": ページ/スライド単位で分割
            max_chars: 1チャンクの最大文字数（charモードで使用）
            overlap_chars: チャンク間の重複文字数（charモードで使用）
        """
        self.chunk_mode = chunk_mode
        self.max_chars = max_chars
        self.overlap_chars = overlap_chars

    def _split_into_sentences(self, text: str) -> List[str]:
        """テキストを文に分割"""
        text = text.replace("\r\n", "\n")
        parts = re.split(r'(?<=[。！？!?])\s*|\n{2,}', text)
        return [p.strip() for p in parts if p and p.strip()]

    def create(self, text: str) -> List[str]:
        """
        テキストをチャンクに分割

        Args:
            text: 入力テキスト

        Returns:
            チャンクのリスト
        """
        if self.chunk_mode == "char":
            return self._create_char_based(text)
        elif self.chunk_mode == "page":
            return self._create_page_based(text)
        else:
            raise ValueError(f"Unknown chunk_mode: {self.chunk_mode}")

    def _create_char_based(self, text: str) -> List[str]:
        """文字数ベースでチャンクに分割（重複あり）"""
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

    def _create_page_based(self, text: str) -> List[str]:
        """
        ページ/スライド単位でチャンクを作成

        テキスト内の "[[スライド{番号}]]" マーカーを検出し、
        スライド番号ごとにグループ化して結合します。

        Args:
            text: "[[スライド{番号}]] {テキスト}" 形式を含むテキスト

        Returns:
            スライドごとに結合されたテキストのリスト（スライド番号順）
        """
        lines = text.split('\n')
        slide_chunks = {}

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # "[[スライド{番号}]]"を抽出
            match = re.search(r'\[\[スライド(\d+)\]\]', line)
            if match:
                slide_num = match.group(1)
                slide_text = line.replace(match.group(0), '').strip()
                if slide_num not in slide_chunks:
                    slide_chunks[slide_num] = []
                if slide_text:  # 空文字列は追加しない
                    slide_chunks[slide_num].append(slide_text)

        # スライドごとに結合してリストを作成（番号順）
        chunks = [
            " ".join(texts)
            for slide_num, texts in sorted(slide_chunks.items(), key=lambda x: int(x[0]))
        ]

        return chunks