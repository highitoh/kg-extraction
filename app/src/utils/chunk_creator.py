import re
from typing import List, Literal


class ChunkCreator:
    """
    Utility class for splitting text into chunks
    Reusable in ViewExtractor
    """

    def __init__(self, chunk_mode: Literal["char", "page"] = "char", max_chars: int = 400, overlap_chars: int = 50):
        """
        Args:
            chunk_mode: Chunk splitting mode
                - "char": Split by character count (default)
                - "page": Split by page/slide unit
            max_chars: Maximum characters per chunk (used in char mode)
            overlap_chars: Overlapping characters between chunks (used in char mode)
        """
        self.chunk_mode = chunk_mode
        self.max_chars = max_chars
        self.overlap_chars = overlap_chars

    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences (supports both Japanese and English)"""
        text = text.replace("\r\n", "\n")
        # Split by:
        #   - Japanese sentence-end markers (。！？!?)
        #   - English sentence boundary: word of 2+ chars ending with period,
        #     followed by whitespace and uppercase letter
        #     (avoids splitting abbreviations like "Dr.", "Mr.", "U.")
        #   - Double newlines (paragraph boundary)
        parts = re.split(
            r"(?<=[。！？!?])\s*"
            r"|(?<=[a-z]{2}\.)\s+(?=[A-Z])"
            r"|\n{2,}",
            text,
        )
        sentences = [p.strip() for p in parts if p and p.strip()]

        # Split oversized sentences that exceed max_chars
        result = []
        for s in sentences:
            if len(s) <= self.max_chars:
                result.append(s)
            else:
                result.extend(self._split_oversized(s))
        return result

    def split_at_line_boundaries(self, text: str) -> List[str]:
        """Break text at line ends, and only when it exceeds max_chars.

        Shorter text is returned as a single piece.
        """
        text = text.replace("\r\n", "\n").strip()
        if not text:
            return []
        if len(text) <= self.max_chars:
            return [text]
        return self._split_oversized(text)

    def _split_oversized(self, text: str) -> List[str]:
        """Split an oversized sentence into pieces that fit within max_chars.
        Tries splitting by newline first, then by space, then by hard cut."""
        # Try splitting by single newline
        parts = [p.strip() for p in text.split("\n") if p.strip()]
        if len(parts) > 1:
            result = []
            for p in parts:
                if len(p) <= self.max_chars:
                    result.append(p)
                else:
                    result.extend(self._split_by_space_or_hard(p))
            return result

        return self._split_by_space_or_hard(text)

    def _split_by_space_or_hard(self, text: str) -> List[str]:
        """Split text by space boundaries, falling back to hard character cut."""
        # Try splitting by spaces
        tokens = re.split(r"(\s+)", text)
        chunks, cur = [], ""
        for token in tokens:
            if len(cur) + len(token) <= self.max_chars:
                cur += token
            else:
                if cur.strip():
                    chunks.append(cur.strip())
                if len(token.strip()) > self.max_chars:
                    # Hard cut for tokens with no whitespace at all
                    remaining = token.strip()
                    while remaining:
                        chunks.append(remaining[: self.max_chars])
                        remaining = remaining[self.max_chars :]
                    cur = ""
                else:
                    cur = token
        if cur.strip():
            chunks.append(cur.strip())
        return chunks if chunks else [text]

    def create(self, text: str) -> List[str]:
        """
        Split text into chunks

        Args:
            text: Input text

        Returns:
            List of chunks
        """
        if self.chunk_mode == "char":
            return self._create_char_based(text)
        elif self.chunk_mode == "page":
            return self._create_page_based(text)
        else:
            raise ValueError(f"Unknown chunk_mode: {self.chunk_mode}")

    def _create_char_based(self, text: str) -> List[str]:
        """Split into chunks based on character count (with overlap)"""
        sents = self._split_into_sentences(text)
        chunks, cur = [], ""
        for s in sents:
            if len(cur) + len(s) + 1 <= self.max_chars:
                cur = (cur + " " + s).strip()
            else:
                if cur:
                    chunks.append(cur)
                if self.overlap_chars > 0 and cur:
                    tail = cur[-self.overlap_chars :]
                    cur = (tail + " " + s).strip()
                else:
                    cur = s
        if cur:
            chunks.append(cur)
        return chunks

    def _create_page_based(self, text: str) -> List[str]:
        """
        Create chunks by page/slide unit

        Detects "[[Slide{number}]]" markers in text and
        groups and joins them by slide number.

        Args:
            text: Text containing "[[Slide{number}]] {text}" format

        Returns:
            List of text joined by slide (in slide number order)
        """
        lines = text.split("\n")
        slide_chunks = {}

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Extract "[[Slide{number}]]"
            match = re.search(r"\[\[Slide(\d+)\]\]", line)
            if match:
                slide_num = match.group(1)
                slide_text = line.replace(match.group(0), "").strip()
                if slide_num not in slide_chunks:
                    slide_chunks[slide_num] = []
                if slide_text:  # Don't add empty strings
                    slide_chunks[slide_num].append(slide_text)

        # Create list by joining per slide (in number order)
        chunks = [" ".join(texts) for slide_num, texts in sorted(slide_chunks.items(), key=lambda x: int(x[0]))]

        return chunks

    def create_with_section_metadata(self, sentences: List[dict]) -> List[str]:
        """
        Create chunks with section prefixes for Markdown files

        Args:
            sentences: List of sentences with optional "section" metadata
                       [{"text": "...", "section": "Header > Subheader"}, ...]

        Returns:
            List of chunks with [[Section: ...]] prefix at the beginning
        """
        if not sentences:
            return []

        # Build text with section markers
        text_with_markers = []
        for sent in sentences:
            section = sent.get("section", "")
            if section:
                text_with_markers.append(f"[[SECTION:{section}]]")
            text_with_markers.append(sent["text"])

        combined = " ".join(text_with_markers)

        # Use existing character-based chunking logic
        chunks = self._create_char_based(combined)

        # Post-process: convert [[SECTION:...]] markers to proper prefixes
        result = []
        for chunk in chunks:
            # Extract section from first marker in chunk
            match = re.search(r"\[\[SECTION:([^\]]+)\]\]", chunk)
            if match:
                section = match.group(1)
                # Remove all markers from chunk
                clean_chunk = re.sub(r"\[\[SECTION:[^\]]+\]\]\s*", "", chunk)
                result.append(f"[[Section: {section}]] {clean_chunk}")
            else:
                result.append(chunk)

        return result
