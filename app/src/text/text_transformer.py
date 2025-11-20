"""
TextTransformer: Text formatting class for text extracted from PDFs

Removes mid-line breaks from text extracted from PDFs, etc., and processes
line breaks only at sentence end markers (sentence finalization).

Features:
- Preserves empty lines as paragraph separators
- Preserves bullet point line heads as-is
- Absorbs English line-end hyphens: re-\nport → report
- Concatenates CJK characters without spaces, other characters with half-width spaces
- Finalizes sentences at sentence end markers (。．！？!?) + closing brackets (」』）】》"'")
"""

from typing import Any, Dict, List, Iterable
import re
import uuid

from langchain_core.runnables import Runnable
from langchain_core.runnables.config import RunnableConfig
from src.utils.logger import Logger


class TextTransformer(Runnable):
    """Text formatting class"""

    def __init__(self):
        self.logger = Logger("/workspace/app/log/text_transformer")
        # Definition of sentence end markers and closing brackets
        self.EOS_CHARS = '。．！？!?'
        self.CLOSE_CHARS = '」』）】》"\'"'
        self.EOS_PAT = re.compile(rf'[{re.escape(self.EOS_CHARS)}][{re.escape(self.CLOSE_CHARS)}]*')
        self.BULLET = re.compile(r'^(\d+[\.\)]|[A-Za-z]\)|[・•\-–—◇◆●■□①-⑩])\s*')

    def _is_cjk(self, ch: str) -> bool:
        """Determine if character is CJK (Chinese-Japanese-Korean) character"""
        if not ch:
            return False
        o = ord(ch)
        # Simple detection of Hiragana, Katakana, CJK Unified Ideographs, CJK Compatibility Ideographs, etc.
        return (
            0x3040 <= o <= 0x30FF or  # Hiragana/Katakana
            0x3400 <= o <= 0x4DBF or  # CJK Ext-A
            0x4E00 <= o <= 0x9FFF or  # CJK Unified
            0xF900 <= o <= 0xFAFF     # CJK Compatibility Ideographs
        )

    def _needs_space(self, a: str, b: str) -> bool:
        """CJK×CJK needs no space, otherwise insert half-width space"""
        return not (self._is_cjk(a) and self._is_cjk(b))

    def _flush_sentences(self, buf: str, out: List[str]) -> str:
        """
        Repeatedly extract "sentence end marker + closing bracket sequence" from buffer and add to out.
        Returns the remainder (incomplete sentence).
        """
        while True:
            m = self.EOS_PAT.search(buf)
            if not m:
                break
            j = m.end()
            sent = buf[:j].strip()
            if sent:
                out.append(sent)
            buf = buf[j:].lstrip()
        return buf

    def normalize_lines(self, texts: List[str]) -> List[str]:
        """
        Remove unnatural mid-line breaks from line arrays and insert line breaks only at sentence ends.

        Args:
            texts: List of sentence texts

        Returns:
            List of normalized sentence texts
        """
        out: List[str] = []
        buf: str = ""

        for text in texts:
            cur = text.strip()

            # Paragraph separator (empty line)
            if cur == "":
                if buf:
                    out.append(buf)
                    buf = ""
                continue

            # Output bullet point lines independently
            if self.BULLET.match(cur):
                if buf:
                    out.append(buf)
                    buf = ""
                out.append(cur)
                continue

            # Concatenation processing
            if not buf:
                buf = cur
            else:
                if buf.endswith('-'):  # re-\nport → report
                    buf = buf[:-1] + cur
                else:
                    buf += (" " if self._needs_space(buf[-1], cur[0]) else "") + cur

            # Finalize and output sentences where end markers appear
            temp_out = []
            remaining_buf = self._flush_sentences(buf, temp_out)

            # Add finalized sentences
            out.extend(temp_out)
            buf = remaining_buf

        # Output any remaining fragments (incomplete sentences) at the end
        if buf:
            out.append(buf)

        return out

    def transform(self, input_data: Dict) -> Dict:
        """
        Format input data and generate output data

        Args:
            input_data: Input data in DocTextChainOutput format

        Returns:
            Output data in formatted DocTextChainOutput format
        """
        sentences = input_data.get("sentences", [])

        # Prepare text for each sentence
        texts = [sentence.get("text", "") for sentence in sentences]

        # Normalize text
        normalized_texts = self.normalize_lines(texts)

        # Convert normalized results to output format
        new_sentences = []
        for normalized_text in normalized_texts:
            if normalized_text.strip():  # Skip empty strings
                new_sentences.append({
                    "text": normalized_text
                })

        # Build output data
        output_data = {
            "id": input_data.get("id", str(uuid.uuid4())),
            "file_name": input_data.get("file_name", ""),
            "sentences": new_sentences
        }

        return output_data

    def invoke(self, input: Dict[str, Any]) -> Dict[str, Any]:
        """
        LangChain Runnable invoke method implementation

        Args:
            input: Input data in TextTransformerInput format
            config: RunnableConfig (optional)

        Returns:
            Output data in TextTransformerOutput format
        """
        # Call existing transform method
        output = self.transform(input)

        # Save output JSON to log
        self.logger.save_log(output, "text_transformer_output_")

        return output


if __name__ == "__main__":
    # Sample data for testing
    sample_data = {
        "id": "test-001",
        "file_name": "sample.pdf",
        "sentences": [
            {"text": "これは途中で改行された"},
            {"text": "文章の例です。次の文も"},
            {"text": "同様に途中で改行されて"},
            {"text": "います！"},
            {"text": ""},
            {"text": "• 箇条書きの項目1"},
            {"text": "• 箇条書きの項目2"},
            {"text": ""},
            {"text": "英語のハイフン処理 re-"},
            {"text": "port example です。"}
        ]
    }

    transformer = TextTransformer()
    result = transformer.transform(sample_data)

    print("=== TextTransformer テスト結果 ===")
    print(f"入力文章数: {len(sample_data['sentences'])}")
    print(f"出力文章数: {len(result['sentences'])}")
    print()
    print("Sentences:")
    for i, sentence in enumerate(result['sentences'], 1):
        print(f"  {i}. Text: '{sentence['text']}'")