"""
TextTransformer: PDFから抽出されたテキストの整形クラス

PDF等から抽出したテキストの「途中改行」を除去し、
文末記号が出現した地点でのみ改行（文確定）する処理を行います。

特長:
- 空行は段落区切りとして保持
- 箇条書き行頭はそのまま保持
- 英語の行末ハイフン re-\nport → report を吸収
- CJK同士は無空白で連結、それ以外は半角スペースで連結
- 文末記号（。．！？!?）＋閉じ記号（」』）】》"'"）までを文として確定
"""

from typing import Any, Dict, List, Iterable
import re
import uuid

from langchain.schema.runnable import Runnable
from langchain_core.runnables.config import RunnableConfig
from logger import Logger


class TextTransformer(Runnable):
    """テキスト整形クラス"""

    def __init__(self):
        self.logger = Logger("./log/text_transformer")
        # 文末記号と閉じ記号の定義
        self.EOS_CHARS = '。．！？!?'
        self.CLOSE_CHARS = '」』）】》"\'"'
        self.EOS_PAT = re.compile(rf'[{re.escape(self.EOS_CHARS)}][{re.escape(self.CLOSE_CHARS)}]*')
        self.BULLET = re.compile(r'^(\d+[\.\)]|[A-Za-z]\)|[・•\-–—◇◆●■□①-⑩])\s*')

    def _is_cjk(self, ch: str) -> bool:
        """文字がCJK（中日韓）文字かどうかを判定"""
        if not ch:
            return False
        o = ord(ch)
        # ひらがな・カタカナ・CJK統合漢字・CJK互換漢字等の簡易判定
        return (
            0x3040 <= o <= 0x30FF or  # Hiragana/Katakana
            0x3400 <= o <= 0x4DBF or  # CJK Ext-A
            0x4E00 <= o <= 0x9FFF or  # CJK Unified
            0xF900 <= o <= 0xFAFF     # CJK Compatibility Ideographs
        )

    def _needs_space(self, a: str, b: str) -> bool:
        """CJK×CJKはスペース不要、それ以外は半角スペースを挿入"""
        return not (self._is_cjk(a) and self._is_cjk(b))

    def _flush_sentences(self, buf: str, out: List[str]) -> str:
        """
        バッファから「文末記号＋閉じ記号列」までを繰り返し切り出して out に積む。
        残り（未完文）を返す。
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
        行列から不自然な途中改行を除去し、文末でのみ改行する。

        Args:
            texts: 文章テキストのリスト

        Returns:
            正規化された文章テキストのリスト
        """
        out: List[str] = []
        buf: str = ""

        for text in texts:
            cur = text.strip()

            # 段落区切り（空行）
            if cur == "":
                if buf:
                    out.append(buf)
                    buf = ""
                continue

            # 箇条書き行は独立で出力
            if self.BULLET.match(cur):
                if buf:
                    out.append(buf)
                    buf = ""
                out.append(cur)
                continue

            # 連結処理
            if not buf:
                buf = cur
            else:
                if buf.endswith('-'):  # re-\nport → report
                    buf = buf[:-1] + cur
                else:
                    buf += (" " if self._needs_space(buf[-1], cur[0]) else "") + cur

            # 文末が出現した分は確定して出力
            temp_out = []
            remaining_buf = self._flush_sentences(buf, temp_out)

            # 確定した文章を追加
            out.extend(temp_out)
            buf = remaining_buf

        # 端数（未完文）が残っていれば末尾に出力
        if buf:
            out.append(buf)

        return out

    def transform(self, input_data: Dict) -> Dict:
        """
        入力データを整形して出力データを生成

        Args:
            input_data: PDFTextChainOutputフォーマットの入力データ

        Returns:
            整形されたPDFTextChainOutputフォーマットの出力データ
        """
        sentences = input_data.get("sentences", [])

        # 各文章のテキストを準備
        texts = [sentence.get("text", "") for sentence in sentences]

        # テキストを正規化
        normalized_texts = self.normalize_lines(texts)

        # 正規化された結果を出力形式に変換
        new_sentences = []
        for normalized_text in normalized_texts:
            if normalized_text.strip():  # 空文字列はスキップ
                new_sentences.append({
                    "text": normalized_text
                })

        # 出力データを構築
        output_data = {
            "id": input_data.get("id", str(uuid.uuid4())),
            "file_name": input_data.get("file_name", ""),
            "sentences": new_sentences
        }

        return output_data

    def invoke(self, input: Dict[str, Any]) -> Dict[str, Any]:
        """
        LangChain Runnableのinvokeメソッド実装

        Args:
            input: TextTransformerInputフォーマットの入力データ
            config: RunnableConfig（オプション）

        Returns:
            TextTransformerOutputフォーマットの出力データ
        """
        # 既存のtransformメソッドを呼び出し
        output = self.transform(input)

        # 出力JSONをログに保存
        self.logger.save_log(output, "text_transformer_output_")

        return output


if __name__ == "__main__":
    # テスト用のサンプルデータ
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