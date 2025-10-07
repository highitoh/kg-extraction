# doc_text_filter.py (修正版: 目次・ページ番号を削除)

from __future__ import annotations
from typing import Any, Dict, List, Tuple
import re
import uuid

from langchain.schema.runnable import Runnable
from langchain_core.runnables.config import RunnableConfig

try:
    from logger import Logger
except Exception:
    class Logger:
        def __init__(self, *_args, **_kwargs): ...
        def save_log(self, *_args, **_kwargs): ...
        def info(self, *_args, **_kwargs): ...
        def error(self, *_args, **_kwargs): ...

try:
    from chunk_creator import ChunkCreator
except Exception:
    ChunkCreator = None


class DocTextFilter(Runnable):
    BULLET_PATTERNS = [
        r"^\s*[\u2022\u30fb\-\–\—\*▶・•]\s+",
        r"^\s*\(\d+\)\s+",
        r"^\s*\d+[\.\)]\s+",
        r"^\s*[a-zA-Z][\.\)]\s+",
        r"^\s*[①-⑳]\s+",
        r"^\s*-\s+\[\s?[xX]?\s?\]\s+",
    ]
    BULLET_RE = re.compile("|".join(BULLET_PATTERNS))
    SENT_END_RE = re.compile(r"[。！？!?](?:[)”\]\}』」】〉》]*)\s*$")

    # === 追加: 除外パターン（目次・ページ番号） ===
    EXCLUDE_PATTERNS = [
        re.compile(r"^\s*目\s*次\s*$"),          # 「目 次」
        re.compile(r"^\s*別\d+-\d+\s*$"),       # 「別8-1」「別8-5」など
    ]

    def __init__(self):
        self.logger = Logger("./log/doc_text_filter")

    def invoke(self, input: Dict[str, Any], config: RunnableConfig = None) -> Dict[str, Any]:
        self._validate_input_minimally(input)
        src = input["source"]
        params = input.get("filter", {}).get("params", {}) or {}

        file_name, lines = self._collect_lines(src)
        max_chars = int(params.get("max_chars", 0)) or 0
        if ChunkCreator and max_chars > 0:
            chunks = ChunkCreator(max_chars=max_chars).create("\n".join(lines))
            sentences: List[str] = []
            for ch in chunks:
                chunk_lines = ch.splitlines()
                sentences.extend(self._lines_to_sentences(chunk_lines))
        else:
            sentences = self._lines_to_sentences(lines)

        output = {
            "id": str(uuid.uuid4()),
            "file_name": file_name,
            "sentences": [{"text": s} for s in sentences if s.strip()]
        }
        try:
            self.logger.save_log(output, "doc_text_filter_output_")
        except Exception:
            pass
        return output

    def _collect_lines(self, src: Dict[str, Any]) -> Tuple[str, List[str]]:
        file_name = src.get("file_name", src.get("filename", "unknown"))
        sents = src.get("sentences", [])
        lines = []
        for s in sents:
            t = (s.get("text") if isinstance(s, dict) else str(s)).rstrip("\n")
            if t is not None:
                lines.append(t)
        return file_name, lines

    def _lines_to_sentences(self, lines: List[str]) -> List[str]:
        sentences: List[str] = []
        i = 0
        N = len(lines)

        while i < N:
            line = self._normalize_inline_whitespace(lines[i])

            # --- 除外パターンの判定 ---
            if any(p.match(line) for p in self.EXCLUDE_PATTERNS):
                i += 1
                continue

            if not line.strip():
                i += 1
                continue

            if self._is_bullet(line):
                sentence, jump = self._consume_bullet_item(lines, i)
                sentences.append(sentence)
                i = jump
                continue

            buf = [line]
            i += 1
            while i < N:
                nxt = self._normalize_inline_whitespace(lines[i])
                if not nxt.strip():
                    break
                if self._is_bullet(nxt):
                    break
                if not self._is_sentence_end(buf[-1]):
                    buf[-1] = self._smart_join(buf[-1], nxt)
                else:
                    break
                i += 1
            sentences.append(self._final_clean("".join(buf)))
        return sentences

    def _is_bullet(self, text: str) -> bool:
        return bool(self.BULLET_RE.search(text))

    def _consume_bullet_item(self, lines: List[str], start_idx: int) -> Tuple[str, int]:
        i = start_idx
        N = len(lines)
        head = self._normalize_inline_whitespace(lines[i])
        buf = [head]
        i += 1
        while i < N:
            cur = self._normalize_inline_whitespace(lines[i])
            if not cur.strip():
                break
            if self._is_bullet(cur):
                break
            buf.append(self._soft_wrap_join(buf.pop(), cur))
            i += 1
        return self._final_clean("".join(buf)), i

    def _is_sentence_end(self, text: str) -> bool:
        return bool(self.SENT_END_RE.search(text))

    def _normalize_inline_whitespace(self, text: str) -> str:
        t = text.replace("\u00AD", "")
        t = re.sub(r"[ \t]{2,}", " ", t)
        return t.strip("\r")

    def _smart_join(self, prev: str, cur: str) -> str:
        if prev.endswith("-"):
            return prev[:-1] + cur.lstrip()
        if cur[:1] in "、。，．）」』】〉》" or prev.endswith(("（", "「", "『", "【", "〈", "《")):
            return prev + cur
        if (prev.endswith(" ") or cur.startswith(" ")):
            return prev + cur
        return prev + " " + cur

    def _soft_wrap_join(self, prev: str, cur: str) -> str:
        if prev.endswith("-"):
            return prev[:-1] + cur.lstrip()
        if prev.endswith(("（", "「", "『", "【", "〈", "《")):
            return prev + cur
        if cur[:1] in "、。，．）」』】〉》":
            return prev + cur
        if prev.endswith(" ") or cur.startswith(" "):
            return prev + cur
        return prev + " " + cur

    def _final_clean(self, text: str) -> str:
        return text.rstrip()

    def _validate_input_minimally(self, input: Dict[str, Any]) -> None:
        if "source" not in input or "filter" not in input:
            raise ValueError("Input must include 'source' and 'filter'.")
        src = input["source"]
        if not isinstance(src, dict) or "sentences" not in src:
            raise ValueError("source must be an object containing 'sentences'.")
        if not isinstance(src["sentences"], list):
            raise ValueError("source.sentences must be a list.")



if __name__ == "__main__":
    # ダミー入力データ（TextExtractorの出力を模倣）
    source = {
        "file_name": "dummy.pdf",
        "sentences": [
            {"text": "これは1行目の文章です。"},
            {"text": "これは2行目で"},
            {"text": "途中で改行されています。"},
            {"text": "・箇条書きの最初の項目です"},
            {"text": "折返し行が続きます"},
            {"text": "・箇条書きの2つ目"},
            {"text": "最後の通常文です。"}
        ]
    }

    # DocTextFilterを初期化
    filter_instance = DocTextFilter()

    # フィルタを実行
    result = filter_instance.invoke({
        "source": source,
        "filter": {"name": "sentence_extraction", "params": {}}
    })

    # 出力確認
    print("=== DocTextFilter 出力結果 ===")
    for idx, s in enumerate(result["sentences"], 1):
        print(f"{idx:02d}: {s['text']}")
