"""Relevance screening against a project profile.

Drops the content that is not about the project (another organization's own
offerings, another project) before view extraction. Each LLM call sees a whole
segment - a Miro region, a window of consecutive text chunks, an image or a
table - so that an element naming no one is judged by the subject of the
segment it sits in, while the judgment itself is made per element.
"""

import asyncio
import csv
import io
import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

from langchain_core.messages import HumanMessage
from langchain_core.runnables import Runnable

from src.utils.chunk_creator import ChunkCreator
from src.utils.llm_factory import LLMFactory
from src.utils.llm_retry import call_with_retry
from src.utils.logger import Logger
from src.utils.mermaid_parser import is_structural_diagram
from src.utils.project_profile import render_profile
from src.utils.prompt_utils import fill_prompt_template
from src.utils.table_structure import render_table_context

# Section prefix that Markdown chunks start with (ChunkCreator.create_with_section_metadata)
_SECTION_PREFIX_RE = re.compile(r"^\s*\[\[Section: ([^\]]*)\]\]\s*")


def is_image_chunk(chunk: Dict[str, Any]) -> bool:
    """Whether a merged chunk came from the image branch."""
    return (chunk.get("source") or {}).get("kind") == "image"


def table_outline(block: Dict[str, Any]) -> str:
    """A table as one element: its headers, which is what class extraction reads from it."""
    table = block.get("table") or {}
    rows = [h.strip() for h in table.get("row_headers") or [] if (h or "").strip()]
    cols = [h.strip() for h in table.get("col_headers") or [] if (h or "").strip()]
    return f"Table with rows: {', '.join(rows) or '-'} / columns: {', '.join(cols) or '-'}"


@dataclass
class _ChunkParts:
    """A chunk taken apart into the elements that are judged."""

    elements: List[str]
    joiner: str = " "  # How the kept elements are joined back into the chunk text
    section: str = ""  # Section the chunk sits in (Markdown)
    prefix: str = ""  # Section prefix of the chunk text, put back in front of what is kept


@dataclass
class _Segment:
    """The elements one LLM call judges."""

    file_id: str
    context: str = ""
    elements: List[Tuple[str, str]] = field(default_factory=list)  # (section, text), unique by text

    def add(self, text: str, section: str = "") -> None:
        # A sentence repeated by chunk overlap is judged once
        if text and all(text != t for _, t in self.elements):
            self.elements.append((section, text))


class RelevanceScreener(Runnable):
    """
    Task to drop content that is not about the project described by a project profile
    Input: {"profile": dict, "source_type": str, "file_id": str, "chunks": [...], "structured_blocks": [...]}
    Output: {"chunks": [...], "structured_blocks": [...], "kept_sentences": [...], "screening": {...}}

    An element without a valid judgment is kept: screening never loses content
    because of an unreadable answer.
    """

    def __init__(
        self,
        model: str | None = None,
        temperature: float = 1.0,
        max_concurrency: int = 8,
        segment_chars: int = 2000,
        progress: bool = True,
        log_dir: str = "/workspace/app/log/relevance_screener",
    ):
        """
        Args:
            model: LLM model name (if None, use the default model)
            temperature: LLM temperature
            max_concurrency: Concurrent LLM calls
            segment_chars: Size of a window of consecutive text chunks judged together (non-Miro sources)
            progress: Progress display flag
            log_dir: Log output directory
        """
        self.llm = LLMFactory.create_llm(model=model, temperature=temperature, reasoning_effort="minimal")
        schema = self._load_schema()
        self.llm_json = self.llm.bind(
            response_format={"type": "json_schema", "json_schema": {"name": "relevance_screener", "schema": schema}}
        )
        self.prompt = self._load_prompt()
        self.max_concurrency = max_concurrency
        self.segment_chars = segment_chars
        self.progress = progress
        self.chunk_creator = ChunkCreator()
        self.logger = Logger(log_dir)

    @staticmethod
    def _app_dir() -> str:
        # Go up 4 levels from src/extract/relevance/relevance_screener.py to /workspace/app/
        return os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

    def _load_schema(self) -> dict:
        """Load JSON schema file"""
        schema_path = os.path.join(self._app_dir(), "schemas", "relevance-screener", "llm.schema.json")
        with open(schema_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _load_prompt(self) -> str:
        """Load prompt file"""
        prompt_path = os.path.join(self._app_dir(), "prompts", "relevance_screener.txt")
        with open(prompt_path, "r", encoding="utf-8") as f:
            return f.read()

    @staticmethod
    def _to_text(c: Any) -> str:
        """Convert LLM message to text"""
        if isinstance(c, str):
            return c
        for part in c:
            if isinstance(part, dict) and part.get("type") in ("output_text", "text"):
                return part.get("text", "")
        return ""

    # ---------- Taking chunks apart ----------

    def _split_chunk(self, chunk: Dict[str, Any]) -> _ChunkParts:
        """Split a chunk into elements, the same way ViewExtractor splits it into sentences."""
        text = chunk.get("text", "") or ""
        # Image: the whole description is one element
        if is_image_chunk(chunk):
            return _ChunkParts(elements=[text] if text.strip() else [])
        # Miro board items: one element per item
        if chunk.get("elements") is not None:
            return _ChunkParts(elements=list(chunk["elements"]))

        # Markdown: set the section prefix aside
        section, prefix = "", ""
        match = _SECTION_PREFIX_RE.match(text)
        if match:
            section, prefix = match.group(1), match.group(0).strip()
            text = text[match.end() :]

        # Diagram text: one element per line; other text: one element per sentence
        if is_structural_diagram(text):
            return _ChunkParts(self.chunk_creator.split_at_line_boundaries(text), "\n", section, prefix)
        return _ChunkParts(self.chunk_creator._split_into_sentences(text), " ", section, prefix)

    def _build_segments(
        self, chunks: List[Dict[str, Any]], parts: List[_ChunkParts], source_type: str
    ) -> Tuple[List[_Segment], List[int]]:
        """Group chunks into segments.

        A Miro region is one segment, together with its rendered image. For other
        sources, consecutive text chunks are windowed up to segment_chars (their
        chunks are small and rarely hold their own subject), and each image is a
        segment of its own.

        Returns:
            (segments, segment index of each chunk)
        """
        segments: List[_Segment] = []
        segment_of: List[int] = []

        if source_type == "miro_board":
            by_file_id: Dict[str, int] = {}
            for chunk in chunks:
                file_id = chunk.get("file_id", "")
                if file_id not in by_file_id:
                    by_file_id[file_id] = len(segments)
                    segments.append(_Segment(file_id=file_id))
                segment = segments[by_file_id[file_id]]
                if chunk.get("title") and not segment.context:
                    segment.context = f"Frame title: {chunk['title']}"
                segment_of.append(by_file_id[file_id])
        else:
            window, window_chars = -1, 0
            for chunk in chunks:
                if is_image_chunk(chunk):
                    segment_of.append(len(segments))
                    segments.append(_Segment(file_id=chunk.get("file_id", "")))
                    window = -1
                    continue
                size = len(chunk.get("text", "") or "")
                if window < 0 or window_chars + size > self.segment_chars:
                    window, window_chars = len(segments), 0
                    segments.append(_Segment(file_id=chunk.get("file_id", "")))
                segment_of.append(window)
                window_chars += size

        for chunk_parts, index in zip(parts, segment_of, strict=True):
            for text in chunk_parts.elements:
                segments[index].add(text, chunk_parts.section)

        return segments, segment_of

    # ---------- Judging ----------

    @staticmethod
    def _format_elements(segment: _Segment) -> Tuple[str, Dict[str, str]]:
        """Format a segment's elements as CSV; returns (csv, {element id: text})."""
        output = io.StringIO()
        writer = csv.writer(output, quoting=csv.QUOTE_MINIMAL)
        writer.writerow(["id", "section", "text"])
        id_map = {}
        for i, (section, text) in enumerate(segment.elements):
            element_id = f"e{i}"
            writer.writerow([element_id, section, text])
            id_map[element_id] = text
        return output.getvalue(), id_map

    async def _judge_segment(self, segment: _Segment, profile_text: str) -> Dict[str, Tuple[str, str]]:
        """Judge one segment; returns {element text: (label, reason)} for the valid judgments."""
        elements_csv, id_map = self._format_elements(segment)
        prompt_text = fill_prompt_template(
            self.prompt,
            project_profile=profile_text,
            segment_context=segment.context or "(none)",
            elements_csv=elements_csv,
        )
        msg = HumanMessage(content=[{"type": "text", "text": prompt_text}])
        ai = await call_with_retry(
            lambda: self.llm_json.ainvoke([msg]),
            timeout=90.0,
            label="RelevanceScreener",
        )

        try:
            data = json.loads(self._to_text(ai.content))
        except json.JSONDecodeError:
            return {}
        judgments = data.get("judgments", []) if isinstance(data, dict) else []

        judged: Dict[str, Tuple[str, str]] = {}
        for judgment in judgments if isinstance(judgments, list) else []:
            if not isinstance(judgment, dict):
                continue
            text = id_map.get(judgment.get("id", ""))
            label = judgment.get("label")
            if text is None or label not in ("keep", "drop"):
                continue
            judged[text] = (label, (judgment.get("reason") or "").strip())
        return judged

    async def _judge_all(self, segments: List[_Segment], profile_text: str) -> List[Dict[str, Tuple[str, str]]]:
        """Judge all segments in parallel."""
        sem = asyncio.Semaphore(self.max_concurrency)
        total = sum(1 for s in segments if s.elements)
        counter = 0
        lock = asyncio.Lock()

        if self.progress:
            print(f"[RelevanceScreener] Starting parallel processing: {total} segments")

        async def _wrapped(segment: _Segment) -> Dict[str, Tuple[str, str]]:
            nonlocal counter
            if not segment.elements:
                return {}
            async with sem:
                judged = await self._judge_segment(segment, profile_text)
            async with lock:
                counter += 1
                if self.progress:
                    drops = sum(1 for label, _ in judged.values() if label == "drop")
                    print(
                        f"[{counter}/{total}] RelevanceScreener: {segment.file_id or 'segment'} "
                        f"dropped {drops}/{len(segment.elements)}"
                    )
            return judged

        return await asyncio.gather(*[_wrapped(s) for s in segments])

    # ---------- Putting chunks back together ----------

    def invoke(self, input: Dict[str, Any], config=None) -> Dict[str, Any]:
        profile = input["profile"]
        source_type = input.get("source_type", "")
        fallback_file_id = input.get("file_id", "")
        chunks: List[Dict[str, Any]] = input.get("chunks", []) or []
        blocks: List[Dict[str, Any]] = input.get("structured_blocks", []) or []

        parts = [self._split_chunk(chunk) for chunk in chunks]
        segments, segment_of = self._build_segments(chunks, parts, source_type)

        table_segment_of: Dict[int, int] = {}
        for block_index, block in enumerate(blocks):
            if block.get("block_type") != "table":
                continue
            table_segment_of[block_index] = len(segments)
            segment = _Segment(file_id=block.get("file_id", ""), context=render_table_context(block))
            segment.add(table_outline(block))
            segments.append(segment)

        judged_by_segment = asyncio.run(self._judge_all(segments, render_profile(profile)))

        def is_dropped(segment_index: int, text: str) -> bool:
            return judged_by_segment[segment_index].get(text, ("keep", ""))[0] == "drop"

        # Chunks: rebuild from the kept elements, drop the ones with nothing left
        kept_chunks: List[Dict[str, Any]] = []
        kept_sentences: List[Dict[str, Any]] = []
        for chunk, chunk_parts, index in zip(chunks, parts, segment_of, strict=True):
            kept = [text for text in chunk_parts.elements if not is_dropped(index, text)]
            if len(kept) == len(chunk_parts.elements):
                kept_chunks.append(chunk)
            elif kept:
                rebuilt = dict(chunk)
                body = chunk_parts.joiner.join(kept)
                rebuilt["text"] = f"{chunk_parts.prefix} {body}" if chunk_parts.prefix else body
                if "elements" in chunk:
                    rebuilt["elements"] = kept
                kept_chunks.append(rebuilt)

            if not is_image_chunk(chunk):
                for text in kept:
                    if not kept_sentences or kept_sentences[-1]["text"] != text:
                        kept_sentences.append({"text": text})

        kept_blocks = [
            block
            for block_index, block in enumerate(blocks)
            if block_index not in table_segment_of
            or not is_dropped(table_segment_of[block_index], table_outline(block))
        ]

        # Summary, counted once per element of each segment
        dropped: List[Dict[str, str]] = []
        kept_count = unjudged = 0
        log_segments = []
        for segment, judged in zip(segments, judged_by_segment, strict=True):
            log_judgments = []
            for _, text in segment.elements:
                label, reason = judged.get(text, ("keep", ""))
                if text not in judged:
                    unjudged += 1
                if label == "drop":
                    dropped.append({"file_id": segment.file_id or fallback_file_id, "text": text, "reason": reason})
                else:
                    kept_count += 1
                log_judgments.append({"text": text, "label": label, "reason": reason, "judged": text in judged})
            log_segments.append({"file_id": segment.file_id, "context": segment.context, "judgments": log_judgments})

        screening = {
            "segments": sum(1 for s in segments if s.elements),
            "elements": kept_count + len(dropped),
            "kept": kept_count,
            "unjudged": unjudged,
            "dropped": dropped,
        }

        if self.progress:
            print(
                f"RelevanceScreener: kept {kept_count}/{screening['elements']} elements "
                f"({unjudged} unjudged), chunks {len(chunks)} -> {len(kept_chunks)}, "
                f"tables {len(blocks)} -> {len(kept_blocks)}"
            )

        self.logger.save_log(
            {"segments": log_segments, "screening": screening}, filename_prefix="relevance_screener_output_"
        )

        return {
            "chunks": kept_chunks,
            "structured_blocks": kept_blocks,
            "kept_sentences": kept_sentences,
            "screening": screening,
        }
