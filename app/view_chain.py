import os
import uuid
import json
import asyncio
import logging
from typing import Dict, Any, List, Union, Tuple
from langchain.schema.runnable import Runnable, RunnableSequence
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from .ChunkCreator import ChunkCreator
from .logger import Logger


class ViewExtractor(Runnable):
    """
    ビュー記述を抽出するタスク
    入力: ViewChainInput
    出力: ViewChainOutput
    """

    def __init__(
        self,
        llm: ChatOpenAI | None = None,
        model: str = "gpt-5-nano",
        temperature: float = 0.0,
        max_concurrency: int = 8,
        progress: bool = True,
        max_spans_per_label: int = 3,
        log_dir: str = "log/view_extractor",
        chunk_max_chars: int = 1600,
        chunk_overlap_chars: int = 200,
    ):
        self.llm = llm or ChatOpenAI(
            model=model,
            temperature=temperature,
            reasoning={"effort": "minimal"},
            output_version="responses/v1",
        )
        self.llm_json = self.llm.bind(response_format={"type": "json_object"})
        self.max_concurrency = max_concurrency
        self.progress = progress
        self.max_spans_per_label = max_spans_per_label

        # ChunkCreatorを初期化
        self.chunk_creator = ChunkCreator(max_chars=chunk_max_chars, overlap_chars=chunk_overlap_chars)

        # プロンプトファイルを読み込み
        self.prompt = self._load_prompt()

        # 対象ビュー
        self.views = [
            "value_analysis",
            "business_concept",
            "business_requirement",
            "system_requirement",
            "system_component",
            "data_analysis",
        ]

        # Loggerを初期化
        self.logger = Logger(log_dir)

    def _load_prompt(self) -> str:
        """プロンプトファイルを読み込む"""
        prompt_path = os.path.join(os.path.dirname(__file__), "prompts", "view_extractor.txt")
        with open(prompt_path, "r", encoding="utf-8") as f:
            return f.read()

    def _combine_sentences(self, sentences: List[Dict[str, Any]]) -> str:
        """ViewChainInputのsentencesを統合してテキストにする"""
        return " ".join([sentence["text"] for sentence in sentences])

    @staticmethod
    def _to_text(c: Union[str, List[Any]]) -> str:
        """LLMメッセージをテキストに変換"""
        if isinstance(c, str):
            return c
        for part in c:
            if isinstance(part, dict) and part.get("type") in ("output_text", "text"):
                return part.get("text", "")
        return ""

    async def _extract_spans_for_chunk(self, chunk: str) -> Dict[str, List[str]]:
        """1チャンク分のビュー別スパン抽出"""
        msg = HumanMessage(
            content=[
                {"type": "text", "text": self.prompt},
                {"type": "text", "text": f"# 対象テキスト\n{chunk}"},
            ]
        )
        ai = await self.llm_json.ainvoke([msg])
        raw = self._to_text(ai.content)
        try:
            data = json.loads(raw)
        except Exception:
            data = {}

        # 正規化（既知キーのみ、重複除去、最大件数制限）
        spans: Dict[str, List[str]] = {k: [] for k in self.views}
        for k in self.views:
            v = data.get(k, [])
            if not isinstance(v, list):
                continue
            seen = set()
            out = []
            for s in v:
                if not isinstance(s, str):
                    continue
                s2 = s.strip()
                if not s2 or s2 in seen:
                    continue
                seen.add(s2)
                out.append(s2)
                if len(out) >= self.max_spans_per_label:
                    break
            spans[k] = out
        return spans

    async def _process_chunks_parallel(self, chunks: List[str]) -> List[Dict[str, Any]]:
        """チャンクを並列処理でビュー抽出"""
        sem = asyncio.Semaphore(self.max_concurrency)
        total = len(chunks)
        counter = 0
        lock = asyncio.Lock()

        async def _wrapped(i: int, c: str) -> Dict[str, Any]:
            nonlocal counter
            async with sem:
                spans = {k: [] for k in self.views}
                for attempt in range(5):
                    try:
                        spans = await self._extract_spans_for_chunk(c)
                        break
                    except Exception:
                        await asyncio.sleep(min(2 ** attempt, 10))
                async with lock:
                    counter += 1
                    total_spans = sum(len(v) for v in spans.values())
                    if self.progress:
                        print(f"[{counter}/{total}] done (chunk {i}, total_spans={total_spans})")
                return {"index": i, "spans": spans, "chunk": c}

        results = await asyncio.gather(*[_wrapped(i, c) for i, c in enumerate(chunks)])
        return results

    def invoke(self, input: Dict[str, Any], config=None) -> Dict[str, Any]:
        target = input["target"]
        sentences: List[Dict[str, Any]] = target["sentences"]
        file_id = target.get("file_id", str(uuid.uuid4()))

        # ViewChainInputのテキストを統合
        combined_text = self._combine_sentences(sentences)

        # チャンクに分割
        chunks = self.chunk_creator.create(combined_text)

        # 並列実行でビュー抽出
        chunk_results = asyncio.run(self._process_chunks_parallel(chunks))

        # 結果をViewChainOutput形式に変換
        views = []
        for view_type in self.views:
            texts = []
            for result in chunk_results:
                spans = result["spans"].get(view_type, [])
                for span in spans:
                    # 元の文からlines番号を推定（簡易実装）
                    matching_sentence = next(
                        (s for s in sentences if span in s["text"]),
                        sentences[0] if sentences else {"lines": [1]}
                    )
                    texts.append({
                        "file_id": file_id,
                        "lines": matching_sentence.get("lines", [1]),
                        "text": span,
                    })

            if texts:  # 抽出されたテキストがある場合のみビューを追加
                views.append({
                    "type": view_type,
                    "texts": texts,
                })

        output: Dict[str, Any] = {
            "id": str(uuid.uuid4()),
            "views": views,
        }

        # Loggerを使ってViewChainOutputをログ出力
        self.logger.save_log(output)

        return output


class ViewFilter(Runnable):
    """
    ビュー記述抽出結果にフィルタ処理を適用するタスク
    入力: ViewChainOutput
    出力: ViewChainOutput（同一スキーマ）

    DuplicateSentenceRemovalRuleを適用して重複文章を削除
    """

    def __init__(self, progress: bool = True, log_dir: str = "log/view_filter"):
        self.progress = progress
        # Loggerを初期化
        self.logger = Logger(log_dir)

    def _remove_duplicate_sentences(self, views: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        ビュー間で重複する文章を削除
        - 完全一致または包含関係にある場合に重複と判定
        - 完全一致の場合は先の文章を残し、包含の場合は長い方を残す
        """
        if not views:
            return views

        # ビューごとにテキストをグループ化
        view_text_map: Dict[str, List[Tuple[int, int, Dict[str, Any]]]] = {}

        for view_idx, view in enumerate(views):
            view_type = view.get("type", "unknown")
            texts = view.get("texts", [])

            if view_type not in view_text_map:
                view_text_map[view_type] = []

            for text_idx, text_obj in enumerate(texts):
                view_text_map[view_type].append((view_idx, text_idx, text_obj))

        # 各ビュー内で重複を検出・削除
        to_remove: List[Tuple[int, int]] = []  # (view_index, text_index)

        for view_type, text_list in view_text_map.items():
            if len(text_list) < 2:
                continue

            i = 0
            while i < len(text_list):
                view_idx1, text_idx1, text_obj1 = text_list[i]
                text1 = text_obj1.get("text", "")

                j = i + 1
                while j < len(text_list):
                    view_idx2, text_idx2, text_obj2 = text_list[j]
                    text2 = text_obj2.get("text", "")

                    # 重複判定：完全一致または包含関係
                    is_duplicate = False
                    text_to_remove = None

                    if text1 == text2:
                        # 完全一致の場合は後の文章を削除
                        is_duplicate = True
                        text_to_remove = (view_idx2, text_idx2)
                    elif text1 in text2:
                        # text1がtext2に包含されている場合、短い方(text1)を削除
                        is_duplicate = True
                        text_to_remove = (view_idx1, text_idx1)
                    elif text2 in text1:
                        # text2がtext1に包含されている場合、短い方(text2)を削除
                        is_duplicate = True
                        text_to_remove = (view_idx2, text_idx2)

                    if is_duplicate and text_to_remove:
                        to_remove.append(text_to_remove)
                        if self.progress:
                            print(f"Removing duplicate text from {view_type}: {text_to_remove[0]}.{text_to_remove[1]}")

                        # 削除対象のレコードをリストから除去
                        if text_to_remove == (view_idx1, text_idx1):
                            text_list.pop(i)
                            j = i  # iの位置が変わったのでjをリセット
                            break
                        else:
                            text_list.pop(j)
                            continue

                    j += 1
                i += 1

        # to_removeをソートして後ろから削除（インデックスのずれを防ぐ）
        to_remove = sorted(set(to_remove), key=lambda x: (x[0], x[1]), reverse=True)

        # 新しいviewsリストを作成（重複テキストを除去）
        filtered_views = []
        for view_idx, view in enumerate(views):
            texts = view.get("texts", [])
            filtered_texts = []

            for text_idx, text_obj in enumerate(texts):
                if (view_idx, text_idx) not in to_remove:
                    filtered_texts.append(text_obj)

            # テキストがある場合のみビューを追加
            if filtered_texts:
                filtered_view = view.copy()
                filtered_view["texts"] = filtered_texts
                filtered_views.append(filtered_view)

        return filtered_views

    def invoke(self, input: Dict[str, Any], config=None) -> Dict[str, Any]:
        """ViewChainOutputの重複文章を削除してフィルタリング"""
        views = input.get("views", [])

        # 重複文章を削除
        filtered_views = self._remove_duplicate_sentences(views)

        # 結果を返す
        output = input.copy()
        output["views"] = filtered_views

        if self.progress:
            original_count = sum(len(v.get("texts", [])) for v in views)
            filtered_count = sum(len(v.get("texts", [])) for v in filtered_views)
            print(f"ViewFilter: {original_count} -> {filtered_count} texts (removed {original_count - filtered_count} duplicates)")

        # Loggerを使ってViewChainOutputをログ出力
        self.logger.save_log(output)

        return output


def create_view_chain() -> RunnableSequence:
    """
    ビュー記述抽出チェイン:
      ViewExtractor -> ViewFilter -> ViewChainLogger
    いずれもテンプレート（抽出/フィルタの中身は未実装）
    """
    extractor = ViewExtractor()
    view_filter = ViewFilter()
    return RunnableSequence(steps=[extractor, view_filter])

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    chain = create_view_chain()
    sample_input = {
        "target": {
            "id": "pdf-text-id-001",
            "file_id": "file-abc",
            "file_name": "sample.pdf",
            "sentences": [
                {"lines": [1], "text": "本書はシステムのビジョンを述べる。"},
                {"lines": [2], "text": "利用者は注文をアプリから行える。"},
            ],
        },
        "metamodel": {
            # 実装時: 参照するメタモデル（JSON Schema / OWL等）の情報をここに配置
        },
    }

    result = chain.invoke(sample_input)
    print(result)
