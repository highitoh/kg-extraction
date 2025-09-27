from typing import Dict, Any, List, Union
import asyncio
import json
import os
import uuid

from langchain.schema.runnable import Runnable
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

from chunk_creator import ChunkCreator
from logger import Logger

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


if __name__ == "__main__":
    # テスト用のサンプルデータ
    sample_input = {
        "target": {
            "id": "test_001",
            "file_id": "sample_file",
            "sentences": [
                {
                    "lines": [1, 2],
                    "text": "システムは顧客情報を管理する必要がある。"
                },
                {
                    "lines": [3, 4],
                    "text": "顧客データベースには氏名、住所、電話番号を格納する。"
                },
                {
                    "lines": [5, 6],
                    "text": "プロジェクトの目的は売上向上である。"
                },
                {
                    "lines": [7, 8],
                    "text": "APIサーバーはRESTfulな設計で構築する。"
                }
            ]
        },
        "metamodel": {
            "version": "1.0",
            "domain": "business"
        }
    }

    # ViewExtractorのインスタンス作成
    print("ViewExtractor テスト開始...")
    extractor = ViewExtractor(
        model="gpt-5-nano",
        temperature=0.0,
        max_concurrency=2,  # テスト用に並列数を制限
        progress=True,
        max_spans_per_label=2  # テスト用に制限
    )

    print(f"対象ビュー: {extractor.views}")

    try:
        # テスト実行
        result = extractor.invoke(sample_input)

        print("\n=== テスト結果 ===")
        print(f"出力ID: {result['id']}")
        print(f"抽出されたビュー数: {len(result['views'])}")

        for view in result['views']:
            print(f"\nビュー種別: {view['type']}")
            print(f"抽出テキスト数: {len(view['texts'])}")
            for i, text in enumerate(view['texts'][:2]):  # 最初の2件のみ表示
                print(f"  [{i+1}] {text['text']} (lines: {text['lines']})")

        print("\n✅ ViewExtractor テスト完了")

    except Exception as e:
        print(f"\n❌ テスト実行エラー: {e}")
        import traceback
        traceback.print_exc()

