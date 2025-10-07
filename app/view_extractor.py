from typing import Dict, Any, List, Union
import asyncio
import json
import os
import uuid

from langchain.schema.runnable import Runnable
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from openai import APIConnectionError, RateLimitError, APIError

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
        model: str = "gpt-5-mini",
        temperature: float = 0.0,
        max_concurrency: int = 8,
        max_retries: int = 5,
        progress: bool = True,
        max_spans_per_label: int = 3,
        log_dir: str = "log/view_extractor",
    ):
        self.llm = llm or ChatOpenAI(
            model=model,
            temperature=temperature,
            reasoning={"effort": "minimal"},
            output_version="responses/v1",
        )

        # JSONスキーマを読み込み
        schema = self._load_schema()
        self.llm_json = self.llm.bind(response_format={"type": "json_schema", "json_schema": {"name": "view_extractor", "schema": schema}})
        self.max_concurrency = max_concurrency
        self.max_retries = max_retries
        self.progress = progress
        self.max_spans_per_label = max_spans_per_label

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

    def _load_schema(self) -> dict:
        """JSONスキーマファイルを読み込む"""
        schema_path = os.path.join(os.path.dirname(__file__), "schemas", "view-extractor", "llm.schema.json")
        with open(schema_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _load_prompt(self) -> str:
        """プロンプトファイルを読み込む"""
        prompt_path = os.path.join(os.path.dirname(__file__), "prompts", "view_extractor.txt")
        with open(prompt_path, "r", encoding="utf-8") as f:
            return f.read()

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
        ai = await self._invoke_with_retry(msg)
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

    async def _invoke_with_retry(self, msg: HumanMessage):
        """リトライ機能付きのLLM呼び出し"""
        delay = 1.0
        max_delay = 60.0
        last_exception = None

        for attempt in range(self.max_retries + 1):
            try:
                return await self.llm_json.ainvoke([msg])
            except (APIConnectionError, RateLimitError, APIError) as e:
                last_exception = e

                if attempt == self.max_retries:
                    raise

                wait_time = min(delay, max_delay)
                if self.progress:
                    print(f"[ViewExtractor Retry {attempt + 1}/{self.max_retries}] API error: {type(e).__name__}. Retrying in {wait_time:.1f}s...")
                await asyncio.sleep(wait_time)
                delay *= 2

        if last_exception:
            raise last_exception

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
        chunks_data: List[Dict[str, Any]] = target["chunks"]
        file_id = target.get("file_id", str(uuid.uuid4()))

        # チャンクテキストのリストを取得
        chunks = [chunk["text"] for chunk in chunks_data]

        # 並列実行でビュー抽出
        chunk_results = asyncio.run(self._process_chunks_parallel(chunks))

        # 結果をViewChainOutput形式に変換
        views = []
        for view_type in self.views:
            texts = []
            for result in chunk_results:
                spans = result["spans"].get(view_type, [])
                for span in spans:
                    texts.append({
                        "file_id": file_id,
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
        self.logger.save_log(output, filename_prefix="view_extractor_output_")

        return output


if __name__ == "__main__":
    # テスト用のサンプルデータ
    sample_input = {
        "target": {
            "id": "test_001",
            "file_id": "sample_file",
            "chunks": [
                {
                    "text": "システムは顧客情報を管理する必要がある。顧客データベースには氏名、住所、電話番号を格納する。"
                },
                {
                    "text": "プロジェクトの目的は売上向上である。APIサーバーはRESTfulな設計で構築する。"
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
                print(f"  [{i+1}] {text['text']}")

        print("\n✅ ViewExtractor テスト完了")

    except Exception as e:
        print(f"\n❌ テスト実行エラー: {e}")
        import traceback
        traceback.print_exc()

