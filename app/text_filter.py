from typing import Any, Dict, List
import asyncio
import json
import uuid

from langchain.schema.runnable import Runnable
from langchain_core.runnables.config import RunnableConfig
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from logger import Logger
from chunk_creator import ChunkCreator

class TextFilter(Runnable):
    """抽出テキストをフィルタリングするタスク"""

    def __init__(self):
        with open("./prompts/sentence_filter.txt", "r", encoding="utf-8") as f:
            self.classification_prompt = f.read()

        # Load JSON schema from file
        with open("./schemas/text-filter/llm.schema.json", "r", encoding="utf-8") as f:
            schema = json.load(f)

        self._llm = ChatOpenAI(
            model="gpt-5-nano",
            temperature=0.0
        )
        self._llm_json = self._llm.bind(response_format={"type": "json_schema", "json_schema": {"name": "text_filter_schema", "schema": schema}})
        self.logger = Logger("./log/text_filter")
        self.chunk_creator = ChunkCreator(max_chars=1600, overlap_chars=200)

    async def _classify_chunk(self, chunk_text: str, chunk_sentences: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """チャンクのテキストを分類し、文章のみを抽出"""
        msg = HumanMessage(
            content=self.classification_prompt + f"\n\n# 対象テキスト\n{chunk_text}"
        )

        try:
            ai = await self._llm_json.ainvoke([msg])
            raw_content = ai.content if isinstance(ai.content, str) else str(ai.content)

            response_data = json.loads(raw_content)

            # JSONスキーマ: {"items": [...]} 形式
            classifications = response_data.get("items", [])

            # "文章"に分類された行のみを抽出
            filtered_sentences = []
            for classification in classifications:
                if classification.get("content") == "文章":
                    start_line = classification.get("startLine")
                    end_line = classification.get("endLine")

                    # startLineとendLineが整数として取得されることを確認
                    if isinstance(start_line, int) and isinstance(end_line, int):
                        # 指定された行番号範囲内のすべての行を抽出
                        for sentence in chunk_sentences:
                            sentence_lines = sentence["lines"]
                            # linesが配列の場合、その中の任意の行番号が範囲内にあるかチェック
                            if isinstance(sentence_lines, list):
                                if any(start_line <= line <= end_line for line in sentence_lines):
                                    filtered_sentences.append(sentence)
                            else:
                                # 整数の場合の処理（後方互換性）
                                if start_line <= sentence_lines <= end_line:
                                    filtered_sentences.append(sentence)

            return filtered_sentences

        except Exception as e:
            print(f"チャンク分類中にエラーが発生しました: {e}")
            return []

    def _create_chunks_with_sentences(self, sentences: List[Dict[str, Any]]) -> List[tuple]:
        """文章データをチャンクに分割し、各チャンクに対応する文章リストを作成"""
        # 全体のテキストを作成
        full_text = "\n".join([f"Lines {s['lines'][0] if isinstance(s['lines'], list) else s['lines']}: {s['text']}" for s in sentences])

        # テキストをチャンクに分割
        chunks = self.chunk_creator.create(full_text)

        chunk_data = []
        for chunk in chunks:
            # チャンク内の行番号を抽出
            chunk_sentences = []
            lines = chunk.split("\n")
            for line in lines:
                if line.strip().startswith("Lines "):
                    try:
                        line_num = int(line.split(":")[0].replace("Lines ", ""))
                        # 対応する文章データを検索
                        for sentence in sentences:
                            sentence_lines = sentence["lines"]
                            if isinstance(sentence_lines, list):
                                if line_num in sentence_lines:
                                    chunk_sentences.append(sentence)
                                    break
                            else:
                                if sentence_lines == line_num:
                                    chunk_sentences.append(sentence)
                                    break
                    except (ValueError, IndexError):
                        continue

            if chunk_sentences:
                chunk_data.append((chunk, chunk_sentences))

        return chunk_data

    async def _classify_text(self, sentences: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """LLMを使用してテキストを並列分類し、文章のみを抽出"""
        # チャンクを作成
        chunk_data = self._create_chunks_with_sentences(sentences)

        # 各チャンクを並列処理
        tasks = []
        for chunk_text, chunk_sentences in chunk_data:
            task = self._classify_chunk(chunk_text, chunk_sentences)
            tasks.append(task)

        # 並列実行
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # 結果をマージ（重複排除）
        filtered_sentences = []
        seen_lines = set()

        for result in results:
            if isinstance(result, Exception):
                print(f"チャンク処理中にエラーが発生しました: {result}")
                continue

            for sentence in result:
                sentence_lines = sentence["lines"]
                if isinstance(sentence_lines, list):
                    line_key = tuple(sentence_lines)
                else:
                    line_key = sentence_lines
                if line_key not in seen_lines:
                    filtered_sentences.append(sentence)
                    seen_lines.add(line_key)

        # 行番号順にソート
        filtered_sentences.sort(key=lambda x: x["lines"][0] if isinstance(x["lines"], list) else x["lines"])

        return filtered_sentences

    def invoke(self, input: Dict[str, Any], config: RunnableConfig = None) -> Dict[str, Any]:
        # input: TextFilterInput
        # output: TextFilterOutput

        source = input["source"]

        # TextExtractorOutput/TextFilterOutputの両方から"sentences"を取得
        sentences = source.get("sentences", [])

        # LLMを使用して文章のみを抽出
        filtered_sentences = asyncio.run(self._classify_text(sentences))

        output = {
            "id": str(uuid.uuid4()),
            "file_name": source["file_name"],
            "sentences": filtered_sentences
        }

        # 出力JSONをログに保存
        self.logger.save_log(output, "text_filter_output_")

        return output

if __name__ == "__main__":
    # テスト用の入力データ
    test_input = {
        "source": {
            "file_name": "test.pdf",
            "sentences": [
                {"lines": [1], "text": "これは通常の文章です。"},
                {"lines": [2], "text": "株式会社テスト"},
                {"lines": [3], "text": "もう一つの文章example。"},
                {"lines": [4], "text": "図1: グラフの説明"},
                {"lines": [5], "text": "最後の文章です。"}
            ]
        }
    }

    # TextFilterのインスタンス作成とテスト実行
    text_filter = TextFilter()
    result = text_filter.invoke(test_input)

    print("TextFilter Test Result:")
    print(f"File name: {result['file_name']}")
    print(f"Filtered sentences count: {len(result['sentences'])}")
    for sentence in result["sentences"]:
        print(f"  Line {sentence['lines']}: {sentence['text']}")
    print(f"Output ID: {result['id']}")