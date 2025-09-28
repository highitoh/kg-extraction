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
            temperature=0.0,
            reasoning={"effort": "minimal"}
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

            # ai.contentが配列形式の場合はtext属性を取得
            if isinstance(ai.content, list) and len(ai.content) > 0 and 'text' in ai.content[0]:
                raw_content = ai.content[0]['text']
            elif isinstance(ai.content, str):
                raw_content = ai.content
            else:
                raw_content = str(ai.content)

            response_data = json.loads(raw_content)

            # LLMの分類結果をログ出力
            self.logger.save_log(response_data, "llm_classification_response_")

            # JSONスキーマ: {"items": [...]} 形式
            classifications = response_data.get("items", [])

            # "文章"に分類されたテキストを抽出
            filtered_sentences = []
            for classification in classifications:
                if classification.get("content") == "文章":
                    classified_text = classification.get("text")

                    # 分類されたテキストが文字列として取得されることを確認
                    if isinstance(classified_text, str):
                        # チャンク内の文章と照合して該当するものを抽出
                        for sentence in chunk_sentences:
                            sentence_text = sentence["text"]
                            # 分類されたテキストが文章内に含まれているかチェック
                            if classified_text.strip() in sentence_text or sentence_text.strip() in classified_text:
                                filtered_sentences.append(sentence)

            return filtered_sentences

        except Exception as e:
            print(f"チャンク分類中にエラーが発生しました: {e}")
            return []

    def _create_chunks_with_sentences(self, sentences: List[Dict[str, Any]]) -> List[tuple]:
        """文章データをチャンクに分割し、各チャンクに対応する文章リストを作成"""
        # 全体のテキストを作成
        full_text = "\n".join([s['text'] for s in sentences])

        # テキストをチャンクに分割
        chunks = self.chunk_creator.create(full_text)

        chunk_data = []
        for i, chunk in enumerate(chunks):
            # チャンク内のテキストに対応する文章を特定
            chunk_sentences = []
            chunk_lines = chunk.split("\n")

            for line in chunk_lines:
                line = line.strip()
                if not line:
                    continue

                # テキストマッチングで対応する文章を検索
                for sentence in sentences:
                    sentence_text = sentence["text"].strip()
                    if line in sentence_text or sentence_text in line:
                        if sentence not in chunk_sentences:
                            chunk_sentences.append(sentence)

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
        seen_texts = set()

        for result in results:
            if isinstance(result, Exception):
                print(f"チャンク処理中にエラーが発生しました: {result}")
                continue

            for sentence in result:
                sentence_text = sentence["text"]
                if sentence_text not in seen_texts:
                    filtered_sentences.append(sentence)
                    seen_texts.add(sentence_text)

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
                {"text": "これは通常の文章です。"},
                {"text": "株式会社テスト"},
                {"text": "もう一つの文章example。"},
                {"text": "図1: グラフの説明"},
                {"text": "最後の文章です。"}
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
        print(f"  Text: {sentence['text']}")
    print(f"Output ID: {result['id']}")