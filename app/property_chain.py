import os
import uuid
import json
import asyncio
import logging
from typing import Dict, Any, List, Union
from langchain.schema.runnable import Runnable, RunnableSequence
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from .logger import Logger


class PropertyExtractor(Runnable):
    """
    プロパティ（関係）抽出
    入力: PropertyChainInput (class_info, metamodel)
    出力: PropertyChainOutput (id, properties)
    """

    def __init__(
        self,
        llm: ChatOpenAI | None = None,
        model: str = "gpt-5-nano",
        temperature: float = 0.0,
        max_concurrency: int = 8,
        progress: bool = True,
        log_dir: str = "log/property_extractor",
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

        # プロンプトを読み込み
        self.prompt = self._load_prompt()

        # Logger設定
        self.logger = Logger(log_dir)

    def _load_prompt(self) -> str:
        """プロンプトを読み込み"""
        prompt_path = os.path.join(os.path.dirname(__file__), "prompts", "PropertyExtractor.txt")
        with open(prompt_path, "r", encoding="utf-8") as f:
            return f.read()

    @staticmethod
    def _to_text(c: Union[str, List[Any]]) -> str:
        """LLMの出力をテキスト形式に変換"""
        if isinstance(c, str):
            return c
        for part in c:
            if isinstance(part, dict) and part.get("type") in ("output_text", "text"):
                return part.get("text", "")
        return ""

    async def _extract_properties_from_classes(self, classes: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        """クラス情報からプロパティを抽出"""
        # クラス情報をテキスト形式に整理
        class_texts = []
        for cls in classes:
            cls_id = cls.get("id", "")
            cls_label = cls.get("label", "")
            cls_iri = cls.get("class_iri", "")
            file_id = cls.get("file_id", "")
            class_texts.append(f"- ID: {cls_id}, Label: {cls_label}, IRI: {cls_iri}, File: {file_id}")

        class_text_block = "\n".join(class_texts)

        msg = HumanMessage(
            content=[
                {"type": "text", "text": self.prompt},
                {"type": "text", "text": f"# クラス情報\n{class_text_block}"},
            ]
        )

        ai = await self.llm_json.ainvoke([msg])
        raw = self._to_text(ai.content)
        try:
            data = json.loads(raw)
        except Exception:
            data = {}

        # プロパティ情報を整理
        properties = []
        extracted_properties = data.get("properties", [])
        if isinstance(extracted_properties, list):
            for prop in extracted_properties:
                if not isinstance(prop, dict):
                    continue

                src_id = prop.get("src_id", "")
                property_iri = prop.get("property_iri", "")
                dest_id = prop.get("dest_id", "")

                if src_id and property_iri and dest_id:
                    properties.append({
                        "src_id": src_id.strip(),
                        "property_iri": property_iri.strip(),
                        "dest_id": dest_id.strip()
                    })

        return properties

    def invoke(self, input: Dict[str, Any], config=None) -> Dict[str, Any]:
        """PropertyChainInputからプロパティを抽出してPropertyChainOutputを出力"""
        class_info = input["class_info"]
        classes = class_info.get("classes", [])

        # プロパティ抽出実行
        extracted_properties = asyncio.run(self._extract_properties_from_classes(classes))

        # 出力形式に合わせて整理
        properties = []
        for prop in extracted_properties:
            properties.append({
                "id": str(uuid.uuid4()),
                "src_id": prop["src_id"],
                "property_iri": prop["property_iri"],
                "dest_id": prop["dest_id"],
            })

        output: Dict[str, Any] = {
            "id": str(uuid.uuid4()),
            "properties": properties,
        }

        if self.progress:
            total_properties = len(properties)
            print(f"PropertyExtractor: extracted {total_properties} properties")

        # Loggerでpropertyチェーンの出力を保存
        self.logger.save_log(output)

        return output


class PropertyFilter(Runnable):
    """
    プロパティフィルタリング（空実装）
    入力: PropertyChainOutput
    出力: PropertyChainOutput（フィルタリング済み）

    現在は空実装
    """

    def __init__(self, progress: bool = True, log_dir: str = "log/property_filter"):
        self.progress = progress
        # Logger設定
        self.logger = Logger(log_dir)

    def invoke(self, input: Dict[str, Any], config=None) -> Dict[str, Any]:
        """PropertyChainOutputをそのまま返す（空実装）"""
        output = input.copy()

        if self.progress:
            properties_count = len(output.get("properties", []))
            print(f"PropertyFilter: {properties_count} properties (no filtering applied)")

        # Loggerでpropertyチェーンの出力を保存
        self.logger.save_log(output)

        return output


def create_property_chain() -> RunnableSequence:
    """
    プロパティ抽出チェーン:
      PropertyExtractor -> PropertyFilter
    """
    extractor = PropertyExtractor()
    property_filter = PropertyFilter()
    return RunnableSequence(steps=[extractor, property_filter])


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    chain = create_property_chain()
    sample_input = {
        "class_info": {
            "id": "class-chain-001",
            "classes": [
                {
                    "id": "class-001",
                    "class_iri": "ex:BusinessConceptClass",
                    "label": "顧客管理システム",
                    "file_id": "file-abc",
                },
                {
                    "id": "class-002",
                    "class_iri": "ex:SystemComponentClass",
                    "label": "データベース",
                    "file_id": "file-abc",
                }
            ]
        },
        "metamodel": {
            # メタモデル情報はここに格納
        },
    }

    result = chain.invoke(sample_input)
    print(json.dumps(result, ensure_ascii=False, indent=2))