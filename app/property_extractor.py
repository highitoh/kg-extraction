from typing import Dict, Any, List, Union
import asyncio
import json
import os
import uuid

from langchain.schema.runnable import Runnable
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

from logger import Logger

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


if __name__ == "__main__":
    import json

    # Create PropertyExtractor instance
    extractor = PropertyExtractor()

    # Sample input data for testing (class_info with classes)
    sample_input = {
        "class_info": {
            "id": "test-class-001",
            "classes": [
                {
                    "id": "cls-001",
                    "class_iri": "ex:CustomerManagementSystem",
                    "label": "顧客管理システム",
                    "file_id": "file-001"
                },
                {
                    "id": "cls-002",
                    "class_iri": "ex:UserRegistration",
                    "label": "ユーザー登録",
                    "file_id": "file-001"
                },
                {
                    "id": "cls-003",
                    "class_iri": "ex:Authentication",
                    "label": "認証",
                    "file_id": "file-001"
                },
                {
                    "id": "cls-004",
                    "class_iri": "ex:OrderManagementSystem",
                    "label": "注文管理システム",
                    "file_id": "file-002"
                },
                {
                    "id": "cls-005",
                    "class_iri": "ex:DatabaseModule",
                    "label": "データベース接続モジュール",
                    "file_id": "file-002"
                },
                {
                    "id": "cls-006",
                    "class_iri": "ex:ApiGateway",
                    "label": "APIゲートウェイ",
                    "file_id": "file-002"
                }
            ]
        },
        "metamodel": {}
    }

    print("=== PropertyExtractor Test ===")
    print("Input:")
    print(json.dumps(sample_input, ensure_ascii=False, indent=2))
    print("\n" + "="*50 + "\n")

    try:
        # Execute PropertyExtractor
        result = extractor.invoke(sample_input)

        print("Output:")
        print(json.dumps(result, ensure_ascii=False, indent=2))

        print(f"\nSummary:")
        print(f"- Total properties extracted: {len(result['properties'])}")

        # Display each property relationship
        for i, prop in enumerate(result["properties"], 1):
            # Find source and destination class labels for display
            src_class = next((c for c in sample_input["class_info"]["classes"] if c["id"] == prop["src_id"]), None)
            dest_class = next((c for c in sample_input["class_info"]["classes"] if c["id"] == prop["dest_id"]), None)

            src_label = src_class["label"] if src_class else prop["src_id"]
            dest_label = dest_class["label"] if dest_class else prop["dest_id"]

            print(f"- Property {i}: {src_label} --[{prop['property_iri']}]--> {dest_label}")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
