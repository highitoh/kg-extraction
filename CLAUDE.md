# Project Configuration for Claude Code

## Project Overview
PDFファイルから知識を抽出してTurtleファイルとNeo4j用CSVファイルを生成するLangChainベースのアプリケーション

## Development Environment
- **Language**: Python
- **Framework**: LangChain
- **Database**: Neo4j
- **Container**: Docker Compose

## Key Files and Structure
```
/workspace/
├── app/                    # メインアプリケーション
│   ├── main.py            # 知識グラフ抽出スクリプト
│   ├── schemas/           # 入出力スキーマ定義
│   │   ├── *-chain/
│   │   └── common/        # 共通スキーマ
│   ├── prompts/           # タスク用プロンプトテキスト
│   ├── *.py               # チェイン・タスク実行スクリプト群
│   └── requirements.txt   # Python依存関係
├── neo4j/                 # Neo4jデータ
└── doc/                   # ドキュメント
```

## Schema Structure
`/workspace/app/schemas/`は、各チェインやLLMの入出力のJSONスキーマを定義している
- **Chains**: doc-text-chain, view-chain, class-chain, property-chain
- **Extractors**: text-extractor, view-extractor, class-extractor, property-extractor
- **Filters**: text-filter, view-filter
- **Generators**: turtle-generator, neo4j-csv-generator
- **Common**: Shared schema definitions (metamodel.schema.json)

## Prompt Templates
`/workspace/app/prompts/` は、LLMで使用するプロンプトのテキストを記述している

## Common Commands

### Development
```bash
# 依存関係のインストール
cd app && pip install -r requirements.txt

# メインアプリケーションの実行
cd app && python main.py [PDF_PATH]

# 特定のチェインの実行 (統合前)
cd app && python doc_text_chain.py
cd app && python view_chain.py
cd app && python class_chain.py
cd app && python property_chain.py
```

### Docker Environment
```bash
# 環境の起動
docker-compose up -d

# アプリケーションコンテナに入る
docker exec -it app_container bash

# Neo4jブラウザアクセス
# http://localhost:7474 (neo4j/neo4jpassword)
```

### Testing and Quality
```bash
# Pythonコードの実行
cd app && python -m pytest  # テストがある場合

# 型チェック
cd app && python -m mypy .   # mypyがインストールされている場合
```

## Environment Variables
- `NEO4J_AUTH=neo4j/neo4jpassword` (Docker環境)
- `NEO4J_initial_dbms_default__database=partner`

## Output Files
- **Turtle files**: RDF/Turtle形式の知識グラフ
- **Neo4j CSV files**: Neo4jインポート用CSVファイル
- **Result JSON**: `extraction_result.json` - 処理結果サマリー

## Key Dependencies
- langchain: LLMチェイン構築
- langchain-openai: OpenAI API連携
- pdfplumber, PyPDF2: PDF処理
- rdflib: RDF/Turtle生成
- neo4j: グラフデータベース

## Notes
- PDFファイルは `/workspace/doc/` に配置
- 出力ファイルは `/workspace/app/output/` に生成
- ログファイルは `/workspace/app/log/` に生成
- Neo4jデータは永続化される (`./neo4j/data`)
- スキーマ定義により各チェイン・タスクの入出力仕様が明確化
- プロンプトテキストは外部ファイル管理により調整・保守が容易