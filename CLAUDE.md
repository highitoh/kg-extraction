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
│   ├── src/               # ソースコード (ステージベース構成)
│   │   ├── main.py        # エントリポイント
│   │   ├── core/          # アプリケーション中核 (graph.py, nodes.py, state.py)
│   │   ├── utils/         # 共通ユーティリティ (logger.py, llm_factory.py, chunk_creator.py)
│   │   ├── classification/  # ファイル分類ステージ
│   │   ├── text/          # テキスト抽出ステージ (document/presentation)
│   │   ├── view/          # ビュー抽出ステージ
│   │   ├── class/         # クラス抽出ステージ
│   │   ├── property/      # プロパティ抽出ステージ
│   │   └── output/        # 出力生成ステージ (turtle_generator.py, neo4j_csv_generator.py)
│   ├── schemas/           # 入出力スキーマ定義
│   │   ├── *-chain/
│   │   └── common/        # 共通スキーマ
│   ├── prompts/           # タスク用プロンプトテキスト
│   ├── metamodel/         # メタモデル定義
│   ├── output/            # 生成ファイル出力先
│   ├── log/               # ログファイル
│   └── requirements.txt   # Python依存関係
├── neo4j/                 # Neo4jデータ
└── doc/                   # ドキュメント
```

## Schema Structure
`/workspace/app/schemas/`は、各チェインやLLMの入出力のJSONスキーマを定義している。再利用性を高めるため、上位と同じ入出力の場合はスキーマを参照して定義する。
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
cd app && python src/main.py [PDF_PATH]

# プロジェクト別に出力ファイルを管理
cd app && python src/main.py [PDF_PATH] --project [PROJECT_NAME]

# すべてのオプション
cd app && python src/main.py [PDF_PATH] --project [PROJECT_NAME] --model [MODEL_NAME] --output-dir [OUTPUT_DIR]

# 特定のチェインの実行 (開発・デバッグ用)
cd app && python -c "import sys; sys.path.insert(0, '.'); from src.text.doc_text_chain import *"
cd app && python -c "import sys; sys.path.insert(0, '.'); from src.view.view_chain import *"
cd app && python -c "import sys; sys.path.insert(0, '.'); from src.class.class_chain import *"
cd app && python -c "import sys; sys.path.insert(0, '.'); from src.property.property_chain import *"
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

## Testing and Quality
全体処理を動作確認するときは、コストを抑えるため**gpt-5-nano**を使用する

### Single file extraction
```bash
# デフォルトプロジェクト
python src/main.py ../doc/sample.pdf --model gpt-5-nano

# 特定のプロジェクト
python src/main.py ../doc/sample.pdf --project test-project --model gpt-5-nano
```

### Multiple files extraction
```bash
python src/main.py ../doc/sample.pdf --project test-project --model gpt-5-nano
python src/main.py ../doc/sample2.pdf --project test-project --model gpt-5-nano
```

## Environment Variables
- `NEO4J_AUTH=neo4j/neo4jpassword` (Docker環境)

## Output Files
出力ファイルは`--project`パラメータで指定したプロジェクト別に管理される。

### ディレクトリ構造
```
/workspace/app/output/
├── default/                           # デフォルトプロジェクト
│   ├── knowledge_graph_YYYYMMDD_HHMMSS.ttl
│   ├── nodes.csv
│   ├── relationships.csv
│   ├── knowledge.json                 # 累積知識（ACCEPT済みクラス+有効プロパティ）
│   └── extraction_result_YYYYMMDD_HHMMSS.json
├── project1/                          # カスタムプロジェクト
│   ├── knowledge_graph_YYYYMMDD_HHMMSS.ttl
│   ├── nodes.csv
│   ├── relationships.csv
│   ├── knowledge.json
│   └── extraction_result_YYYYMMDD_HHMMSS.json
└── project2/
    └── ...
```

### ファイル種別
- **Turtle files**: `knowledge_graph_*.ttl` - RDF/Turtle形式の知識グラフ
- **Neo4j CSV files**: `nodes.csv`, `relationships.csv` - Neo4jインポート用CSVファイル
- **Knowledge JSON**: `knowledge.json` - 累積知識（軽量版、次回実行時のインプットとして使用）
  - ACCEPT判定されたクラスのみ
  - 強度>0のプロパティのみ
  - 複数回実行で自動的に累積される
- **Result JSON**: `extraction_result_YYYYMMDD_HHMMSS.json` - 各実行の完全な処理結果（タイムスタンプ付き、履歴・監査用）

## Key Dependencies
- langchain: LLMチェイン構築
- langchain-openai: OpenAI API連携
- pdfplumber, pypdf: PDF処理
- rdflib: RDF/Turtle生成
- neo4j: グラフデータベース

## Notes
- PDFファイルは `/workspace/doc/` に配置
- 出力ファイルは `/workspace/app/output/{project}/` に生成
- ログファイルは `/workspace/app/log/` に生成
- Neo4jデータは永続化される (`./neo4j/data`)
- スキーマ定義により各チェイン・タスクの入出力仕様が明確化
- プロンプトテキストは外部ファイル管理により調整・保守が容易

## Command Line Arguments
```bash
python src/main.py [PDF_PATH] [OPTIONS]

Arguments:
  PDF_PATH              抽出対象のPDFファイルパス (デフォルト: ../doc/sample.pdf)

Options:
  --project PROJECT     プロジェクト名 (デフォルト: default)
  --model MODEL         LLMモデル名 (例: gpt-5-mini, gpt-5-nano)
  --output-dir DIR      出力ディレクトリ (デフォルト: /workspace/app/output)
```