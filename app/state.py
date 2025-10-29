"""
LangGraph State Definition for Knowledge Extraction
知識抽出処理のLangGraph State定義
"""

from typing import TypedDict, List, Dict, Any, Optional


class KnowledgeExtractionState(TypedDict, total=False):
    """
    知識抽出処理全体のState定義

    各ノードはこのStateを受け取り、必要なフィールドを読み取り・更新する。
    total=Falseにより、全フィールドがオプショナル扱いとなり、
    各ノードで必要なフィールドのみを更新可能。
    """

    # ========== 入力情報 ==========
    files: List[Dict[str, str]]  # [{"path": "/path/to/file.pdf"}]
    file_path: str  # 処理対象のファイルパス（filesから取り出した値）
    file_type: str  # "document" | "presentation"

    # ========== メタモデル ==========
    metamodel: Dict[str, Any]  # メタモデル定義（全ノードで参照）

    # ========== LLMモデル設定 ==========
    model: Optional[str]  # LLMモデル名（例: "gpt-5-mini", "gpt-5-nano"）

    # ========== 出力先設定 ==========
    output_dir: str  # 出力ディレクトリ

    # ========== 中間結果: テキスト抽出段階 ==========
    id: str  # 抽出ID（ファイルIDとして使用）
    file_id: str  # ファイルID
    file_name: str  # ファイル名
    sentences: List[Dict[str, Any]]  # 抽出された文章リスト
    # sentence: {"text": str, ...}

    # ========== 中間結果: チャンク分割段階 ==========
    chunks: List[Dict[str, str]]  # テキストチャンク
    # chunk: {"text": str}

    # ========== 中間結果: ビュー抽出段階 ==========
    views: List[Dict[str, Any]]  # ビュー記述抽出結果
    # view: {
    #   "type": str,  # ビュー種別
    #   "texts": [{"file_id": str, "text": str}, ...]
    # }

    # ========== 中間結果: クラス抽出段階 ==========
    classes: List[Dict[str, Any]]  # クラス抽出結果
    # class: {
    #   "id": str,         # クラス個体ID
    #   "class_iri": str,  # メタモデルのクラスIRI
    #   "label": str,      # 抽出クラス名
    #   "sources": List[str],   # 抽出元文章
    #   "file_ids": List[str]   # 抽出元ファイルID
    # }

    # ========== 中間結果: プロパティ抽出段階 ==========
    properties: List[Dict[str, Any]]  # プロパティ（関係）抽出結果
    # property: {
    #   "id": str,           # プロパティID
    #   "src_id": str,       # 出発点クラス個体ID
    #   "property_iri": str, # プロパティIRI
    #   "dest_id": str       # 到達点クラス個体ID
    # }

    # ========== 最終出力 ==========
    turtle_file: str  # Turtleファイルパス
    turtle_content: str  # Turtle内容（大きいので通常はJSONに保存しない）
    nodes_csv: str  # Neo4jノードCSVファイルパス
    relationships_csv: str  # Neo4j関係CSVファイルパス

    # ========== メタ情報 ==========
    current_stage: str  # 現在の処理ステージ（デバッグ用）
    errors: List[str]  # エラーメッセージリスト

    # ========== サマリー ==========
    summary: Dict[str, Any]  # 最終サマリー情報
    # {
    #   "classes_count": int,
    #   "properties_count": int,
    #   "files_generated": List[str]
    # }
