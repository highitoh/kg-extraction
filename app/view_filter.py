from typing import Dict, Any, List, Tuple

from langchain.schema.runnable import Runnable

from logger import Logger

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


if __name__ == "__main__":
    import uuid

    # テスト用のサンプルデータ（重複を含む）
    sample_input = {
        "id": str(uuid.uuid4()),
        "views": [
            {
                "type": "business_concept",
                "texts": [
                    {
                        "file_id": "test_file",
                        "lines": [1, 2],
                        "text": "顧客管理システム"
                    },
                    {
                        "file_id": "test_file",
                        "lines": [3, 4],
                        "text": "顧客管理システムの開発"  # 包含関係（長い方）
                    },
                    {
                        "file_id": "test_file",
                        "lines": [5, 6],
                        "text": "顧客管理システム"  # 完全一致（重複）
                    }
                ]
            },
            {
                "type": "system_requirement",
                "texts": [
                    {
                        "file_id": "test_file",
                        "lines": [7, 8],
                        "text": "データベース連携機能"
                    },
                    {
                        "file_id": "test_file",
                        "lines": [9, 10],
                        "text": "データベース連携"  # 包含関係（短い方）
                    },
                    {
                        "file_id": "test_file",
                        "lines": [11, 12],
                        "text": "API設計要件"
                    }
                ]
            },
            {
                "type": "data_analysis",
                "texts": [
                    {
                        "file_id": "test_file",
                        "lines": [13, 14],
                        "text": "売上データ分析"
                    }
                ]
            }
        ]
    }

    # ViewFilterのインスタンス作成
    print("ViewFilter テスト開始...")
    filter_instance = ViewFilter(progress=True)

    print("\n=== 元のデータ ===")
    for view in sample_input["views"]:
        print(f"ビュー種別: {view['type']}")
        for i, text in enumerate(view["texts"]):
            print(f"  [{i+1}] {text['text']} (lines: {text['lines']})")

    try:
        # テスト実行
        result = filter_instance.invoke(sample_input)

        print("\n=== フィルタリング結果 ===")
        print(f"出力ID: {result['id']}")
        print(f"フィルタ後のビュー数: {len(result['views'])}")

        for view in result['views']:
            print(f"\nビュー種別: {view['type']}")
            print(f"テキスト数: {len(view['texts'])}")
            for i, text in enumerate(view['texts']):
                print(f"  [{i+1}] {text['text']} (lines: {text['lines']})")

        # 重複削除の検証
        original_text_count = sum(len(v.get("texts", [])) for v in sample_input["views"])
        filtered_text_count = sum(len(v.get("texts", [])) for v in result["views"])

        print(f"\n=== 統計情報 ===")
        print(f"元のテキスト数: {original_text_count}")
        print(f"フィルタ後のテキスト数: {filtered_text_count}")
        print(f"削除されたテキスト数: {original_text_count - filtered_text_count}")

        print("\n✅ ViewFilter テスト完了")

    except Exception as e:
        print(f"\n❌ テスト実行エラー: {e}")
        import traceback
        traceback.print_exc()
