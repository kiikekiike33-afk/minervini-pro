# ============================================
# results_io.py - スクリーニング結果JSONの読み書き
# ============================================
# 結果をSQLite DBだけでなく、リポジトリ内の data/ 以下に
# JSONとしても保存・読込するためのモジュール。
#
# これにより:
#   - GitHub Actions / PCバッチが書いた結果JSONをコミットできる
#   - Streamlit Cloud のアプリが、起動時に同梱JSONを取り込んで表示できる
#     (Cloud のファイルシステムは揮発性で DB が残らないため)
#   - 外部ツール (Muse の定期モニタリング等) が画面ではなく
#     data/latest.json を直接読める
#
# ファイル配置:
#   data/latest.json          最新日の結果 {"date": ..., "results": [...]}
#   data/history/YYYY-MM-DD.json  日別の結果 (同形式)
#   ※ results の各要素は screening_results テーブルの1行分 (辞書)
# ============================================

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def data_dir() -> Path:
    """リポジトリの data/ ディレクトリ (code/ の1つ上) を返す"""
    return Path(__file__).resolve().parent.parent / "data"


def find_data_dir() -> Path | None:
    """同梱データの探索 (code/ の1つ上 → code/ 内 の順に探す)"""
    candidates = [
        Path(__file__).resolve().parent.parent / "data",
        Path(__file__).resolve().parent / "data",
    ]
    for c in candidates:
        if c.is_dir():
            return c
    return None


def export_results(rows: list[dict], date_str: str, base: Path | None = None) -> list[Path]:
    """
    結果行のリストを data/latest.json と data/history/<date>.json に書き出す。
    戻り値: 書き出したファイルのパスのリスト。
    """
    base = base or data_dir()
    history_dir = base / "history"
    history_dir.mkdir(parents=True, exist_ok=True)

    payload = {"date": date_str, "results": rows}
    text = json.dumps(payload, ensure_ascii=False, indent=1, default=str)

    written = []
    for path in (base / "latest.json", history_dir / f"{date_str}.json"):
        path.write_text(text, encoding="utf-8")
        written.append(path)
        logger.info(f"結果JSONを書き出しました: {path} ({len(rows)}件)")
    return written


def load_bundled_results(base: Path | None = None) -> dict[str, list[dict]]:
    """
    data/history/*.json と data/latest.json を読み、
    {日付: 結果行リスト} の辞書で返す。壊れたファイルは読み飛ばす。
    """
    base = base or find_data_dir()
    if base is None:
        return {}

    by_date: dict[str, list[dict]] = {}

    files: list[Path] = []
    history_dir = base / "history"
    if history_dir.is_dir():
        files.extend(sorted(history_dir.glob("*.json")))
    latest = base / "latest.json"
    if latest.is_file():
        files.append(latest)  # latest を最後に読む (同日なら上書きで同一内容)

    for path in files:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            date_str = payload.get("date")
            rows = payload.get("results")
            if date_str and isinstance(rows, list):
                by_date[date_str] = rows
        except Exception as e:
            logger.warning(f"同梱JSONの読込に失敗 ({path}): {e}")

    return by_date
