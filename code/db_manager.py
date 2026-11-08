# ============================================
# db_manager.py - SQLite データベース管理モジュール
# ============================================
# スクリーニング結果とAI分析レポートの永続化を担当する。
# テーブル設計:
#   screening_results - 日次スクリーニング結果（カレンダー機能の基盤）
#   ai_reports        - Gemini AI分析レポートのキャッシュ
# ============================================

import sqlite3
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# DB ファイルパス（このスクリプトと同じディレクトリに配置）
DB_PATH = Path(__file__).parent / "minervini_pro.db"


def get_connection() -> sqlite3.Connection:
    """SQLiteコネクションを返す（WALモードで高速化）"""
    conn = sqlite3.connect(str(DB_PATH), timeout=30)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """データベースの初期化（テーブル作成）"""
    conn = get_connection()
    cur = conn.cursor()

    # スクリーニング結果テーブル
    cur.execute("""
        CREATE TABLE IF NOT EXISTS screening_results (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            date        TEXT    NOT NULL,
            code        TEXT    NOT NULL,
            name        TEXT    NOT NULL,
            close_price REAL,
            sma50       REAL,
            sma150      REAL,
            sma200      REAL,
            diff_52w_high REAL,
            market_cap  REAL,
            rs_outperform REAL,
            avg_turnover  REAL,
            sma20_diff  REAL,
            rsi14       REAL,
            cup_depth   REAL,
            recovery    REAL,
            vcp_contraction    REAL,
            breakout_vol_ratio REAL,
            signal_type        TEXT,
            forward_eps        REAL,
            operating_cf       REAL,
            comment     TEXT,
            UNIQUE(date, code)
        )
    """)

    # 既存DBへのカラム追加対応（既に存在する場合はエラーになるため無視する）
    migration_columns = [
        "sma20_diff REAL", "rsi14 REAL", "cup_depth REAL", "recovery REAL",
        "vcp_contraction REAL", "breakout_vol_ratio REAL", "signal_type TEXT",
        "forward_eps REAL", "operating_cf REAL",
    ]
    for col_def in migration_columns:
        try:
            cur.execute(f"ALTER TABLE screening_results ADD COLUMN {col_def}")
        except sqlite3.OperationalError:
            pass

    # AI分析レポートテーブル
    cur.execute("""
        CREATE TABLE IF NOT EXISTS ai_reports (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            code        TEXT    NOT NULL,
            date        TEXT    NOT NULL,
            web_info    TEXT,
            report_text TEXT,
            created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(date, code)
        )
    """)

    conn.commit()
    conn.close()
    logger.info("データベースを初期化しました。")


# ============================================
# スクリーニング結果のCRUD
# ============================================
def save_screening_results(results: list[dict], target_date: str = None) -> int:
    """スクリーニング結果をDBに保存。当日の既存データを削除してからUPSERTする"""
    conn = get_connection()
    cur = conn.cursor()
    
    # 既存の同日データをクリア（0件になった場合も古いデータを残さないため）
    if target_date:
        cur.execute("DELETE FROM screening_results WHERE date = ?", (target_date,))
    elif results:
        # target_dateが明示されていない場合はresultsの最初の要素から推測
        cur.execute("DELETE FROM screening_results WHERE date = ?", (results[0]['date'],))

    if not results:
        conn.commit()
        conn.close()
        return 0
        
    saved = 0
    for r in results:
        try:
            cur.execute("""
                INSERT OR REPLACE INTO screening_results
                (date, code, name, close_price, sma50, sma150, sma200,
                 diff_52w_high, market_cap, rs_outperform, avg_turnover,
                 sma20_diff, rsi14, vcp_contraction, breakout_vol_ratio,
                 signal_type, forward_eps, operating_cf, comment)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                r["date"], r["code"], r["name"], r.get("close_price"),
                r.get("sma50"), r.get("sma150"), r.get("sma200"),
                r.get("diff_52w_high"), r.get("market_cap"),
                r.get("rs_outperform"), r.get("avg_turnover"),
                r.get("sma20_diff"), r.get("rsi14"),
                r.get("vcp_contraction"), r.get("breakout_vol_ratio"),
                r.get("signal_type"), r.get("forward_eps"), r.get("operating_cf"),
                r.get("comment", "")
            ))
            saved += 1
        except Exception as e:
            logger.warning(f"DB保存エラー ({r.get('code')}): {e}")
    conn.commit()
    conn.close()
    logger.info(f"スクリーニング結果 {saved} 件を保存しました。")
    return saved


def get_available_dates() -> list[str]:
    """DBに保存されている日付のリスト（降順）を取得"""
    try:
        conn = get_connection()
        rows = conn.execute(
            "SELECT DISTINCT date FROM screening_results ORDER BY date DESC"
        ).fetchall()
        conn.close()
        return [row["date"] for row in rows]
    except Exception:
        return []


def get_stocks_by_date(date_str: str) -> list[dict]:
    """指定日のスクリーニング結果を取得"""
    try:
        conn = get_connection()
        rows = conn.execute(
            "SELECT * FROM screening_results WHERE date = ? ORDER BY rs_outperform DESC",
            (date_str,)
        ).fetchall()
        conn.close()
        return [dict(row) for row in rows]
    except Exception:
        return []


# ============================================
# AI レポートのCRUD
# ============================================
def save_ai_report(code: str, date: str, web_info: str, report_text: str) -> None:
    """AI分析レポートをDBに保存"""
    try:
        conn = get_connection()
        conn.execute("""
            INSERT OR REPLACE INTO ai_reports (code, date, web_info, report_text)
            VALUES (?, ?, ?, ?)
        """, (code, date, web_info, report_text))
        conn.commit()
        conn.close()
    except Exception as e:
        logger.warning(f"AIレポート保存エラー ({code}): {e}")


def get_ai_report(code: str, date: str) -> dict | None:
    """指定銘柄・日付のAIレポートを取得（キャッシュ用）"""
    try:
        conn = get_connection()
        row = conn.execute(
            "SELECT * FROM ai_reports WHERE code = ? AND date = ?",
            (code, date)
        ).fetchone()
        conn.close()
        return dict(row) if row else None
    except Exception:
        return None


def get_ai_reports_by_date(date: str) -> dict[str, dict]:
    """指定日の全AIレポートをコード→レポートの辞書で返す"""
    try:
        conn = get_connection()
        rows = conn.execute(
            "SELECT * FROM ai_reports WHERE date = ?", (date,)
        ).fetchall()
        conn.close()
        return {row["code"]: dict(row) for row in rows}
    except Exception:
        return {}
