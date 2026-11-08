# ============================================
# data_fetcher.py - データ取得モジュール
# ============================================
# JPX（日本取引所グループ）から全上場企業リストをダウンロードし、
# yfinanceから株価データ・日経平均のリターンを取得する。
# ============================================

import datetime
import logging
import os
import tempfile
import time

import pandas as pd
import requests
import yfinance as yf

logger = logging.getLogger(__name__)


def fetch_all_tse_stocks() -> dict[str, str]:
    """
    JPXの公開Excelファイルから全上場銘柄（ETF/REIT等を除く）を取得し、
    {ティッカーコード: 銘柄名} の辞書で返す。
    """
    url = "https://www.jpx.co.jp/markets/statistics-equities/misc/tvdivq0000001vg2-att/data_j.xls"
    logger.info("JPXから上場企業リストをダウンロード中...")

    try:
        response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=60)
        response.raise_for_status()

        with tempfile.NamedTemporaryFile(suffix=".xls", delete=False) as tmp:
            tmp.write(response.content)
            tmp_path = tmp.name

        # xlrd / openpyxl の両方にフォールバック対応
        try:
            df = pd.read_excel(tmp_path, engine="xlrd")
        except Exception:
            try:
                df = pd.read_excel(tmp_path)
            except Exception:
                df = pd.read_excel(tmp_path, engine="openpyxl")
        finally:
            os.unlink(tmp_path)

        cols = df.columns.tolist()
        if len(cols) < 4:
            return {}

        code_col, name_col, market_col = cols[1], cols[2], cols[3]
        stocks = {}
        skipped = 0

        for _, row in df.iterrows():
            code = str(row[code_col]).strip().replace(".0", "")
            name = str(row[name_col]).strip()
            market = str(row[market_col]).strip()

            # ETF/ETN/PRO Market/REITは除外
            if any(kw in market for kw in ("ETF", "ETN", "PRO", "REIT")):
                skipped += 1
                continue

            if len(code) >= 4 and code[:4].replace("A", "").replace("B", "").replace("C", "").isdigit():
                stocks[f"{code}.T"] = name

        logger.info(f"JPXから {len(stocks)} 銘柄を取得（ETF等 {skipped} 件除外）")
        return stocks

    except Exception as e:
        logger.error(f"JPXからのデータ取得に失敗: {e}")
        return {}


def get_nikkei_60d_return() -> float:
    """日経平均(^N225)の直近60営業日（約3ヶ月）の騰落率を取得"""
    try:
        tk = yf.Ticker("^N225")
        hist = tk.history(period="6mo")
        if len(hist) < 60:
            logger.warning("日経平均のデータが不十分です（60日未満）")
            return 0.0
        return float((hist["Close"].iloc[-1] / hist["Close"].iloc[-60]) - 1.0)
    except Exception as e:
        logger.error(f"日経平均取得エラー: {e}")
        return 0.0


def fetch_stock_history(code: str, period: str = "2y") -> pd.DataFrame | None:
    """個別銘柄の株価履歴を取得"""
    try:
        ticker = yf.Ticker(code)
        hist = ticker.history(period=period)
        if hist.empty:
            return None
        return hist
    except Exception:
        return None


def fetch_stock_info(code: str) -> dict:
    """個別銘柄のinfo（時価総額など）を取得"""
    try:
        ticker = yf.Ticker(code)
        return ticker.info
    except Exception:
        return {}


def batch_download(codes: list[str], period: str = "2y") -> pd.DataFrame:
    """
    複数銘柄の株価データをyf.downloadで一括取得。
    threads=False で安定性を確保。
    """
    try:
        data = yf.download(codes, period=period, group_by="ticker", threads=False, progress=False)
        return data
    except Exception as e:
        logger.error(f"一括ダウンロードエラー: {e}")
        return pd.DataFrame()


def is_business_day(d: datetime.date) -> bool:
    """土日・祝日を除いた日本の営業日判定"""
    if d.weekday() >= 5:
        return False
    try:
        import jpholiday
        if jpholiday.is_holiday(d):
            return False
    except ImportError:
        pass  # jpholidayがない場合は土日のみで判定
    return True


def get_business_days(start_date: datetime.date, end_date: datetime.date) -> list[datetime.date]:
    """指定期間の営業日リストを取得"""
    days = []
    current = start_date
    while current <= end_date:
        if is_business_day(current):
            days.append(current)
        current += datetime.timedelta(days=1)
    return days
