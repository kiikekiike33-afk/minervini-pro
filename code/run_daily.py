# ============================================
# run_daily.py - 日次自動バッチオーケストレーター
# ============================================
# 毎日15:30にWindowsタスクスケジューラから実行される。
# JPX全上場企業に対してミネルヴィニ条件のスクリーニングを行い、
# 結果をSQLiteに保存する。
# ============================================

import datetime
import logging
import time

from data_fetcher import is_business_day
from db_manager import init_db, get_stocks_by_date
from results_io import export_results
from screener import run_full_screening

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def main():
    today = datetime.date.today()

    if not is_business_day(today):
        logger.info(f"{today} は休場日のため、処理をスキップします。")
        return

    logger.info(f"=== Minervini Pro 日次バッチ開始: {today} ===")
    start = time.time()

    init_db()
    results = run_full_screening(today)

    # 結果をリポジトリの data/ にJSONでも書き出す (GitHub経由で共有するため)。
    # DB保存済みの行を正として書き出す。0件の日も「実行した」記録として残す。
    try:
        rows = get_stocks_by_date(today.isoformat())
        export_results(rows, today.isoformat())
    except Exception as e:
        logger.warning(f"結果JSONの書き出しに失敗: {e}")

    elapsed = time.time() - start
    logger.info(f"=== 完了: {len(results)} 銘柄 ({elapsed/60:.1f}分) ===")

    if results:
        import pandas as pd
        df = pd.DataFrame(results)
        # コンソール出力用フォーマット
        df["SMA20乖離率(%)"] = df["sma20_diff"].map(lambda x: f"{x:+.1f}%")
        df["RSI(14)"] = df["rsi14"].map(lambda x: f"{x:.1f}")
        df["RS(%)"] = (df["rs_outperform"] * 100).map(lambda x: f"{x:+.1f}%")
        df["高値乖離(%)"] = (df["diff_52w_high"] * 100).map(lambda x: f"{x:.1f}%")
        disp_df = df[["code", "name", "close_price", "高値乖離(%)", "SMA20乖離率(%)", "RSI(14)", "RS(%)"]]
        print("\n【CWH/VCP スクリーニング通過銘柄 サマリー】")
        print(disp_df.to_string(index=False))


if __name__ == "__main__":
    main()
