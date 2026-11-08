# ============================================
# screener.py - S・A評価 成長株スクリーニングエンジン
# ============================================
# オニール / ミネルヴィニの成長株投資手法に基づくスクリーニング:
#   Stage 1: トレンドテンプレート（第2ステージ確認・上方乖離制限）
#   Stage 2: 3ヶ月以上ぶりの新高値検出
#   Stage 3: 出来高確認（大口資金流入・1.5倍以上）
#   Stage 4: ファンダメンタルズ健全性（予想純利益黒字・直近営業CFプラス）
# ============================================

import datetime
import logging
import time
from typing import Any

import numpy as np
import pandas as pd
import yfinance as yf

from data_fetcher import (
    fetch_all_tse_stocks, get_nikkei_60d_return,
    batch_download, get_business_days
)
from db_manager import save_screening_results

logger = logging.getLogger(__name__)

# スクリーニング定数
MIN_TURNOVER_20D = 300_000_000  # 流動性フィルタ: 平均売買代金3億円以上


def evaluate_sa_batch(
    codes: list[str],
    names: dict[str, str],
    nikkei_return: float,
    target_date: datetime.date
) -> list[dict]:
    """
    バッチダウンロードしたデータに対して、S・A評価3段階ロジックを適用。

    Stage 1: トレンドテンプレート（第2ステージ確認）
    Stage 2: VCPボラティリティ収縮パターン検出
    Stage 3: ブレイクアウト + 大口資金流入（ポケットピボット）検出
    """
    results = []
    batch_size = 50
    total_batches = (len(codes) + batch_size - 1) // batch_size

    logger.info(f"全 {len(codes)} 銘柄を {total_batches} バッチで処理します")

    for batch_idx in range(total_batches):
        batch_start = batch_idx * batch_size
        batch_end = min(batch_start + batch_size, len(codes))
        batch_codes = codes[batch_start:batch_end]

        logger.info(f"バッチ {batch_idx + 1}/{total_batches} ({batch_start + 1}〜{batch_end})")

        try:
            data = yf.download(
                batch_codes, period="2y", group_by="ticker",
                threads=False, progress=False
            )
            if data.empty:
                continue

            is_multi = isinstance(data.columns, pd.MultiIndex)

            for code in batch_codes:
                try:
                    # データの切り出し
                    if is_multi:
                        if code not in data.columns.get_level_values(0):
                            continue
                        hist = data[code].copy()
                    else:
                        if len(batch_codes) > 1:
                            continue
                        hist = data.copy()

                    hist = hist.dropna(subset=["Close", "Volume"])

                    # 未来のデータ（指定日より後のデータ）を除外
                    target_date_str = target_date.isoformat()
                    hist = hist.loc[:target_date_str]

                    # 最低250営業日（約1年分）のデータが必要
                    if len(hist) < 250:
                        continue

                    close = hist["Close"]
                    high = hist["High"]
                    low = hist["Low"]
                    opn = hist["Open"]
                    volume = hist["Volume"]

                    # ==========================================
                    # 事前計算: 移動平均・RSI
                    # ==========================================
                    sma20  = close.rolling(20).mean()
                    sma50  = close.rolling(50).mean()
                    sma150 = close.rolling(150).mean()
                    sma200 = close.rolling(200).mean()

                    # RSI(14)
                    delta = close.diff()
                    up = delta.clip(lower=0)
                    down = -1 * delta.clip(upper=0)
                    roll_up = up.ewm(alpha=1/14, adjust=False).mean()
                    roll_down = down.ewm(alpha=1/14, adjust=False).mean()
                    rs = roll_up / roll_down
                    rsi = 100.0 - (100.0 / (1.0 + rs))

                    c_close  = float(close.iloc[-1])
                    c_sma20  = float(sma20.iloc[-1])
                    c_sma50  = float(sma50.iloc[-1])
                    c_sma150 = float(sma150.iloc[-1])
                    c_sma200 = float(sma200.iloc[-1])
                    c_rsi14  = float(rsi.iloc[-1])

                    # 52週高値・安値
                    high52 = float(high.iloc[-250:].max())
                    low52  = float(low.iloc[-250:].min())

                    # 流動性フィルタ: 20日平均売買代金 ≥ 3億円
                    turnover_20d = float((close.iloc[-20:] * volume.iloc[-20:]).mean())
                    if turnover_20d < MIN_TURNOVER_20D:
                        continue

                    # ==========================================
                    # Stage 1: 基本トレンド確認
                    # ==========================================
                    # ※ 半年停滞→新高値ブレイクの銘柄はSMA序列が
                    #    崩れていることが多いため、最低限の条件のみ

                    # 条件1: 株価が200日移動平均線を上回っている（上昇トレンド）
                    if not (c_close > c_sma200):
                        continue

                    # 条件2: 200日移動平均線が上向き（20営業日前比）
                    if len(sma200) < 21:
                        continue
                    sma200_20d_ago = float(sma200.iloc[-21])
                    if not (c_sma200 > sma200_20d_ago):
                        continue

                    # 条件3: 移動平均線からの上方乖離が過剰でないこと
                    # 短期的な伸びきり（SMA50から+20%超え）や
                    # 長期的な伸びきり（SMA200から+40%超え）を除外
                    if c_close > c_sma50 * 1.20 or c_close > c_sma200 * 1.40:
                        continue

                    # 52週高値乖離率（表示用）
                    diff_52w = (c_close / high52) - 1.0

                    # ==========================================
                    # Stage 2: 3ヶ月以上ぶりの新高値検出
                    # ==========================================
                    # 過去250日（1年分）の最高値を特定し、
                    # その最高値が60日（3ヶ月）以上前に付けられたものであること。
                    # そして直近5日でその旧高値をついに突破した = 久しぶりの新高値

                    if len(high) < 250:
                        continue

                    # 直近5日の高値
                    high_recent_5d = float(high.iloc[-5:].max())

                    # 直近5日を除く過去250日の最高値
                    highs_before_5d = high.iloc[-250:-5]
                    max_high_before_5d = float(highs_before_5d.max())

                    # 新高値条件: 直近5日で旧最高値を突破
                    if not (high_recent_5d >= max_high_before_5d):
                        continue

                    # その旧最高値が何日前に付けられたかを計算
                    max_idx = int(highs_before_5d.values.argmax())
                    days_since_max = len(highs_before_5d) - max_idx

                    # 核心条件: 旧最高値が60営業日（約3ヶ月）以上前に付けられたこと
                    # = この銘柄は3ヶ月以上、最高値を更新できていなかった
                    if days_since_max < 60:
                        continue

                    # ==========================================
                    # Stage 3: 出来高確認（厳格版）
                    # ==========================================
                    # 新高値更新時に有意な出来高増加を伴っていること
                    vol_50d_avg = float(volume.iloc[-50:].mean())
                    recent_volume = volume.iloc[-5:]

                    # 直近5日の最大出来高が50日平均の1.5倍以上
                    breakout_vol_ratio = float(recent_volume.max() / vol_50d_avg) if vol_50d_avg > 0 else 0
                    if breakout_vol_ratio < 1.5:
                        continue

                    # シグナル種別
                    if breakout_vol_ratio >= 2.0:
                        signal_type = "VOL_SURGE"
                    else:
                        signal_type = "VOL_BREAKOUT"

                    # VCP収縮率（互換用: 簡易計算）
                    if len(hist) >= 120:
                        high_t1 = float(high.iloc[-120:-60].max())
                        low_t1 = float(low.iloc[-120:-60].min())
                        high_t3 = float(high.iloc[-60:].max())
                        low_t3 = float(low.iloc[-60:].min())
                        range_t1 = (high_t1 - low_t1) / high_t1 if high_t1 > 0 else 0
                        range_t3 = (high_t3 - low_t3) / high_t3 if high_t3 > 0 else 0
                        vcp_contraction = (1.0 - range_t3 / range_t1) * 100 if range_t1 > 0 else 0
                    else:
                        vcp_contraction = 0

                    # RS計算（表示用、フィルタなし）
                    ret_60d = float((c_close / close.iloc[-60]) - 1.0) if len(close) >= 60 else 0
                    rs_vs_nikkei = ret_60d - nikkei_return

                    # SMA20乖離率
                    sma20_diff = (c_close / c_sma20 - 1.0) * 100 if not pd.isna(c_sma20) else 0

                    # ==========================================
                    # 注目ポイントの自動生成
                    # ==========================================
                    comments = []
                    if diff_52w >= 0:
                        comments.append("🎯 52週新高値更新")
                    elif diff_52w >= -0.03:
                        comments.append("🔥 52週高値に迫る")
                    if days_since_max >= 200:
                        comments.append(f"⏰ 約{days_since_max // 20}ヶ月ぶり")
                    elif days_since_max >= 120:
                        comments.append(f"⏰ 約{days_since_max // 20}ヶ月ぶり")
                    elif days_since_max >= 60:
                        comments.append(f"⏰ 約{days_since_max // 20}ヶ月ぶり")
                    if breakout_vol_ratio >= 2.0:
                        comments.append(f"🚀 出来高{breakout_vol_ratio:.1f}倍の大商い")
                    elif breakout_vol_ratio >= 1.5:
                        comments.append(f"📈 出来高{breakout_vol_ratio:.1f}倍")
                    if rs_vs_nikkei >= 0.20:
                        comments.append("💪 圧倒的RS")
                    elif rs_vs_nikkei >= 0.05:
                        comments.append("📊 RS良好")

                    comment = " / ".join(comments) if comments else "✅ 3ヶ月以上ぶり新高値"

                    # 全3段階通過 → 結果に追加
                    results.append({
                        "date": target_date.isoformat(),
                        "code": code,
                        "name": names.get(code, code),
                        "close_price": c_close,
                        "sma50": c_sma50,
                        "sma150": c_sma150,
                        "sma200": c_sma200,
                        "diff_52w_high": diff_52w,
                        "rs_outperform": rs_vs_nikkei,
                        "avg_turnover": turnover_20d,
                        "sma20_diff": sma20_diff,
                        "rsi14": c_rsi14,
                        "vcp_contraction": vcp_contraction,
                        "breakout_vol_ratio": breakout_vol_ratio,
                        "signal_type": signal_type,
                        "comment": comment,
                    })
                    logger.info(
                        f"  ✅ 通過: {names.get(code, code)} ({code}) "
                        f"VCP:{vcp_contraction:.0f}% Vol:{breakout_vol_ratio:.1f}x [{signal_type}]"
                    )

                except Exception:
                    continue

        except Exception as e:
            logger.error(f"バッチ {batch_idx + 1} 処理エラー: {e}")

        time.sleep(2)  # yfinance API制限回避

    logger.info(f"3段階フィルタ通過: {len(results)} 件")
    return results


def check_fundamentals(code: str) -> tuple[bool, float | None, float | None, str]:
    """
    ファンダメンタルズ健全性判定:
    1. 予想利益（Forward EPS）が黒字（> 0）であること（取得不能時は実績EPSで判定）
    2. 直近決算の営業キャッシュフローがプラス（> 0）であること
    戻り値: (通過フラグ, forward_eps, operating_cf, 除外理由)
    """
    try:
        ticker = yf.Ticker(code)
        info = ticker.info or {}

        # 1. 予想利益チェック
        forward_eps = info.get("forwardEps")
        trailing_eps = info.get("trailingEps")

        target_eps = forward_eps if forward_eps is not None else trailing_eps
        if target_eps is not None and target_eps <= 0:
            return False, forward_eps, None, f"赤字（予想/実績EPS: {target_eps}）"

        # 2. 営業キャッシュフローチェック
        op_cf = info.get("operatingCashflow")
        if op_cf is None:
            try:
                cf = ticker.cashflow
                if cf is not None and not cf.empty and "Operating Cash Flow" in cf.index:
                    val = cf.loc["Operating Cash Flow"].iloc[0]
                    if pd.notna(val):
                        op_cf = float(val)
            except Exception:
                pass

        if op_cf is not None and op_cf <= 0:
            return False, forward_eps, op_cf, f"営業CFマイナス（{op_cf:,.0f}円）"

        return True, forward_eps, op_cf, ""
    except Exception as e:
        logger.warning(f"ファンダメンタルズ確認失敗 ({code}): {e}")
        return True, None, None, ""


def run_full_screening(target_date: datetime.date | None = None) -> list[dict]:
    """
    全上場企業を対象にS・A評価スクリーニングを実行し、
    結果をDBに保存して返す。
    """
    if target_date is None:
        target_date = datetime.date.today()

    logger.info(f"=== S・A評価 フルスクリーニング開始: {target_date} ===")
    start_time = time.time()

    # 日経平均のリターン取得（RS計算用）
    nikkei_return = get_nikkei_60d_return()
    logger.info(f"日経平均 60日リターン: {nikkei_return*100:+.1f}%")

    # JPXから全銘柄リストを取得
    stocks = fetch_all_tse_stocks()
    if not stocks:
        logger.error("銘柄リストの取得に失敗しました。")
        return []

    codes = list(stocks.keys())

    # 3段階テクニカルフィルタ適用
    candidates = evaluate_sa_batch(codes, stocks, nikkei_return, target_date)
    logger.info(f"テクニカル通過: {len(candidates)} 件。ファンダメンタルズ（黒字予想・営業CFプラス）検証を実施...")

    final_results = []
    for c in candidates:
        passed, f_eps, op_cf, reason = check_fundamentals(c["code"])
        c["forward_eps"] = f_eps
        c["operating_cf"] = op_cf
        if not passed:
            logger.info(f"  ❌ ファンダメンタルズ除外: {c['name']} ({c['code']}) - {reason}")
            continue
        final_results.append(c)

    logger.info(f"最終通過: {len(final_results)} 件")

    # DB保存
    saved = save_screening_results(final_results, target_date.isoformat())

    elapsed = time.time() - start_time
    logger.info(f"=== スクリーニング完了: {len(final_results)} 件 ({elapsed/60:.1f}分) ===")

    return final_results
