# ============================================
# advanced_screen.py - 高度スクリーニング (第2段階)
# ============================================
# 全上場銘柄に対して以下を計算し、data/ に JSON で書き出す。
#
#   ① 地合い判定: 全銘柄の breadth (150/200日線超え比率、加重リターン
#      中央値、プラス比率) から 強気/中立/弱気 を判定する
#   ② 候補抽出: トレンドテンプレート (価格系7項目) >=6、
#      IBD式加重リターンの全銘柄パーセンタイル >=80、
#      52週高値から -25% 以内、20日平均売買代金 >=3億円
#   ③ スコア: テンプレート30 + RS25 + 出来高15 + ベースの質15 +
#      簡易ファンダ15 = 100点。S>=85 / A>=70 / B<70
#   ④ ピボット (52週高値) と損切りライン (ピボット -8%) を付与
#   ⑤ 業種 (JPX 33業種) 別の候補分布を集計
#   ⑥ スコア70以上の候補には次回決算予定日 (yfinance) を付与
#   ⑦ data/outcomes.json に候補を記録し、5営業日後/20営業日後の
#      リターンを後続の実行で自動記入する (事後検証)
#
# 出力:
#   data/advanced_latest.json / data/advanced_history/<date>.json
#   data/outcomes.json
#
# ファンダ (EPS成長・売上成長・ROE) は集計サイト (yfinance) の値で、
# 一次資料で検証したものではない「速報扱い」である点に注意。
# EPS成長が +500% 超 / -90% 未満の場合は異常値として中立評価に補正し
# eps_flag を付ける。
# ============================================

import datetime
import json
import logging
import os
import tempfile
import time
from pathlib import Path

import pandas as pd
import requests
import yfinance as yf

from data_fetcher import fetch_all_tse_stocks, get_business_days

logger = logging.getLogger(__name__)

JPX_URL = "https://www.jpx.co.jp/markets/statistics-equities/misc/tvdivq0000001vg2-att/data_j.xlsx"
CHUNK = 120
HISTORY_DAYS = 500  # 営業日ベースの下限 (SMA200 + 加重RSに必要)


def data_dir() -> Path:
    return Path(__file__).resolve().parent.parent / "data"


def fetch_sector_map() -> dict[str, str]:
    """JPXの銘柄一覧から {ティッカー: 33業種区分} を作る。失敗時は空。"""
    try:
        response = requests.get(JPX_URL, headers={"User-Agent": "Mozilla/5.0"}, timeout=60)
        response.raise_for_status()
        with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as tmp:
            tmp.write(response.content)
            tmp_path = tmp.name
        try:
            df = pd.read_excel(tmp_path)
        finally:
            os.unlink(tmp_path)
        cols = df.columns.tolist()
        code_col, market_col, sector_col = cols[1], cols[3], cols[5]
        sectors: dict[str, str] = {}
        for _, row in df.iterrows():
            market = str(row[market_col]).strip()
            if any(kw in market for kw in ("ETF", "ETN", "PRO", "REIT")):
                continue
            code = str(row[code_col]).strip().replace(".0", "")
            sector = str(row[sector_col]).strip()
            if len(code) >= 4 and code[:4].replace("A", "").replace("B", "").replace("C", "").isdigit():
                sectors[f"{code}.T"] = sector
        return sectors
    except Exception as e:
        logger.warning(f"業種マップの取得に失敗: {e}")
        return {}


def _ret(s: pd.Series, n: int):
    if len(s) <= n:
        return None
    return (s.iloc[-1] / s.iloc[-1 - n] - 1) * 100


def _compute_one(t: str, name: str, df: pd.DataFrame, asof: str):
    df = df.dropna(subset=["Close"])
    df = df[df.index <= asof]
    if len(df) < HISTORY_DAYS:
        return None
    close, high, low, vol = df["Close"], df["High"], df["Low"], df["Volume"]
    p = float(close.iloc[-1])
    sma50 = float(close.rolling(50).mean().iloc[-1])
    sma150 = float(close.rolling(150).mean().iloc[-1])
    sma200s = close.rolling(200).mean()
    sma200 = float(sma200s.iloc[-1])
    sma200_1m = float(sma200s.iloc[-22])
    hi52 = float(high.tail(252).max())
    lo52 = float(low.tail(252).min())
    tt = {
        "p_gt_150_200": bool(p > sma150 and p > sma200),
        "150_gt_200": bool(sma150 > sma200),
        "200_uptrend": bool(sma200 > sma200_1m),
        "50_gt_150_200": bool(sma50 > sma150 and sma50 > sma200),
        "p_gt_50": bool(p > sma50),
        "above_low_30": bool(p >= lo52 * 1.30),
        "near_high_15": bool(p >= hi52 * 0.85),
    }
    weights = [(63, 0.4), (126, 0.2), (189, 0.2), (252, 0.2)]
    w_ret = sum(_ret(close, n) * w for n, w in weights)
    turnover = float((close.tail(20) * vol.tail(20)).mean())

    prior_max = float(close.iloc[-65:-5].max())
    vol50 = vol.rolling(50).mean()
    breakout = False
    bo_vol = None
    for i in range(-5, 0):
        if float(close.iloc[i]) > prior_max and float(vol50.iloc[i]) > 0:
            vr = float(vol.iloc[i]) / float(vol50.iloc[i])
            if vr >= 1.5:
                breakout = True
                bo_vol = vr
                break
    dryup = None
    if float(vol.tail(50).mean()) > 0:
        dryup = float(vol.tail(10).mean()) / float(vol.tail(50).mean())

    tr = pd.concat(
        [(high - low), (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1
    ).max(axis=1)
    tr_pct = tr / close * 100
    base_early = float(tr_pct.iloc[-60:-30].mean())
    base_late = float(tr_pct.iloc[-30:-5].mean())
    contraction = base_early / base_late if base_late > 0 else None
    base_close = close.iloc[-50:-5]
    base_depth = (float(base_close.max()) / float(base_close.min()) - 1) * 100

    return {
        "ticker": t,
        "name": name,
        "price": p,
        "tt_pass": int(sum(tt.values())),
        "tt": tt,
        "w_ret": round(float(w_ret), 2),
        "dist_high_pct": round((p / hi52 - 1) * 100, 1),
        "turnover_oku": round(turnover / 1e8, 1),
        "breakout": breakout,
        "bo_vol_ratio": (round(bo_vol, 2) if bo_vol else None),
        "dryup": (round(dryup, 2) if dryup is not None else None),
        "contraction": (round(contraction, 2) if contraction else None),
        "base_depth_pct": round(base_depth, 1),
        "ext50_pct": round((p / sma50 - 1) * 100, 1),
        "pivot_price": round(hi52, 1),
        "stop_price": round(hi52 * 0.92, 1),
    }


def compute_universe(asof: str) -> list[dict]:
    names = fetch_all_tse_stocks()
    tickers = list(names.keys())
    logger.info(f"高度スクリーニング対象: {len(tickers)} 銘柄 (基準日 {asof})")
    start = (datetime.date.fromisoformat(asof) - datetime.timedelta(days=800)).isoformat()
    end = (datetime.date.fromisoformat(asof) + datetime.timedelta(days=1)).isoformat()
    metrics: list[dict] = []
    t0 = time.time()
    for i in range(0, len(tickers), CHUNK):
        batch = tickers[i:i + CHUNK]
        try:
            px = yf.download(batch, start=start, end=end, group_by="ticker",
                             auto_adjust=False, progress=False, threads=True)
        except Exception as e:
            logger.warning(f"チャンク {i // CHUNK} のダウンロード失敗: {e}")
            continue
        for t in batch:
            try:
                df = px[t] if t in px else None
            except Exception:
                df = None
            if df is None:
                continue
            m = _compute_one(t, names[t], df, asof)
            if m:
                metrics.append(m)
        logger.info(f"チャンク {i // CHUNK + 1}/{(len(tickers) + CHUNK - 1) // CHUNK}: "
                    f"累計 {len(metrics)} 件 [{time.time() - t0:.0f}s]")
    # RSパーセンタイル (全銘柄)
    metrics.sort(key=lambda m: m["w_ret"])
    n = len(metrics)
    for rank, m in enumerate(metrics):
        m["rs_pct"] = round(rank / max(n - 1, 1) * 100, 1)
    return metrics


def compute_regime(metrics: list[dict]) -> dict:
    n = len(metrics)
    above = sum(1 for m in metrics if m["tt"]["p_gt_150_200"]) / n * 100
    pos = sum(1 for m in metrics if m["w_ret"] > 0) / n * 100
    tt6 = sum(1 for m in metrics if m["tt_pass"] >= 6) / n * 100
    wrets = sorted(m["w_ret"] for m in metrics)
    median = wrets[n // 2] if n else 0.0

    def vote(value, hi, lo):
        return 1 if value >= hi else (-1 if value < lo else 0)

    total = vote(above, 40, 25) + vote(median, 5, -5) + vote(pos, 55, 40)
    label = "強気" if total >= 2 else ("弱気" if total <= -2 else "中立")
    return {
        "label": label,
        "stocks": n,
        "above_ma_pct": round(above, 1),
        "positive_pct": round(pos, 1),
        "tt6_pct": round(tt6, 1),
        "median_wret_pct": round(median, 1),
    }


def _score(m: dict) -> float:
    s = m["tt_pass"] / 7 * 30
    s += m["rs_pct"] / 100 * 25
    if m["breakout"]:
        s += min(m["bo_vol_ratio"] or 0, 4) / 4 * 15
    else:
        d = m["dryup"] or 1.2
        s += 12 if d <= 0.7 else (9 if d <= 0.85 else (6 if d <= 1.0 else 3))
    base = 0.0
    c = m["contraction"] or 0
    base += 7.5 if c >= 1.15 else (4.0 if c >= 1.0 else 0.0)
    base += 7.5 if m["base_depth_pct"] <= 15 else (4.0 if m["base_depth_pct"] <= 25 else 0.0)
    s += base
    f = 0.0
    eg, rg, roe = m.get("eps_growth_q"), m.get("rev_growth"), m.get("roe")
    if eg is not None:
        if eg > 500 or eg < -90:
            m["eps_flag"] = "異常値・要確認"
            f += 3.5
        else:
            f += 7 if eg >= 25 else (4 if eg > 0 else 0)
    if rg is not None:
        f += 4 if rg >= 10 else (2 if rg > 0 else 0)
    if roe is not None:
        f += 4 if roe >= 17 else (2 if roe >= 10 else 0)
    s += f
    if m["ext50_pct"] > 12:
        s -= 5
    return round(s, 1)


def select_candidates(metrics: list[dict]) -> list[dict]:
    cands = [m for m in metrics
             if m["tt_pass"] >= 6 and m["rs_pct"] >= 80
             and m["dist_high_pct"] >= -25 and m["turnover_oku"] >= 3.0]
    logger.info(f"候補: {len(cands)} 件。ファンダメンタルズ取得中...")
    for m in cands:
        try:
            info = yf.Ticker(m["ticker"]).info
            eg, rg, roe = info.get("earningsQuarterlyGrowth"), info.get("revenueGrowth"), info.get("returnOnEquity")
            m["eps_growth_q"] = round(eg * 100, 1) if isinstance(eg, (int, float)) else None
            m["rev_growth"] = round(rg * 100, 1) if isinstance(rg, (int, float)) else None
            m["roe"] = round(roe * 100, 1) if isinstance(roe, (int, float)) else None
        except Exception:
            m["eps_growth_q"] = m["rev_growth"] = m["roe"] = None
    for m in cands:
        m["score"] = _score(m)
        m["grade"] = "S" if m["score"] >= 85 else ("A" if m["score"] >= 70 else "B")
        m["type"] = "ブレイク" if m["breakout"] else "ベース形成"
    cands.sort(key=lambda m: -m["score"])
    return cands


def enrich_earnings(cands: list[dict], asof: str, min_score: float = 70.0, limit: int = 80):
    """スコア上位の候補に次回決算予定日を付ける (取得できない場合は None)。"""
    targets = [m for m in cands if m["score"] >= min_score][:limit]
    logger.info(f"決算日取得対象: {len(targets)} 件")
    asof_d = datetime.date.fromisoformat(asof)
    for m in targets:
        m["next_earnings"] = None
        try:
            df = yf.Ticker(m["ticker"]).get_earnings_dates(limit=8)
            if df is not None and len(df):
                future = [d for d in df.index if d.date() > asof_d]
                if future:
                    m["next_earnings"] = min(future).date().isoformat()
        except Exception:
            pass


def sector_summary(cands: list[dict], sectors: dict[str, str]) -> list[dict]:
    by_sector: dict[str, dict] = {}
    for m in cands:
        sec = sectors.get(m["ticker"], "不明")
        m["sector"] = sec
        agg = by_sector.setdefault(sec, {"sector": sec, "count": 0, "s_count": 0, "score_sum": 0.0})
        agg["count"] += 1
        agg["s_count"] += 1 if m["grade"] == "S" else 0
        agg["score_sum"] += m["score"]
    out = []
    for agg in by_sector.values():
        agg["avg_score"] = round(agg.pop("score_sum") / agg["count"], 1)
        out.append(agg)
    out.sort(key=lambda a: (-a["count"], -a["avg_score"]))
    return out[:10]


def update_outcomes(asof: str, cands: list[dict], price_map: dict[str, float], base: Path) -> dict:
    """data/outcomes.json に当日候補を追記し、期日が来た過去候補のリターンを記入する。"""
    path = base / "outcomes.json"
    data = {"picks": []}
    if path.is_file():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            pass
    picks: list[dict] = data.get("picks", [])
    asof_d = datetime.date.fromisoformat(asof)

    known = {(p["date"], p["ticker"]) for p in picks}
    for m in cands:
        key = (asof, m["ticker"])
        if key in known:
            continue
        picks.append({
            "date": asof, "ticker": m["ticker"], "name": m["name"],
            "grade": m["grade"], "type": m["type"], "score": m["score"],
            "entry_price": m["price"], "ret_5d": None, "ret_20d": None,
        })

    for p in picks:
        cur = price_map.get(p["ticker"])
        if cur is None or not p["entry_price"]:
            continue
        pick_d = datetime.date.fromisoformat(p["date"])
        if pick_d >= asof_d:
            continue
        bd = len(get_business_days(pick_d + datetime.timedelta(days=1), asof_d))
        if p["ret_5d"] is None and bd >= 5:
            p["ret_5d"] = round((cur / p["entry_price"] - 1) * 100, 2)
        if p["ret_20d"] is None and bd >= 20:
            p["ret_20d"] = round((cur / p["entry_price"] - 1) * 100, 2)

    cutoff = (asof_d - datetime.timedelta(days=180)).isoformat()
    picks = [p for p in picks if p["date"] >= cutoff]
    data = {"updated": asof, "picks": picks}
    path.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    logger.info(f"outcomes.json 更新: 累計 {len(picks)} 件")
    return data


def run_advanced_screening(asof: str) -> dict:
    base = data_dir()
    (base / "advanced_history").mkdir(parents=True, exist_ok=True)
    metrics = compute_universe(asof)
    regime = compute_regime(metrics)
    cands = select_candidates(metrics)
    enrich_earnings(cands, asof)
    sectors = sector_summary(cands, fetch_sector_map())
    payload = {"date": asof, "regime": regime, "sectors": sectors, "candidates": cands}
    text = json.dumps(payload, ensure_ascii=False, indent=1, default=str)
    (base / "advanced_latest.json").write_text(text, encoding="utf-8")
    (base / "advanced_history" / f"{asof}.json").write_text(text, encoding="utf-8")
    price_map = {m["ticker"]: m["price"] for m in metrics}
    update_outcomes(asof, cands, price_map, base)
    logger.info(f"高度スクリーニング完了: {asof} / 候補 {len(cands)} 件 / 地合い {regime['label']}")
    return payload


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    run_advanced_screening(datetime.date.today().isoformat())
