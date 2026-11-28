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
#   ③ スコア: テンプレート35 + RS30 + 出来高20 + ベースの質15 = 100点。
#      S>=85 / A>=70 / B<70。ファンダはスコアに入れず表示のみ
#      (2026-11-28 改訂: バックテスト検証済み。旧配点は 30/25/15/15+ファンダ15)
#   ④ ピボット2種を付与: 従来の52週高値ベース (pivot_price / stop_price) と、
#      ベース右側の最終収縮高値ベースの pivot_v2 (直近5営業日を除く過去60営業日
#      の高値の最大) / stop_v2 = max(pivot_v2 x 0.92, 直近10営業日の安値 x 0.99)
#   ⑤ プール分類: A = ベース形成中 (終値 < pivot_v2)、
#      B = ブレイク成立または買いゾーン内 (pivot_v2 〜 pivot_v2 x 1.05)、
#      C = 伸びきり (終値 > pivot_v2 x 1.05、監視のみ)
#   ⑥ 業種 (JPX 33業種) 別の候補分布を集計
#   ⑦ スコア70以上の候補には次回決算予定日 (yfinance) を付与
#   ⑧ data/outcomes.json に候補を記録し、5/20/40/60営業日後の絶対リターンと
#      TOPIX連動ETF (1306.T) 比の超過リターン、保有期間中の MFE/MAE を
#      後続の実行で自動記入する (事後検証)。あわせて売り判定 (S3: ストップ /
#      トレンド崩壊 / 分配 / 60営業日の時間) を日次評価し、売却が確定した
#      pick には exit_* を記録する (2026-11-28 バックテストで採用検証済み)
#   ⑨ 地合いにディストリビューション・デイ相当カウント (1306.T、
#      直近25営業日窓) を併記する (表示のみ)
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

    # pivot_v2: ベース右側の最終収縮高値 = 直近5営業日を除く過去60営業日の高値の最大
    # (ブレイク判定の prior_max と同じ窓。高値ベース)
    pivot_v2 = float(high.iloc[-65:-5].max())
    # stop_v2: 構造的損切り。pivot_v2 の -8% と直近10営業日の安値の少し下の高い方。
    # -8% が絶対上限 (pivot_v2 x 0.92 を下回ることはない)
    stop_v2 = max(pivot_v2 * 0.92, float(low.tail(10).min()) * 0.99)

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
        "pivot_v2": round(pivot_v2, 1),
        "stop_v2": round(stop_v2, 1),
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


BENCH_TICKER = "1306.T"  # TOPIX連動ETF (超過リターンとDDカウントのベンチマーク)


def fetch_benchmark(asof: str):
    """1306.T の価格履歴を asof まで取得する。失敗時は None。"""
    try:
        start = (datetime.date.fromisoformat(asof) - datetime.timedelta(days=120)).isoformat()
        end = (datetime.date.fromisoformat(asof) + datetime.timedelta(days=1)).isoformat()
        df = yf.download(BENCH_TICKER, start=start, end=end, auto_adjust=False, progress=False)
        if df is None or len(df) == 0:
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df.dropna(subset=["Close"])
    except Exception as e:
        logger.warning(f"ベンチマーク取得に失敗: {e}")
        return None


def distribution_days(bench, window: int = 25):
    """ディストリビューション・デイ相当のカウント (表示のみ、地合い判定には使わない)。

    1306.T が前日比 -0.2% 以下に下落し出来高が前日より多い日を、直近 window
    営業日で数える。データ不足時は None。
    """
    if bench is None or len(bench) < window + 1:
        return None
    tail = bench.tail(window + 1)
    chg = tail["Close"].pct_change()
    vol_up = tail["Volume"] > tail["Volume"].shift(1)
    return int(((chg <= -0.002) & vol_up).iloc[1:].sum())


def _score(m: dict) -> float:
    """総合スコア (2026-11-28 改訂配点): TT 35 + RS 30 + 出来高 20 + ベースの質 15。

    ファンダメンタルズはスコアに入れない (表示のみ)。バックテスト
    (2026-11-28、2独立実行で一致) で検証済みの配点。
    """
    s = m["tt_pass"] / 7 * 35
    s += m["rs_pct"] / 100 * 30
    if m["breakout"]:
        s += min(m["bo_vol_ratio"] or 0, 4) / 4 * 20
    else:
        d = m["dryup"] or 1.2
        s += 16 if d <= 0.7 else (12 if d <= 0.85 else (8 if d <= 1.0 else 4))
    base = 0.0
    c = m["contraction"] or 0
    base += 7.5 if c >= 1.15 else (4.0 if c >= 1.0 else 0.0)
    base += 7.5 if m["base_depth_pct"] <= 15 else (4.0 if m["base_depth_pct"] <= 25 else 0.0)
    s += base
    if m["ext50_pct"] > 12:
        s -= 5
    return round(min(s, 100.0), 1)


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
        # EPS成長の異常値は表示上の注意フラグとしてのみ付ける (スコアには入れない)
        eg = m.get("eps_growth_q")
        if eg is not None and (eg > 500 or eg < -90):
            m["eps_flag"] = "異常値・要確認"
        m["score"] = _score(m)
        m["grade"] = "S" if m["score"] >= 85 else ("A" if m["score"] >= 70 else "B")
        m["type"] = "ブレイク" if m["breakout"] else "ベース形成"
        # プール分類 (状態ラベル。Pool B を優遇する根拠はバックテストに無い)
        # C (伸びきり) を最優先し、次に B (ブレイク成立 or 買いゾーン内)、残りが A
        pv2 = m["pivot_v2"]
        m["in_buy_zone"] = bool(pv2 <= m["price"] <= pv2 * 1.05)
        if m["price"] > pv2 * 1.05:
            m["pool"] = "C"
        elif m["breakout"] or m["in_buy_zone"]:
            m["pool"] = "B"
        else:
            m["pool"] = "A"
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


def funnel_counts(metrics: list[dict], n_candidates: int) -> list[dict]:
    """候補関門の段階別通過件数。select_candidates の関門を同じ順序・同じ条件で数える。

    最終段の「候補」は select_candidates の実件数で、直前段と一致するのが正常。
    ズレたら関門の定義がどちらかで変わったサインになる。
    """
    n_tt = n_rs = n_high = n_turnover = 0
    for m in metrics:
        if m["tt_pass"] < 6:
            continue
        n_tt += 1
        if m["rs_pct"] < 80:
            continue
        n_rs += 1
        if m["dist_high_pct"] < -25:
            continue
        n_high += 1
        if m["turnover_oku"] < 3.0:
            continue
        n_turnover += 1
    return [
        {"stage": "全銘柄", "count": len(metrics)},
        {"stage": "トレンドテンプレート 6/7以上", "count": n_tt},
        {"stage": "RS 80以上", "count": n_rs},
        {"stage": "52週高値から -25%以内", "count": n_high},
        {"stage": "売買代金 3億円以上", "count": n_turnover},
        {"stage": "候補", "count": n_candidates},
    ]


def sector_momentum(metrics: list[dict], sector_map: dict[str, str]) -> list[dict]:
    """全銘柄の加重リターン (w_ret) を業種別に集計した中央値。

    候補銘柄だけの集計 (sector_summary) と違い、ユニバース全体で業種の
    強弱を見るためのもの。中央値は compute_regime と同じ上限中央値の流儀。
    """
    by_sector: dict[str, list[float]] = {}
    for m in metrics:
        by_sector.setdefault(sector_map.get(m["ticker"], "不明"), []).append(m["w_ret"])
    out = []
    for sec, vals in by_sector.items():
        vals.sort()
        out.append({
            "sector": sec,
            "count": len(vals),
            "median_wret_pct": round(vals[len(vals) // 2], 2),
        })
    out.sort(key=lambda a: -a["median_wret_pct"])
    return out


def update_outcomes(asof: str, cands: list[dict], price_map: dict[str, float], base: Path,
                    bench_price: float | None = None) -> dict:
    """data/outcomes.json に当日候補を追記し、期日が来た過去候補のリターンを記入する。

    絶対リターン (ret_5d/20d/40d/60d) に加え、bench_entry (pick 日の 1306.T 終値)
    がある pick には TOPIX 比の超過リターン (exc_*) も記入する。旧形式の pick
    (bench_entry なし) は絶対リターンのみで後方互換。MFE/MAE は pick 後
    60営業日まで日次で累積する。
    """
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
            "pool": m.get("pool"), "pivot_v2": m.get("pivot_v2"), "stop_v2": m.get("stop_v2"),
            "entry_price": m["price"], "bench_entry": bench_price,
            "ret_5d": None, "ret_20d": None, "ret_40d": None, "ret_60d": None,
            "exc_5d": None, "exc_20d": None, "exc_40d": None, "exc_60d": None,
            "mfe_pct": None, "mae_pct": None,
            "exit_date": None, "exit_price": None, "exit_reason": None,
            "exit_ret": None, "exit_exc": None,
        })

    for p in picks:
        cur = price_map.get(p["ticker"])
        if cur is None or not p["entry_price"]:
            continue
        pick_d = datetime.date.fromisoformat(p["date"])
        if pick_d >= asof_d:
            continue
        bd = len(get_business_days(pick_d + datetime.timedelta(days=1), asof_d))
        r = (cur / p["entry_price"] - 1) * 100
        if bd <= 60 and not p.get("exit_date"):
            prev_hi = p.get("mfe_pct")
            prev_lo = p.get("mae_pct")
            p["mfe_pct"] = round(max(prev_hi if prev_hi is not None else r, r), 2)
            p["mae_pct"] = round(min(prev_lo if prev_lo is not None else r, r), 2)
        bench_ret = None
        if bench_price and p.get("bench_entry"):
            bench_ret = (bench_price / p["bench_entry"] - 1) * 100
        for days, rkey, ekey in ((5, "ret_5d", "exc_5d"), (20, "ret_20d", "exc_20d"),
                                 (40, "ret_40d", "exc_40d"), (60, "ret_60d", "exc_60d")):
            if p.get(rkey) is None and bd >= days:
                p[rkey] = round(r, 2)
                if bench_ret is not None:
                    p[ekey] = round(r - bench_ret, 2)

    cutoff = (asof_d - datetime.timedelta(days=180)).isoformat()
    picks = [p for p in picks if p["date"] >= cutoff]
    data = {"updated": asof, "picks": picks}
    path.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    logger.info(f"outcomes.json 更新: 累計 {len(picks)} 件")
    return data


def _walk_exit(df: pd.DataFrame, pick: dict):
    """1銘柄の価格履歴で売り判定 (S3) を pick 日の翌営業日から走査する。

    優先順位: ストップ → トレンド崩壊 → 分配 → 60営業日 (時間)。
    売却が確定したら (exit_date, exit_price, exit_reason) を返し、
    データの範囲内で未確定なら None を返す。定義はバックテスト
    (backtest_sell) の検証済み実装と同一。
    """
    df = df.dropna(subset=["Close"]).sort_index()
    dates = [d.date().isoformat() for d in df.index]
    if pick["date"] not in dates:
        return None
    i0 = dates.index(pick["date"])
    close = df["Close"]
    entry = pick["entry_price"]
    stop_v2 = pick["stop_v2"]
    sma20 = close.rolling(20).mean()
    sma50 = close.rolling(50).mean()
    volavg20 = df["Volume"].rolling(20).mean()
    chg = close.pct_change() * 100
    rng = df["High"] - df["Low"]
    low_pos = (close - df["Low"]) / rng.where(rng > 0)
    dist = ((chg <= -2.0) & (df["Volume"] > df["Volume"].shift(1))
            & (df["Volume"] >= 1.2 * volavg20) & (low_pos < 0.4)).fillna(False)

    for day in range(1, 61):
        j = i0 + day
        if j >= len(df):
            return None
        c = float(close.iloc[j])
        gain = (c / entry - 1) * 100
        # 1) ストップ
        if c < stop_v2:
            return dates[j], round(c, 1), "ストップ"
        # 2) トレンド崩壊 (含み益 +30%以上では参照線を 20日SMA に切替)
        ref = sma20 if gain >= 30.0 else sma50
        ref_j = ref.iloc[j]
        if pd.notna(ref_j) and c < ref_j:
            vol_ok = pd.notna(volavg20.iloc[j]) and float(df["Volume"].iloc[j]) >= 1.2 * float(volavg20.iloc[j])
            ref_prev = ref.iloc[j - 1]
            two_days = pd.notna(ref_prev) and float(close.iloc[j - 1]) < float(ref_prev)
            if vol_ok or two_days:
                return dates[j], round(c, 1), "トレンド崩壊"
        # 3) 分配 (直近15営業日 = 当日を含む窓、保有期間内のみ)
        lo = max(i0 + 1, j - 14)
        cnt = int(dist.iloc[lo:j + 1].sum())
        peak = float(close.iloc[i0:j + 1].max())
        if cnt >= 3 and c <= peak * 0.95:
            return dates[j], round(c, 1), "分配"
        # 4) 時間 (60営業日)
        if day == 60:
            return dates[j], round(c, 1), "時間"
    return None


def evaluate_exits(asof: str, base: Path, bench=None) -> int:
    """outcomes.json の未売却 pick に売り判定 (S3) を適用し、exit_* を記録する。

    対象は stop_v2 を持つ pick (2026-11-30 以降の新形式) のみ。旧形式の
    pick は従来の固定ホライズン記録のまま対象外。新規に売却が確定した
    件数を返す。
    """
    path = base / "outcomes.json"
    if not path.is_file():
        return 0
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return 0
    picks: list[dict] = data.get("picks", [])
    open_picks = [p for p in picks
                  if p.get("stop_v2") is not None and not p.get("exit_date")
                  and p["date"] < asof]
    if not open_picks:
        return 0

    tickers = sorted({p["ticker"] for p in open_picks})
    earliest = min(p["date"] for p in open_picks)
    start = (datetime.date.fromisoformat(earliest) - datetime.timedelta(days=130)).isoformat()
    end = (datetime.date.fromisoformat(asof) + datetime.timedelta(days=1)).isoformat()
    frames: dict[str, pd.DataFrame] = {}
    for i in range(0, len(tickers), CHUNK):
        batch = tickers[i:i + CHUNK]
        try:
            px = yf.download(batch, start=start, end=end, group_by="ticker",
                             auto_adjust=False, progress=False, threads=True)
        except Exception as e:
            logger.warning(f"売り判定用ダウンロード失敗 (チャンク {i // CHUNK}): {e}")
            continue
        for t in batch:
            try:
                df = px[t] if t in px else None
            except Exception:
                df = None
            if df is not None and len(df):
                frames[t] = df

    bench_close: dict[str, float] = {}
    if bench is not None and len(bench):
        for d, v in zip(bench.index, bench["Close"]):
            bench_close[d.date().isoformat()] = float(v)

    def bench_on(date_str: str):
        if date_str in bench_close:
            return bench_close[date_str]
        prior = [d for d in bench_close if d <= date_str]
        return bench_close[max(prior)] if prior else None

    n_exit = 0
    for p in open_picks:
        df = frames.get(p["ticker"])
        if df is None:
            continue
        hit = _walk_exit(df, p)
        if not hit:
            continue
        exit_date, exit_price, reason = hit
        if exit_date > asof:
            continue
        p["exit_date"] = exit_date
        p["exit_price"] = exit_price
        p["exit_reason"] = reason
        p["exit_ret"] = round((exit_price / p["entry_price"] - 1) * 100, 2)
        b0 = p.get("bench_entry") or bench_on(p["date"])
        b1 = bench_on(exit_date)
        p["exit_exc"] = (round(p["exit_ret"] - (b1 / b0 - 1) * 100, 2)
                         if b0 and b1 else None)
        n_exit += 1
        logger.info(f"売り判定: {p['date']} {p['ticker']} {p['name']} → {reason} "
                    f"({exit_date}, {p['exit_ret']:+.2f}%)")
    if n_exit:
        data["updated"] = asof
        path.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    logger.info(f"売り判定: 対象 {len(open_picks)} 件 / 新規売却 {n_exit} 件")
    return n_exit


def run_advanced_screening(asof: str) -> dict:
    base = data_dir()
    (base / "advanced_history").mkdir(parents=True, exist_ok=True)
    metrics = compute_universe(asof)
    regime = compute_regime(metrics)
    bench = fetch_benchmark(asof)
    bench_price = float(bench["Close"].iloc[-1]) if bench is not None and len(bench) else None
    regime["dd_count_25d"] = distribution_days(bench)
    cands = select_candidates(metrics)
    enrich_earnings(cands, asof)
    sector_map = fetch_sector_map()
    sectors = sector_summary(cands, sector_map)
    payload = {
        "date": asof, "regime": regime, "sectors": sectors,
        "funnel": funnel_counts(metrics, len(cands)),
        "sector_momentum": sector_momentum(metrics, sector_map),
        "candidates": cands,
    }
    text = json.dumps(payload, ensure_ascii=False, indent=1, default=str)
    (base / "advanced_latest.json").write_text(text, encoding="utf-8")
    (base / "advanced_history" / f"{asof}.json").write_text(text, encoding="utf-8")
    price_map = {m["ticker"]: m["price"] for m in metrics}
    update_outcomes(asof, cands, price_map, base, bench_price=bench_price)
    evaluate_exits(asof, base, bench=bench)
    logger.info(f"高度スクリーニング完了: {asof} / 候補 {len(cands)} 件 / 地合い {regime['label']}")
    return payload


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    run_advanced_screening(datetime.date.today().isoformat())
