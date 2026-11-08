# ============================================
# app.py - Minervini Pro メインUI
# ============================================
# 3つのアプリの全機能を統合したプレミアムダッシュボード。
# 5タブ構成: ダッシュボード / カレンダー / チャート / AI分析 / 設定
# ============================================

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import time

from db_manager import (
    init_db, get_available_dates, get_stocks_by_date,
    get_ai_reports_by_date, save_ai_report, import_bundled_results
)
from data_fetcher import fetch_stock_history
from ai_analyzer import resolve_api_key, generate_ai_report

# ========================================
# ページ設定
# ========================================
st.set_page_config(
    page_title="Minervini Pro v2",
    page_icon="🦅",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ========================================
# プレミアムCSS
# ========================================
st.markdown("""
<style>
    /* Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+JP:wght@300;400;500;700&family=Inter:wght@300;400;500;600;700&display=swap');

    /* 全体のダークテーマ */
    .stApp {
        background: linear-gradient(135deg, #0a0e1a 0%, #0d1526 50%, #111827 100%);
        color: #ffffff;
        font-family: 'Inter', 'Noto Sans JP', sans-serif;
    }

    /* ヘッダー非表示 */
    header[data-testid="stHeader"] {
        background: transparent;
    }

    /* サイドバー */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
        border-right: 1px solid rgba(99, 102, 241, 0.2);
    }
    
    /* 全体テキストの明るさ底上げ */
    label p, label span, div[data-testid="stMarkdownContainer"] > p, 
    .stCaption, div[data-testid="stCaptionContainer"], small,
    div[data-testid="stText"] {
        color: #ffffff !important;
    }

    /* タブスタイル */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
        background: rgba(15, 23, 42, 0.6);
        border-radius: 12px;
        padding: 4px;
        border: 1px solid rgba(99, 102, 241, 0.15);
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 10px;
        padding: 10px 20px;
        color: #ffffff;
        font-weight: 500;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #4f46e5, #7c3aed) !important;
        color: #ffffff !important;
        border-radius: 10px;
    }

    /* メトリクスカード */
    div[data-testid="metric-container"] {
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.08), rgba(124, 58, 237, 0.05));
        border: 1px solid rgba(99, 102, 241, 0.2);
        padding: 18px 20px;
        border-radius: 16px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
        backdrop-filter: blur(10px);
    }
    div[data-testid="metric-container"] label {
        color: #ffffff !important;
    }
    div[data-testid="metric-container"] [data-testid="stMetricValue"] {
        color: #ffffff !important;
        font-weight: 700;
    }

    /* データフレーム */
    [data-testid="stDataFrame"] {
        border-radius: 12px;
        overflow: hidden;
        border: 1px solid rgba(99, 102, 241, 0.15);
    }

    /* ボタン */
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #4f46e5, #7c3aed);
        border: none;
        border-radius: 12px;
        font-weight: 600;
        padding: 12px 28px;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(79, 70, 229, 0.3);
    }
    .stButton > button[kind="primary"]:hover {
        box-shadow: 0 6px 25px rgba(79, 70, 229, 0.5);
        transform: translateY(-1px);
    }

    /* カスタムカード */
    .glass-card {
        background: rgba(15, 23, 42, 0.6);
        border: 1px solid rgba(99, 102, 241, 0.15);
        border-radius: 16px;
        padding: 24px;
        backdrop-filter: blur(10px);
        margin-bottom: 16px;
    }
    .report-card {
        background: linear-gradient(135deg, rgba(15, 23, 42, 0.8), rgba(30, 41, 59, 0.6));
        border: 1px solid rgba(99, 102, 241, 0.2);
        border-radius: 16px;
        padding: 28px;
        margin-top: 16px;
        line-height: 1.8;
        color: #ffffff;
    }
    .report-card h3, .report-card h4 {
        color: #c4b5fd;
        margin-top: 16px;
    }
    .report-card strong {
        color: #ddd6fe;
    }

    /* アラートスタイル */
    .big-alert {
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.1), rgba(124, 58, 237, 0.05));
        border: 1px solid rgba(99, 102, 241, 0.3);
        border-radius: 16px;
        padding: 32px;
        text-align: center;
        margin: 24px 0;
    }
    .big-alert h2 {
        color: #a78bfa;
        margin-bottom: 8px;
    }
    .big-alert p {
        color: #ffffff;
        font-size: 1.05rem;
    }

    /* タイトルグラデーション */
    .gradient-title {
        background: linear-gradient(135deg, #818cf8, #a78bfa, #c084fc);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.2rem;
        font-weight: 700;
        letter-spacing: -0.02em;
    }
    .subtitle {
        color: #ffffff;
        font-size: 1.0rem;
        margin-top: -8px;
    }
</style>
""", unsafe_allow_html=True)

# ========================================
# DB初期化
# ========================================
init_db()
# リポジトリ同梱の結果JSON (data/) があれば、DBに無い日付を取り込む。
# Streamlit Cloud は DB が揮発性のため、コミット済みJSONが表示の正になる。
try:
    import_bundled_results()
except Exception:
    pass

# ========================================
# サイドバー
# ========================================
with st.sidebar:
    st.divider()

    api_key = resolve_api_key(None)
    if api_key:
        st.success("APIキー: ✅ 設定済み（環境変数）", icon="🔑")
    else:
        st.warning("APIキー: ⚠️ 未設定（AI分析タブで必要）", icon="🔑")

    st.divider()

    # 日付選択
    dates = get_available_dates()
    if dates:
        selected_date = st.selectbox("📅 日付を選択", dates, index=0)
    else:
        selected_date = None
        st.info("まだスクリーニングデータがありません。")

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🚀 最新データでスクリーニング実行", type="primary", use_container_width=True):
        with st.spinner("フルスクリーニングを実行中...（約3〜5分）"):
            from screener import run_full_screening
            import datetime
            results = run_full_screening(datetime.date.today())
            st.success(f"完了！ {len(results)}件抽出。")
            time.sleep(2)
            st.rerun()

    st.divider()
    st.markdown("### 📊 スクリーニング条件")
    st.markdown("""
    <small>
    <b>【基本トレンド】</b><br>
    ① 株価 > SMA200（上昇トレンド）<br>
    ② SMA200が上向き<br>
    ③ 上方乖離の制限（SMA50の+20%未満、SMA200の+40%未満）<br>
    <br>
    <b>【3ヶ月以上ぶりの新高値】</b><br>
    ④ 過去1年の最高値が3ヶ月以上前に形成<br>
    ⑤ 直近5日でその旧高値を突破<br>
    <br>
    <b>【出来高確認】</b><br>
    ⑥ 出来高 ≥ 50日平均×1.5倍<br>
    <br>
    <b>【流動性フィルタ】</b><br>
    ⑦ 20日平均売買代金 ≥ 3億円<br>
    <br>
    <b>【ファンダメンタルズ健全性】</b><br>
    ⑧ 今期予想純利益が黒字（赤字除外）<br>
    ⑨ 直近決算の営業CFがプラス
    </small>
    """, unsafe_allow_html=True)



# ========================================
# セッションステート初期化
# ========================================
if "selected_stock_code" not in st.session_state:
    st.session_state.selected_stock_code = None
if "ai_results" not in st.session_state:
    st.session_state.ai_results = {}

# ========================================
# 選択日のデータ取得
# ========================================
stocks_data = get_stocks_by_date(selected_date) if selected_date else []

# 表示用DataFrame生成関数
def make_display_df(stocks: list[dict]) -> pd.DataFrame:
    if not stocks:
        return pd.DataFrame()
    df = pd.DataFrame(stocks)
    
    # 時価総額（旧データ互換）
    if "market_cap" in df.columns and df["market_cap"].notna().any():
        df["時価総額(億円)"] = (df["market_cap"] / 1_0000_0000).fillna(0).astype(int)
    else:
        df["時価総額(億円)"] = "-"
    
    df["高値乖離(%)"] = (df["diff_52w_high"] * 100).round(1)
    df["RS(%)"] = (df["rs_outperform"] * 100).round(1)
    
    # 過去データでカラムが存在しない場合への対応
    if "sma20_diff" not in df.columns:
        df["sma20_diff"] = None
    if "rsi14" not in df.columns:
        df["rsi14"] = None
    if "vcp_contraction" not in df.columns:
        df["vcp_contraction"] = None
    if "breakout_vol_ratio" not in df.columns:
        df["breakout_vol_ratio"] = None
    if "signal_type" not in df.columns:
        df["signal_type"] = None
        
    df["VCP収縮(%)"] = df["vcp_contraction"].astype(float).round(0)
    df["出来高倍率"] = df["breakout_vol_ratio"].astype(float).round(1)
    df["シグナル"] = df["signal_type"].fillna("-")
    df["RSI(14)"] = df["rsi14"].astype(float).round(1)
    
    df["株価"] = df["close_price"].round(0).astype(int)

    # 予想EPS・営業CF
    if "forward_eps" in df.columns:
        df["予想EPS"] = df["forward_eps"].apply(lambda x: f"¥{x:.1f}" if pd.notna(x) and x != "" else "-")
    else:
        df["予想EPS"] = "-"

    if "operating_cf" in df.columns:
        df["営業CF"] = df["operating_cf"].apply(lambda x: f"¥{x/1_0000_0000:,.1f}億" if pd.notna(x) and x != "" else "-")
    else:
        df["営業CF"] = "-"
    
    # 銘柄スカウター用のURLを構築
    base_code = df["code"].str.replace(".T", "", regex=False)
    df["銘柄スカウターURL"] = "https://monex.ifis.co.jp/index.php?sa=report_zaimu&bcode=" + base_code + "&name=" + df["name"]
    
    disp = df[["code", "銘柄スカウターURL", "株価", "予想EPS", "営業CF", "VCP収縮(%)", "出来高倍率", "シグナル", "高値乖離(%)", "RS(%)", "RSI(14)", "comment"]]
    disp.columns = ["コード", "銘柄名", "株価(円)", "予想EPS", "営業CF", "VCP収縮(%)", "出来高倍率", "シグナル", "高値乖離(%)", "RS(%)", "RSI(14)", "💡 注目ポイント"]
    return disp

# ========================================
# チャート生成関数
# ========================================
def create_stock_chart(code: str) -> go.Figure | None:
    hist = fetch_stock_history(code, "2y")
    if hist is None or hist.empty:
        return None

    hist["SMA20"]  = hist["Close"].rolling(20).mean()
    hist["SMA50"]  = hist["Close"].rolling(50).mean()
    hist["SMA150"] = hist["Close"].rolling(150).mean()
    hist["SMA200"] = hist["Close"].rolling(200).mean()
    high_52w = hist["High"].rolling(window=250, min_periods=1).max().iloc[-1]

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        vertical_spacing=0.03, row_heights=[0.7, 0.3],
    )

    fig.add_trace(go.Candlestick(
        x=hist.index, open=hist["Open"], high=hist["High"],
        low=hist["Low"], close=hist["Close"], name="株価",
        increasing_line_color="#10b981", decreasing_line_color="#f43f5e",
        increasing_fillcolor="#10b981", decreasing_fillcolor="#f43f5e",
        line_width=1.5
    ), row=1, col=1)

    fig.add_trace(go.Scatter(x=hist.index, y=hist["SMA20"], name="SMA20", line=dict(color="#facc15", width=1.5, dash="dot"), hoverinfo="skip"), row=1, col=1)
    fig.add_trace(go.Scatter(x=hist.index, y=hist["SMA50"], name="SMA50", line=dict(color="#38bdf8", width=1.8), hoverinfo="skip"), row=1, col=1)
    fig.add_trace(go.Scatter(x=hist.index, y=hist["SMA150"], name="SMA150", line=dict(color="#a78bfa", width=1.8), hoverinfo="skip"), row=1, col=1)
    fig.add_trace(go.Scatter(x=hist.index, y=hist["SMA200"], name="SMA200", line=dict(color="#fb923c", width=1.8), hoverinfo="skip"), row=1, col=1)

    fig.add_hline(
        y=high_52w, line_dash="dash", line_color="#ec4899", line_width=1.5,
        annotation_text=f" 52週高値 (¥{high_52w:,.0f})", annotation_position="bottom right",
        annotation_font=dict(color="#ec4899", size=11, family="Inter"), row=1, col=1
    )

    vol_colors = ["rgba(16, 185, 129, 0.6)" if c >= o else "rgba(244, 63, 94, 0.6)" for c, o in zip(hist["Close"], hist["Open"])]
    fig.add_trace(go.Bar(
        x=hist.index, y=hist["Volume"], name="出来高",
        marker_color=vol_colors, marker_line_width=0, showlegend=False, hoverinfo="y"
    ), row=2, col=1)

    fig.update_layout(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(15, 23, 42, 0.4)",
        font=dict(family="Inter, Noto Sans JP, sans-serif", size=11, color="#e2e8f0"),
        xaxis_rangeslider_visible=False, height=500, margin=dict(l=10, r=10, t=10, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, bgcolor="rgba(0,0,0,0)"),
        hovermode="x unified", hoverlabel=dict(bgcolor="rgba(15, 23, 42, 0.9)", font_size=12, font_family="Inter"), bargap=0,
    )
    fig.update_xaxes(
        gridcolor="rgba(99,102,241,0.08)", zeroline=False, showspikes=True, spikecolor="rgba(167, 139, 250, 0.5)",
        spikesnap="cursor", spikemode="across", spikethickness=1,
        rangeselector=dict(
            buttons=list([
                dict(count=3, label="3M", step="month", stepmode="backward"),
                dict(count=6, label="6M", step="month", stepmode="backward"),
                dict(count=1, label="1Y", step="year", stepmode="backward"),
                dict(step="all", label="2Y")
            ]), bgcolor="rgba(30, 41, 59, 0.8)", activecolor="rgba(99, 102, 241, 0.8)", font=dict(color="#e2e8f0", size=10), y=1.05, x=0
        )
    )
    fig.update_yaxes(gridcolor="rgba(99,102,241,0.08)", zeroline=False, showspikes=True, spikecolor="rgba(167, 139, 250, 0.5)", spikesnap="cursor", spikemode="across", spikethickness=1)
    return fig

# ========================================
# メイン画面ヘッダー & KPI
# ========================================
st.markdown('<p class="gradient-title">⚡ Minervini Pro V2</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">ミネルヴィニ・トレンドテンプレート(CWH) × Gemini AI 統合投資分析プラットフォーム</p>', unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)
col1.metric("総監視銘柄数", "約4,000 銘柄", "全上場企業")
col2.metric("条件クリア銘柄", f"{len(stocks_data)} 銘柄" if stocks_data else "0 銘柄", "厳格化(CWH/VCP)")
col3.metric("日経平均トレンド", "確認済", "Bull")
col4.metric("API ステータス", "正常" if api_key else "未設定", "Gemini-Flash")

st.markdown("<br>", unsafe_allow_html=True)

# ========================================
# 4タブ構成
# ========================================
tab_data, tab_cal, tab_chart, tab_insights = st.tabs([
    "🏠 ダッシュボード", "📅 カレンダー", "📈 チャート", "🤖 AI分析"
])

# ========================================
# Tab 1: ダッシュボード (Data)
# ========================================
with tab_data:
    if not selected_date:
        st.markdown("""
        <div class="big-alert">
            <h2>📊 データがありません</h2>
            <p>まだスクリーニングが実行されていません。<br>
            「⚙️ 設定」タブからバッチ実行するか、日次自動バッチの結果をお待ちください。</p>
        </div>
        """, unsafe_allow_html=True)
    elif not stocks_data:
        st.markdown(f"""
        <div class="big-alert">
            <h2>📉 該当銘柄なし</h2>
            <p>{selected_date} のS・A評価スクリーニング条件をクリアした銘柄はありませんでした。<br>
            3段階すべて（トレンド → VCP → ブレイクアウト）を満たす銘柄が出現するまで待ちましょう。</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        # サマリーメトリクス
        n_stocks = len(stocks_data)
        avg_rs = sum(s["rs_outperform"] for s in stocks_data) / n_stocks * 100
        best_stock = max(stocks_data, key=lambda x: x["rs_outperform"])

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("📊 通過銘柄数", f"{n_stocks} 銘柄")
        m2.metric("📈 平均RS", f"+{avg_rs:.1f}%")
        m3.metric("🏆 最強RS", f"{best_stock['name']}", f"+{best_stock['rs_outperform']*100:.1f}%")
        m4.metric("📅 分析日", selected_date)

        st.markdown("")

        disp_df = make_display_df(stocks_data)
        st.markdown("##### 👇 銘柄の行をクリックすると「📈 チャート」タブで詳細を確認できます")

        # st.dataframe with Column Config for formatting
        selection = st.dataframe(
            disp_df,
            use_container_width=True,
            hide_index=True,
            on_select="rerun",
            selection_mode="single-row",
            column_config={
                "銘柄名": st.column_config.LinkColumn(
                    "銘柄名",
                    display_text=r"name=(.+)$"
                ),
                "VCP収縮(%)": st.column_config.NumberColumn(format="%.0f%%"),
                "出来高倍率": st.column_config.NumberColumn(format="%.1fx"),
                "高値乖離(%)": st.column_config.NumberColumn(format="%.1f%%"),
                "RS(%)": st.column_config.NumberColumn(format="+%.1f%%"),
                "RSI(14)": st.column_config.NumberColumn(format="%.1f"),
                "株価(円)": st.column_config.NumberColumn(format="¥%d"),
            }
        )

        # 行選択時にチャートタブ用にセッションに保存
        if selection and selection.selection and selection.selection.rows:
            idx = selection.selection.rows[0]
            st.session_state.selected_stock_code = stocks_data[idx]["code"]

# ========================================
# Tab 2: テクニカルチャート
# ========================================
with tab_chart:
    st.subheader("📈 テクニカル分析チャート")

    if not stocks_data:
        st.info("スクリーニングデータがありません。ダッシュボードから銘柄を選択してください。")
    else:
        # 銘柄選択
        stock_options = {f"{s['name']} ({s['code']})": s for s in stocks_data}
        default_key = None
        if st.session_state.selected_stock_code:
            for k, v in stock_options.items():
                if v["code"] == st.session_state.selected_stock_code:
                    default_key = k
                    break

        selected_label = st.selectbox(
            "表示する銘柄を選択",
            list(stock_options.keys()),
            index=list(stock_options.keys()).index(default_key) if default_key else 0
        )
        selected_stock = stock_options[selected_label]
        
        st.session_state.selected_stock_code = selected_stock["code"]

        # 指標表示
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("株価", f"¥{selected_stock['close_price']:,.0f}")
        
        f_eps = selected_stock.get('forward_eps')
        c2.metric("予想EPS", f"¥{f_eps:.1f}" if pd.notna(f_eps) and f_eps != "" else "-")

        op_cf = selected_stock.get('operating_cf')
        c3.metric("営業CF", f"¥{op_cf/1_0000_0000:,.1f}億" if pd.notna(op_cf) and op_cf != "" else "-")
            
        diff_52w = selected_stock.get('diff_52w_high', 0.0)
        c4.metric("高値乖離", f"{float(diff_52w) * 100:.1f}%")
        
        c4.metric("RS (対日経)", f"+{selected_stock['rs_outperform']*100:.1f}%")

        if selected_stock.get("comment"):
            st.markdown(f"**💡 {selected_stock['comment']}**")

        # チャート描画
        with st.spinner("チャートデータを取得中..."):
            fig = create_stock_chart(selected_stock["code"])
            
        if fig:
            st.plotly_chart(fig, use_container_width=True, key="main_chart")
        else:
            st.error("チャートデータの取得に失敗しました。")

# ========================================
# Tab 2: AI 銘柄分析 (Insights)
# ========================================
with tab_insights:
    st.subheader("🤖 Gemini AI カタリスト ＆ ファンダメンタルズ評価")

    if not stocks_data:
        st.info("スクリーニング結果がありません。")
    elif not api_key:
        st.warning("AI分析を実行するには、サイドバーからGemini APIキーを設定してください。")
    else:
        cached_reports = get_ai_reports_by_date(selected_date) if selected_date else {}
        
        # 抽出された銘柄ごとにExpander（アコーディオン）でレポートを折りたたんで表示
        for stock in stocks_data:
            ticker = stock['code']
            name = stock['name']
            has_report = ticker in cached_reports
            
            icon = "✅" if has_report else "🔥"
            with st.expander(f"{icon} {ticker} {name} - 詳細レポートを開く", expanded=False):
                chart_col, text_col = st.columns([1, 1.2])
                
                with chart_col:
                    st.markdown("**日足チャート (過去2年)**")
                    with st.spinner("チャート読込中..."):
                        fig = create_stock_chart(ticker)
                        if fig:
                            st.plotly_chart(fig, use_container_width=True, key=f"chart_{ticker}")
                        else:
                            st.error("チャート取得失敗")
                
                with text_col:
                    st.markdown("#### 🤖 AI 分析ハイライト")
                    
                    if has_report:
                        rpt = cached_reports[ticker]
                        report_text = rpt.get("report_text", "レポートがありません。")
                        st.markdown(f'<div class="report-card" style="margin-top:0;">{report_text.replace(chr(10), "<br>")}</div>', unsafe_allow_html=True)
                    else:
                        st.info("この銘柄のAI分析はまだ実行されていません。")
                        if st.button(f"▶️ {name} のAI分析を実行", key=f"btn_ai_{ticker}", type="primary"):
                            with st.spinner("Gemini API で最新カタリストを分析中..."):
                                web_info, report = generate_ai_report(stock, api_key)
                                save_ai_report(ticker, selected_date, web_info, report)
                                st.rerun()

# ========================================
# Tab 3: カレンダー（履歴閲覧）
# ========================================
with tab_cal:
    st.subheader("📅 スクリーニング履歴カレンダー")

    if not dates:
        st.info("まだ履歴データがありません。")
    else:
        # 日付ごとの銘柄数を一覧表示
        cal_data = []
        for d in dates:
            day_stocks = get_stocks_by_date(d)
            cal_data.append({
                "日付": d,
                "通過銘柄数": len(day_stocks),
                "銘柄": ", ".join([s["name"] for s in day_stocks[:5]]) + ("..." if len(day_stocks) > 5 else "") if day_stocks else "—"
            })

        cal_df = pd.DataFrame(cal_data)
        st.dataframe(
            cal_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "通過銘柄数": st.column_config.NumberColumn(format="%d 銘柄"),
            }
        )

        st.divider()

        # 選択日の詳細表示
        if selected_date and stocks_data:
            st.markdown(f"##### 📋 {selected_date} の通過銘柄一覧")
            disp_df = make_display_df(stocks_data)
            st.dataframe(disp_df, use_container_width=True, hide_index=True)
        elif selected_date:
            st.markdown(f"""
            <div class="big-alert">
                <h2>📉 {selected_date}</h2>
                <p>この日はトレンドテンプレート条件をクリアした銘柄はありませんでした。</p>
            </div>
            """, unsafe_allow_html=True)


