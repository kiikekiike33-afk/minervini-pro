# ============================================
# ai_analyzer.py - AI銘柄分析モジュール
# ============================================
# DuckDuckGoで最新ニュース・材料を収集し、
# Gemini API (google-genai SDK) で投資判断レポートを生成する。
# tenacityによる指数バックオフ自動リトライ付き。
# ============================================

import logging
import os
import time
from typing import Any

# Windowsレジストリ（Linux/Streamlit Cloud環境では存在しないため安全にフォールバック）
try:
    import winreg
except ImportError:
    winreg = None

from duckduckgo_search import DDGS
from google import genai
from google.genai import errors
from tenacity import (
    retry, wait_exponential, stop_after_attempt,
    retry_if_exception_type, RetryError
)

from db_manager import save_ai_report, get_ai_report

logger = logging.getLogger(__name__)


def get_api_key_from_registry() -> str:
    """
    Windowsレジストリからユーザー環境変数 GEMINI_API_KEY を直接読み取る。
    setx で設定した値はここに保存されているため、
    プロセスの環境変数に依存せず確実に取得できる。
    """
    if winreg is None:
        return ""

    try:
        key_handle = winreg.OpenKey(
            winreg.HKEY_CURRENT_USER, "Environment"
        )
        value, _ = winreg.QueryValueEx(key_handle, "GEMINI_API_KEY")
        winreg.CloseKey(key_handle)
        return str(value).strip()
    except FileNotFoundError:
        return ""
    except Exception as e:
        logger.debug(f"レジストリからのAPIキー読み取りエラー: {e}")
        return ""


def resolve_api_key(user_input: str = "") -> str:
    """
    APIキーの解決優先順位:
    1. ユーザーがUI上で入力した値
    2. Streamlit Secrets (GEMINI_API_KEY)
    3. Windowsレジストリ（setxで設定された値）
    4. 環境変数 GEMINI_API_KEY
    """
    if user_input and user_input.strip():
        return user_input.strip()

    # Streamlit Cloud の Secrets からの取得
    try:
        import streamlit as st
        if "GEMINI_API_KEY" in st.secrets:
            return str(st.secrets["GEMINI_API_KEY"]).strip()
    except Exception:
        pass

    # レジストリから直接読み取り (Windows)
    registry_key = get_api_key_from_registry()
    if registry_key:
        return registry_key

    # フォールバック: プロセスの環境変数
    return os.environ.get("GEMINI_API_KEY", "")


def gather_web_info(code: str, name: str) -> str:
    """DuckDuckGoで最新のカタリスト（材料）を検索"""
    query = f"{name} {code} 決算 サプライズ 理由 新高値 材料 強み"
    try:
        results = DDGS().text(query, max_results=3)
        texts = []
        for r in results:
            title = r.get("title", "")
            body = r.get("body", "")
            texts.append(f"・{title}\n  {body}")
        return "\n".join(texts) if texts else "有益なウェブ情報が見つかりませんでした。"
    except Exception as e:
        logger.warning(f"ウェブ検索エラー ({code}): {e}")
        return "検索中にエラーが発生しました。"


@retry(
    retry=retry_if_exception_type(errors.APIError),
    wait=wait_exponential(multiplier=2, min=2, max=60),
    stop=stop_after_attempt(5)
)
def _call_gemini(api_key: str, prompt: str) -> str:
    """Gemini APIを呼び出す（リトライ付き）"""
    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
    )
    return response.text


def generate_ai_report(stock: dict, api_key: str) -> tuple[str, str]:
    """
    指定銘柄に対して:
    1. ウェブから最新情報を収集
    2. Gemini AIで投資判断レポートを生成
    3. DBにキャッシュ保存
    戻り値: (web_info, report_text)
    """
    code = stock["code"]
    name = stock["name"]
    date = stock.get("date", "")

    # キャッシュチェック: 同日・同銘柄のレポートが既にあれば再利用
    cached = get_ai_report(code, date)
    if cached and cached.get("report_text"):
        logger.info(f"キャッシュ済みレポートを使用: {name} ({code})")
        return cached.get("web_info", ""), cached["report_text"]

    # ウェブ情報収集
    time.sleep(1)  # API制限配慮
    web_info = gather_web_info(code, name)

    # AIプロンプト構築
    mc_oku = stock.get("market_cap", 0) / 1_0000_0000
    diff_pct = stock.get("diff_52w_high", 0) * 100
    rs_pct = stock.get("rs_outperform", 0) * 100

    prompt = f"""
あなたはプロのクオンツアナリストであり、厳格な成長株投資家です。
以下の日本株データと最新のウェブ情報を元に、投資判断レポートを作成してください。

【対象銘柄】
コード: {code}
銘柄名: {name}
時価総額: 約 {mc_oku:.0f} 億円
52週高値からの乖離率: {diff_pct:.1f}%
日経平均に対するアウトパフォーム（直近3ヶ月）: +{rs_pct:.1f}%

【ウェブ検索から得られた最新のカタリスト・ニュース】
{web_info}

【出力構成（日本語で出力）】
① 【ビジネスモデルと直近の成長材料】なぜ今買われているのか、何が業績を牽引しているか
② 【業績のモメンタム】直近の決算内容やサプライズ要素について、ネットの反応や開示情報から読み取れる点
③ 【投資判断とリスク】ブレイクアウトとしての期待値と、懸念されるリスク（地合い、しこり玉、業績の持続性など）
"""

    # Gemini API呼び出し
    time.sleep(2)  # 無料枠レート制限配慮
    try:
        report_text = _call_gemini(api_key, prompt)
    except RetryError as e:
        # 5回リトライしても失敗した場合、最後の例外を表示
        last_err = e.last_attempt.exception() if e.last_attempt else e
        report_text = f"⚠️ AIレポート生成に失敗しました: {last_err}"
        logger.error(f"AIレポート生成失敗 ({code}): {last_err}")
    except Exception as e:
        report_text = f"⚠️ AIレポート生成に失敗しました: {e}"
        logger.error(f"AIレポート生成失敗 ({code}): {e}")

    # DBにキャッシュ保存
    save_ai_report(code, date, web_info, report_text)

    return web_info, report_text
