import logging
logging.basicConfig(level=logging.INFO)
import datetime
from screener import run_full_screening

results = run_full_screening(datetime.date(2026, 7, 16))

print(f"\n=== 最終結果: {len(results)}件 ===")
for r in results:
    print(f"  {r['name']} ({r['code']}) VCP:{r['vcp_contraction']:.0f}% Vol:{r['breakout_vol_ratio']:.1f}x [{r['signal_type']}]")
