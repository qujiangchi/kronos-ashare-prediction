# 徐工机械 + 对照股 方向验证（walk-forward 回测 + 系统性看空 bias 检验）
# 目的：验证"Kronos 是否对所有股票都系统性预测下跌"（用户质疑方向反了）。
import sys, os
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import requests
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(HERE)
from model import Kronos, KronosTokenizer, KronosPredictor

TOK = "NeoQuasar/Kronos-Tokenizer-base"
MODEL = "NeoQuasar/Kronos-base"
DEV = "cpu"
URL = "https://web.ifzq.gtimg.cn/appstock/app/fqkline/get"
H = {"User-Agent": "Mozilla/5.0"}
OUT = os.path.join(HERE, "xugong_v2_output")
os.makedirs(OUT, exist_ok=True)


def fetch(sec, start, end, count=1024):
    try:
        r = requests.get(URL, params={"param": f"{sec},day,{start},{end},{count},"}, headers=H, timeout=20)
        j = r.json()
        node = j.get("data", {}).get(sec)
        if isinstance(node, dict):
            rows = node.get("day") or node.get("qfqday")
        elif isinstance(node, list):
            rows = node
        else:
            rows = None
        if not rows:
            return None
        df = pd.DataFrame([r[:6] for r in rows], columns=["ts", "open", "close", "high", "low", "vol"])
        for c in ["open", "close", "high", "low", "vol"]:
            df[c] = df[c].astype(float)
        df["ts"] = pd.to_datetime(df["ts"])
        return df.sort_values("ts").reset_index(drop=True)
    except Exception as e:
        print("  fetch err", e)
        return None


print("加载模型 ...")
tok = KronosTokenizer.from_pretrained(TOK)
model = Kronos.from_pretrained(MODEL)
pred_inst = KronosPredictor(model, tok, device=DEV, max_context=512)


def predict_from(x_df, x_ts, y_ts, pl):
    x = x_df[["open", "high", "low", "close", "vol"]].copy()
    x["amount"] = 0.0
    pred = pred_inst.predict(df=x, x_timestamp=x_ts, y_timestamp=y_ts, pred_len=pl,
                             T=1.0, top_p=0.95, sample_count=5, verbose=False)
    pred.index = y_ts.values
    return pred


# ---------- Part A: 徐工机械 walk-forward 回测 2024 -> 2025H1 ----------
print("\n[Part A] 徐工机械 walk-forward 回测 (2024 数据 -> 2025H1 真实)")
dfx = fetch("sz000425", "2024-01-01", "2025-12-31")
cut = pd.Timestamp("2024-12-31")
train = dfx[dfx.ts <= cut]
test = dfx[(dfx.ts > cut) & (dfx.ts <= pd.Timestamp("2025-06-30"))]
LB, PL = 150, len(test)
xA, x_tsA = train.iloc[-LB:], train.iloc[-LB:]["ts"]
predA = predict_from(xA, x_tsA, test["ts"], PL)
realA, predA_c = test["close"].values, predA["close"].values
rchg = realA[-1] / realA[0] - 1
pchg = predA_c[-1] / realA[0] - 1
print(f"  真实 2025H1: {realA[0]:.2f} -> {realA[-1]:.2f} ({rchg:+.1%}) 方向={'涨' if rchg>0 else '跌'}")
print(f"  预测 2025H1: {realA[0]:.2f} -> {predA_c[-1]:.2f} ({pchg:+.1%}) 方向={'涨' if pchg>0 else '跌'}")
print(f"  => 方向一致: {(rchg>0)==(pchg>0)}")

# ---------- Part B: 找近期最涨 A 股，验证系统性看空 bias ----------
print("\n[Part B] 找近期最涨 A 股，验证 Kronos 是否系统性看空")
cands = [("300308", "sz"), ("688256", "sh"), ("300502", "sz"), ("002594", "sz"),
         ("601318", "sh"), ("600519", "sh"), ("000858", "sz"), ("601127", "sh")]
best = None
for code, pre in cands:
    dfc = fetch(pre + code, "2025-06-01", "2026-08-25")
    if dfc is None or len(dfc) < 90:
        continue
    cl = dfc["close"].values
    r60 = cl[-1] / cl[-60] - 1
    if best is None or r60 > best[2]:
        best = (pre + code, dfc, r60)
print(f"  近期最涨股: {best[0]}  60日收益 {best[2]:+.1%}")
dfb = best[1]
LB, PL = 150, 30
xB, x_tsB = dfb.iloc[-(LB + PL):-PL], dfb.iloc[-(LB + PL):-PL]["ts"]
yB, y_tsB = dfb.iloc[-PL:], dfb.iloc[-PL:]["ts"]
predB = predict_from(xB, x_tsB, y_tsB, PL)
realB, predB_c = yB["close"].values, predB["close"].values
rchgB = realB[-1] / realB[0] - 1
pchgB = predB_c[-1] / realB[0] - 1
print(f"  真实 近30日: {realB[0]:.2f} -> {realB[-1]:.2f} ({rchgB:+.1%}) 方向={'涨' if rchgB>0 else '跌'}")
print(f"  预测 近30日: {realB[0]:.2f} -> {predB_c[-1]:.2f} ({pchgB:+.1%}) 方向={'涨' if pchgB>0 else '跌'}")
print(f"  => 方向一致: {(rchgB>0)==(pchgB>0)}")

# ---------- 图 ----------
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial"]
plt.rcParams["axes.unicode_minus"] = False
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
ax1.plot(test["ts"], realA, label=f"真实 2025H1 ({rchg:+.1%})", color="#185FA5", lw=2)
ax1.plot(predA.index, predA_c, label=f"Kronos预测 ({pchg:+.1%})", color="#C0392B", lw=1.8, ls="--")
ax1.set_title(f"Part A 徐工回测: 真实 vs 预测  方向一致={(rchg>0)==(pchg>0)}")
ax1.legend(); ax1.grid(alpha=0.3)
ax2.plot(yB["ts"], realB, label=f"真实 近30日 ({rchgB:+.1%})", color="#185FA5", lw=2)
ax2.plot(predB.index, predB_c, label=f"Kronos预测 ({pchgB:+.1%})", color="#C0392B", lw=1.8, ls="--")
ax2.set_title(f"Part B {best[0]}: 真实 vs 预测  方向一致={(rchgB>0)==(pchgB>0)}")
ax2.legend(); ax2.grid(alpha=0.3)
plt.tight_layout()
png = os.path.join(OUT, "xugong_backtest.png")
plt.savefig(png, dpi=150, bbox_inches="tight")
print(f"\n图: {png}")
print("回测与对照完成 ✅")
