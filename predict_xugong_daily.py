# 徐工机械 (000425.SZ) 每日自动预测脚本
# 用途：定时任务每天调用，拉最新行情 -> Kronos-base 预测未来 30 个交易日 -> 生成图+CSV。
# 设计：短期(30日)预测更有参考价值；用 base 模型避免首日 level-shift；只给方向参考，不夸大幅度。
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
TODAY = datetime.now()
OUT = os.path.join(HERE, "daily_output", TODAY.strftime("%Y-%m-%d"))
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
        print("fetch err", e)
        return None


def build_future(last_date, n):
    out, d = [], last_date + timedelta(days=1)
    while len(out) < n:
        if d.weekday() < 5:
            out.append(d)
        d += timedelta(days=1)
    return out


def main():
    print(f"[{TODAY:%Y-%m-%d}] 拉取徐工机械(000425) 最新行情 ...")
    df = fetch("sz000425", "2024-01-01", TODAY.strftime("%Y-%m-%d"))
    if df is None or df.empty:
        raise RuntimeError("行情拉取失败")
    print(f"  共 {len(df)} 条，最新 {df['ts'].iloc[-1]:%Y-%m-%d} 收盘 {df['close'].iloc[-1]:.2f}")

    LB, PL = 150, 30
    x = df.iloc[-LB:][["open", "high", "low", "close", "vol"]].copy()
    x["amount"] = 0.0
    x_ts = df.iloc[-LB:]["ts"]
    future = build_future(df["ts"].iloc[-1], PL)
    y_ts = pd.Series(future)
    tok = KronosTokenizer.from_pretrained(TOK)
    model = Kronos.from_pretrained(MODEL)
    pred = KronosPredictor(model, tok, device=DEV, max_context=512).predict(
        df=x, x_timestamp=x_ts, y_timestamp=y_ts, pred_len=PL, T=1.0, top_p=0.95, sample_count=5, verbose=False)
    pred.index = future

    last = df["close"].iloc[-1]
    first = pred["close"].iloc[0]
    lastp = pred["close"].iloc[-1]
    fchg = (first / last - 1) * 100
    lchg = (lastp / last - 1) * 100
    direction = "偏强" if lchg > 0 else "偏弱"
    print(f"  当前 {last:.2f} -> 预测首日 {first:.2f} ({fchg:+.1f}%) 期末 {lastp:.2f} ({lchg:+.1f}%) 短期方向{direction}")

    # CSV
    rep = pd.DataFrame({
        "日期": [d.strftime("%Y-%m-%d") for d in future],
        "预测收盘": pred["close"].values,
        "较当前涨跌幅(%)": ((pred["close"].values / last - 1) * 100),
    })
    csvp = os.path.join(OUT, "徐工机械_预测.csv")
    rep.to_csv(csvp, index=False, encoding="utf-8-sig")

    # 图
    plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9))
    hist = df.set_index("ts")["close"].iloc[-250:]
    ax1.plot(hist.index, hist.values, label="历史收盘", color="#185FA5", lw=1.8)
    ax1.plot(pred.index, pred["close"].values, label=f"Kronos预测({lchg:+.1f}%)", color="#C0392B", lw=1.8, marker="o", ms=2)
    ax1.axhline(last, color="gray", ls=":", alpha=0.6, label=f"当前 {last:.2f}")
    ax1.set_title(f"徐工机械(000425) Kronos 每日预测 {TODAY:%Y-%m-%d}\n当前 {last:.2f} -> 期末 {lastp:.2f} ({lchg:+.1f}%)  短期方向{direction}")
    ax1.set_ylabel("收盘价(元)")
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3)
    ax2.bar(pred.index, (pred["close"].values / last - 1) * 100, color="#C0392B", alpha=0.8)
    ax2.axhline(0, color="black", lw=1)
    ax2.set_title("逐日预测涨跌幅 (%)")
    ax2.set_ylabel("涨跌幅 (%)")
    ax2.grid(alpha=0.3)
    plt.tight_layout()
    pngp = os.path.join(OUT, "徐工机械_预测.png")
    plt.savefig(pngp, dpi=150, bbox_inches="tight")
    print(f"  图: {pngp}")
    print(f"  CSV: {csvp}")
    print("每日预测完成 ✅")


if __name__ == "__main__":
    main()
