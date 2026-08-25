# 徐工机械 (000425.SZ) 重新预测 + 偏差诊断
# 目标：定位上一版 -34% 偏离现实的根因，并通过缩短回看/预测窗口给出更合理的重预测。
#
# 关键洞察（来自 model/kronos.py 的 predict）：
#   模型用【整个回看窗口】的均值/标准差对价格做 z-score 标准化，预测后再反标准化回同一均值。
#   下行趋势中窗口均值高于现价，任何在标准化空间的适度向下偏移，经"高均值锚点"反标准化后
#   会被放大成对绝对价位的显著下跌（level-shift）。缩短回看窗口(LB)可让锚点贴近现价，显著削弱该效应。

import sys
import os
from datetime import datetime, timedelta
import math

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import akshare as ak

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(HERE)
from model import Kronos, KronosTokenizer, KronosPredictor

STOCK_CODE = "000425"
STOCK_NAME = "徐工机械"
START_DATE = "20230101"
END_DATE = datetime.now().strftime("%Y%m%d")
TOKENIZER_NAME = "NeoQuasar/Kronos-Tokenizer-base"
DEVICE = "cpu"
COLS = ["open", "high", "low", "close", "volume", "amount"]
OUT_DIR = os.path.join(HERE, "xugong_v2_output")
os.makedirs(OUT_DIR, exist_ok=True)


def fetch_data():
    print(f"[数据] akshare 拉取 {STOCK_NAME}({STOCK_CODE}) 日线 {START_DATE}~{END_DATE} ...")
    df, last_err = None, None
    for attempt in range(3):
        try:
            df = ak.stock_zh_a_daily(
                symbol="sz" + STOCK_CODE, start_date=START_DATE,
                end_date=END_DATE, adjust="qfq",
            )
            if df is not None and not df.empty:
                break
        except Exception as e:
            last_err = e
            print(f"      ⚠ 第 {attempt + 1} 次重试: {type(e).__name__}")
    if df is None or df.empty:
        raise RuntimeError(f"无法获取行情: {last_err}")
    df = df.rename(columns={"date": "timestamps"})
    df["timestamps"] = pd.to_datetime(df["timestamps"])
    for col in COLS:
        if col not in df.columns:
            df[col] = 0.0
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    df = df.sort_values("timestamps").reset_index(drop=True)
    print(f"      ✅ {len(df)} 条，区间 {df['timestamps'].min().date()}~{df['timestamps'].max().date()}，最新收盘 {df['close'].iloc[-1]:.2f}")
    return df


def build_future_dates(last_date, n):
    future, d = [], last_date + timedelta(days=1)
    while len(future) < n:
        if d.weekday() < 5:
            future.append(d)
        d += timedelta(days=1)
    return future


def diagnose(df):
    close = df["close"]
    last = close.iloc[-1]
    print("\n[诊断] 价格统计与归一化锚点")
    for lb in (400, 250, 150, 120):
        seg = close.iloc[-lb:]
        print(f"      回看 {lb:3d} 日: 均值 {seg.mean():.3f}  标准差 {seg.std():.3f}  (现价 {last:.2f}, 均值偏离现价 {(seg.mean()/last-1)*100:+.1f}%)")
    ret20 = close.iloc[-1] / close.iloc[-20] - 1
    ret60 = close.iloc[-1] / close.iloc[-60] - 1
    ret250 = close.iloc[-1] / close.iloc[-250] - 1
    daily = close.pct_change().iloc[-250:]
    ann_vol = daily.std() * math.sqrt(252)
    recent_daily_vol = close.pct_change().iloc[-30:].std()
    print(f"      近期收益: 20日 {ret20:+.1%}  60日 {ret60:+.1%}  250日 {ret250:+.1%}")
    print(f"      年化波动率 {ann_vol:.1%}  | 近30日日均波动(1σ) {recent_daily_vol:.2%}")
    print(f"      ⚠ A股主板日涨跌停限制 ±10%；原预测首日跳空需远小于此才合理")
    return {"last": last, "ann_vol": ann_vol, "recent_daily_vol": recent_daily_vol}


def get_predictor(model_name, tokenizer, cache):
    if model_name not in cache:
        print(f"      ⏬ 加载模型 {model_name} ...")
        model = Kronos.from_pretrained(model_name)
        cache[model_name] = KronosPredictor(model, tokenizer, device=DEVICE, max_context=512)
    return cache[model_name]


def run_config(df, name, model_name, lb, pl, T, top_p, sc, tokenizer, cache):
    pred_inst = get_predictor(model_name, tokenizer, cache)
    x_df = df.loc[-lb:, COLS].reset_index(drop=True)
    x_ts = df.loc[-lb:, "timestamps"].reset_index(drop=True)
    future = build_future_dates(df["timestamps"].iloc[-1], pl)
    y_ts = pd.Series(future)
    pred = pred_inst.predict(df=x_df, x_timestamp=x_ts, y_timestamp=y_ts, pred_len=pl,
                             T=T, top_p=top_p, sample_count=sc, verbose=False)
    pred = pred.iloc[:len(future)]
    pred.index = future
    last = df["close"].iloc[-1]
    first_chg = (pred["close"].iloc[0] / last - 1) * 100
    last_chg = (pred["close"].iloc[-1] / last - 1) * 100
    print(f"      [{name}] LB={lb} PL={pl} T={T} sc={sc}: 首日 {pred['close'].iloc[0]:.2f} ({first_chg:+.1f}%) 期末 {pred['close'].iloc[-1]:.2f} ({last_chg:+.1f}%)")
    return {"name": name, "pred": pred, "first_chg": first_chg, "last_chg": last_chg,
            "lb": lb, "pl": pl, "T": T, "sc": sc, "model": model_name.split('/')[-1]}


def main():
    df = fetch_data()
    diag = diagnose(df)
    last = diag["last"]

    tokenizer = KronosTokenizer.from_pretrained(TOKENIZER_NAME)
    cache = {}

    configs = [
        # 原配置（复现偏离）
        ("A_原配置(LB400/PL100)", "NeoQuasar/Kronos-small", 400, 100, 1.2, 0.95, 3),
        # 仅缩短预测窗口
        ("B_短预测(LB400/PL30)",  "NeoQuasar/Kronos-small", 400, 30,  1.0, 0.95, 5),
        # 缩短回看窗口（锚点贴近现价）
        ("C_短回看(LB150/PL30)",  "NeoQuasar/Kronos-small", 150, 30,  1.0, 0.95, 5),
        # Kronos-base（更大容量，验证漂移是否由模型容量导致）
        ("D_base(LB150/PL30)",    "NeoQuasar/Kronos-base",  150, 30,  1.0, 0.95, 5),
    ]
    results = []
    print("\n[预测] 多配置对比")
    for c in configs:
        try:
            results.append(run_config(df, *c, tokenizer, cache))
        except Exception as e:
            print(f"      ⚠ 配置 {c[0]} 运行失败: {type(e).__name__}: {str(e)[:80]}; 已跳过")

    # ---- 汇总表 ----
    print("\n[汇总]")
    summary_rows = []
    for r in results:
        print(f"  {r['name']:24s} 首日 {r['first_chg']:+7.1f}%  期末 {r['last_chg']:+7.1f}%")
        summary_rows.append({
            "配置": r["name"], "模型": r["model"], "回看": r["lb"], "预测步": r["pl"],
            "温度T": r["T"], "采样数": r["sc"], "首日涨跌幅%": round(r["first_chg"], 2),
            "期末涨跌幅%": round(r["last_chg"], 2),
        })
    sum_df = pd.DataFrame(summary_rows)
    sum_df.to_csv(os.path.join(OUT_DIR, "summary.csv"), index=False, encoding="utf-8-sig")
    print(f"      💾 汇总: {os.path.join(OUT_DIR, 'summary.csv')}")

    # 各配置预测 CSV
    for r in results:
        rep = pd.DataFrame({
            "日期": [d.strftime("%Y-%m-%d") for d in r["pred"].index],
            "预测收盘": r["pred"]["close"].values,
            "较当前涨跌幅(%)": ((r["pred"]["close"].values / last - 1) * 100),
        })
        safe = r["name"].replace("/", "_")
        rep.to_csv(os.path.join(OUT_DIR, f"pred_{safe}.csv"), index=False, encoding="utf-8-sig")

    # ---- 绘图 ----
    plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial"]
    plt.rcParams["axes.unicode_minus"] = False
    colors = {"A_原配置(LB400/PL100)": "#C0392B",
              "B_短预测(LB400/PL30)": "#BA7517",
              "C_短回看(LB150/PL30)": "#1E8449"}
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 11))
    hist_plot = df.set_index("timestamps")["close"].iloc[-min(250, len(df)):]
    ax1.plot(hist_plot.index, hist_plot.values, label="历史收盘", color="#185FA5", lw=1.8)
    for r in results:
        ax1.plot(r["pred"].index, r["pred"]["close"].values,
                 label=f"{r['name']} ({r['first_chg']:+.1f}%→{r['last_chg']:+.1f}%)",
                 color=colors.get(r["name"], None), lw=1.8, marker="o", ms=2)
    ax1.axhline(last, color="gray", ls=":", alpha=0.6, label=f"当前价 {last:.2f}")
    ax1.set_title(f"{STOCK_NAME}({STOCK_CODE}) 多配置 Kronos 预测对比（红=原偏离 / 绿=改进重预测）")
    ax1.set_ylabel("收盘价 (元)")
    ax1.legend(fontsize=8); ax1.grid(alpha=0.3)

    # 各配置相对当前价的涨跌幅
    for r in results:
        ax2.plot(r["pred"].index, (r["pred"]["close"].values / last - 1) * 100,
                 label=r["name"], color=colors.get(r["name"], None), lw=1.5)
    ax2.axhline(0, color="black", lw=1)
    ax2.set_title("逐日预测涨跌幅 (%)  —  对比 A股 ±10% 涨跌停限制")
    ax2.set_ylabel("涨跌幅 (%)")
    ax2.legend(fontsize=8); ax2.grid(alpha=0.3)

    plt.tight_layout()
    png_path = os.path.join(OUT_DIR, "xugong_v2_compare.png")
    plt.savefig(png_path, dpi=150, bbox_inches="tight")
    print(f"      📊 图: {png_path}")
    print("徐工重预测与诊断完成 ✅")


if __name__ == "__main__":
    main()
