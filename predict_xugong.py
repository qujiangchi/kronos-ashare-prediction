# 徐工机械 (000425.SZ) Kronos 实测脚本
# 数据：akshare 真实日线（前复权）；模型：Kronos-small（CPU）。
# 修正原 examples 脚本的 cuda:0 硬编码与绝对路径问题，自包含可复现。

import sys
import os
from datetime import datetime, timedelta

import pandas as pd
import matplotlib
matplotlib.use("Agg")  # 无界面环境，直接存图
import matplotlib.pyplot as plt

import akshare as ak

# 让脚本能 import 项目内的 model 包
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(HERE)
from model import Kronos, KronosTokenizer, KronosPredictor

# ---- 配置 ----
STOCK_CODE = "000425"
STOCK_NAME = "徐工机械"
START_DATE = "20230101"                       # 拉取起点，数据越多回看越稳
END_DATE = datetime.now().strftime("%Y%m%d")  # 拉到今天
LOOKBACK = 400                                # 输入历史根数（<=512 上下文）
PRED_LEN = 100                                # 预测未来交易日数
MODEL_NAME = "NeoQuasar/Kronos-small"         # 已缓存；可换 Kronos-base 提精度
TOKENIZER_NAME = "NeoQuasar/Kronos-Tokenizer-base"
DEVICE = "cpu"
OUT_DIR = os.path.join(HERE, "xugong_output")
os.makedirs(OUT_DIR, exist_ok=True)


def fetch_data():
    print(f"[1/4] akshare 拉取 {STOCK_NAME}({STOCK_CODE}) 日线 {START_DATE}~{END_DATE} ...")
    # 优先用新浪源（本环境 eastmoney 直连被重置，新浪稳定）
    df, last_err = None, None
    for attempt in range(3):
        try:
            df = ak.stock_zh_a_daily(
                symbol="sz" + STOCK_CODE, start_date=START_DATE,
                end_date=END_DATE, adjust="qfq",
            )
            if df is not None and not df.empty:
                break
        except Exception as e:  # 网络抖动则重试
            last_err = e
            print(f"      ⚠ 第 {attempt + 1} 次重试: {type(e).__name__}")
    if df is None or df.empty:
        raise RuntimeError(f"无法获取行情: {last_err}")

    df = df.rename(columns={"date": "timestamps"})
    df["timestamps"] = pd.to_datetime(df["timestamps"])
    for col in ["open", "high", "low", "close", "volume", "amount"]:
        if col not in df.columns:
            df[col] = 0.0
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    df = df.sort_values("timestamps").reset_index(drop=True)
    print(f"      ✅ 共 {len(df)} 条，区间 {df['timestamps'].min().date()} ~ {df['timestamps'].max().date()}")
    print(f"      最新收盘: {df['close'].iloc[-1]:.2f} 元")
    return df


def build_future_dates(last_date, n):
    """生成未来 n 个自然交易日（跳过周末，粗略，不查真实休市）"""
    future, d = [], last_date + timedelta(days=1)
    while len(future) < n:
        if d.weekday() < 5:
            future.append(d)
        d += timedelta(days=1)
    return future


def main():
    df = fetch_data()

    print(f"[2/4] 加载模型 {MODEL_NAME} (device={DEVICE}) ...")
    tokenizer = KronosTokenizer.from_pretrained(TOKENIZER_NAME)
    model = Kronos.from_pretrained(MODEL_NAME)
    predictor = KronosPredictor(model, tokenizer, device=DEVICE, max_context=512)

    print(f"[3/4] 预测未来 {PRED_LEN} 个交易日 (lookback={LOOKBACK}) ...")
    x_df = df.loc[-LOOKBACK:, ["open", "high", "low", "close", "volume", "amount"]].reset_index(drop=True)
    x_ts = df.loc[-LOOKBACK:, "timestamps"].reset_index(drop=True)
    future = build_future_dates(df["timestamps"].iloc[-1], PRED_LEN)
    y_ts = pd.Series(future)

    pred_df = predictor.predict(
        df=x_df, x_timestamp=x_ts, y_timestamp=y_ts, pred_len=PRED_LEN,
        T=1.2, top_p=0.95, sample_count=3, verbose=True,
    )
    pred_df = pred_df.iloc[:len(future)]
    pred_df.index = future

    # ---- 报告 ----
    hist_close = df["close"].iloc[-1]
    pred_close = pred_df["close"].iloc[-1]
    chg = (pred_close / hist_close - 1) * 100
    print(f"\n[4/4] 生成图表与报告 ...")
    print(f"      当前价 {hist_close:.2f} -> 预测期末 {pred_close:.2f}  ({chg:+.2f}%)")
    print(f"      预测区间最高 {pred_df['close'].max():.2f} / 最低 {pred_df['close'].min():.2f}")

    # 保存预测 CSV
    rep = pd.DataFrame({
        "日期": [d.strftime("%Y-%m-%d") for d in future],
        "预测开盘": pred_df["open"].values,
        "预测最高": pred_df["high"].values,
        "预测最低": pred_df["low"].values,
        "预测收盘": pred_df["close"].values,
        "预测成交量": pred_df["volume"].values,
        "预测成交额": pred_df["amount"].values,
        "较当前涨跌幅(%)": ((pred_df["close"].values / hist_close - 1) * 100),
    })
    csv_path = os.path.join(OUT_DIR, f"{STOCK_CODE}_prediction.csv")
    rep.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"      💾 CSV: {csv_path}")

    # 绘图
    plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))

    hist_plot = df.set_index("timestamps")["close"].iloc[-min(250, len(df)):]
    ax1.plot(hist_plot.index, hist_plot.values, label="历史收盘", color="#185FA5", lw=1.8)
    ax1.plot(pred_df.index, pred_df["close"].values, label="Kronos 预测", color="#BA7517", lw=1.8, marker="o", ms=2)
    ax1.axvline(x=pred_df.index[0], color="red", ls="--", alpha=0.6)
    ax1.set_title(f"{STOCK_NAME}({STOCK_CODE}) Kronos 预测  当前 {hist_close:.2f} -> 期末 {pred_close:.2f} ({chg:+.2f}%)")
    ax1.set_ylabel("收盘价 (元)")
    ax1.legend(); ax1.grid(alpha=0.3)

    ax2.bar(pred_df.index, (pred_df["close"].values / hist_close - 1) * 100, color="#BA7517", alpha=0.8)
    ax2.axhline(0, color="black", lw=1)
    ax2.set_title("逐日预测涨跌幅 (%)")
    ax2.set_ylabel("涨跌幅 (%)")
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    png_path = os.path.join(OUT_DIR, f"{STOCK_CODE}_prediction.png")
    plt.savefig(png_path, dpi=150, bbox_inches="tight")
    print(f"      📊 图: {png_path}")
    print("徐工机械实测完成 ✅")


if __name__ == "__main__":
    main()
