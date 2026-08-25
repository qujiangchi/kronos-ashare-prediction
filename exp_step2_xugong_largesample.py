# exp_step2_xugong_largesample.py
# 假设（基于 STEP 1）：徐工 20/30 日方向命中率 61.3% 但不显著(p=0.281, n=31)，
#   主因是样本太小。月度锚点天然相互独立（相邻锚点测试窗不重叠），
#   把历史拉到 ~2015 可得 ~100+ 独立锚点，足以检验 61% 是否真显著。
# 实验：徐工(000425)，全历史(2013~今，分块拉取避免 count 截断)，月度锚点，
#   LB=150，SC=4，记录 H∈{5,10,15,20,30} 模型方向命中率 + 动量基线 + 二项 p + Wilson 95% CI。
# 决策：H=20/30 在 n≈100+ 时 p<0.05 → 采纳并进入"逐票选 horizon"；
#       仍不显著 → 结论"方向信号不可靠"，转向幅度/波动率参考或元模型。
import sys, os, json, math
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(HERE)
from calibrate_xugong import (raw_predict, sample_daily_returns, fetch,
                              build_future, CLOSE_IDX)

TOK = "NeoQuasar/Kronos-Tokenizer-base"
MODEL = "NeoQuasar/Kronos-base"
DEV = "cpu"
OUT = os.path.join(HERE, "optimization_output")
os.makedirs(OUT, exist_ok=True)

HORIZONS = [5, 10, 15, 20, 30]
LB, PL, SC = 150, 30, 4
SEC, NAME = "sz000425", "徐工机械"
START_HISTORY = "2013-01-01"


def binom_p_exact(n, k, p0=0.5):
    if n == 0:
        return 1.0
    k = min(k, n)
    p_k = math.comb(n, k) * (p0 ** k) * ((1 - p0) ** (n - k))
    total = 0.0
    for x in range(0, n + 1):
        px = math.comb(n, x) * (p0 ** x) * ((1 - p0) ** (n - x))
        if px <= p_k + 1e-15:
            total += px
    return min(1.0, total)


def wilson_ci(k, n, z=1.96):
    """Wilson 95% 置信区间（比例），对小样本比正态近似更稳。"""
    if n == 0:
        return (0.0, 0.0)
    phat = k / n
    denom = 1 + z * z / n
    centre = (phat + z * z / (2 * n)) / denom
    half = (z * math.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n))) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def fetch_full(sec, start, end):
    """分块拉取全历史（gtimg count 上限~1024，分窗避免截断），去重合并。"""
    end_d = datetime.strptime(end, "%Y-%m-%d")
    chunks = []
    y0 = int(start[:4])
    for y in range(y0, end_d.year + 1, 3):  # 每 3 年一窗，约 <= 1024 条
        cs = f"{y}-01-01"
        ce = f"{min(y + 3, end_d.year + 1)}-01-01"
        if ce > end:
            ce = end
        d = fetch(sec, cs, ce, count=1024)
        if d is not None and len(d):
            chunks.append(d)
    if not chunks:
        return None
    df = pd.concat(chunks, ignore_index=True)
    df = df.drop_duplicates("ts").sort_values("ts").reset_index(drop=True)
    return df


def main():
    from model import Kronos, KronosTokenizer, KronosPredictor
    print("加载模型 ...")
    tok = KronosTokenizer.from_pretrained(TOK)
    model = Kronos.from_pretrained(MODEL)
    pred_inst = KronosPredictor(model, tok, device=DEV, max_context=512)
    print("模型就绪。")

    today = datetime.now().strftime("%Y-%m-%d")
    print(f"分块拉取 {NAME}({SEC}) 全历史 {START_HISTORY}~{today} ...")
    df = fetch_full(SEC, START_HISTORY, today)
    if df is None or len(df) < 250:
        raise RuntimeError("行情拉取失败")
    print(f"  共 {len(df)} 条，{df['ts'].iloc[0]:%Y-%m-%d}~{df['ts'].iloc[-1]:%Y-%m-%d}")

    # 月度锚点（同校准逻辑）
    anchors = []
    for y in range(df['ts'].dt.year.min(), df['ts'].dt.year.max() + 1):
        for m in range(1, 13):
            cand = df[(df.ts.dt.year == y) & (df.ts.dt.month == m)]
            if len(cand) == 0:
                continue
            a = cand.iloc[0]["ts"]
            if a >= df["ts"].iloc[-1] - timedelta(days=PL + 5):
                continue
            if (df["ts"] <= a).sum() >= LB:
                anchors.append(a)
    print(f"锚点数: {len(anchors)}")

    hits = {h: {"model": 0, "mom20": 0, "mom5": 0} for h in HORIZONS}
    n_valid = 0

    for ai, a in enumerate(anchors):
        train = df[df.ts <= a]
        test = df[df.ts > a].iloc[:PL]
        if len(test) < PL:
            continue
        last = float(train["close"].iloc[-1])
        x_df = train.iloc[-LB:][["open", "high", "low", "close", "vol"]].copy()
        x_df["amount"] = 0.0
        x_ts = train.iloc[-LB:]["ts"]
        y_ts = test["ts"].reset_index(drop=True)
        try:
            raw = raw_predict(pred_inst, x_df, x_ts, y_ts, PL, T=1.0, top_p=0.95, sample_count=SC)
        except Exception as e:
            if (ai + 1) % 20 == 0:
                print(f"  ⚠ 锚点 {a:%Y-%m} 失败: {e}")
            continue
        gs = np.array([sample_daily_returns(raw[s, :, CLOSE_IDX], last) for s in range(raw.shape[0])])
        g_med = np.median(gs, axis=0)
        g_use = g_med.copy()
        g_use[0] = np.median(g_med[1:])  # 去首日 level-shift 伪影
        mom20_dir = np.sign(last / float(train["close"].iloc[-21]) - 1.0) if len(train) >= 21 else 0.0
        mom5_dir = np.sign(last / float(train["close"].iloc[-6]) - 1.0) if len(train) >= 6 else 0.0

        act = {h: np.sign(float(test["close"].iloc[h - 1]) / last - 1.0) for h in HORIZONS}
        pred = {h: np.sign(np.prod(1.0 + g_use[:h]) - 1.0) for h in HORIZONS}
        for h in HORIZONS:
            if act[h] == 0:
                continue
            if pred[h] == act[h]:
                hits[h]["model"] += 1
            if mom20_dir != 0 and mom20_dir == act[h]:
                hits[h]["mom20"] += 1
            if mom5_dir != 0 and mom5_dir == act[h]:
                hits[h]["mom5"] += 1
        n_valid += 1
        if (ai + 1) % 20 == 0:
            print(f"  已处理 {ai+1}/{len(anchors)} 锚点 (n_valid={n_valid})")

    print(f"\n===== {NAME} 方向命中率（独立月度锚点 n={n_valid}）=====")
    print(f"{'H':>3} | {'模型':>7} {'95%CI':>14} {'动量20':>7} {'动量5':>6} | {'模型p':>8}")
    rows = []
    for h in HORIZONS:
        nh = n_valid
        m = hits[h]["model"] / nh if nh else 0
        b20 = hits[h]["mom20"] / nh if nh else 0
        b5 = hits[h]["mom5"] / nh if nh else 0
        p = binom_p_exact(nh, hits[h]["model"])
        lo, hi = wilson_ci(hits[h]["model"], nh)
        print(f"{h:>3} | {m:>6.1%} [{lo:>5.1%},{hi:>5.1%}] {b20:>7.1%} {b5:>6.1%} | {p:>8.4f}")
        rows.append({"horizon": h, "model_hit": round(m, 4), "wilson_lo": round(lo, 4),
                    "wilson_hi": round(hi, 4), "mom20_hit": round(b20, 4),
                    "mom5_hit": round(b5, 4), "model_correct": hits[h]["model"],
                    "n": nh, "model_p": round(p, 5)})
    out_csv = os.path.join(OUT, "step2_xugong_largesample.csv")
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"CSV: {out_csv}")


if __name__ == "__main__":
    main()
