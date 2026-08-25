# exp_step1_xugong_horizon.py
# 假设：Kronos 作为趋势外推器，较短 horizon 方向可能比 30 日更准；
#       且可能并不优于"直接用近 20/5 日动量方向"这种朴素基线。
# 实验：徐工(000425)，2024-01~2026-07 月度锚点，LB=150，SC=8，
#       记录模型在 H∈{5,10,15,20,30} 的方向命中率，
#       同时算两个朴素动量基线(近20日/近5日方向)的同 horizon 命中率，
#       并对每个 horizon 做二项检验(vs 50%, 精确双尾 p)。
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
LB, PL, SC = 150, 30, 8
SEC, NAME = "sz000425", "徐工机械"


def binom_p_exact(n, k, p0=0.5):
    """二项检验精确双尾 p 值：累加所有概率 <= P(X=k) 的结果。"""
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


def main():
    from model import Kronos, KronosTokenizer, KronosPredictor
    print("加载模型 ...")
    tok = KronosTokenizer.from_pretrained(TOK)
    model = Kronos.from_pretrained(MODEL)
    pred_inst = KronosPredictor(model, tok, device=DEV, max_context=512)
    print("模型就绪。")

    print(f"拉取 {NAME}({SEC}) 全量历史 ...")
    df = fetch(SEC, "2022-01-01", datetime.now().strftime("%Y-%m-%d"))
    if df is None or len(df) < 250:
        raise RuntimeError("行情拉取失败")
    print(f"  共 {len(df)} 条，{df['ts'].iloc[0]:%Y-%m-%d}~{df['ts'].iloc[-1]:%Y-%m-%d}")

    # 月度锚点（同 calibrate_xugong）
    anchors = []
    for y in range(2024, 2027):
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

    # 累计命中计数：每个 horizon 一个 (model, mom20, mom5)
    hits = {h: {"model": 0, "mom20": 0, "mom5": 0} for h in HORIZONS}
    n_valid = 0
    rec_rows = []

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
            print(f"  ⚠ 锚点 {a:%Y-%m} 失败: {e}")
            continue
        gs = np.array([sample_daily_returns(raw[s, :, CLOSE_IDX], last) for s in range(raw.shape[0])])
        g_med = np.median(gs, axis=0)
        g_use = g_med.copy()
        g_use[0] = np.median(g_med[1:])  # 去首日 level-shift 伪影
        # 朴素动量基线（锚点前）
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
        if (ai + 1) % 10 == 0:
            print(f"  已处理 {ai+1}/{len(anchors)} 锚点")

    # 汇总
    print(f"\n===== {NAME} 方向命中率（有效锚点 {n_valid}）=====")
    print(f"{'H':>3} | {'模型':>6} {'动量20':>7} {'动量5':>6} | {'模型p':>8}")
    rows = []
    for h in HORIZONS:
        nh = n_valid
        m = hits[h]["model"] / nh if nh else 0
        b20 = hits[h]["mom20"] / nh if nh else 0
        b5 = hits[h]["mom5"] / nh if nh else 0
        p = binom_p_exact(nh, hits[h]["model"])
        print(f"{h:>3} | {m:>6.1%} {b20:>7.1%} {b5:>6.1%} | {p:>8.4f}")
        rows.append({"horizon": h, "model_hit": round(m, 4), "mom20_hit": round(b20, 4),
                    "mom5_hit": round(b5, 4), "model_correct": hits[h]["model"],
                    "n": nh, "model_p": round(p, 5)})
    out_csv = os.path.join(OUT, "step1_xugong_horizon.csv")
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"CSV: {out_csv}")

    # 交叉验证：30 日模型命中率是否≈校准JSON的61.3%
    cal_path = os.path.join(HERE, "calibration_output", SEC, "calibration.json")
    if os.path.exists(cal_path):
        cal = json.load(open(cal_path, encoding="utf-8"))
        print(f"校准JSON 30日命中率={cal.get('direction_hit_rate'):.3f} (本次30日={rows[-1]['model_hit']:.3f}, n={rows[-1]['n']})")


if __name__ == "__main__":
    main()
