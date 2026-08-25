# exp_step3_multistock.py
# 用途：在 STEP 2（徐工大样本显著性）确认方向信号有效后，复刻到其他票，检验"方向信号是否普适"。
# 方法：与 STEP 2 完全一致的大样本月度锚点法（分块拉全历史、LB=150、SC=4、H∈{5,10,15,20,30}、
#        模型方向命中率 + 动量20/5 基线 + 二项精确 p + Wilson 95% CI），逐票跑、汇总成一张表。
# 运行：HF_ENDPOINT=https://hf-mirror.com venv/Scripts/python.exe exp_step3_multistock.py
# 产物：optimization_output/step3_multistock.csv（每票每 horizon 一行）+ 控制台摘要。
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
START_HISTORY = "2013-01-01"

# 复刻目标：取校准中 h 较高且有代表性的票（高/中 h、不同行业）
STOCKS = [
    ("sh601318", "中国平安"),
    ("sz002185", "华天科技"),
    ("sh600036", "招商银行"),
]

# 在最前面追加"徐工"基准（STEP 2 已跑，可重跑对齐口径，或仅用于对照）
# 这里先不含徐工；若需对照可取消下一行注释并把它加入 STOCKS。
# STOCKS.insert(0, ("sz000425", "徐工机械"))


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
    if n == 0:
        return (0.0, 0.0)
    phat = k / n
    denom = 1 + z * z / n
    centre = (phat + z * z / (2 * n)) / denom
    half = (z * math.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n))) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def fetch_full(sec, start, end):
    end_d = datetime.strptime(end, "%Y-%m-%d")
    chunks = []
    for y in range(int(start[:4]), end_d.year + 1, 3):
        ce = min(y + 3, end_d.year + 1)
        d = fetch(sec, f"{y}-01-01", f"{ce}-01-01", count=1024)
        if d is not None and len(d):
            chunks.append(d)
    if not chunks:
        return None
    return pd.concat(chunks, ignore_index=True).drop_duplicates("ts").sort_values("ts").reset_index(drop=True)


def run_one(pred_inst, sec, name, today):
    df = fetch_full(sec, START_HISTORY, today)
    if df is None or len(df) < 250:
        print(f"  ⚠ {name}({sec}) 行情拉取失败，跳过")
        return []
    print(f"  {name}({sec}): {len(df)} 条，{df['ts'].iloc[0]:%Y-%m-%d}~{df['ts'].iloc[-1]:%Y-%m-%d}")
    anchors = []
    for y in range(df['ts'].dt.year.min(), df['ts'].dt.year.max() + 1):
        for m in range(1, 13):
            c = df[(df.ts.dt.year == y) & (df.ts.dt.month == m)]
            if len(c) == 0:
                continue
            a = c.iloc[0]["ts"]
            if a >= df["ts"].iloc[-1] - timedelta(days=PL + 5):
                continue
            if (df["ts"] <= a).sum() >= LB:
                anchors.append(a)
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
                print(f"    ⚠ 锚点 {a:%Y-%m} 失败: {e}")
            continue
        gs = np.array([sample_daily_returns(raw[s, :, CLOSE_IDX], last) for s in range(raw.shape[0])])
        g_med = np.median(gs, axis=0)
        g_use = g_med.copy()
        g_use[0] = np.median(g_med[1:])
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
            print(f"    {name} 已处理 {ai+1}/{len(anchors)} (n_valid={n_valid})")
    rows = []
    for h in HORIZONS:
        nh = n_valid
        m = hits[h]["model"] / nh if nh else 0
        lo, hi = wilson_ci(hits[h]["model"], nh)
        p = binom_p_exact(nh, hits[h]["model"])
        rows.append({"sec": sec, "name": name, "horizon": h, "n": nh,
                     "model_hit": round(m, 4), "wilson_lo": round(lo, 4),
                     "wilson_hi": round(hi, 4), "mom20_hit": round(hits[h]["mom20"] / nh, 4) if nh else 0,
                     "mom5_hit": round(hits[h]["mom5"] / nh, 4) if nh else 0,
                     "model_correct": hits[h]["model"], "model_p": round(p, 5)})
    return rows


def main():
    from model import Kronos, KronosTokenizer, KronosPredictor
    print("加载模型 ...")
    tok = KronosTokenizer.from_pretrained(TOK)
    model = Kronos.from_pretrained(MODEL)
    pred_inst = KronosPredictor(model, tok, device=DEV, max_context=512)
    print("模型就绪。")
    today = datetime.now().strftime("%Y-%m-%d")

    all_rows = []
    for sec, name in STOCKS:
        print(f"\n##### {name}({sec}) #####")
        rows = run_one(pred_inst, sec, name, today)
        all_rows.extend(rows)

    df_out = pd.DataFrame(all_rows)
    out_csv = os.path.join(OUT, "step3_multistock.csv")
    df_out.to_csv(out_csv, index=False)
    print(f"\nCSV: {out_csv}")
    # 摘要：每只票 H=20/30 的命中率与 p
    print("\n===== 多票方向命中率摘要（重点 H=20,30）=====")
    for sec, name in STOCKS:
        sub = df_out[df_out.sec == sec]
        print(f"\n{name}({sec}) n={int(sub['n'].iloc[0])}:")
        for h in (20, 30):
            r = sub[sub.horizon == h].iloc[0]
            flag = "✅显著" if r.model_p < 0.05 and r.wilson_lo > 0.5 else "❌不显著"
            print(f"  H={h:>2}: 模型 {r.model_hit:>6.1%} [CI {r.wilson_lo:>5.1%},{r.wilson_hi:>5.1%}] p={r.model_p:.4f} {flag}")


if __name__ == "__main__":
    main()
