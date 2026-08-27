"""
STEP 5 — MA 规则方向基线（与 Kronos head-to-head）
=================================================
假设：若简单 MA20/60 金叉死叉规则的方向命中率已 ≥ Kronos 的 ~54%（且不显著优于 50%），
      则 Kronos 对 A 股个股"无方向增量价值"，系统方向信号应改为可证伪的简单规则。

实验：
  - 对 watchlist 11 只票，拉全历史（2013~今，分块 fetch）。
  - 月度锚点（每月首个交易日，需 ≥60 日前置 + ≥30 日未来）。
  - 在每锚点构造两类"方向信号"并检验其 20/30 日方向命中率 + 二项精确 p(vs 50%) + Wilson CI：
      1) MA 信号：pred = sign(MA20 - MA60)  （金叉看多 / 死叉看空）
      2) 动量信号：pred = sign(近20日收益)   （趋势延续）
  - 同时跑一个标准"日频 long/flat" MA(20,60) 交叉策略，算累计收益 vs 买入持有，作经济价值佐证。
  - 与 STEP3 的 Kronos 方向命中率(~54%, 均不显著) 直接对比。

指标：每票 MA_hit20/30, mom_hit20/30, 对应 p 与 Wilson CI；全样本策略累计收益 vs buy&hold。

输出：optimization_output/step5_ma_baseline.csv + 控制台摘要。
"""
import os, sys, math
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "optimization_output")
os.makedirs(OUT, exist_ok=True)
OUT_CSV = os.path.join(OUT, "step5_ma_baseline.csv")

# ---------- 复用 calibrate_xugong 的 fetch（带分块拉全历史）----------
sys.path.insert(0, HERE)
from calibrate_xugong import fetch  # fetch(sec, start, end, count=1024) -> df[ts,open,close,high,low,vol]

def fetch_full(sec, start="2013-01-01", end=None):
    end = end or datetime.now().strftime("%Y-%m-%d")
    chunks = []
    for y in range(int(start[:4]), int(end[:4]) + 1, 3):
        ce = min(y + 3, int(end[:4]) + 1)
        d = fetch(sec, f"{y}-01-01", f"{ce}-01-01", count=1024)
        if d is not None and len(d):
            chunks.append(d)
    if not chunks:
        return None
    df = pd.concat(chunks, ignore_index=True).drop_duplicates("ts").sort_values("ts").reset_index(drop=True)
    return df

# ---------- 统计辅助 ----------
def wilson_ci(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return (max(0.0, center - half), min(1.0, center + half))

def binom_p_exact(k, n, p0=0.5):
    try:
        from scipy.stats import binomtest
        return float(binomtest(k, n, p0).pvalue)
    except Exception:
        # 正态近似兜底
        if n == 0:
            return 1.0
        p = k / n
        se = math.sqrt(p0 * (1 - p0) / n)
        if se == 0:
            return 1.0
        z = abs(p - p0) / se
        return 2 * (1 - 0.5 * (1 + math.erf(z / math.sqrt(2))))

# ---------- WATCHLIST（取自 predict_multi.py）----------
WATCHLIST = [
    ("招商积余", "001914", "sz"), ("今世缘", "603369", "sh"), ("源飞宠物", "001222", "sz"),
    ("同仁堂", "600085", "sh"), ("启明星辰", "002439", "sz"), ("徐工机械", "000425", "sz"),
    ("泰瑞机器", "603289", "sh"), ("华天科技", "002185", "sz"), ("中国平安", "601318", "sh"),
    ("招商银行", "600036", "sh"), ("铜陵有色", "000630", "sz"),
]

H = [20, 30]
LB_MA = 60

def analyze(sec, name, prefix):
    code = prefix + sec
    df = fetch_full(code)
    if df is None or len(df) < 250:
        print(f"  [{name}] 数据不足，跳过"); return None
    close = df["close"].values
    n = len(df)
    last_year = int(df["ts"].iloc[-1].year)
    ma20 = pd.Series(close).rolling(20).mean().values
    ma60 = pd.Series(close).rolling(60).mean().values

    # 月度锚点
    anchors = []
    for y in range(2015, last_year + 1):
        for m in range(1, 13):
            cand = df[(df.ts.dt.year == y) & (df.ts.dt.month == m)]
            if len(cand) == 0:
                continue
            a = cand.iloc[0]
            ai = df.index[df.ts == a["ts"]][0]
            if ai < LB_MA or ai + max(H) >= n:
                continue
            anchors.append(ai)
    if not anchors:
        print(f"  [{name}] 无合格锚点，跳过"); return None

    ma_hit = {20: 0, 30: 0}
    mom_hit = {20: 0, 30: 0}
    tot = len(anchors)
    for ai in anchors:
        ma_sig = 1 if ma20[ai] > ma60[ai] else -1
        mom_sig = 1 if close[ai] > close[ai - 20] else -1
        for h in H:
            fwd = close[ai + h] / close[ai] - 1.0
            act = 1 if fwd >= 0 else -1
            if ma_sig == act:
                ma_hit[h] += 1
            if mom_sig == act:
                mom_hit[h] += 1

    # 日频 long/flat 策略
    pos = np.zeros(n)
    valid = ~np.isnan(ma20) & ~np.isnan(ma60)
    pos[valid] = (ma20[valid] > ma60[valid]).astype(float)
    ret = pd.Series(close).pct_change().values
    strat_ret = np.zeros(n)
    for t in range(1, n):
        strat_ret[t] = pos[t - 1] * ret[t]   # 用前一日仓位，避免前视
    strat_cum = np.prod(1 + strat_ret[1:]) - 1
    buyhold_cum = np.prod(1 + ret[1:]) - 1

    row = {"sec": code, "name": name, "n": tot,
           "ma_hit20": ma_hit[20] / tot, "ma_p20": binom_p_exact(ma_hit[20], tot),
           "ma_hit30": ma_hit[30] / tot, "ma_p30": binom_p_exact(ma_hit[30], tot),
           "mom_hit20": mom_hit[20] / tot, "mom_p20": binom_p_exact(mom_hit[20], tot),
           "mom_hit30": mom_hit[30] / tot, "mom_p30": binom_p_exact(mom_hit[30], tot),
           "strat_cumret": strat_cum, "buyhold_cumret": buyhold_cum}
    return row

def main():
    print("=" * 70)
    print("STEP 5 — MA 规则方向基线（head-to-head vs Kronos）")
    rows = []
    for name, sec, prefix in WATCHLIST:
        print(f"\n##### {name}({prefix}{sec}) #####", flush=True)
        r = analyze(sec, name, prefix)
        if r:
            rows.append(r)
            lo20, hi20 = wilson_ci(int(r["ma_hit20"] * r["n"]), r["n"])
            print(f"  MA命中 H20={r['ma_hit20']:.1%}(p={r['ma_p20']:.3f}) H30={r['ma_hit30']:.1%}(p={r['ma_p30']:.3f})")
            print(f"  动量命中 H20={r['mom_hit20']:.1%} H30={r['mom_hit30']:.1%}")
            print(f"  策略累计={r['strat_cumret']:.1%} vs 买入持有={r['buyhold_cumret']:.1%}")

    if rows:
        pd.DataFrame(rows).to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
        print(f"\n已写出 {OUT_CSV}")

    # 汇总对比 Kronos（STEP3 参考值 ~54% 且不显著）
    print("\n" + "=" * 70)
    print("汇总：MA规则 vs Kronos 方向命中率（参考 STEP3：Kronos ~54%, 均不显著 p>0.05）")
    any_sig = False
    for r in rows:
        sig = (r["ma_p20"] < 0.05) or (r["ma_p30"] < 0.05)
        any_sig = any_sig or sig
        print(f"  {r['name']:6s} MA20={r['ma_hit20']:.1%}(p={r['ma_p20']:.3f}) "
              f"MA30={r['ma_hit30']:.1%}(p={r['ma_p30']:.3f}) "
              f"{'✅显著' if sig else '❌不显著'}")
    print("=" * 70)
    if not any_sig:
        print("结论：MA规则方向命中率与 Kronos 同样不显著优于 50% → 两者皆无方向 alpha；")
        print("      系统方向信号应废弃，改用'不预测方向、仅做幅度/波动率参考'或元模型。")
    else:
        print("结论：存在 MA 规则显著优于 50% 的票 → 方向可由简单规则给出，Kronos 方向可废弃改用 MA。")

if __name__ == "__main__":
    main()
