# exp_step4_magnitude.py
# 用途：STEP 3 已证明 Kronos-base 在 A 股个股层面【方向】信号普遍不可靠（0/3 票 H20/30 显著，
#   全 Wilson CI 跨 50%）。本步检验系统【仅存的潜在价值】—— Kronos 预测路径的"幅度/波动率"
#   是否携带可用信息：预测 |端点收益| / 预测路径波动率 是否能在秩意义上跟踪真实实现值。
# 方法：与 STEP 2/3 完全一致的大样本月度锚点法（LB=150, SC=4, PL=30, 2013~今, 月度锚点,
#   独立锚点）, 对 4 只代表票（含 STEP 2 徐工 + STEP 3 高 h 三票）各 n≈155 锚点；
#   每锚点取 Kronos 中位预测路径，计算 predicted |R|、predicted 路径波动率、realized |R|、
#   realized 波动率，并与"近期(锚点前30日)波动率/收益幅度"朴素基线对照（排除纯波动聚集伪信号）。
# 指标：每票 + pooled(z-score 后跨票合并) 的 Spearman 秩相关 (rho, 正态近似 p)。
#   - abs_endpoint: Spearman(pred|R|, act|R|) vs 基线 Spearman(recent|R|, act|R|)
#   - path_vol:     Spearman(pred_vol, act_vol) vs 基线 Spearman(recent_vol, act_vol)
#   - signed_ret:   Spearman(pred R, act R)（方向与幅度合并，对照方向纯噪声预期）
# 运行：HF_ENDPOINT=https://hf-mirror.com venv/Scripts/python.exe exp_step4_magnitude.py
# 产物：optimization_output/step4_magnitude.csv + 控制台摘要 + 自动追加 OPTIMIZATION_LOG.md STEP 4 段
#       + 本地 git commit（push 尽力，失败不影响本地提交）+ 清除 .running。
import sys, os, json, math, subprocess
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(HERE)
from calibrate_xugong import (raw_predict, sample_daily_returns, fetch, CLOSE_IDX)

TOK = "NeoQuasar/Kronos-Tokenizer-base"
MODEL = "NeoQuasar/Kronos-base"
DEV = "cpu"
OUT = os.path.join(HERE, "optimization_output")
os.makedirs(OUT, exist_ok=True)

LB, PL, SC = 150, 30, 4
START_HISTORY = "2013-01-01"
STOCKS = [
    ("sh601318", "中国平安"),
    ("sz002185", "华天科技"),
    ("sh600036", "招商银行"),
    ("sz000425", "徐工机械"),
]


def norm_cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def spearman_p(rho, n):
    if n < 3:
        return 1.0
    if abs(rho) >= 1.0:
        return 1e-300
    t = rho * math.sqrt((n - 2) / max(1e-12, 1.0 - rho * rho))
    return max(1e-300, min(1.0, 2.0 * (1.0 - norm_cdf(abs(t)))))


def spearman(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    if len(x) < 3:
        return (0.0, 1.0)
    rx = pd.Series(x).rank().values
    ry = pd.Series(y).rank().values
    mx, my = rx.mean(), ry.mean()
    num = np.sum((rx - mx) * (ry - my))
    den = math.sqrt(np.sum((rx - mx) ** 2) * np.sum((ry - my) ** 2))
    rho = num / den if den > 1e-12 else 0.0
    return (rho, spearman_p(rho, len(x)))


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
        return None
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
    P_abs = []; A_abs = []; P_vol = []; A_vol = []; R_abs = []; R_vol = []; P_ret = []; A_ret = []
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
        pred_ret = float(np.prod(1.0 + g_use) - 1.0)
        pred_abs = abs(pred_ret)
        pred_vol = float(np.std(g_use))
        act_ret = float(test["close"].iloc[-1] / last - 1.0)
        act_abs = abs(act_ret)
        act_returns = test["close"].values[1:] / test["close"].values[:-1] - 1.0
        act_vol = float(np.std(act_returns))
        recent = train.iloc[-31:]
        recent_returns = recent["close"].values[1:] / recent["close"].values[:-1] - 1.0
        recent_vol = float(np.std(recent_returns))
        recent_abs = abs(float(recent["close"].iloc[-1] / recent["close"].iloc[0] - 1.0))
        P_abs.append(pred_abs); A_abs.append(act_abs); P_vol.append(pred_vol)
        A_vol.append(act_vol); R_abs.append(recent_abs); R_vol.append(recent_vol)
        P_ret.append(pred_ret); A_ret.append(act_ret)
        n_valid += 1
        if (ai + 1) % 20 == 0:
            print(f"    {name} 已处理 {ai+1}/{len(anchors)} (n_valid={n_valid})")
    if n_valid < 10:
        return None
    return dict(sec=sec, name=name, n=n_valid,
                P_abs=np.array(P_abs), A_abs=np.array(A_abs),
                P_vol=np.array(P_vol), A_vol=np.array(A_vol),
                R_abs=np.array(R_abs), R_vol=np.array(R_vol),
                P_ret=np.array(P_ret), A_ret=np.array(A_ret))


def main():
    from model import Kronos, KronosTokenizer, KronosPredictor
    print("加载模型 ...")
    tok = KronosTokenizer.from_pretrained(TOK)
    model = Kronos.from_pretrained(MODEL)
    pred_inst = KronosPredictor(model, tok, device=DEV, max_context=512)
    print("模型就绪。")
    today = datetime.now().strftime("%Y-%m-%d")

    results = []
    pooled = dict(P_abs=[], A_abs=[], P_vol=[], A_vol=[], R_abs=[], R_vol=[], P_ret=[], A_ret=[])
    for sec, name in STOCKS:
        print(f"\n##### {name}({sec}) #####")
        r = run_one(pred_inst, sec, name, today)
        if r is None:
            print(f"  {name} 数据不足，跳过")
            continue
        # 每票 Spearman
        a_k, a_kp = spearman(r["P_abs"], r["A_abs"])
        a_b, a_bp = spearman(r["R_abs"], r["A_abs"])
        v_k, v_kp = spearman(r["P_vol"], r["A_vol"])
        v_b, v_bp = spearman(r["R_vol"], r["A_vol"])
        s_k, s_kp = spearman(r["P_ret"], r["A_ret"])
        results.append(dict(sec=sec, name=name, n=r["n"],
                            abs_kronos_rho=round(a_k, 4), abs_kronos_p=round(a_kp, 5),
                            abs_base_rho=round(a_b, 4), abs_base_p=round(a_bp, 5),
                            vol_kronos_rho=round(v_k, 4), vol_kronos_p=round(v_kp, 5),
                            vol_base_rho=round(v_b, 4), vol_base_p=round(v_bp, 5),
                            signed_kronos_rho=round(s_k, 4), signed_kronos_p=round(s_kp, 5)))
        # 累计 pooled（票内 z-score 消除量纲差异）
        for k in ("P_abs", "A_abs", "P_vol", "A_vol", "R_abs", "R_vol", "P_ret", "A_ret"):
            z = (r[k] - r[k].mean()) / (r[k].std() + 1e-9)
            pooled[k].append(z)
        print(f"  |R|: Kronos rho={a_k:+.3f}(p={a_kp:.3f}) vs 基线 rho={a_b:+.3f}(p={a_bp:.3f})")
        print(f"  vol: Kronos rho={v_k:+.3f}(p={v_kp:.3f}) vs 基线 rho={v_b:+.3f}(p={v_bp:.3f})")
        print(f"  signed: Kronos rho={s_k:+.3f}(p={s_kp:.3f})")

    # pooled 合并检验
    for k in pooled:
        pooled[k] = np.concatenate(pooled[k]) if pooled[k] else np.array([])
    if len(pooled["P_abs"]) >= 3:
        pa_k, pa_kp = spearman(pooled["P_abs"], pooled["A_abs"])
        pa_b, pa_bp = spearman(pooled["R_abs"], pooled["A_abs"])
        pv_k, pv_kp = spearman(pooled["P_vol"], pooled["A_vol"])
        pv_b, pv_bp = spearman(pooled["R_vol"], pooled["A_vol"])
        ps_k, ps_kp = spearman(pooled["P_ret"], pooled["A_ret"])
        results.append(dict(sec="POOLED", name="跨票合并(z-score)", n=len(pooled["P_abs"]),
                            abs_kronos_rho=round(pa_k, 4), abs_kronos_p=round(pa_kp, 5),
                            abs_base_rho=round(pa_b, 4), abs_base_p=round(pa_bp, 5),
                            vol_kronos_rho=round(pv_k, 4), vol_kronos_p=round(pv_kp, 5),
                            vol_base_rho=round(pv_b, 4), vol_base_p=round(pv_bp, 5),
                            signed_kronos_rho=round(ps_k, 4), signed_kronos_p=round(ps_kp, 5)))

    df_out = pd.DataFrame(results)
    out_csv = os.path.join(OUT, "step4_magnitude.csv")
    df_out.to_csv(out_csv, index=False)
    print(f"\nCSV: {out_csv}")

    # ---- 控制台摘要 ----
    print("\n===== STEP 4 幅度/波动率信息量摘要 =====")
    for _, row in df_out.iterrows():
        print(f"{row['name']:>10}(n={int(row['n'])}): "
              f"|R| Kronos={row['abs_kronos_rho']:+.3f}(p={row['abs_kronos_p']:.3f}) / 基线={row['abs_base_rho']:+.3f}(p={row['abs_base_p']:.3f}) | "
              f"vol Kronos={row['vol_kronos_rho']:+.3f}(p={row['vol_kronos_p']:.3f}) / 基线={row['vol_base_rho']:+.3f}(p={row['vol_base_p']:.3f}) | "
              f"signed={row['signed_kronos_rho']:+.3f}(p={row['signed_kronos_p']:.3f})")

    # ---- 判定 ----
    sig = df_out[(df_out.sec != "POOLED")]
    n_stocks = len(sig)
    abs_sig = (sig.abs_kronos_p < 0.05).sum()
    vol_sig = (sig.vol_kronos_p < 0.05).sum()
    signed_sig = (sig.signed_kronos_p < 0.05).sum()
    pooled_row = df_out[df_out.sec == "POOLED"]
    pooled_abs_sig = bool(pooled_row.size and pooled_row.iloc[0]["abs_kronos_p"] < 0.05)
    pooled_vol_sig = bool(pooled_row.size and pooled_row.iloc[0]["vol_kronos_p"] < 0.05)
    # Kronos 是否优于基线
    better_abs = (sig.abs_kronos_rho > sig.abs_base_rho).sum()
    better_vol = (sig.vol_kronos_rho > sig.vol_base_rho).sum()

    verdict = []
    if (abs_sig >= 2 or pooled_abs_sig) or (vol_sig >= 2 or pooled_vol_sig):
        verdict.append("Kronos 幅度/波动率含显著信息 → 可服务于波动择时/仓位尺度")
    else:
        verdict.append("Kronos 幅度/波动率秩相关在个股层面不显著 → 对单名几乎零可用信息")
    if better_abs >= max(1, n_stocks - 1) and better_vol >= max(1, n_stocks - 1):
        verdict.append("且 Kronos 优于近期波动基线（非纯波动聚集回声）")
    else:
        verdict.append("Kronos 未系统性优于近期波动基线（若有微弱相关，可能只是波动聚集回声）")

    decision = (
        "决策：Kronos-base 对 A 股单名【方向】与【幅度/波动率】均无可证伪的可用信号"
        if (abs_sig == 0 and vol_sig == 0 and signed_sig == 0 and not (pooled_abs_sig or pooled_vol_sig))
        else "决策：幅度/波动率存在微弱信号，下一步用其构建仓位尺度叠加层并回测（backlog 9）"
    )

    # ---- 追加 OPTIMIZATION_LOG.md STEP 4 段 ----
    md = build_step4_md(df_out, sig, n_stocks, abs_sig, vol_sig, signed_sig,
                        pooled_abs_sig, pooled_vol_sig, better_abs, better_vol,
                        verdict, decision)
    log_path = os.path.join(HERE, "OPTIMIZATION_LOG.md")
    with open(log_path, "a", encoding="utf-8") as f:
        f.write("\n" + md)
    print("\n已追加 STEP 4 段到 OPTIMIZATION_LOG.md")

    # ---- 本地提交（push 尽力，失败不影响本地）----
    try:
        subprocess.run(["git", "add", out_csv, log_path], cwd=HERE, check=True)
        msg = ("exp(step 4): 幅度/波动率信息量检验 — " +
               ("Kronos幅度/波动率秩相关不显著/未优于基线→单名零可用信号" if (abs_sig == 0 and vol_sig == 0 and signed_sig == 0 and not (pooled_abs_sig or pooled_vol_sig)) else "Kronos幅度/波动率存微弱信号→下一步构建仓位尺度层"))
        subprocess.run(["git", "commit", "-m", msg], cwd=HERE, check=True)
        print("本地提交完成。")
        try:
            subprocess.run(["timeout", "60", "git", "push", "origin", "master"], cwd=HERE, check=False)
            print("push 已尝试。")
        except Exception as e:
            print("push 失败（保留本地提交）:", e)
    except Exception as e:
        print("提交失败:", e)

    # ---- 清除 .running（仅成功收尾时）----
    running = os.path.join(OUT, ".running")
    try:
        if os.path.exists(running):
            os.remove(running)
            print("已清除 .running。")
    except Exception:
        pass

    print("\nSTEP 4 完成。判定：", "; ".join(verdict))
    print(decision)


def build_step4_md(df_out, sig, n_stocks, abs_sig, vol_sig, signed_sig,
                   pooled_abs_sig, pooled_vol_sig, better_abs, better_vol,
                   verdict, decision):
    lines = []
    lines.append("## STEP 4 — 幅度/波动率信息量检验（方向失效后的仅存价值）")
    lines.append(f"- **假设**：STEP 3 证明方向不可靠，但 Kronos 预测路径的【幅度/波动率】可能仍携带信息——")
    lines.append(f"  预测 |端点收益| / 路径波动率 或许能在秩意义上跟踪真实实现值，服务于波动择时或仓位尺度。")
    lines.append(f"- **实验**：对 中国平安/华天科技/招商银行/徐工机械 用大样本月度锚点法（LB=150, SC=4, PL=30, 2013~今, 各 n≈155），")
    lines.append(f"  每锚点取 Kronos 中位预测路径，算 预测|R|/预测波动率/真实|R|/真实波动率，并与【锚点前30日近期波动】朴素基线对照；")
    lines.append(f"  每票 + 跨票合并(z-score) 计算 Spearman 秩相关(rho, p)。")
    lines.append(f"- **指标**：每票与 pooled 的 |R|、路径波动率、signed 收益的 Kronos 秩相关 vs 基线秩相关。")
    lines.append(f"- **脚本**：`exp_step4_magnitude.py`，产物 `optimization_output/step4_magnitude.csv`。")
    lines.append(f"- **状态**：✅ 完成（由夜间自动化代理后台运行并自提交）。")
    lines.append("")
    lines.append("  | 票 | n | |R| Kronos rho(p) | |R| 基线 rho(p) | vol Kronos rho(p) | vol 基线 rho(p) | signed rho(p) |")
    lines.append("  |---|---|---|---|---|---|---|---|")
    for _, row in df_out.iterrows():
        lines.append(
            f"  | {row['name']} | {int(row['n'])} | {row['abs_kronos_rho']:+.3f}({row['abs_kronos_p']:.3f}) | "
            f"{row['abs_base_rho']:+.3f}({row['abs_base_p']:.3f}) | {row['vol_kronos_rho']:+.3f}({row['vol_kronos_p']:.3f}) | "
            f"{row['vol_base_rho']:+.3f}({row['vol_base_p']:.3f}) | {row['signed_kronos_rho']:+.3f}({row['signed_kronos_p']:.3f}) |")
    lines.append("")
    lines.append(f"- **结论**：")
    lines.append(f"  1. 个股层面：|R| 显著(p<0.05)票数 = {abs_sig}/{n_stocks}；路径波动率显著票数 = {vol_sig}/{n_stocks}；signed 显著票数 = {signed_sig}/{n_stocks}。")
    lines.append(f"  2. 跨票合并(pooled)检验：|R| Kronos p={'%.3f'%df_out[df_out.sec=='POOLED'].iloc[0]['abs_kronos_p'] if (df_out.sec=='POOLED').any() else float('nan')}"
                 f"（{'显著' if pooled_abs_sig else '不显著'}）；路径波动率 p="
                 f"{'%.3f'%df_out[df_out.sec=='POOLED'].iloc[0]['vol_kronos_p'] if (df_out.sec=='POOLED').any() else float('nan')}"
                 f"（{'显著' if pooled_vol_sig else '不显著'}）。")
    lines.append(f"  3. 优于基线：|R| Kronos>基线 的票数 = {better_abs}/{n_stocks}；波动率 Kronos>基线 = {better_vol}/{n_stocks}。")
    lines.append(f"  4. {'；'.join(verdict)}")
    lines.append(f"- **{decision}**")
    lines.append(f"- **下一步（STEP 5）**：")
    if (abs_sig == 0 and vol_sig == 0 and signed_sig == 0 and not (pooled_abs_sig or pooled_vol_sig)):
        lines.append(f"  幅度/波动率同样无可证伪信号 → 正式结论：Kronos-base 对 A 股单名【方向+幅度】均不可用。")
        lines.append(f"  系统转向：用可证伪简单规则（MA20/60 金叉死叉 / 趋势）替代方向预测，并废弃 h 校准项；")
        lines.append(f"  先做 backlog 8『Kronos 方向 vs 朴素动量 head-to-head』与『MA 规则回测』确认替代方案有边缘（exp_step5_ma_baseline.py），再重构 predict_multi。")
    else:
        lines.append(f"  幅度/波动率存在微弱信号 → 用其构建仓位尺度叠加层（高预测波动→降仓）并回测（backlog 9 元模型）。")
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
