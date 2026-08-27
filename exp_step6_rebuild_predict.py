# STEP 6 — predict_multi 重构版（诚实化 + MA 风控开关）
# 设计依据（见 OPTIMIZATION_LOG.md STEP 1-5）：
#   - Kronos-base 在 A 股个股月频上**无方向 alpha**（STEP 1-3 证伪，h 校准项是噪声）。
#   - 朴素 MA / 动量规则也**无方向 alpha**（STEP 5 证伪），但日频 long/flat MA(20,60)
#     是稳健的"回撤控制/风控开关"（躲过下跌、牺牲单边牛市），属可证伪机制。
#   - 因此本版：删除"看多/看空"方向预测叙事与 h 依赖；Kronos 仅输出"趋势延续情景参考路径"
#     （中位外推形状，明确标注非预测）；新增 MA(20,60) long/flat 作为"风控开关"列。
# 用法：HF_ENDPOINT=https://hf-mirror.com python exp_step6_rebuild_predict.py [--limit N] [--only 000425]
#   --limit N   只跑前 N 只（快速自测）
#   --only CODE 只跑指定 code（如 000425）
# 输出：daily_output_v2/YYYY-MM-DD/ 下每只票的 png/csv/md + 汇总报告
import sys, os, json, warnings, argparse
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(HERE)
from calibrate_xugong import raw_predict, sample_daily_returns, fetch, build_future  # noqa

TOK = "NeoQuasar/Kronos-Tokenizer-base"
MODEL = "NeoQuasar/Kronos-base"
DEV = "cpu"
TODAY = datetime.now()
OUT = os.path.join(HERE, "daily_output_v2", TODAY.strftime("%Y-%m-%d"))
os.makedirs(OUT, exist_ok=True)
CLOSE_IDX = 3

WATCHLIST = [
    {"name": "招商积余", "code": "001914", "prefix": "sz"},
    {"name": "今世缘",   "code": "603369", "prefix": "sh"},
    {"name": "源飞宠物", "code": "001222", "prefix": "sz"},
    {"name": "同仁堂",   "code": "600085", "prefix": "sh"},
    {"name": "启明星辰", "code": "002439", "prefix": "sz"},
    {"name": "徐工机械", "code": "000425", "prefix": "sz"},
    {"name": "泰瑞机器", "code": "603289", "prefix": "sh"},
    {"name": "华天科技", "code": "002185", "prefix": "sz"},
    {"name": "中国平安", "code": "601318", "prefix": "sh"},
    {"name": "招商银行", "code": "600036", "prefix": "sh"},
    {"name": "铜陵有色", "code": "000630", "prefix": "sz"},
]

LB, PL, SC = 150, 30, 8
FETCH_START = "2023-01-01"


def sec_id(item):
    return f"{item['prefix']}{item['code']}"


def ma_risk_switch(close):
    """基于收盘序列末端的 MA(20,60) long/flat 风控开关。返回 (开关, 趋势强度%)。"""
    s = pd.Series(np.asarray(close, dtype=float))
    if len(s) < 60:
        return "数据不足", np.nan
    ma20 = float(s.rolling(20).mean().iloc[-1])
    ma60 = float(s.rolling(60).mean().iloc[-1])
    if ma20 > ma60:
        switch = "多头持有"
    else:
        switch = "空仓观望"
    gap = (ma20 / ma60 - 1.0) * 100.0
    return switch, gap


def predict_one(item, pred_inst):
    sec = sec_id(item)
    name = item["name"]

    df = fetch(sec, FETCH_START, TODAY.strftime("%Y-%m-%d"))
    if df is None or len(df) < LB + PL:
        print(f"  [{name}] 行情不足或拉取失败，跳过")
        return None

    last = float(df["close"].iloc[-1])
    last_dt = df["ts"].iloc[-1]
    print(f"  [{name}] 最新 {last_dt:%Y-%m-%d} 收盘 {last:.2f}")

    x_df = df.iloc[-LB:][["open", "high", "low", "close", "vol"]].copy(); x_df["amount"] = 0.0
    x_ts = df.iloc[-LB:]["ts"]
    future = build_future(last_dt, PL)
    y_ts = pd.Series(future)

    try:
        raw = raw_predict(pred_inst, x_df, x_ts, y_ts, PL, T=1.0, top_p=0.95, sample_count=SC)
    except Exception as e:
        print(f"  [{name}] 推理失败: {e}")
        return None

    gs = np.array([sample_daily_returns(raw[s, :, CLOSE_IDX], last) for s in range(SC)])
    g_med = np.median(gs, axis=0)
    g_use = g_med.copy()
    g_use[0] = np.median(g_med[1:])  # 去首日伪影

    raw_close = np.empty(PL); c = last
    for t in range(PL):
        c = c * (1.0 + g_use[t]); raw_close[t] = c
    # k 压缩仅作"情景幅度参考"，不再声称是幅度预测
    k = 0.5
    cal_close = np.empty(PL); c = last
    for t in range(PL):
        c = c * (1.0 + k * g_use[t]); cal_close[t] = c

    end_rets = np.array([float(np.prod(1.0 + g[1:]) - 1.0) for g in gs])
    up_pct = float(np.mean(end_rets > 0)) * 100
    end_med = float(np.median(end_rets))
    end_q1 = float(np.percentile(end_rets, 25))
    end_q3 = float(np.percentile(end_rets, 75))

    raw_chg = (raw_close[-1] / last - 1) * 100
    cal_chg = (cal_close[-1] / last - 1) * 100
    model_trend = "上行(趋势延续)" if raw_chg > 0 else "下行(趋势延续)"

    cl = df["close"].values
    ret20 = cl[-1] / cl[-20] - 1 if len(cl) >= 20 else np.nan
    ret60 = cl[-1] / cl[-60] - 1 if len(cl) >= 60 else np.nan
    recent_vol = float(pd.Series(cl).pct_change().iloc[-30:].std() * 100) if len(cl) >= 30 else np.nan
    switch, ma_gap = ma_risk_switch(cl)
    ma_consistent = "一致" if (raw_chg > 0) == (ret20 > 0) else "背离"

    # 单品 CSV
    csvp = os.path.join(OUT, f"{name}_预测_参考.csv")
    rep = pd.DataFrame({
        "日期": [d.strftime("%Y-%m-%d") for d in future],
        "情景参考收盘(k=0.5)": np.round(cal_close, 3),
        "情景参考涨跌幅(%)": np.round((cal_close / last - 1) * 100, 2),
        "模型外推收盘(未校准)": np.round(raw_close, 3),
        "模型外推涨跌幅(%)": np.round((raw_close / last - 1) * 100, 2),
    })
    rep.to_csv(csvp, index=False, encoding="utf-8-sig")

    # 单品图
    import matplotlib
    matplotlib.use("Agg"); import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial"]
    plt.rcParams["axes.unicode_minus"] = False

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    hist = df.set_index("ts")["close"].iloc[-200:]
    ax1.plot(hist.index, hist.values, label="历史收盘", color="#185FA5", lw=1.6)
    ax1.plot(future, raw_close, label=f"模型外推(未校准 {raw_chg:+.1f}%)", color="#BA7517", lw=1.4, ls="--", marker="o", ms=2)
    ax1.plot(future, cal_close, label=f"情景参考(k=0.5 {cal_chg:+.1f}%)", color="#C0392B", lw=1.6, marker="o", ms=2)
    ax1.axhline(last, color="gray", ls=":", alpha=0.6, label=f"当前 {last:.2f}")
    ax1.set_title(f"{name}({item['code']}) Kronos 情景参考 {TODAY:%Y-%m-%d}\n模型外推趋势{model_trend} | MA风控开关:{switch}(趋势强度{ma_gap:+.1f}%)")
    ax1.set_ylabel("收盘价"); ax1.legend(fontsize=8); ax1.grid(alpha=0.3)
    ax2.bar(future, (cal_close / last - 1) * 100, color="#C0392B", alpha=0.8)
    ax2.axhline(0, color="black", lw=1)
    if not np.isnan(recent_vol):
        ax2.axhline(recent_vol, color="#185FA5", ls=":", lw=1, label=f"近30日1σ {recent_vol:.2f}%")
        ax2.axhline(-recent_vol, color="#185FA5", ls=":", lw=1)
    ax2.set_title("情景参考逐日涨跌幅 (%)"); ax2.set_ylabel("涨跌幅 (%)"); ax2.legend(fontsize=8); ax2.grid(alpha=0.3)
    plt.tight_layout()
    pngp = os.path.join(OUT, f"{name}_预测_参考.png")
    plt.savefig(pngp, dpi=120, bbox_inches="tight")
    plt.close(fig)

    # 单品解释 md
    expl = f"""# {name}({item['code']}) Kronos 情景参考 + MA 风控开关  {TODAY:%Y-%m-%d}

## 一、结论速览
- 当前价：**{last:.2f}**（{last_dt:%Y-%m-%d} 收盘）
- **模型外推趋势（趋势延续假设，非预测）**：{model_trend}，未校准外推期末 {raw_chg:+.1f}%
- **情景参考幅度（k=0.5 压缩，不可当价格目标）**：期末约 {cal_close[-1]:.2f}（{cal_chg:+.1f}%）
- **MA(20,60) 风控开关：{switch}**（趋势强度 {ma_gap:+.1f}%）——可证伪的回撤控制机制
- 采样看多占比 {up_pct:.0f}%（{SC} 次采样中正向占比，仅反映外推形态，非置信度）；样本期末收益中位 {end_med:+.1%}（IQR {end_q1:+.1%}~{end_q3:+.1%})

## 二、模型在外推什么（诚实边界）
Kronos 是**趋势外推型**模型，只会把最近一段 K 线惯性延续下去，**不预测拐点、无方向 alpha**（已用大样本严格证伪，方向命中率 ≈ 抛硬币）。
- 模型外推趋势：{model_trend}；近期真实：20 日 {ret20:+.1%}/60 日 {ret60:+.1%}；模型外推与近 20 日真实 {ma_consistent}。
- 上图"情景参考路径"是**形状/幅度量级参考**，不是方向预测；任何 >±10% 的单日预测都视为噪声。

## 三、怎么用（实战口径）
1. **不据此开仓/平仓**：模型方向无可预测 alpha，仅作"趋势延续情景"参考。
2. **仓位/风控看 MA 开关**：`{switch}` 是日频 long/flat 回测验证过的回撤控制机制——多头持有=趋势在上方持股，空仓观望=趋势走弱减仓避险（会牺牲单边牛市收益，但降低回撤）。
3. 近 30 日日均波动(1σ) 约 {recent_vol:.2f}%，用于仓位尺度参考（波动大→降仓）。
4. ⚠️ 本结果**仅供研究/参考，不构成任何投资建议**。
"""
    md_path = os.path.join(OUT, f"{name}_预测_参考.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(expl)

    return {
        "name": name, "code": item["code"], "sec": sec,
        "last": last, "last_dt": last_dt,
        "raw_chg": raw_chg, "cal_chg": cal_chg, "model_trend": model_trend,
        "up_pct": up_pct, "ma_switch": switch, "ma_gap": ma_gap,
        "ret20": ret20, "ret60": ret60, "recent_vol": recent_vol,
        "ma_consistent": ma_consistent, "csvp": csvp, "pngp": pngp, "md_path": md_path,
        "cal_close": cal_close, "raw_close": raw_close, "future": future,
    }


def build_summary(records):
    df = pd.DataFrame([{
        "股票": r["name"], "代码": r["code"],
        "当前价": f"{r['last']:.2f}",
        "模型外推趋势": r["model_trend"],
        "情景参考30日": f"{r['cal_chg']:+.1f}%",
        "MA风控开关": r["ma_switch"],
        "趋势强度%": f"{r['ma_gap']:+.1f}",
        "近30日波动1σ": f"{r['recent_vol']:.2f}%",
        "外推/真实20日": r["ma_consistent"],
    } for r in records])
    return df


def build_summary_chart(records):
    import matplotlib
    matplotlib.use("Agg"); import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial"]
    plt.rcParams["axes.unicode_minus"] = False

    n = len(records)
    cols = 3
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(18, 4.5 * rows))
    axes = axes.flatten()
    for ax in axes[n:]:
        ax.axis("off")

    for ax, r in zip(axes, records):
        future = r["future"]
        ax.plot(future, r["raw_close"], color="#BA7517", lw=1.2, ls="--", marker="o", ms=2, label=f"外推 {r['raw_chg']:+.1f}%")
        ax.plot(future, r["cal_close"], color="#C0392B", lw=1.4, marker="o", ms=2, label=f"情景 {r['cal_chg']:+.1f}%")
        ax.axhline(r["last"], color="gray", ls=":", alpha=0.6)
        color = "#185FA5" if "多头" in r["ma_switch"] else "#7F8C8D"
        ax.set_title(f"{r['name']}({r['code']}) {r['ma_switch']} | {r['ma_gap']:+.1f}%", fontsize=10, color=color)
        ax.set_ylabel("收盘价")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
        ax.tick_params(axis='x', rotation=30, labelsize=7)
    plt.tight_layout()
    p = os.path.join(OUT, f"00_汇总_情景参考图_{TODAY:%Y-%m-%d}.png")
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 只（自测）")
    ap.add_argument("--only", type=str, default="", help="只跑指定 code，如 000425")
    args = ap.parse_args()

    from model import Kronos, KronosTokenizer, KronosPredictor
    print(f"===== STEP6 重构版 情景参考+MA风控  {TODAY:%Y-%m-%d} =====")
    print(f"加载模型 {MODEL} ...")
    tok = KronosTokenizer.from_pretrained(TOK)
    model = Kronos.from_pretrained(MODEL)
    pred_inst = KronosPredictor(model, tok, device=DEV, max_context=512)
    print("模型就绪。开始批量推理 ...\n")

    items = WATCHLIST
    if args.only:
        items = [it for it in WATCHLIST if it["code"] == args.only]
    elif args.limit:
        items = WATCHLIST[:args.limit]

    records = []
    for item in items:
        r = predict_one(item, pred_inst)
        if r:
            records.append(r)
        print()

    if not records:
        raise RuntimeError("没有成功生成任何预测")

    summary = build_summary(records)
    csv_summary = os.path.join(OUT, f"00_汇总_情景参考表_{TODAY:%Y-%m-%d}.csv")
    summary.to_csv(csv_summary, index=False, encoding="utf-8-sig")

    png_summary = build_summary_chart(records)

    md_lines = [
        f"# Kronos 情景参考 + MA 风控开关 汇总  {TODAY:%Y-%m-%d}",
        "",
        f"共 {len(records)} 只自选股，预测未来 {PL} 个交易日。",
        "",
        "## 系统定位（基于 STEP 1-5 实证）",
        "- Kronos-base 在 A 股个股月频上**无方向 alpha**（大样本证伪）；不再输出\"看多/看空\"方向预测。",
        "- Kronos 仅作**趋势延续情景参考路径**（中位外推形状，明确标注非预测）。",
        "- **MA(20,60) long/flat 作为可证伪的回撤控制开关**（明确标注\"风控\"而非\"方向预测\"）。",
        "",
        "## 汇总表",
        "",
        summary.to_markdown(index=False),
        "",
        "## 使用口径（必读）",
        "1. **不据此开仓**：模型方向无可预测 alpha，仅当趋势延续情景参考。",
        "2. **仓位/风控看 MA 开关**：多头持有=持股，空仓观望=减仓避险（回撤控制，会牺牲单边牛市收益）。",
        "3. **幅度不可信**：k=0.5 是情景压缩系数；情景 30 日涨跌幅不可当价格目标。",
        "4. 任何单日预测 >±10% 视为噪声；本结果仅供研究参考，不构成投资建议。",
        "",
        "## 各票详细参考",
        "",
    ]
    for r in records:
        md_lines.append(f"- [{r['name']}({r['code']})]({os.path.basename(r['md_path'])}): 模型外推 {r['model_trend']}，MA风控 {r['ma_switch']}({r['ma_gap']:+.1f}%)，与20日真实 {r['ma_consistent']}")
    md_lines.append("")
    md_lines.append(f"![汇总情景参考图]({os.path.basename(png_summary)})")

    md_path = os.path.join(OUT, f"00_汇总_情景参考解释_{TODAY:%Y-%m-%d}.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))

    print(f"\n===== 汇总完成 =====")
    print(f"汇总表: {csv_summary}")
    print(f"汇总图: {png_summary}")
    print(f"汇总解释: {md_path}")
    print(f"\n{summary.to_string(index=False)}")


if __name__ == "__main__":
    main()
