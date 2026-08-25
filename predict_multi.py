# 多票 Kronos 校准版批量预测
# 用法：HF_ENDPOINT=https://hf-mirror.com python predict_multi.py
# 输出：daily_output/YYYYMMDD/ 下每只票的 png/csv/md，以及一份汇总报告
import sys, os, json, warnings
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
OUT = os.path.join(HERE, "daily_output", TODAY.strftime("%Y-%m-%d"))
os.makedirs(OUT, exist_ok=True)
CLOSE_IDX = 3

# 同花顺自选股（图中识别出的 11 只）
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


def load_calib(item):
    """优先读取 per-stock 校准文件，否则根目录徐工 calibration.json（仅对 000425 有效），否则兜底。"""
    code = item["code"]
    per_stock = os.path.join(HERE, "calibration_output", sec_id(item), "calibration.json")
    if os.path.exists(per_stock):
        with open(per_stock, encoding="utf-8") as f:
            return json.load(f), False
    root = os.path.join(HERE, "calibration_output", "calibration.json")
    if code == "000425" and os.path.exists(root):
        with open(root, encoding="utf-8") as f:
            return json.load(f), False
    print(f"  [{item['name']}] 未找到专属校准，使用兜底 k=0.5/h=0.55")
    return {"k_return_shrinkage": 0.5, "direction_hit_rate": 0.55,
            "n_anchors": 0, "pred_len": PL, "lookback": LB}, True


def predict_one(item, pred_inst):
    sec = sec_id(item)
    name = item["name"]
    calib, fallback = load_calib(item)
    k = float(calib.get("k_return_shrinkage", 0.5))
    h = float(calib.get("direction_hit_rate", 0.55))
    n_anchors = int(calib.get("n_anchors", 0))

    df = fetch(sec, FETCH_START, TODAY.strftime("%Y-%m-%d"))
    if df is None or len(df) < LB + PL:
        print(f"  [{name}] 行情不足或拉取失败，跳过")
        return None

    last = float(df["close"].iloc[-1])
    last_dt = df["ts"].iloc[-1]
    print(f"  [{name}] 最新 {last_dt:%Y-%m-%d} 收盘 {last:.2f}  k={k:.3f} h={h:.1%}{' [兜底]' if fallback else ''}")

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
    direction = "偏强" if raw_chg > 0 else "偏弱"
    cl = df["close"].values
    ret20 = cl[-1] / cl[-20] - 1 if len(cl) >= 20 else np.nan
    ret60 = cl[-1] / cl[-60] - 1 if len(cl) >= 60 else np.nan
    recent_vol = float(pd.Series(cl).pct_change().iloc[-30:].std() * 100) if len(cl) >= 30 else np.nan

    # 单品 CSV
    csvp = os.path.join(OUT, f"{name}_预测_校准.csv")
    rep = pd.DataFrame({
        "日期": [d.strftime("%Y-%m-%d") for d in future],
        "校准预测收盘": np.round(cal_close, 3),
        "校准涨跌幅(%)": np.round((cal_close / last - 1) * 100, 2),
        "未校准预测收盘": np.round(raw_close, 3),
        "未校准涨跌幅(%)": np.round((raw_close / last - 1) * 100, 2),
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
    ax1.plot(future, raw_close, label=f"未校准 ({raw_chg:+.1f}%)", color="#BA7517", lw=1.4, ls="--", marker="o", ms=2)
    ax1.plot(future, cal_close, label=f"校准×k={k:.2f} ({cal_chg:+.1f}%)", color="#C0392B", lw=1.6, marker="o", ms=2)
    ax1.axhline(last, color="gray", ls=":", alpha=0.6, label=f"当前 {last:.2f}")
    ax1.set_title(f"{name}({item['code']}) Kronos 校准预测 {TODAY:%Y-%m-%d}\n当前 {last:.2f} -> 校准期末 {cal_close[-1]:.2f} ({cal_chg:+.1f}%)  方向{direction}  置信度{up_pct:.0f}%")
    ax1.set_ylabel("收盘价"); ax1.legend(fontsize=8); ax1.grid(alpha=0.3)
    ax2.bar(future, (cal_close / last - 1) * 100, color="#C0392B", alpha=0.8)
    ax2.axhline(0, color="black", lw=1)
    if not np.isnan(recent_vol):
        ax2.axhline(recent_vol, color="#185FA5", ls=":", lw=1, label=f"近30日1σ {recent_vol:.2f}%")
        ax2.axhline(-recent_vol, color="#185FA5", ls=":", lw=1)
    ax2.set_title("校准后逐日涨跌幅 (%)"); ax2.set_ylabel("涨跌幅 (%)"); ax2.legend(fontsize=8); ax2.grid(alpha=0.3)
    plt.tight_layout()
    pngp = os.path.join(OUT, f"{name}_预测_校准.png")
    plt.savefig(pngp, dpi=120, bbox_inches="tight")
    plt.close(fig)

    # 单品解释 md
    trend_word = "上行" if ret20 > 0 else "下行"
    model_trend_word = "上行" if raw_chg > 0 else "下行"
    consistency = "一致" if (raw_chg > 0) == (ret20 > 0) else "背离"
    mag_note = "接近零，仅表示净方向偏正" if cal_chg > 0 else "接近零，仅表示净方向偏负"
    fallback_warn = "\n> ⚠️ 该标的尚未完成专属校准，当前使用 k=0.5/h=0.55 兜底值，结果仅供参考，建议尽快跑专属校准。\n" if fallback else ""

    expl = f"""# {name}({item['code']}) Kronos 预测解释  {TODAY:%Y-%m-%d}

## 一、结论速览
- 当前价：**{last:.2f}**（{last_dt:%Y-%m-%d} 收盘）
- 模型方向（样本一致性 {up_pct:.0f}%）：**{direction}**；未校准外推期末收益 **{raw_chg:+.1f}%**
- 校准量级参考（期末 {future[-1]:%Y-%m-%d}）：约 **{cal_close[-1]:.2f}（{cal_chg:+.1f}%）**——因 k={k:.3f} 已把幅度压到{mag_note}，此数**不可读作价格目标**
- 方向置信度：**{up_pct:.0f}%**（{SC} 次采样中看多样本占比）；样本期末收益中位 {end_med:+.1%}（IQR {end_q1:+.1%}~{end_q3:+.1%}){fallback_warn}

## 二、模型在外推什么
Kronos 是**趋势外推型**金融 K 线基础模型。它不预测"拐点"，只会把**最近一段走势的形状**延续下去。
- **模型自己外推的方向**：{model_trend_word}（未校准期末收益 {raw_chg:+.1f}%）。
- **近期真实背景**：20 日收益 **{ret20:+.1%}**、60 日收益 **{ret60:+.1%}**；近 30 日日均波动（1σ）约 **{recent_vol:.2f}%**。
- ⚠️ 模型外推方向（{model_trend_word}）与近 20 日真实走势（{trend_word}）**{consistency}**。趋势外推模型读的是 K 线局部惯性，不等于对中期方向有判断。

## 三、怎么用（实战口径）
1. **不单独据此开仓**；模型方向只配当你的其他分析（均线/动量/基本面）的**一致性校验**。
2. **同向** → 略增强信心；**反向** → 仅作"警惕拐点"提醒，不改变你的原有计划。
3. 未校准幅度 **{raw_chg:+.1f}%** 已被 k={k:.3f} 压缩到 **{cal_chg:+.1f}%**；任何 >±10% 的单日预测都视为噪声。
4. 方向命中率 h={h:.0%}（{n_anchors} 个锚点历史估计）→ 仅略高于随机，**远不到可下注的把握**。

## 四、诚实边界
- ⚠️ 本结果**仅供研究/参考，不构成任何投资建议**。
"""
    md_path = os.path.join(OUT, f"{name}_预测_解释.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(expl)

    return {
        "name": name, "code": item["code"], "sec": sec,
        "last": last, "last_dt": last_dt,
        "raw_chg": raw_chg, "cal_chg": cal_chg, "direction": direction,
        "up_pct": up_pct, "k": k, "h": h, "n_anchors": n_anchors,
        "ret20": ret20, "ret60": ret60, "recent_vol": recent_vol,
        "fallback": fallback, "csvp": csvp, "pngp": pngp, "md_path": md_path,
        "cal_close": cal_close, "raw_close": raw_close, "future": future,
    }


def build_summary(records):
    df = pd.DataFrame([{
        "股票": r["name"], "代码": r["code"],
        "当前价": f"{r['last']:.2f}",
        "模型方向": r["direction"],
        "未校准30日": f"{r['raw_chg']:+.1f}%",
        "校准30日": f"{r['cal_chg']:+.1f}%",
        "置信度": f"{r['up_pct']:.0f}%",
        "k": f"{r['k']:.3f}{'*' if r['fallback'] else ''}",
        "h": f"{r['h']:.0%}{'*' if r['fallback'] else ''}",
        "20日真实": f"{r['ret20']:+.1%}",
        "方向/真实": "一致" if (r["raw_chg"] > 0) == (r["ret20"] > 0) else "背离",
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
        ax.plot(future, r["raw_close"], color="#BA7517", lw=1.2, ls="--", marker="o", ms=2, label=f"未校准 {r['raw_chg']:+.1f}%")
        ax.plot(future, r["cal_close"], color="#C0392B", lw=1.4, marker="o", ms=2, label=f"校准 {r['cal_chg']:+.1f}%")
        ax.axhline(r["last"], color="gray", ls=":", alpha=0.6)
        ax.set_title(f"{r['name']}({r['code']}) {r['direction']} | k={r['k']:.2f} h={r['h']:.0%}{'*' if r['fallback'] else ''}", fontsize=10)
        ax.set_ylabel("收盘价")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
        ax.tick_params(axis='x', rotation=30, labelsize=7)
    plt.tight_layout()
    p = os.path.join(OUT, f"00_汇总_预测走势图_{TODAY:%Y-%m-%d}.png")
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return p


def main():
    from model import Kronos, KronosTokenizer, KronosPredictor
    print(f"===== 多票 Kronos 校准预测 {TODAY:%Y-%m-%d} =====")
    print(f"加载模型 {MODEL} ...")
    tok = KronosTokenizer.from_pretrained(TOK)
    model = Kronos.from_pretrained(MODEL)
    pred_inst = KronosPredictor(model, tok, device=DEV, max_context=512)
    print("模型就绪。开始批量推理 ...\n")

    records = []
    for item in WATCHLIST:
        r = predict_one(item, pred_inst)
        if r:
            records.append(r)
        print()

    if not records:
        raise RuntimeError("没有成功生成任何预测")

    summary = build_summary(records)
    csv_summary = os.path.join(OUT, f"00_汇总_预测表_{TODAY:%Y-%m-%d}.csv")
    summary.to_csv(csv_summary, index=False, encoding="utf-8-sig")

    png_summary = build_summary_chart(records)

    md_lines = [
        f"# Kronos 多票校准预测汇总  {TODAY:%Y-%m-%d}",
        "",
        f"共 {len(records)} 只自选股，预测未来 {PL} 个交易日。",
        "",
        "## 汇总表",
        "",
        summary.to_markdown(index=False),
        "",
        "> 注：k/h 带 `*` 表示该标的尚未完成专属校准，使用兜底值 k=0.5/h=0.55，结果仅供临时参考。",
        "",
        "## 使用口径（必读）",
        "1. **不单独据此开仓**：模型方向命中率 h 多数仅略高于随机，只能当一致性校验。",
        "2. **幅度不可信**：k 是对模型幅度的压缩系数；校准后 30 日涨跌幅通常被压到接近 0，不能当价格目标。",
        "3. **方向/真实背离要警惕**：若模型方向与近 20 日真实走势相反，说明模型在延续旧趋势惯性，可能错过拐点。",
        "4. 任何单日预测 >±10% 视为噪声；本结果仅供研究参考，不构成投资建议。",
        "",
        "## 各票详细解释",
        "",
    ]
    for r in records:
        md_lines.append(f"- [{r['name']}({r['code']})]({os.path.basename(r['md_path'])}): 模型方向 **{r['direction']}**，校准 30 日 **{r['cal_chg']:+.1f}%**，与 20 日真实走势 **{'一致' if (r['raw_chg']>0)==(r['ret20']>0) else '背离'}**")
    md_lines.append("")
    md_lines.append(f"![汇总走势图]({os.path.basename(png_summary)})")

    md_path = os.path.join(OUT, f"00_汇总_预测解释_{TODAY:%Y-%m-%d}.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))

    print(f"\n===== 汇总完成 =====")
    print(f"汇总表: {csv_summary}")
    print(f"汇总图: {png_summary}")
    print(f"汇总解释: {md_path}")
    print(f"\n{summary.to_string(index=False)}")


if __name__ == "__main__":
    main()
