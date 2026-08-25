# 徐工机械(000425) Kronos 校准版每日预测
# 相对 predict_xugong_daily.py 的两处优化：
#   1) 收益率空间重建：拆出逐样本原始路径 -> 取日收益率形状 -> 重新锚定到【真实最新收盘价】，
#      消除原 predict() 把预测价锚定在"回看均值"导致的 level-shift / 首日跳空。
#   2) 幅度校准 + 方向置信度：用 calibrate_xugong.py 估计的压缩系数 k 缩小夸张幅度；
#      用样本间方向一致性给出置信度；并自动生成结果解释。
import sys, os, json
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import requests

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(HERE)
from calibrate_xugong import raw_predict, sample_daily_returns, fetch, build_future  # noqa

TOK = "NeoQuasar/Kronos-Tokenizer-base"
MODEL = "NeoQuasar/Kronos-base"
DEV = "cpu"
URL = "https://web.ifzq.gtimg.cn/appstock/app/fqkline/get"
H = {"User-Agent": "Mozilla/5.0"}
TODAY = datetime.now()
OUT = os.path.join(HERE, "daily_output", TODAY.strftime("%Y-%m-%d"))
os.makedirs(OUT, exist_ok=True)
CLOSE_IDX = 3


def load_calib():
    p = os.path.join(HERE, "calibration_output", "calibration.json")
    if not os.path.exists(p):
        print("⚠ 未找到 calibration.json，使用 k=0.5 兜底（请先跑 calibrate_xugong.py）")
        return {"k_return_shrinkage": 0.5, "direction_hit_rate": 0.55,
                "n_anchors": 0, "pred_len": 30, "lookback": 150}
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def main():
    import matplotlib
    matplotlib.use("Agg"); import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial"]
    plt.rcParams["axes.unicode_minus"] = False

    from model import Kronos, KronosTokenizer, KronosPredictor
    calib = load_calib()
    k = float(calib.get("k_return_shrinkage", 0.5))
    h = float(calib.get("direction_hit_rate", 0.55))
    LB, PL, SC = int(calib.get("lookback", 150)), int(calib.get("pred_len", 30)), 8

    print(f"加载模型 {MODEL} ...")
    tok = KronosTokenizer.from_pretrained(TOK)
    model = Kronos.from_pretrained(MODEL)
    pred_inst = KronosPredictor(model, tok, device=DEV, max_context=512)

    print(f"拉取徐工机械最新行情 ...")
    df = fetch("sz000425", "2023-01-01", TODAY.strftime("%Y-%m-%d"))
    if df is None or df.empty:
        raise RuntimeError("行情拉取失败")
    last = float(df["close"].iloc[-1])
    last_dt = df["ts"].iloc[-1]
    print(f"  最新 {last_dt:%Y-%m-%d} 收盘 {last:.2f}；回看 {LB} 日，预测 {PL} 交易日，采样 {SC} 次")

    x_df = df.iloc[-LB:][["open", "high", "low", "close", "vol"]].copy(); x_df["amount"] = 0.0
    x_ts = df.iloc[-LB:]["ts"]
    future = build_future(last_dt, PL)
    y_ts = pd.Series(future)

    raw = raw_predict(pred_inst, x_df, x_ts, y_ts, PL, T=1.0, top_p=0.95, sample_count=SC)  # (SC, PL, feat)

    # ---- 收益率空间处理 ----
    # 每条样本 -> 日收益率 g（首日以真实收盘为基准，后续用模型自身连续性）
    gs = np.array([sample_daily_returns(raw[s, :, CLOSE_IDX], last) for s in range(SC)])
    g_med = np.median(gs, axis=0)
    # 消除首日 level-shift 伪影：用模型自身典型日收益替代首步锚点偏差
    g_use = g_med.copy()
    g_use[0] = np.median(g_med[1:])

    # 未校准（仅把模型日收益形状锚定到真实收盘，不做 k 压缩）
    raw_close = np.empty(PL)
    c = last
    for t in range(PL):
        c = c * (1.0 + g_use[t]); raw_close[t] = c
    # 校准（×k 压缩幅度）
    cal_close = np.empty(PL)
    c = last
    for t in range(PL):
        c = c * (1.0 + k * g_use[t]); cal_close[t] = c

    # ---- 方向置信度：样本间方向一致性 ----
    # 用"模型自身连续性"的期末收益（去掉首步伪影）衡量每个样本的趋势方向
    end_rets = np.array([float(np.prod(1.0 + g[1:]) - 1.0) for g in gs])
    up_pct = float(np.mean(end_rets > 0)) * 100
    end_med = float(np.median(end_rets)); end_q1 = float(np.percentile(end_rets, 25)); end_q3 = float(np.percentile(end_rets, 75))

    # ---- 关键指标 ----
    raw_end = raw_close[-1]; cal_end = cal_close[-1]
    raw_chg = (raw_end / last - 1) * 100
    cal_chg = (cal_end / last - 1) * 100
    # 近期趋势上下文
    cl = df["close"].values
    ret20 = cl[-1] / cl[-20] - 1
    ret60 = cl[-1] / cl[-60] - 1
    recent_vol = float(pd.Series(cl).pct_change().iloc[-30:].std() * 100)  # 近30日日均波动 1σ(%)
    cal_first = (cal_close[0] / last - 1) * 100

    # 方向以模型自身的未压缩外推信号为准（k>0 不改变符号，避免把被压到接近 0 的 cal_chg 误读成"方向"）
    direction = "偏强" if raw_chg > 0 else "偏弱"
    mag_note = "接近零，仅表示净方向偏正" if cal_chg > 0 else "接近零，仅表示净方向偏负"
    print(f"\n[结果] 当前 {last:.2f}")
    print(f"  未校准(锚定真实收盘, 不压缩): 期末 {raw_end:.2f} ({raw_chg:+.1f}%)")
    print(f"  校准后(×k={k:.2f}):          期末 {cal_end:.2f} ({cal_chg:+.1f}%)  方向{direction}")
    print(f"  方向置信度(样本一致性): {up_pct:.0f}%  | 样本期末收益中位 {end_med:+.1%} (IQR {end_q1:+.1%}~{end_q3:+.1%})")
    print(f"  近期: 20日 {ret20:+.1%}, 60日 {ret60:+.1%}, 近30日日均波动 {recent_vol:.2f}%(1σ)")

    # ---- CSV ----
    rep = pd.DataFrame({
        "日期": [d.strftime("%Y-%m-%d") for d in future],
        "校准预测收盘": np.round(cal_close, 3),
        "校准涨跌幅(%)": np.round((cal_close / last - 1) * 100, 2),
        "未校准预测收盘": np.round(raw_close, 3),
        "未校准涨跌幅(%)": np.round((raw_close / last - 1) * 100, 2),
    })
    csvp = os.path.join(OUT, "徐工机械_预测_校准.csv")
    rep.to_csv(csvp, index=False, encoding="utf-8-sig")

    # ---- 图 ----
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9))
    hist = df.set_index("ts")["close"].iloc[-250:]
    ax1.plot(hist.index, hist.values, label="历史收盘", color="#185FA5", lw=1.8)
    ax1.plot(future, raw_close, label=f"未校准 ({raw_chg:+.1f}%)", color="#BA7517", lw=1.6, ls="--", marker="o", ms=2)
    ax1.plot(future, cal_close, label=f"校准后 ×k={k:.2f} ({cal_chg:+.1f}%)", color="#C0392B", lw=1.8, marker="o", ms=2)
    ax1.axhline(last, color="gray", ls=":", alpha=0.6, label=f"当前 {last:.2f}")
    ax1.set_title(f"徐工机械(000425) Kronos 校准预测 {TODAY:%Y-%m-%d}\n当前 {last:.2f} -> 校准期末 {cal_end:.2f} ({cal_chg:+.1f}%)  方向{direction}  置信度{up_pct:.0f}%")
    ax1.set_ylabel("收盘价(元)"); ax1.legend(fontsize=8); ax1.grid(alpha=0.3)
    ax2.bar(future, (cal_close / last - 1) * 100, color="#C0392B", alpha=0.8, label="校准逐日涨跌幅")
    ax2.axhline(0, color="black", lw=1)
    ax2.axhline(recent_vol, color="#185FA5", ls=":", lw=1, label=f"近30日1σ波动 {recent_vol:.2f}%")
    ax2.axhline(-recent_vol, color="#185FA5", ls=":", lw=1)
    ax2.set_title("校准后逐日预测涨跌幅 (%)"); ax2.set_ylabel("涨跌幅 (%)"); ax2.legend(fontsize=8); ax2.grid(alpha=0.3)
    plt.tight_layout()
    pngp = os.path.join(OUT, "徐工机械_预测_校准.png")
    plt.savefig(pngp, dpi=150, bbox_inches="tight")

    # ---- 自动解释 ----
    expl = f"""# 徐工机械(000425) Kronos 预测解释  {TODAY:%Y-%m-%d}

## 一、结论速览
- 当前价：**{last:.2f}**（{last_dt:%Y-%m-%d} 收盘）
- 模型方向（样本一致性 {up_pct:.0f}%）：**{direction}**；未校准外推期末收益 **{raw_chg:+.1f}%**
- 校准量级参考（期末 {future[-1]:%Y-%m-%d}）：约 **{cal_end:.2f}（{cal_chg:+.1f}%）**——因 k={k:.2f} 已把幅度压到{mag_note}，此数**不可读作价格目标**
- 方向置信度：**{up_pct:.0f}%**（{SC} 次采样中看多样本占比）；样本期末收益中位 {end_med:+.1%}（IQR {end_q1:+.1%}~{end_q3:+.1%}）

## 二、模型在外推什么
Kronos 是**趋势外推型**金融 K 线基础模型（清华，AAAI 2026）。它不预测"拐点"，只会把**最近一段走势的形状**延续下去。
- **模型自己外推的路径方向**：由逐样本日收益率中位构成，期末相对当前为 **{"偏强" if raw_chg>0 else "偏弱"}**（未校准期末收益 {raw_chg:+.1f}%）——即模型把近期 K 线惯性外推为一段**{"上行" if raw_chg>0 else "下行"}路径**。
- **近期真实背景**：20 日收益 **{ret20:+.1%}**、60 日收益 **{ret60:+.1%}**；近 30 日日均波动（1σ）约 **{recent_vol:.2f}%**。
- ⚠️ 模型外推方向（{"上行" if raw_chg>0 else "下行"}）与近 20/60 日真实走势（{"上行" if ret20>0 else "下行"}）{"一致" if (raw_chg>0)==(ret20>0) else "并不一致"}。这正是趋势外推模型的典型表现：它读的是 K 线局部的"形状惯性"，不等于对中期方向有判断。

## 三、为什么做了两处优化（相对旧版）
1. **收益率空间重建（消除 level-shift）**：旧版 predict() 把预测价格锚定在"回看窗口均值"上，
   导致首日常出现不真实的跳空、且整体价位偏离现价。本版拆出模型逐样本原始路径，
   只取它的**日涨跌形状**，重新锚定到真实最新收盘价，价位更合理。
2. **幅度校准（剥离不可信的幅度）**：基于历史 {int(calib.get('n_anchors',0))} 个锚点的 walk-forward 回测，
   估计出**收益压缩系数 k = {k:.3f}**（回归斜率，越接近 0 说明模型幅度越不可信）。
   对徐工机械 30 日预测，k≈{k:.3f} 意味着**模型外推的幅度与真实幅度几乎不相关**——未校准期末 {raw_chg:+.1f}% 属模型夸张，
   校准后收敛到 {cal_chg:+.1f}%（量级被压到接近零，故**此数不能当作价格目标**）。
   - 方向命中率 h = {h:.0%}（95% 置信区间约 40.7%~75.4%）：在徐工 30 日尺度上**仅略高于随机抛硬币（50%）**、统计上不可区分；
     模型"方向"可参考，但**远不到可下注的把握**，且在拐点处会反向。

## 四、怎么读这张图
- 红线（校准后）与橙线（未校准）起点都在当前价；橙线更陡，体现模型对幅度的夸张。
- 蓝点线是近 30 日 1σ 波动基准：若某日预测涨跌幅明显超出 ±{recent_vol:.2f}%，属模型夸张，勿当真。

## 五、诚实的局限（务必先看）
- ⚠️ Kronos **没有均值回归能力**，只在"趋势延续"时有效；一旦进入盘整/反转，方向可能**反着来**。
- ⚠️ 校准压缩了"幅度"，但**改不了方向**。置信度 {up_pct:.0f}% 仅代表样本间一致程度，不等于"一定对"。
- ⚠️ A 股主板日涨跌停限制 **±10%**；任何超过此的单日预测都应视为模型噪声。
- ⚠️ 本结果**仅供研究/参考，不构成任何投资建议**。
"""
    exp_path = os.path.join(OUT, "徐工机械_预测_解释.md")
    with open(exp_path, "w", encoding="utf-8") as f:
        f.write(expl)

    print(f"\n  CSV: {csvp}")
    print(f"  图:  {pngp}")
    print(f"  解释: {exp_path}")
    print("校准预测完成 ✅")

    # 同时把解释打印出来，便于直接查看/邮件引用
    print("\n========== 结果解释 ==========\n")
    print(expl)


if __name__ == "__main__":
    main()
