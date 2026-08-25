# 回测对照：Kronos 未校准预测 vs 真实走势（兼与校准/改良版比较）
# 目的：客观回答"未校准的结果是不是更像真实走势"——用误差数据而非观感。
#   - 未校准(raw)：模型原始幅度，重新锚定到真实最新收盘、做首日 level-shift 修复（即每日脚本里的橙线）
#   - 校准(×k)：raw 的日收益整体 ×k（calibration.json 的 k_return_shrinkage）
#   - 改良(方向保留+幅度约束)：保留模型路径形状与方向，但把幅度缩放到"徐工历史典型 30 日波动"量级
# 指标：端点误差%、方向命中率、路径 MAE%（逐日 |预测/真实-1| 的均值）
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
CLOSE_IDX = 3
OUT = os.path.join(HERE, "backtest_output")
os.makedirs(OUT, exist_ok=True)


def load_calib():
    p = os.path.join(HERE, "calibration_output", "calibration.json")
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def main():
    import matplotlib
    matplotlib.use("Agg"); import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial"]
    plt.rcParams["axes.unicode_minus"] = False

    from model import Kronos, KronosTokenizer, KronosPredictor
    calib = load_calib()
    k = float(calib["k_return_shrinkage"])

    print("加载模型 ...")
    tok = KronosTokenizer.from_pretrained(TOK)
    model = Kronos.from_pretrained(MODEL)
    pred_inst = KronosPredictor(model, tok, device=DEV, max_context=512)

    print("拉取徐工机械全量历史 ...")
    df = fetch("sz000425", "2022-01-01", datetime.now().strftime("%Y-%m-%d"))
    LB, PL, SC = 150, 30, 2

    anchors = []
    for y in range(2024, 2027):
        for m in range(1, 13):
            cand = df[(df.ts.dt.year == y) & (df.ts.dt.month == m)]
            if len(cand) == 0:
                continue
            a = cand.iloc[0]["ts"]
            if a >= df["ts"].iloc[-1] - timedelta(days=PL + 5):
                continue
            anchors.append(a)
    anchors = [a for a in anchors if (df["ts"] <= a).sum() >= LB]
    print(f"回测锚点 {len(anchors)} 个，SC={SC}")

    recs = []
    for ai, a in enumerate(anchors):
        train = df[df.ts <= a]
        test = df[df.ts > a].iloc[:PL]
        if len(test) < PL:
            continue
        last = float(train["close"].iloc[-1])
        x_df = train.iloc[-LB:][["open", "high", "low", "close", "vol"]].copy(); x_df["amount"] = 0.0
        x_ts = train.iloc[-LB:]["ts"]
        y_ts = test["ts"].reset_index(drop=True)
        raw = raw_predict(pred_inst, x_df, x_ts, y_ts, PL, T=1.0, top_p=0.95, sample_count=SC)
        gs = np.array([sample_daily_returns(raw[s, :, CLOSE_IDX], last) for s in range(raw.shape[0])])
        g_med = np.median(gs, axis=0)
        g_use = g_med.copy(); g_use[0] = np.median(g_med[1:])  # 首日修复

        # 未校准路径
        raw_path = np.empty(PL); c = last
        for t in range(PL):
            c = c * (1.0 + g_use[t]); raw_path[t] = c
        # 校准路径
        cal_path = np.empty(PL); c = last
        for t in range(PL):
            c = c * (1.0 + k * g_use[t]); cal_path[t] = c
        # 改良路径：保留形状与方向，幅度缩放到历史典型 30 日波动量级（稍后统一缩）
        actual = test["close"].values.astype(float)

        rec = {
            "anchor": a, "last": last, "actual": actual,
            "raw": raw_path, "cal": cal_path,
            "R_act": actual[-1] / last - 1.0,
            "R_raw": raw_path[-1] / last - 1.0,
            "R_cal": cal_path[-1] / last - 1.0,
        }
        recs.append(rec)
        if (ai + 1) % 5 == 0:
            print(f"  处理 {ai+1}/{len(anchors)}")

    # 历史典型 30 日波动幅度 M = median|R_act|
    M = float(np.median([abs(r["R_act"]) for r in recs]))
    # 构建改良路径（方向保留 + 幅度约束到 M）
    for r in recs:
        target = r["last"] * (1.0 + np.sign(r["R_raw"]) * M)
        f = (target - r["last"]) / (r["raw"][-1] - r["last"]) if (r["raw"][-1] - r["last"]) != 0 else 0.0
        r["bounded"] = r["last"] + (r["raw"] - r["last"]) * f
        r["R_bounded"] = r["bounded"][-1] / r["last"] - 1.0

    def ep_err(path, act, last):  # 端点误差 %
        return abs(path[-1] / act[-1] - 1.0) * 100
    def path_mae(path, act, last):  # 路径 MAE %
        return float(np.mean(np.abs(path / act - 1.0)) * 100)

    rows = []
    for r in recs:
        rows.append({
            "anchor": r["anchor"].strftime("%Y-%m"),
            "R_act%": r["R_act"] * 100,
            "R_raw%": r["R_raw"] * 100,
            "R_cal%": r["R_cal"] * 100,
            "R_bounded%": r["R_bounded"] * 100,
            "ep_raw": ep_err(r["raw"], r["actual"], r["last"]),
            "ep_cal": ep_err(r["cal"], r["actual"], r["last"]),
            "ep_bounded": ep_err(r["bounded"], r["actual"], r["last"]),
            "mae_raw": path_mae(r["raw"], r["actual"], r["last"]),
            "mae_cal": path_mae(r["cal"], r["actual"], r["last"]),
            "mae_bounded": path_mae(r["bounded"], r["actual"], r["last"]),
            "dir_raw_hit": np.sign(r["R_raw"]) == np.sign(r["R_act"]),
        })
    sdf = pd.DataFrame(rows)
    csvp = os.path.join(OUT, "backtest_uncal_vs_actual.csv")
    sdf.to_csv(csvp, index=False, encoding="utf-8-sig")

    print(f"\n[M={M:+.1%} 为徐工历史典型 30 日波动幅度]")
    print(f"端点误差%(越低越好):  未校准 {sdf.ep_raw.mean():.1f} | 校准×k {sdf.ep_cal.mean():.1f} | 方向保留+约束 {sdf.ep_bounded.mean():.1f}")
    print(f"路径MAE%(越低越好):    未校准 {sdf.mae_raw.mean():.1f} | 校准×k {sdf.mae_cal.mean():.1f} | 方向保留+约束 {sdf.mae_bounded.mean():.1f}")
    print(f"方向命中率:           未校准 {sdf.dir_raw_hit.mean():.0%}")

    # 图：最近 8 个锚点，actual vs 未校准(橙) vs 校准(红) vs 改良(绿)
    pick = recs[-8:]
    fig, axes = plt.subplots(4, 2, figsize=(15, 16))
    axes = axes.ravel()
    for i, r in enumerate(pick):
        ax = axes[i]
        fut = [r["anchor"] + timedelta(days=d + 1) for d in range(PL)]
        ax.plot(fut, r["actual"], color="#185FA5", lw=2, label="真实")
        ax.plot(fut, r["raw"], color="#BA7517", lw=1.4, ls="--", label=f"未校准 {r['R_raw']:+.1%}")
        ax.plot(fut, r["cal"], color="#C0392B", lw=1.4, ls=":", label=f"校准×k {r['R_cal']:+.1%}")
        ax.plot(fut, r["bounded"], color="#27C08A", lw=1.4, ls="-.", label=f"方向保留+约束 {r['R_bounded']:+.1%}")
        ax.axhline(r["last"], color="gray", ls=":", alpha=0.6)
        ax.set_title(f"{r['anchor']:%Y-%m}  真实{r['R_act']:+.1%} | 未校准端点误差 {ep_err(r['raw'],r['actual'],r['last']):.0f}%")
        ax.legend(fontsize=7); ax.grid(alpha=0.3)
    plt.tight_layout()
    pngp = os.path.join(OUT, "backtest_overlay.png")
    plt.savefig(pngp, dpi=130, bbox_inches="tight")
    print(f"\nCSV: {csvp}\n图:  {pngp}")


if __name__ == "__main__":
    main()
