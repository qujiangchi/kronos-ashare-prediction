# 多票 Kronos 专属校准脚本
# ---------------------------------------------------------------------------
# 给自选股清单里的每一只票各跑一次 walk-forward 月度锚点校准，
# 估计专属的两个校准量：
#   1) 收益压缩系数 k   —— 模型对收益率(幅度)系统性高估，实测 k<1
#   2) 方向命中率 h      —— 模型作为"趋势外推器"，方向仅略高于抛硬币
# 结果按票写入 calibration_output/{sec_id}/calibration.json，
# 供 predict_multi.py 用 load_calib() 按票读取（不再用统一兜底值）。
#
# 用法（需在 D:/Creater/Kronos 目录下，已建 venv）：
#   HF_ENDPOINT=https://hf-mirror.com venv/Scripts/python.exe calibrate_multi.py
#
# 方法（同 calibrate_xugong.py，只是循环全部标的）：
#   - 拆出 Kronos 逐样本原始预测路径
#   - 每条样本路径 -> 日收益率序列 g
#   - 月频锚点 t 预测其后 PL=30 交易日，比较 预测期末收益 R_pred 与 真实期末收益 R_act
#   - k = Σ(R_act·R_pred) / Σ(R_pred²)  （过原点回归）
#   - h = mean(sign(R_act) == sign(R_pred))
# ---------------------------------------------------------------------------
import sys, os, json
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(HERE)
from calibrate_xugong import (  # noqa: 复用原始推理与行情拉取
    raw_predict, sample_daily_returns, fetch, build_future,
    TOK, MODEL, DEV, CLOSE_IDX,
)

# 与 predict_multi.py 的 WATCHLIST 保持一致
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

LB, PL, SC = 150, 30, 2          # 校准用较少采样即可（SC=2 足够估 k/h）
FETCH_START = "2022-01-01"
MIN_ANCHORS = 8                  # 锚点过少则放弃该票校准，避免过拟合噪声
OUT_ROOT = os.path.join(HERE, "calibration_output")


def sec_id(item):
    return f"{item['prefix']}{item['code']}"


def build_anchors(df, lb, pl):
    """月频锚点：每月第一个交易日，且需有足够历史与未来 PL 个交易日。"""
    anchors = []
    for y in range(2024, 2027):
        for m in range(1, 13):
            cand = df[(df.ts.dt.year == y) & (df.ts.dt.month == m)]
            if len(cand) == 0:
                continue
            a = cand.iloc[0]["ts"]
            if a >= df["ts"].iloc[-1] - timedelta(days=pl + 5):
                continue
            anchors.append(a)
    return [a for a in anchors if (df["ts"] <= a).sum() >= lb]


def calibrate_one(item, pred_inst):
    sec = sec_id(item)
    name = item["name"]
    print(f"\n===== 校准 {name}({item['code']}) =====")
    df = fetch(sec, FETCH_START, datetime.now().strftime("%Y-%m-%d"))
    if df is None or len(df) < LB + PL:
        print(f"  [{name}] 行情不足或拉取失败，跳过")
        return False
    print(f"  共 {len(df)} 条，区间 {df['ts'].iloc[0]:%Y-%m-%d}~{df['ts'].iloc[-1]:%Y-%m-%d}")

    anchors = build_anchors(df, LB, PL)
    print(f"  回测锚点数: {len(anchors)}")
    if len(anchors) < MIN_ANCHORS:
        print(f"  锚点过少(<{MIN_ANCHORS})，跳过以保校准稳健")
        return False

    recs = []
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
        # 每条样本 -> 日收益 -> 中位路径 -> 预测期末收益
        gs = np.array([sample_daily_returns(raw[s, :, CLOSE_IDX], last) for s in range(raw.shape[0])])
        g_med = np.median(gs, axis=0)
        g_use = g_med.copy()
        g_use[0] = np.median(g_med[1:])  # 去首日 level-shift 伪影
        R_pred = float(np.prod(1.0 + g_use) - 1.0)
        R_act = float(test["close"].iloc[-1] / last - 1.0)
        recs.append({"anchor": a.strftime("%Y-%m-%d"), "R_pred": R_pred, "R_act": R_act})

    if len(recs) < MIN_ANCHORS:
        print(f"  有效锚点 {len(recs)} < {MIN_ANCHORS}，跳过")
        return False

    Rp = np.array([r["R_pred"] for r in recs])
    Ra = np.array([r["R_act"] for r in recs])
    if (Rp ** 2).sum() > 1e-12:
        k = float(np.sum(Ra * Rp) / np.sum(Rp ** 2))
    else:
        k = 1.0
    k = max(0.0, min(1.5, k))
    hit = float(np.mean(np.sign(Ra) == np.sign(Rp)))
    n = len(recs)
    se = (hit * (1 - hit) / n) ** 0.5
    print(f"  k={k:.3f}  h={hit:.1%} (95%CI {max(0, hit - 1.96 * se):.1%}~{min(1, hit + 1.96 * se):.1%})  n={n}")

    calib = {
        "symbol": item["code"], "name": name, "sec_id": sec,
        "model": MODEL, "tokenizer": TOK,
        "lookback": LB, "pred_len": PL, "sample_count_calib": SC,
        "k_return_shrinkage": k, "direction_hit_rate": hit,
        "n_anchors": n, "date_range": [recs[0]["anchor"], recs[-1]["anchor"]],
        "pred_ret_mean": float(Rp.mean()), "act_ret_mean": float(Ra.mean()),
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "method": "walk-forward monthly anchors; k = slope(origin regression) of actual vs predicted 30d return; returns in sample-path space re-anchored to true last close",
    }
    out_dir = os.path.join(OUT_ROOT, sec)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "calibration.json"), "w", encoding="utf-8") as f:
        json.dump(calib, f, ensure_ascii=False, indent=2)
    print(f"  已写入 {out_dir}/calibration.json")
    return True


def main():
    from model import Kronos, KronosTokenizer, KronosPredictor
    print("加载模型 ...")
    tok = KronosTokenizer.from_pretrained(TOK)
    model = Kronos.from_pretrained(MODEL)
    pred_inst = KronosPredictor(model, tok, device=DEV, max_context=512)
    print("模型就绪。开始逐票校准 ...\n")

    ok, skip = [], []
    for item in WATCHLIST:
        if calibrate_one(item, pred_inst):
            ok.append(f"{item['name']}({item['code']})")
        else:
            skip.append(f"{item['name']}({item['code']})")

    print(f"\n===== 校准完成 =====")
    print(f"成功 {len(ok)} 只: {ok}")
    print(f"跳过 {len(skip)} 只: {skip}")


if __name__ == "__main__":
    main()
