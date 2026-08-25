# 徐工机械(000425) Kronos 预测 校准脚本
# 目的：用历史多锚点 walk-forward 回测，估计两个关键校准量：
#   1) 收益压缩系数 k   —— 模型对收益率(幅度)系统性高估，实测 k<1（通常 0.3~0.7）
#   2) 方向命中率 h      —— 模型作为"趋势外推器"，在趋势延续时方向对、拐点处反转
# 结果写入 calibration.json 供 predict_xugong_calibrated.py 读取。
#
# 方法要点（详见实战文档）：
#   - 拆出 Kronos 的【逐样本原始预测路径】（默认 predict() 会做 np.mean 平均掉，拿不到分布）
#   - 把每条样本路径转换成"日收益率序列 g"（用模型预测的日常涨跌形状）
#   - 用历史锚点 t 预测其后 PL=30 交易日，比较 预测期末收益 R_pred 与 真实期末收益 R_act
#   - k = Σ(R_act·R_pred) / Σ(R_pred²)  （过原点回归，衡量"幅度被放大几倍"）
#   - h = mean(sign(R_act) == sign(R_pred))
import sys, os, json
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import requests
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(HERE)
from model import Kronos, KronosTokenizer, KronosPredictor
from model.kronos import auto_regressive_inference, calc_time_stamps, sample_from_logits, top_k_top_p_filtering

TOK = "NeoQuasar/Kronos-Tokenizer-base"
MODEL = "NeoQuasar/Kronos-base"
DEV = "cpu"
URL = "https://web.ifzq.gtimg.cn/appstock/app/fqkline/get"
H = {"User-Agent": "Mozilla/5.0"}
OUT = os.path.join(HERE, "calibration_output")
os.makedirs(OUT, exist_ok=True)
CLOSE_IDX = 3  # ['open','high','low','close','volume','amount'] -> close=3


# ---------- 复刻 auto_regressive_inference，但返回逐样本原始路径 ----------
def raw_auto_regressive_inference(tokenizer, model, x, x_stamp, y_stamp, max_context, pred_len,
                                  clip=5, T=1.0, top_k=0, top_p=0.99, sample_count=5,
                                  verbose=False, return_mean=True):
    with torch.no_grad():
        x = torch.clip(x, -clip, clip)
        device = x.device
        x = x.unsqueeze(1).repeat(1, sample_count, 1, 1).reshape(-1, x.size(1), x.size(2)).to(device)
        x_stamp = x_stamp.unsqueeze(1).repeat(1, sample_count, 1, 1).reshape(-1, x_stamp.size(1), x_stamp.size(2)).to(device)
        y_stamp = y_stamp.unsqueeze(1).repeat(1, sample_count, 1, 1).reshape(-1, y_stamp.size(1), y_stamp.size(2)).to(device)

        x_token = tokenizer.encode(x, half=True)
        initial_seq_len = x.size(1)
        batch_size = x_token[0].size(0)
        total_seq_len = initial_seq_len + pred_len
        full_stamp = torch.cat([x_stamp, y_stamp], dim=1)

        generated_pre = x_token[0].new_empty(batch_size, pred_len)
        generated_post = x_token[1].new_empty(batch_size, pred_len)
        pre_buffer = x_token[0].new_zeros(batch_size, max_context)
        post_buffer = x_token[1].new_zeros(batch_size, max_context)
        buffer_len = min(initial_seq_len, max_context)
        if buffer_len > 0:
            start_idx = max(0, initial_seq_len - max_context)
            pre_buffer[:, :buffer_len] = x_token[0][:, start_idx:start_idx + buffer_len]
            post_buffer[:, :buffer_len] = x_token[1][:, start_idx:start_idx + buffer_len]

        ran = trange if verbose else range
        try:
            from tqdm import trange
            ran = trange
        except Exception:
            ran = range
        for i in ran(pred_len):
            current_seq_len = initial_seq_len + i
            window_len = min(current_seq_len, max_context)
            if current_seq_len <= max_context:
                input_tokens = [pre_buffer[:, :window_len], post_buffer[:, :window_len]]
            else:
                input_tokens = [pre_buffer, post_buffer]
            context_end = current_seq_len
            context_start = max(0, context_end - max_context)
            current_stamp = full_stamp[:, context_start:context_end, :].contiguous()
            s1_logits, context = model.decode_s1(input_tokens[0], input_tokens[1], current_stamp)
            s1_logits = s1_logits[:, -1, :]
            sample_pre = sample_from_logits(s1_logits, temperature=T, top_k=top_k, top_p=top_p, sample_logits=True)
            s2_logits = model.decode_s2(context, sample_pre)
            s2_logits = s2_logits[:, -1, :]
            sample_post = sample_from_logits(s2_logits, temperature=T, top_k=top_k, top_p=top_p, sample_logits=True)
            generated_pre[:, i] = sample_pre.squeeze(-1)
            generated_post[:, i] = sample_post.squeeze(-1)
            if current_seq_len < max_context:
                pre_buffer[:, current_seq_len] = sample_pre.squeeze(-1)
                post_buffer[:, current_seq_len] = sample_post.squeeze(-1)
            else:
                pre_buffer.copy_(torch.roll(pre_buffer, shifts=-1, dims=1))
                post_buffer.copy_(torch.roll(post_buffer, shifts=-1, dims=1))
                pre_buffer[:, -1] = sample_pre.squeeze(-1)
                post_buffer[:, -1] = sample_post.squeeze(-1)

        full_pre = torch.cat([x_token[0], generated_pre], dim=1)
        full_post = torch.cat([x_token[1], generated_post], dim=1)
        context_start = max(0, total_seq_len - max_context)
        input_tokens = [full_pre[:, context_start:total_seq_len].contiguous(),
                        full_post[:, context_start:total_seq_len].contiguous()]
        z = tokenizer.decode(input_tokens, half=True)
        z = z.reshape(-1, sample_count, z.size(1), z.size(2))
        preds = z.cpu().numpy()
        if return_mean:
            preds = np.mean(preds, axis=1)
        return preds  # (1, sample_count, total, feat) when return_mean=False


def raw_predict(predictor, df, x_timestamp, y_timestamp, pred_len, T=1.0, top_k=0, top_p=0.95, sample_count=5):
    price_cols = ['open', 'high', 'low', 'close']; vol_col = 'volume'; amt = 'amount'
    df = df.copy()
    if vol_col not in df.columns:
        df[vol_col] = 0.0; df[amt] = 0.0
    if amt not in df.columns and vol_col in df.columns:
        df[amt] = df[vol_col] * df[price_cols].mean(axis=1)
    x_time_df = calc_time_stamps(x_timestamp); y_time_df = calc_time_stamps(y_timestamp)
    x = df[price_cols + [vol_col, amt]].values.astype(np.float32)
    x_stamp = x_time_df.values.astype(np.float32); y_stamp = y_time_df.values.astype(np.float32)
    x_mean, x_std = np.mean(x, axis=0), np.std(x, axis=0)
    xnorm = (x - x_mean) / (x_std + 1e-5); xnorm = np.clip(xnorm, -predictor.clip, predictor.clip)
    x_tensor = torch.from_numpy(xnorm[np.newaxis, ...]).to(predictor.device)
    x_stamp_tensor = torch.from_numpy(x_stamp[np.newaxis, ...]).to(predictor.device)
    y_stamp_tensor = torch.from_numpy(y_stamp[np.newaxis, ...]).to(predictor.device)
    allpreds = raw_auto_regressive_inference(predictor.tokenizer, predictor.model, x_tensor, x_stamp_tensor,
                                             y_stamp_tensor, predictor.max_context, pred_len, predictor.clip,
                                             T, top_k, top_p, sample_count, False, return_mean=False)
    preds = allpreds[0, :, -pred_len:, :]          # (sample_count, pred_len, feat)
    preds_den = preds * (x_std + 1e-5) + x_mean    # 反标准化（用同一组 mean/std）
    return preds_den                               # (sample_count, pred_len, feat)


def sample_daily_returns(pred_close_sample, last_close):
    """把一条样本路径转成日收益率序列 g（长度 pred_len），首日以真实收盘价做基准。"""
    P = np.concatenate([[last_close], pred_close_sample])
    return P[1:] / P[:-1] - 1.0


def fetch(sec, start, end, count=1024):
    try:
        r = requests.get(URL, params={"param": f"{sec},day,{start},{end},{count},"}, headers=H, timeout=20)
        j = r.json()
        node = j.get("data", {}).get(sec)
        rows = node.get("day") or node.get("qfqday") if isinstance(node, dict) else (node if isinstance(node, list) else None)
        if not rows:
            return None
        df = pd.DataFrame([r[:6] for r in rows], columns=["ts", "open", "close", "high", "low", "vol"])
        for c in ["open", "close", "high", "low", "vol"]:
            df[c] = df[c].astype(float)
        df["ts"] = pd.to_datetime(df["ts"])
        return df.sort_values("ts").reset_index(drop=True)
    except Exception as e:
        print("  fetch err", e); return None


def build_future(last_date, n):
    out, d = [], last_date + timedelta(days=1)
    while len(out) < n:
        if d.weekday() < 5:
            out.append(d)
        d += timedelta(days=1)
    return out


def main():
    print("加载模型 ...")
    tok = KronosTokenizer.from_pretrained(TOK)
    model = Kronos.from_pretrained(MODEL)
    pred_inst = KronosPredictor(model, tok, device=DEV, max_context=512)
    print("模型就绪。")

    print("拉取徐工机械全量历史 ...")
    df = fetch("sz000425", "2022-01-01", datetime.now().strftime("%Y-%m-%d"))
    if df is None or len(df) < 250:
        raise RuntimeError("行情拉取失败")
    print(f"  共 {len(df)} 条，区间 {df['ts'].iloc[0]:%Y-%m-%d}~{df['ts'].iloc[-1]:%Y-%m-%d}")

    LB, PL, SC = 150, 30, 2
    # 锚点：每月第一个交易日，且需有足够历史与未来 PL 个交易日
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
    print(f"回测锚点数: {len(anchors)}")

    recs = []
    for ai, a in enumerate(anchors):
        train = df[df.ts <= a]
        test = df[df.ts > a].iloc[:PL]
        if len(test) < PL:
            continue
        last = train["close"].iloc[-1]
        x_df = train.iloc[-LB:][["open", "high", "low", "close", "vol"]].copy(); x_df["amount"] = 0.0
        x_ts = train.iloc[-LB:]["ts"]
        y_ts = test["ts"].reset_index(drop=True)
        try:
            raw = raw_predict(pred_inst, x_df, x_ts, y_ts, PL, T=1.0, top_p=0.95, sample_count=SC)
        except Exception as e:
            print(f"  ⚠ 锚点 {a:%Y-%m} 失败: {e}"); continue
        # 每条样本 -> 日收益 -> 中位路径 -> 预测期末收益
        gs = np.array([sample_daily_returns(raw[s, :, CLOSE_IDX], last) for s in range(raw.shape[0])])
        g_med = np.median(gs, axis=0)
        # 去掉首日 level-shift 伪影：预测首日常被锚定到回看均值(非真实趋势)，
        # 用模型自身典型日收益替代首步，只保留"趋势形状"
        g_use = g_med.copy()
        g_use[0] = np.median(g_med[1:])
        R_pred = float(np.prod(1.0 + g_use) - 1.0)
        R_act = float(test["close"].iloc[-1] / last - 1.0)
        recs.append({"anchor": a.strftime("%Y-%m-%d"), "R_pred": R_pred, "R_act": R_act})
        if (ai + 1) % 5 == 0:
            print(f"  已处理 {ai+1}/{len(anchors)} 锚点")

    Rp = np.array([r["R_pred"] for r in recs])
    Ra = np.array([r["R_act"] for r in recs])
    # 过原点回归估计 k
    if (Rp ** 2).sum() > 1e-12:
        k = float(np.sum(Ra * Rp) / np.sum(Rp ** 2))
    else:
        k = 1.0
    k = max(0.0, min(1.5, k))
    hit = float(np.mean(np.sign(Ra) == np.sign(Rp)))
    # 方向命中率的 95% 置信区间（Wald）
    n = len(recs)
    se = (hit * (1 - hit) / n) ** 0.5
    print(f"\n[校准结果] 锚点 {n} 个")
    print(f"  收益压缩系数 k = {k:.3f}  (模型期末收益需 ×k 才接近真实量级)")
    print(f"  方向命中率 h = {hit:.1%}  (95%CI ≈ {max(0,hit-1.96*se):.1%}~{min(1,hit+1.96*se):.1%})")
    print(f"  预测收益均值 {Rp.mean():+.2%} / 真实收益均值 {Ra.mean():+.2%}")

    calib = {
        "symbol": "000425", "name": "徐工机械", "model": MODEL, "tokenizer": TOK,
        "lookback": LB, "pred_len": PL, "sample_count_calib": SC,
        "k_return_shrinkage": k, "direction_hit_rate": hit,
        "n_anchors": n, "date_range": [recs[0]["anchor"], recs[-1]["anchor"]],
        "pred_ret_mean": float(Rp.mean()), "act_ret_mean": float(Ra.mean()),
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "method": "walk-forward monthly anchors; k = slope(origin regression) of actual vs predicted 30d return; returns in sample-path space re-anchored to true last close",
    }
    with open(os.path.join(OUT, "calibration.json"), "w", encoding="utf-8") as f:
        json.dump(calib, f, ensure_ascii=False, indent=2)

    # 散点图：真实 vs 预测（过原点，斜率 k）
    import matplotlib
    matplotlib.use("Agg"); import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(Rp * 100, Ra * 100, s=18, alpha=0.6, color="#185FA5", label="各历史锚点")
    lim = max(abs(Rp).max(), abs(Ra).max()) * 100 * 1.1
    ax.plot([-lim, lim], [-lim, lim], "k--", lw=1, label="理想(无偏)")
    ax.plot([-lim, lim], [-lim * k, lim * k], color="#C0392B", lw=2, label=f"校准线 k={k:.2f}")
    ax.set_xlabel("Kronos 预测 30日收益 (%)")
    ax.set_ylabel("真实 30日收益 (%)")
    ax.set_title(f"徐工机械 校准散点 (n={n}, 方向命中率 {hit:.0%})")
    ax.legend(); ax.grid(alpha=0.3)
    png = os.path.join(OUT, "calibration_scatter.png")
    plt.savefig(png, dpi=150, bbox_inches="tight")
    print(f"  散点图: {png}")
    print("校准完成 ✅")


if __name__ == "__main__":
    main()
