"""
Kronos 跑通演示脚本（自包含，无需官方缺失的示例数据）
- 使用仓库自带的 tests/data/regression_input.csv（字段与官方示例一致：
  timestamps,open,high,low,close,volume,amount，5 分钟 K 线）
- 使用 Agg 后端保存 PNG，避免在无显示环境调用 plt.show() 卡死
- 模型权重默认从 HuggingFace 下载（中国网络建议设置 HF_ENDPOINT=https://hf-mirror.com）
"""
import os
import sys

# 确保能 import 到同级的 model 包
ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # 无显示环境：用非交互后端
import matplotlib.pyplot as plt

from model import Kronos, KronosTokenizer, KronosPredictor


def plot_prediction(kline_df, pred_df, out_path):
    pred_df.index = kline_df.index[-pred_df.shape[0]:]
    sr_close = kline_df['close']
    sr_pred_close = pred_df['close']
    sr_close.name = 'Ground Truth'
    sr_pred_close.name = "Prediction"

    sr_volume = kline_df['volume']
    sr_pred_volume = pred_df['volume']
    sr_volume.name = 'Ground Truth'
    sr_pred_volume.name = "Prediction"

    close_df = pd.concat([sr_close, sr_pred_close], axis=1)
    volume_df = pd.concat([sr_volume, sr_pred_volume], axis=1)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    ax1.plot(close_df['Ground Truth'], label='Ground Truth', color='blue', linewidth=1.5)
    ax1.plot(close_df['Prediction'], label='Prediction', color='red', linewidth=1.5)
    ax1.set_ylabel('Close Price', fontsize=14)
    ax1.legend(loc='lower left', fontsize=12)
    ax1.grid(True)

    ax2.plot(volume_df['Ground Truth'], label='Ground Truth', color='blue', linewidth=1.5)
    ax2.plot(volume_df['Prediction'], label='Prediction', color='red', linewidth=1.5)
    ax2.set_ylabel('Volume', fontsize=14)
    ax2.legend(loc='upper left', fontsize=12)
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"[OK] 走势对比图已保存: {out_path}")


def main():
    # 1. 加载分词器与模型（HuggingFace Hub）
    print("[1/4] 加载分词器与模型（首次会从 HuggingFace 下载）...")
    tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
    model = Kronos.from_pretrained("NeoQuasar/Kronos-small")

    # 2. 实例化预测器（自动选择 cpu / cuda / mps）
    predictor = KronosPredictor(model, tokenizer, max_context=512)
    print(f"      运行设备: {predictor.device}")

    # 3. 准备数据：使用仓库自带示例数据（替代官方缺失的 XSHG_5min_600977.csv）
    data_path = os.path.join(ROOT, "tests", "data", "regression_input.csv")
    df = pd.read_csv(data_path)
    df['timestamps'] = pd.to_datetime(df['timestamps'])

    lookback, pred_len = 400, 120
    x_df = df.loc[:lookback - 1, ['open', 'high', 'low', 'close', 'volume', 'amount']]
    x_timestamp = df.loc[:lookback - 1, 'timestamps']
    y_timestamp = df.loc[lookback:lookback + pred_len - 1, 'timestamps']

    # 4. 生成预测
    print("[2/4] 生成预测中...")
    pred_df = predictor.predict(
        df=x_df,
        x_timestamp=x_timestamp,
        y_timestamp=y_timestamp,
        pred_len=pred_len,
        T=1.0,
        top_p=0.9,
        sample_count=1,
        verbose=True,
    )

    print("[3/4] 预测结果（前 5 行）:")
    print(pred_df.head())

    # 5. 可视化
    print("[4/4] 绘制对比图...")
    kline_df = df.loc[:lookback + pred_len - 1]
    out_path = os.path.join(ROOT, "kronos_prediction_result.png")
    plot_prediction(kline_df, pred_df, out_path)

    # 保存预测结果 CSV 便于查看
    csv_out = os.path.join(ROOT, "kronos_prediction_result.csv")
    pred_df.to_csv(csv_out)
    print(f"[OK] 预测结果 CSV 已保存: {csv_out}")
    print("跑通完成 ✅")


if __name__ == "__main__":
    main()
