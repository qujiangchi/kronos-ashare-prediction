# Kronos A股/ETF 走势预测实战

> 基于清华开源金融 K 线基础模型 **Kronos** 的本地实战笔记与脚本集合。
> ⚠️ 仅供研究学习，Kronos 为研究模型，**不构成任何投资建议**。

## 1. 项目来源

本仓库基于清华开源的金融 K 线基础模型 **Kronos**（上游：[shiyu-coder/Kronos](https://github.com/shiyu-coder/Kronos)，AAAI 2026，arXiv:2508.02739）。

Kronos 的核心思路是把 K 线当作「语言」来建模：

1. **Tokenizer**：把连续的 OHLCV（开/高/低/收/量/额）量化成分层离散 token；
2. **Transformer 解码器**：在 token 序列上做自回归预训练，学会「给定历史 K 线，续写未来 K 线」。

模型家族（已开源）：`mini`(4.1M) / `small`(24.7M) / `base`(102.3M) / `large`(499M，未开源)。
**small / base 最大上下文 512 根 K 线，预测上限约 120 根。**

本仓库在官方代码基础上，**新增了针对 A 股 / ETF 的实战脚本与踩坑记录**，用于本地复现
「拉真实行情 → Kronos 预测 → 回测证伪」的完整链路。

## 2. 新增脚本一览

| 脚本 | 说明 |
|---|---|
| `run_kronos_demo.py` | 最基础 demo：用仓库自带数据跑通 small 模型 |
| `predict_xugong.py` | 徐工机械(000425.SZ) 实测：akshare 新浪源取日线 → small 预测 100 日 |
| `predict_513160.py` | 港股科技ETF银华(513160) 实测：腾讯接口取日线 → small 预测 |
| `predict_xugong_v2.py` | 徐工重预测与偏差诊断：small / base × 多回看 / 预测窗口对照 |
| `predict_xugong_backtest.py` | walk-forward 回测 + 系统性看空对照（中国平安），验证方向能力 |
| `predict_xugong_daily.py` | **每日定时脚本**：取最新行情 → Kronos-base 预测未来 30 日 → 出图 + CSV |

所有脚本均使用**相对路径**（`os.path.dirname(os.path.abspath(__file__))`），克隆到任意目录均可运行。

## 3. 环境搭建（国内必看）

```bash
# 1. 用 Python 3.13 建 venv（不要升级 pip，见第 5 节坑 1）
python -m venv venv

# 2. 装依赖（torch 用 CPU 版省空间；如需 GPU 去掉 --index-url 并装对应 CUDA 版）
venv/Scripts/python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cpu
venv/Scripts/python.exe -m pip install numpy pandas einops huggingface_hub matplotlib tqdm safetensors akshare requests

# 3. 国内必须设置 HF 镜像，否则 from_pretrained 会超时
export HF_ENDPOINT=https://hf-mirror.com
```

## 4. 运行示例

```bash
cd <仓库目录>
export HF_ENDPOINT=https://hf-mirror.com

# 每日预测（徐工机械，未来 30 日）
venv/Scripts/python.exe predict_xugong_daily.py

# 单只回测证伪（徐工 + 中国平安对照）
venv/Scripts/python.exe predict_xugong_backtest.py
```

输出目录（已加入 `.gitignore`，不入库，可随时重跑生成）：
`daily_output/`、`xugong_output/`、`513160_output/`、`xugong_v2_output/`。

## 5. 数据源（踩坑重点）

| 标的类型 | 可用接口 | 备注 |
|---|---|---|
| A 股普通股票（如 000425） | `akshare.stock_zh_a_daily`（新浪源） | 本机可用 |
| ETF / 基金（如 513160） | 腾讯财经 `web.ifzq.gtimg.cn/appstock/app/fqkline/get` | 新浪源不支持 ETF；东方财富源被远端重置 |

腾讯接口要点：日期必须带连字符（`2024-01-01`），`count ≤ 1024`，返回行只有 6 列（无成交额，`amount` 填 0）。

模型权重：`NeoQuasar/Kronos-Tokenizer-base` + `NeoQuasar/Kronos-small`（或 `-base`），
经 HF 镜像下载后缓存到 `~/.cache/huggingface`。

## 6. 三个关键坑

1. **venv 内不要 `pip install --upgrade pip`**：本机环境会触发「安全删除失败」导致安装中断。
   全新 venv 直接装依赖即可（pip 旧一点没关系）。
2. **HF 直连超时**：国内必须 `HF_ENDPOINT=https://hf-mirror.com`。
3. **官方示例数据缺失**：`examples/prediction_example.py` 引用的 `./data/XSHG_5min_600977.csv`
   不存在，可改用仓库自带 `tests/data/regression_input.csv`。

## 7. 诚实结论：Kronos 能信到什么程度

> 以下为实测结论，不构成投资建议。

- **Kronos 是一个「趋势外推器」**：它顺着最近的价格动量往外画，**没有均值回归能力**。
- 对**趋势延续**的标的，方向大致对；对处于**拐点 / 反转**的标的（如超跌后企稳的徐工），方向会**反**：
  - 回测实测：徐工用 2024 数据 → 预测 2025H1，真实 **+0.9%**，模型预测 **-7.1%**（方向反）；
  - 对照：中国平安近期真涨，模型预测也涨（方向对）。
- **点位严重失真**：即便方向对，幅度也常被放大（平安真实 +8%，模型 +22%）；
  small 模型还会出现首日 -20% 以上的 level-shift 跳空。
- **正确用法**：只作「近期动量是否延续」的弱参考；任何方向结论都必须先 walk-forward 回测证伪，
  且不超过 ±10% 的常识 sanity check。
- **提升方向**：换 Kronos-base（容量更大，首日跳空从 -24% 降到 -2%）、缩短 horizon、采样置信带、
  或在你的标的池上做域适应微调（`finetune/`）。

## 8. 每日定时预测

`predict_xugong_daily.py` 设计为被定时任务调用：取最新行情 → Kronos-base 预测未来 30 个交易日 →
在 `daily_output/YYYY-MM-DD/` 生成走势图与 CSV。注意结果仅作方向参考。

## 9. 许可证

原 Kronos 代码与许可证见 `LICENSE` / `README.md`（上游：[shiyu-coder/Kronos](https://github.com/shiyu-coder/Kronos)）。
本仓库新增脚本按相同许可证发布，仅供研究学习。
