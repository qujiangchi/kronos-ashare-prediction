# -*- coding: utf-8 -*-
"""Build a clean HTML email (styled table) for the multi-stock Kronos
prediction summary and write daily_output/<today>/email_body.html.

Chart is intentionally NOT attached: the email channel is unstable for
large base64 attachments, so the summary chart is shown in-conversation
via present_files and kept locally at SRC_PNG.
"""
import os, csv
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
TODAY = datetime.now()
DATE = TODAY.strftime("%Y-%m-%d")
BASE = os.path.join(HERE, "daily_output", DATE)
CSV_PATH = os.path.join(BASE, f"00_汇总_预测表_{DATE}.csv")
SRC_PNG = os.path.join(BASE, f"00_汇总_预测走势图_{DATE}.png")
OUT_HTML = os.path.join(BASE, "email_body.html")

# read summary csv
rows = []
with open(CSV_PATH, encoding="utf-8-sig", newline="") as f:
    for r in csv.DictReader(f):
        rows.append(r)
print("rows:", len(rows))

def esc(s):
    return (s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))

def dir_color(d):
    return "#d4380d" if d == "偏强" else "#389e0d"   # 涨红跌绿

def ret_color(v):
    try:
        f = float(v.replace("%", "").replace("+", ""))
    except Exception:
        return "#000"
    return "#d4380d" if f >= 0 else "#389e0d"

def consistency_bg(c):
    return "background:#fff7e6;" if c == "背离" else ""

# any uncalibrated (fallback) row still carries a '*' in k/h
has_star = any("*" in (r.get("k", "") + r.get("h", "")) for r in rows)

# build table rows
trs = []
for r in rows:
    name = esc(r["股票"]); code = esc(r["代码"])
    price = esc(r["当前价"]); direc = esc(r["模型方向"])
    raw30 = esc(r["未校准30日"]); cal30 = esc(r["校准30日"])
    conf = esc(r["置信度"]); k = esc(r["k"]); h = esc(r["h"])
    r20 = esc(r["20日真实"]); cons = esc(r["方向/真实"])
    trs.append(f"""<tr>
<td>{name}<br><span style="color:#888;font-size:12px">{code}</span></td>
<td style="text-align:right">{price}</td>
<td style="text-align:center;color:{dir_color(direc)};font-weight:bold">{direc}</td>
<td style="text-align:right;color:{ret_color(raw30)}">{raw30}</td>
<td style="text-align:right;color:{ret_color(cal30)}">{cal30}</td>
<td style="text-align:right">{conf}</td>
<td style="text-align:right">{k}</td>
<td style="text-align:right">{h}</td>
<td style="text-align:right;color:{ret_color(r20)}">{r20}</td>
<td style="text-align:center;{consistency_bg(cons)}">{cons}</td>
</tr>""")

table = f"""<table border="1" cellspacing="0" cellpadding="6" style="border-collapse:collapse;font-family:'Microsoft YaHei',Arial,sans-serif;font-size:13px;width:100%">
<thead><tr style="background:#1f3a5f;color:#fff">
<th>股票</th><th>当前价</th><th>模型方向</th><th>未校准30日</th><th>校准30日</th>
<th>置信度</th><th>k</th><th>h</th><th>20日真实</th><th>方向/真实</th>
</tr></thead>
<tbody>{"".join(trs)}</tbody>
</table>"""

if has_star:
    rate_note = ("带 <b>*</b> 的 k/h 表示该票尚未跑专属 walk-forward 校准，使用兜底值（k=0.5 / h=0.55），不可信；"
                "其余票已有真实校准（来自各自 31 锚点历史回测）。")
else:
    rate_note = ("全部 11 只均已跑专属 walk-forward 校准，k/h 来自各自 31 锚点历史估计，<b>不再有 * 兜底</b>；"
                "但样本量很小（单票仅 2 个校准样本），命中率多数仅略高于随机，幅度仍不可信。")

html = f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
</head><body style="margin:0;padding:16px;background:#f5f6f8;color:#222;font-family:'Microsoft YaHei',Arial,sans-serif">
<div style="max-width:880px;margin:0 auto;background:#fff;border-radius:10px;padding:20px;box-shadow:0 1px 4px rgba(0,0,0,.08)">
<h2 style="margin:0 0 4px;color:#1f3a5f">Kronos 多票校准预测 · 汇总（{DATE}）</h2>
<p style="margin:0 0 14px;color:#666;font-size:13px">同花顺自选股 11 只 · 模型：Kronos-base · 预测未来 30 个交易日</p>

{table}

<p style="margin:14px 0 6px;font-size:12px;color:#a00">{rate_note}</p>

<h3 style="margin:18px 0 8px;color:#1f3a5f;font-size:15px">汇总走势图</h3>
<p style="margin:0 0 6px;font-size:13px;color:#666">📎 走势图保存在本地：<code>{SRC_PNG}</code>，并在每日对话中通过 present_files 直接展示；邮件不附大图（通道对大 base64 附件不稳定）。</p>

<h3 style="margin:18px 0 8px;color:#1f3a5f;font-size:15px">使用口径（务必先读）</h3>
<ol style="margin:0;padding-left:20px;font-size:13px;line-height:1.7">
<li><b>不单独据此开仓</b>：模型方向只配当你的其他分析（均线/动量/基本面）的<b>一致性校验</b>。</li>
<li><b>幅度不可信</b>：k 是对模型夸张幅度的压缩系数，校准后 30 日涨跌幅通常被压到接近 0，<b>不能当价格目标</b>。</li>
<li><b>方向/真实「背离」要警惕</b>：说明模型在延续旧 K 线惯性，可能错过拐点。</li>
<li><b>任何单日预测 &gt; ±10% 一律视为噪声</b>（A股涨跌停幅度）。</li>
<li>本结果<b>仅供研究参考，不构成任何投资建议</b>。是否交易、交易什么、交易多少，由你自负盈亏决定。</li>
</ol>
</div>
</body></html>"""

with open(OUT_HTML, "w", encoding="utf-8") as f:
    f.write(html)
print("html written:", OUT_HTML, os.path.getsize(OUT_HTML), "bytes")
