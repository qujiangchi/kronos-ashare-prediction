# -*- coding: utf-8 -*-
"""Build a clean HTML email (styled table + embedded downscaled chart) for the
multi-stock Kronos prediction summary, and emit email_payload.json."""
import os, csv, json, base64, hashlib, io
from PIL import Image

BASE = r"D:\Creater\Kronos\daily_output\2026-08-25"
SRC_PNG = os.path.join(BASE, "00_汇总_预测走势图_2026-08-25.png")
CSV_PATH = os.path.join(BASE, "00_汇总_预测表_2026-08-25.csv")
OUT_HTML = os.path.join(BASE, "email_body.html")
OUT_JSON = os.path.join(BASE, "email_payload.json")
EMAIL_PNG = os.path.join(BASE, "00_汇总_预测走势图_email.png")

# 1) downscale chart to <=800px wide, optimize
im = Image.open(SRC_PNG).convert("RGB")
w, h = im.size
if w > 300:
    new_w = 300
    new_h = int(h * new_w / w)
    im = im.resize((new_w, new_h), Image.LANCZOS)
im.save(EMAIL_PNG, "PNG", optimize=True)
print("chart saved:", EMAIL_PNG, os.path.getsize(EMAIL_PNG), "bytes")

with open(EMAIL_PNG, "rb") as f:
    raw = f.read()
b64 = base64.b64encode(raw).decode()
sha1 = hashlib.sha1(raw).hexdigest()
png_size = len(raw)
print("chart b64 len:", len(b64))

# 2) read summary csv
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

# 3) build table rows
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

rate_note = (
    "带 <b>*</b> 的 k/h 表示该票尚未跑专属 walk-forward 校准，当前使用兜底值（k=0.5 / h=0.55），不可信；"
    "仅 <b>徐工机械(000425)</b> 有真实校准（k=0.026 / h=58%，来自 31 锚点历史回测）。"
)

html = f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
</head><body style="margin:0;padding:16px;background:#f5f6f8;color:#222;font-family:'Microsoft YaHei',Arial,sans-serif">
<div style="max-width:880px;margin:0 auto;background:#fff;border-radius:10px;padding:20px;box-shadow:0 1px 4px rgba(0,0,0,.08)">
<h2 style="margin:0 0 4px;color:#1f3a5f">Kronos 多票校准预测 · 汇总（2026-08-25）</h2>
<p style="margin:0 0 14px;color:#666;font-size:13px">同花顺自选股 11 只 · 模型：Kronos-base · 预测未来 30 个交易日</p>

{table}

<p style="margin:14px 0 6px;font-size:12px;color:#a00">{rate_note}</p>

<h3 style="margin:18px 0 8px;color:#1f3a5f;font-size:15px">汇总走势图</h3>
<p style="margin:0 0 6px;font-size:13px;color:#666">📎 走势图已作为邮件附件附上（PNG），下载或在线预览即可查看 11 只票的预测 vs 真实叠加图。</p>

<h3 style="margin:18px 0 8px;color:#1f3a5f;font-size:15px">使用口径（务必先读）</h3>
<ol style="margin:0;padding-left:20px;font-size:13px;line-height:1.7">
<li><b>不单独据此开仓</b>：模型方向只配当你的其他分析（均线/动量/基本面）的<b>一致性校验</b>。</li>
<li><b>幅度不可信</b>：k 是对模型夸张幅度的压缩系数，校准后 30 日涨跌幅通常被压到接近 0，<b>不能当价格目标</b>。</li>
<li><b>方向/真实「背离」要警惕</b>：说明模型在延续旧 K 线惯性，可能错过拐点（如徐工、同仁堂、启明星辰等）。</li>
<li><b>任何单日预测 &gt; ±10% 一律视为噪声</b>（A股涨跌停幅度）。</li>
<li>本结果<b>仅供研究参考，不构成任何投资建议</b>。是否交易、交易什么、交易多少，由你自负盈亏决定。</li>
</ol>
</div>
</body></html>"""

with open(OUT_HTML, "w", encoding="utf-8") as f:
    f.write(html)
print("html written:", OUT_HTML, os.path.getsize(OUT_HTML), "bytes")

payload = {
    "to": [{"email": "chenao2@foxmail.com", "name": "chenao2"}],
    "subject": "【Kronos多票校准预测】2026-08-25 · 11只自选股（HTML含图）",
    "body": html,
    "body_format": "HTML",
    "skip_confirmation": True,
    "attachments": [
        {
            "filename": "00_汇总_预测走势图_2026-08-25.png",
            "content_type": "image/png",
            "content": b64,
            "size": png_size,
            "sha1": sha1,
        }
    ],
}
with open(OUT_JSON, "w", encoding="utf-8") as f:
    json.dump(payload, f, ensure_ascii=False)
print("payload written:", OUT_JSON)

# wrapped base64 (1900 chars/line) so it can be read back without truncation
WRAP = os.path.join(BASE, "chart_b64_wrapped.txt")
with open(WRAP, "w", encoding="utf-8") as f:
    for i in range(0, len(b64), 1900):
        f.write(b64[i:i+1900] + "\n")
print("wrapped b64 written:", WRAP)
print("attachment b64 len:", len(b64), "sha1:", sha1, "size:", png_size)
