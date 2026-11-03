# -*- coding: utf-8 -*-
"""
生成第三阶段技术方案 PDF（中文）· v2 美化版
设计系统：
  - 封面：顶部品牌色带 + 大标题 + 关键信息框 + 底部页脚
  - 目录页（两遍生成法，页码精确）
  - 章节：左侧色条 + 大标题 + 分隔线；小节带小色块
  - 正文：宋体 10.5pt + 首行缩进两字符；要点框 / 代码块（标签+底纹）/ 成绩链路流程图
  - 表格：表头深蓝反白 + 隔行浅色 + 网格线 + 高亮行
  - 页眉页脚：文档名 + 当前章节 + 页码/总页数
依赖：fpdf2（>=2.7.9，dry_run/LINES 特性）；字体：Windows simsun.ttc / simhei.ttf。
运行：python gen_tech_pdf.py  ->  产出 技术方案.pdf（与本脚本同级）
"""
from fpdf import FPDF
from fpdf.enums import XPos, YPos
import os

# ---------------- 设计常量 ----------------
PRIMARY      = (31, 78, 121)     # 深蓝 #1F4E79
PRIMARY_L    = (46, 117, 182)    # 亮蓝 #2E75B6
TEXT_DARK    = (38, 38, 38)      # 正文深灰
TEXT_MUTED   = (107, 114, 128)   # 弱化灰
RULE_LIGHT   = (216, 222, 231)   # 分隔线浅蓝灰
ROW_ALT      = (247, 249, 252)   # 表格隔行
HILITE_FILL  = (255, 247, 214)   # 高亮行 浅金
HILITE_TEXT  = (178, 0, 0)       # 高亮行 深红（中式：红=优）
CODE_BG      = (246, 248, 251)
CODE_BORDER  = (222, 228, 235)
CALL_BG      = (234, 243, 252)   # 要点框底
CALL_BORDER  = (206, 220, 238)
WARN_BG      = (253, 241, 240)   # 合规声明（红系）
WARN_BORDER  = (240, 214, 212)
WARN_TEXT    = (166, 0, 0)

F_HEI  = r"C:/Windows/Fonts/simhei.ttf"    # 黑体：标题
F_SONG = r"C:/Windows/Fonts/simsun.ttc"    # 宋体：正文

PAGE_W, M_L, M_R, M_T, M_B = 210.0, 18.0, 18.0, 22.0, 18.0
CONTENT_W = PAGE_W - M_L - M_R            # 174mm


def space(pdf, h):
    """剩余空间不足则换页（页底安全线 16mm）。"""
    if pdf.get_y() + h > pdf.h - 16:
        pdf.add_page()


class TechPDF(FPDF):
    def __init__(self):
        super().__init__(format="A4")
        self.set_margins(M_L, M_T, M_B)
        self.set_auto_page_break(True, 16)
        self.add_font("Hei", "", F_HEI)
        self.add_font("Song", "", F_SONG)
        self.current_chapter = ""
        self.set_title("无人机低空航拍图像语义分割 · 技术方案报告")
        self.set_author("流水的天才")
        self.set_subject(" 无人机低空航拍图像语义分割 · 第三阶段技术方案")
        self.alias_nb_pages()

    # ---------- 页眉 / 页脚 ----------
    def header(self):
        if self.page_no() == 1:
            return
        self.set_font("Hei", "", 7.5)
        self.set_text_color(*TEXT_MUTED)
        self.set_xy(M_L, 8)
        self.cell(0, 4.5, "无人机低空航拍图像语义分割 · 技术方案", align="L")
        if self.current_chapter:
            self.set_xy(M_L, 8)
            self.cell(0, 4.5, self.current_chapter, align="R")
        self.set_draw_color(*RULE_LIGHT)
        self.set_line_width(0.3)
        self.line(M_L, 14.6, PAGE_W - M_R, 14.6)

    def footer(self):
        if self.page_no() == 1:
            return
        self.set_y(-13)
        self.set_font("Hei", "", 8)
        self.set_text_color(*TEXT_MUTED)
        self.cell(0, 8, f"第 {self.page_no()} 页 · 共 {{nb}} 页", align="C")


# ================= 版式组件 =================

def chapter(pdf, num, title):
    """章节标题（调用方需已 add_page）。"""
    pdf.current_chapter = title
    y = pdf.get_y()
    pdf.set_fill_color(*PRIMARY)
    pdf.rect(M_L, y, 1.8, 9.5, style="F")
    pdf.set_xy(M_L + 5, y)
    pdf.set_font("Hei", "", 16)
    pdf.set_text_color(20, 32, 54)
    pdf.cell(0, 10, f"{num}  {title}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(1.5)
    pdf.set_draw_color(*RULE_LIGHT)
    pdf.set_line_width(0.35)
    pdf.line(M_L, pdf.get_y(), PAGE_W - M_R, pdf.get_y())
    pdf.ln(5)


def h2(pdf, text):
    pdf.ln(3)
    y = pdf.get_y()
    pdf.set_fill_color(*PRIMARY_L)
    pdf.rect(M_L, y + 0.9, 1.1, 4.6, style="F")
    pdf.set_xy(M_L + 4, y)
    pdf.set_font("Hei", "", 12.5)
    pdf.set_text_color(*PRIMARY)
    pdf.cell(0, 6.2, text, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(2.5)


def body(pdf, text, indent=True):
    pdf.set_font("Song", "", 10.5)
    pdf.set_text_color(*TEXT_DARK)
    if indent:
        text = "\u3000\u3000" + text
    pdf.set_x(M_L)
    pdf.multi_cell(0, 6.2, text, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(1.8)


def bullet(pdf, text):
    pdf.set_font("Song", "", 10.5)
    pdf.set_text_color(*TEXT_DARK)
    y = pdf.get_y()
    pdf.set_fill_color(*PRIMARY)
    pdf.ellipse(M_L + 1.1, y + 2.2, 1.7, 1.7, style="F")
    pdf.set_xy(M_L + 5, y)
    pdf.multi_cell(CONTENT_W - 5, 5.9, text, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(1.2)


def num_item(pdf, idx, text):
    """带编号圆标的条目（创新点）。"""
    pdf.set_font("Song", "", 10.5)
    pdf.set_text_color(*TEXT_DARK)
    y = pdf.get_y()
    cx, cy, d = M_L + 3.1, y + 3.0, 6.2
    pdf.set_fill_color(*PRIMARY)
    pdf.ellipse(cx - d / 2, cy - d / 2, d, d, style="F")
    pdf.set_font("Hei", "", 7.5)
    pdf.set_text_color(255, 255, 255)
    pdf.set_xy(M_L + 0.6, cy - 1.9)
    pdf.cell(d, 4, f"{idx:02d}", align="C")
    pdf.set_font("Song", "", 10.5)
    pdf.set_text_color(*TEXT_DARK)
    pdf.set_xy(M_L + 8.5, y)
    pdf.multi_cell(CONTENT_W - 8.5, 5.9, text, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(1.4)


def callout(pdf, title, text, warn=False):
    """要点框：浅色底 + 左侧色条 + 小标题。"""
    pdf.ln(2.5)
    bar, pad = 1.6, 4.0
    bg = WARN_BG if warn else CALL_BG
    bd = WARN_BORDER if warn else CALL_BORDER
    tcol = WARN_TEXT if warn else PRIMARY
    pdf.set_font("Song", "", 9.5)
    avail = CONTENT_W - pad * 2 - bar - 2
    lines = pdf.multi_cell(avail, 5.8, text, dry_run=True, output="LINES")
    h = pad + 6.0 + 1.2 + len(lines) * 5.8 + pad
    space(pdf, h)
    y0 = pdf.get_y()
    pdf.set_fill_color(*bg)
    pdf.set_draw_color(*bd)
    pdf.set_line_width(0.3)
    pdf.rect(M_L, y0, CONTENT_W, h, style="DF", round_corners=True, corner_radius=2.0)
    pdf.set_fill_color(*tcol)
    pdf.rect(M_L, y0, bar, h, style="F")
    pdf.set_font("Hei", "", 9.5)
    pdf.set_text_color(*tcol)
    pdf.set_xy(M_L + bar + pad, y0 + pad - 1.5)
    pdf.cell(CONTENT_W - pad * 2 - bar, 5.5, title)
    pdf.set_font("Song", "", 9.5)
    pdf.set_text_color(*TEXT_DARK)
    pdf.set_xy(M_L + bar + pad, y0 + pad + 5.0)
    pdf.multi_cell(avail, 5.8, text, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_y(y0 + h + 2.5)


def code_block(pdf, label, lines):
    pdf.ln(2)
    n = len(lines)
    lh, pad, chip_h = 5.0, 3.2, 6.5
    block_h = pad + chip_h + n * lh + pad
    space(pdf, block_h)
    y0 = pdf.get_y()
    x0, W = M_L, CONTENT_W
    pdf.set_fill_color(*CODE_BG)
    pdf.set_draw_color(*CODE_BORDER)
    pdf.set_line_width(0.3)
    pdf.rect(x0, y0, W, block_h, style="DF", round_corners=True, corner_radius=2.0)
    pdf.set_fill_color(*PRIMARY)
    pdf.rect(x0 + 0.6, y0 + 0.6, 36, chip_h - 1.2, style="F",
             round_corners=True, corner_radius=1.4)
    pdf.set_font("Hei", "", 8)
    pdf.set_text_color(255, 255, 255)
    pdf.set_xy(x0 + 0.6, y0 + 0.9)
    pdf.cell(36, 4.6, label, align="C")
    pdf.set_font("Courier", "", 8.5)
    pdf.set_text_color(31, 41, 55)
    ty = y0 + pad + chip_h
    for ln in lines:
        pdf.set_xy(x0 + 6, ty)
        pdf.cell(W - 12, lh, ln)
        ty += lh
    pdf.set_y(ty + 2)


def score_flow(pdf, stages):
    """成绩链路流程图：圆角框 + 箭头（节点数自适应宽度）。"""
    n = len(stages)
    bh = 16
    gap = 4.8 if n >= 6 else 6.0
    bw = (CONTENT_W - (n - 1) * gap) / n
    y0 = pdf.get_y() + 4
    space(pdf, bh + 12)
    x, cy = M_L, y0 + bh / 2
    for i, (name, score) in enumerate(stages):
        last = i == n - 1
        if last:
            fill, draw, tcol = HILITE_FILL, (214, 164, 0), HILITE_TEXT
        else:
            fill, draw, tcol = (238, 244, 252), (184, 204, 232), PRIMARY
        pdf.set_fill_color(*fill)
        pdf.set_draw_color(*draw)
        pdf.set_line_width(0.3)
        pdf.rect(x, y0, bw, bh, style="DF", round_corners=True, corner_radius=1.8)
        pdf.set_font("Hei", "", 8)
        pdf.set_text_color(*TEXT_MUTED)
        pdf.set_xy(x + 1, y0 + 2.2)
        pdf.cell(bw - 2, 5, name, align="C")
        pdf.set_font("Hei", "", 12.5)
        pdf.set_text_color(*tcol)
        pdf.set_xy(x + 1, y0 + 7.6)
        pdf.cell(bw - 2, 7, score, align="C")
        if not last:
            pdf.set_draw_color(120, 140, 165)
            pdf.set_line_width(0.5)
            pdf.line(x + bw + 0.6, cy, x + bw + gap - 1.4, cy)
            hx = x + bw + gap - 0.2
            pdf.set_fill_color(120, 140, 165)
            pdf.polygon([(hx - 2.4, cy - 1.5), (hx, cy), (hx - 2.4, cy + 1.5)], style="F")
        x += bw + gap
    pdf.set_y(y0 + bh + 2.5)
    pdf.set_font("Hei", "", 8.5)
    pdf.set_text_color(*TEXT_MUTED)
    pdf.set_x(M_L)
    pdf.cell(0, 5, "图 2-1  成绩链路（队内线，外部指标）", align="C",
             new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(4)


def data_table(pdf, headers, widths, rows, aligns=None, highlight_rows=()):
    W = CONTENT_W
    header_h, lh = 7.5, 6.6
    nrows = len(rows)
    block_h = header_h + nrows * lh
    space(pdf, block_h + 2)
    y0 = pdf.get_y()
    x0 = M_L
    # 表头
    pdf.set_fill_color(*PRIMARY)
    pdf.set_font("Hei", "", 9.5)
    pdf.set_text_color(255, 255, 255)
    x = x0
    for hw, ht in zip(widths, headers):
        pdf.set_xy(x, y0)
        pdf.cell(hw, header_h, ht, align="C", fill=True)
        x += hw
    # 数据行
    y = y0 + header_h
    for i, row in enumerate(rows):
        hl = i in highlight_rows
        fill = HILITE_FILL if hl else (ROW_ALT if i % 2 == 1 else (255, 255, 255))
        pdf.set_fill_color(*fill)
        pdf.set_text_color(*(HILITE_TEXT if hl else TEXT_DARK))
        pdf.set_font("Hei" if hl else "Song", "", 9.5)
        x = x0
        for j, (cw, val) in enumerate(zip(widths, row)):
            al = aligns[j] if aligns else "L"
            pdf.set_xy(x, y)
            pdf.cell(cw, lh, val, align=al, fill=True)
            x += cw
        y += lh
    # 网格线 + 外框
    pdf.set_draw_color(*RULE_LIGHT)
    pdf.set_line_width(0.2)
    for k in range(nrows + 1):
        yy = y0 + header_h + k * lh
        pdf.line(x0, yy, x0 + W, yy)
    xx = x0
    for ww in widths[:-1]:
        xx += ww
        pdf.line(xx, y0 + header_h, xx, y0 + header_h + nrows * lh)
    pdf.rect(x0, y0, W, block_h, style="D")
    pdf.set_y(y + 3)


# ================= 封面 / 目录 =================

def draw_cover(pdf):
    pdf.set_fill_color(*PRIMARY)
    pdf.rect(0, 0, PAGE_W, 4.2, style="F")
    pdf.set_fill_color(*PRIMARY_L)
    pdf.rect(0, 4.2, PAGE_W, 1.4, style="F")

    pdf.set_font("Hei", "", 10.5)
    pdf.set_text_color(*TEXT_MUTED)
    pdf.set_xy(M_L, 62)
    pdf.cell(0, 7, "全球校园 AI 算法精英大赛（赛马制）", align="C")

    pdf.set_font("Hei", "", 27)
    pdf.set_text_color(20, 32, 54)
    pdf.set_xy(M_L, 80)
    pdf.cell(0, 14, "无人机低空航拍图像语义分割", align="C")

    pdf.set_font("Hei", "", 18)
    pdf.set_text_color(*PRIMARY)
    pdf.set_xy(M_L, 99)
    pdf.cell(0, 10, "技  术  方  案  报  告", align="C")

    # 装饰分隔线 + 菱形
    cx, cyy = PAGE_W / 2, 117
    pdf.set_draw_color(164, 190, 216)
    pdf.set_line_width(0.5)
    pdf.line(cx - 24, cyy, cx - 2.4, cyy)
    pdf.line(cx + 2.4, cyy, cx + 24, cyy)
    pdf.set_fill_color(*PRIMARY)
    pdf.polygon([(cx, cyy - 1.9), (cx + 1.9, cyy), (cx, cyy + 1.9), (cx - 1.9, cyy)], style="F")

    # 关键信息框
    bx, by, bw, bh = PAGE_W / 2 - 75, 134, 150, 42
    pdf.set_fill_color(240, 245, 251)
    pdf.set_draw_color(206, 220, 238)
    pdf.set_line_width(0.3)
    pdf.rect(bx, by, bw, bh, style="DF", round_corners=True, corner_radius=2.5)
    pdf.set_font("Hei", "", 10.5)
    pdf.set_text_color(*TEXT_DARK)
    lines = [
        "赛制：赛马制 · 第三阶段为唯一计分阶段（test1 + test2 + test3 共 3594 张）",
        "最终成绩：69.904990（第二阶段 test2 · 单模型 · 合规）",
        "方案：Swin-B + DFR + KAN 解码器 · 1024 全图视野 · DACS 在线自训练",
        "团队：流水的天才",
    ]
    ty = by + 7
    for ln in lines:
        pdf.set_xy(bx + 8, ty)
        pdf.cell(bw - 16, 6.8, ln, align="C")
        ty += 7.6

    pdf.set_draw_color(*RULE_LIGHT)
    pdf.set_line_width(0.3)
    pdf.line(M_L, 228, PAGE_W - M_R, 228)
    pdf.set_font("Hei", "", 9.5)
    pdf.set_text_color(*TEXT_MUTED)
    pdf.set_xy(M_L, 234)
    pdf.cell(0, 6, "全图视野域适应 · 无外部数据 · 端到端可复现", align="C")
    pdf.set_font("Hei", "", 10)
    pdf.set_xy(M_L, 247)
    pdf.cell(0, 6, "2026 年 9 月", align="C")


def draw_toc(pdf, pages):
    y = pdf.get_y()
    pdf.set_fill_color(*PRIMARY)
    pdf.rect(M_L, y, 1.8, 9.5, style="F")
    pdf.set_xy(M_L + 5, y)
    pdf.set_font("Hei", "", 16)
    pdf.set_text_color(20, 32, 54)
    pdf.cell(0, 10, "目  录", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(2)
    pdf.set_draw_color(*RULE_LIGHT)
    pdf.set_line_width(0.35)
    pdf.line(M_L, pdf.get_y(), PAGE_W - M_R, pdf.get_y())
    pdf.ln(8)

    entries = [
        "1  任务与背景",
        "2  整体方案架构",
        "3  核心方法",
        "4  成绩链路与消融",
        "5  创新点总结",
        "6  可复现性说明",
    ]
    for i, ent in enumerate(entries, start=1):
        e = pdf.get_y()
        pdf.set_font("Hei", "", 11)
        pdf.set_text_color(*TEXT_DARK)
        pdf.set_xy(M_L, e)
        pdf.cell(0, 7, ent, align="L")
        pg = pages.get(i) if pages else None
        num = str(pg) if pg else "—"
        pdf.set_draw_color(150, 160, 175)
        pdf.set_line_width(0.2)
        pdf.dashed_line(M_L + 46, e + 3.8, PAGE_W - M_R - 12, e + 3.8,
                        dash_length=0.7, space_length=0.8)
        pdf.set_font("Hei", "", 11)
        pdf.set_text_color(*PRIMARY)
        pdf.set_xy(PAGE_W - M_R - 11, e)
        pdf.cell(11, 7, num, align="R")
        pdf.ln(13)


# ================= 内容 =================

CHAPTERS = [
    ("1", "任务与背景"),
    ("2", "整体方案架构"),
    ("3", "核心方法"),
    ("4", "成绩链路与消融"),
    ("5", "创新点总结"),
    ("6", "可复现性说明"),
]


def build(pages=None):
    pdf = TechPDF()
    # ---- 封面 ----
    pdf.add_page()
    draw_cover(pdf)
    # ---- 目录 ----
    pdf.current_chapter = "目录"
    pdf.add_page()
    draw_toc(pdf, pages)
    # ---- 章节 ----
    starts = {}
    for num, title in CHAPTERS:
        pdf.current_chapter = title
        pdf.add_page()
        starts[int(num)] = pdf.page_no()
        chapter(pdf, num, title)

        if num == "1":
            body(pdf, "本任务为 9 类无人机低空航拍图像语义分割（0=Ignore 不计 mIoU）："
                      "1=背景 2=建筑 3=道路 4=水体 5=荒地 6=植被 7=农田 8=车辆。"
                      "图像统一 1024×1024，训练集 6996 张，test1/test2/test3 为不同难度的目标域。")
            body(pdf, "核心难点——域偏移与亮度捷径：训练集暗图占比 40.8%，而 test1 暗图占 48.6%，"
                      "且测试集农田/水体占比显著高于训练集。模型在训练中习得“暗即植被/背景”的捷径，"
                      "导致暗色农田、荒地整片塌缩为背景。这是所有失败样本的共同根源。")
            body(pdf, "约束：禁外部数据、禁模型集成、允许 TTA、仅可使用学术公开预训练权重。")
            callout(pdf, "赛程与计分",
                    "赛马制：第一阶段 / 第二阶段不计分，第三阶段（test1 / test2 / test3 共 3594 张）"
                    "为唯一计分阶段，因此一切投入都服务于第三阶段的域适应表现。")

        elif num == "2":
            body(pdf, "方案由“强基础模型 + 域适应自训练 + 全图视野”三段构成：")
            bullet(pdf, "基础模型（DeepKANSeg-Swin-B）：N线配方（噪声剔除 + 暗化增广）后"
                        "本地 val mIoU 0.7630，较旧冠军 +0.95。")
            bullet(pdf, "域适应（DACS 在线自训练 + 锚定伪标签回填）：不引入外部标签，仅靠学生自身 EMA 教师"
                        "与跨域混合，把外部指标从 65.3 推到 68.39。")
            bullet(pdf, "全图视野与物理 batch：裁剪视野由 640 提到 1024 全图（配尺度增广），"
                        "并把真实 batch 由 2 提到 8，在目标域上再取 +1.51。")
            body(pdf, "完整成绩链路如下（外部指标）：", indent=False)
            score_flow(pdf, [("基础模型", "62.40"), ("PL+蒸馏", "65.29"),
                             ("DACS", "67.80"), ("N线修复", "68.39"),
                             ("1024 全图", "69.49"), ("batch8 ★", "69.90")])
            body(pdf, "三段缺一不可：仅基础模型≈62.4 分；缺少强锚则域适应没有稳定落点；"
                      "而缺少全图视野时，模型看不到场景级拓扑（连片农田 / 水系 / 路网），"
                      "这正是 1024 迁移成为最大单项增益的原因。")

        elif num == "3":
            h2(pdf, "3.1 基础模型：DeepKANSeg")
            body(pdf, "Swin-B 编码器 + DFR（动态特征精炼）模块 + KAN 解码器，共 98.3M 参数。"
                      "损失为边界加权 CE（形态学边界带 ×3）+ Dice + 0.5×Lovász-Softmax（半分辨率省显存），"
                      "EMA 0.999，稀有类定向裁剪与采样 boost。推理使用 ms2 八路 TTA（尺度×翻转概率平均）。")

            h2(pdf, "3.2 DACS 在线自训练")
            body(pdf, "DACS（Domain Adaptation via Cross-domain Mixed Sampling）是半监督域适应方法："
                      "用学生模型的 EMA 作为“教师”给无标签目标域打伪标签（置信度阈值 0.85 过滤），"
                      "再用 ClassMix 把目标域伪标签随机选取一半类别的像素粘贴到源域图像上，"
                      "逼学生同时学习源域真值与跨域混合上下文。")
            callout(pdf, "关键洞察",
                    "与静态蒸馏的本质区别：教师是学生自身的 EMA，会随训练自我改进，"
                    "因此能突破“静态蒸馏上限 = 教师分”的瓶颈——这正是反超队友 67.33 的关键。")
            code_block(pdf, "伪代码 · DACS 单步", [
                "# DACS single-step pseudocode",
                "teacher = EMA(student)                 # sync every step",
                "for x_src, y_src in source_loader:",
                "    x_tgt = sample_target_crop()       # random target img + crop",
                "    with no_grad():",
                "        pl, conf = teacher(x_tgt)      # pseudo-label + confidence",
                "        pl = where(conf < tau, IGNORE, pl)",
                "        pl = where(low_conf & team, team_label, pl)  # low-conf refill",
                "    x_mix, y_mix = classmix(x_src, y_src, x_tgt, pl)  # paste half classes",
                "    loss = seg_loss(student(x_mix), y_mix)",
                "    loss.backward(); update(student); EMA_update(teacher)",
            ])

            h2(pdf, "3.3 锚定伪标签回填")
            body(pdf, "DACS 的教师只用学生自身 EMA，在源域偏强的模型上容易把目标域像素漂回源分布。"
                      "为此引入“锚”：用一个更强的外部预测作参照，回填教师低置信的像素。")
            bullet(pdf, "早期（飞轮阶段）：锚逐轮变强（队友 67.33 → 自身 67.80 → 68.11 → 68.39），"
                        "模型随之爬升。")
            bullet(pdf, "★ 飞轮的边界：到 R1 之后自我迭代过峰（R2 一致率 0.7949 < R1 的 0.8002），"
                        "因此后期**不再迭代自我锚**，改用固定的最强外部锚，避免循环偏差累积。")
            callout(pdf, "为何有效",
                    "锚的作用是“把模型钉在目标分布上”：去掉锚，模型会漂回源分布；"
                    "但锚若来自自身迭代会积累循环偏差——这正是飞轮必须封顶的原因。")
            code_block(pdf, "伪代码 · 锚定回填（DACS 单步内）", [
                "# anchored refill inside DACS",
                "teacher = EMA(student)",
                "pl, conf = teacher(x_tgt)",
                "pl = where(conf < tau, IGNORE, pl)        # low-conf -> ignore",
                "pl = where(conf < tau & anchor_valid,     # then refill by anchor",
                "           anchor_label, pl)",
                "x_mix, y_mix = classmix(x_src, y_src, x_tgt, pl)",
                "loss = seg_loss(student(x_mix), y_mix)",
            ])

            h2(pdf, "3.4 全图视野与物理 batch")
            body(pdf, "两项后期关键增益来自同一个方向——让学生看到更多信息：")
            bullet(pdf, "全图视野：crop 由 640 提到 1024 并开启尺度增广（窗口恒 resize 到 c²，"
                        "增强语义等价于放大倍率 c/s），视野覆盖率由 47~94% 提升到 75~100%，"
                        "且与 crop640 保持同分布的尺度增强，构成干净的视野单变量实验（+0.5230）。")
            bullet(pdf, "物理 batch：真实 batch 由 2 提到 8，不用梯度累积冒充。"
                        "全图下单张场景组成差异大，更大 batch 使每次更新看到更多完整场景，"
                        "梯度中的类别与域构成更稳定；batch 8→16 出现反转，峰值在 8（+0.2703）。")
            callout(pdf, "机理自洽",
                    "batch8 之后模型自发减少了背景误判——未做任何类别权重干预，"
                    "Background 占比下降 1.61pp、七个前景类齐升，"
                    "印证“更大 batch → 更全局 → 假背景减少”。")
            body(pdf, "工程注意：batch 8 峰值显存约 18.1GB，需 ≥24GB 显卡；"
                      "每 1000 步留轻量快照，峰值常在中间点而非终点。")

        elif num == "4":
            body(pdf, "实测外部指标演进（队内线）：")
            data_table(pdf,
                       ["阶段", "外部指标", "增量", "说明"],
                       [50, 26, 22, 76],
                       [
                           ["基础模型 KAN-16k", "62.3987", "—", "复现起点（公开 ImageNet Swin-B）"],
                           ["PL 自蒸馏", "64.6141", "+2.21", "伪标签自蒸馏"],
                           ["队友蒸馏", "65.2914", "+0.68", "静态蒸馏上限 = 教师分"],
                           ["DACS（队友锚）", "67.8000", "+2.51", "反超队友 67.33"],
                           ["自我锚定", "68.1069", "+0.31", "飞轮阶段"],
                           ["N线-DACS R1", "68.3946", "+0.29", "根因修复（噪声剔除 + 暗化增广）"],
                           ["cw7 @1024+scale-up", "69.4939", "+1.10", "★ 全图视野（最大单项增益）"],
                           ["relaxbg", "69.6674", "+0.17", "背景类权重低谷点"],
                           ["batch8（第二阶段输出）", "69.904990", "+0.2703", "★ 物理 batch 2→8 · test2 实测"],
                       ],
                       aligns=["L", "C", "C", "L"],
                       highlight_rows={8})
            body(pdf, "已证伪方向（节省后续投入）：")
            data_table(pdf,
                       ["已证伪方向", "结论 / 代价"],
                       [76, 98],
                       [
                           ["旋转 TTA ms2r", "-3.6 分，废弃"],
                           ["纯自训练（无锚）", "漂回源分布，指标不升"],
                           ["自我重标清洗标签", "-0.09，循环偏差"],
                           ["Swin-T@1024", "容量不足（val 0.7310）"],
                           ["DINOv3 冻结特征", "粒度不够（val 0.6730）"],
                           ["N线-自我锚 R2", "一致率 0.7949 < R1 的 0.8002，飞轮过峰"],
                           ["batch16", "68.59 < batch8 的 69.41，过峰反转"],
                           ["目标域 BN 统计校准", "b8 上一致率 96.01 < 96.07，无正贡献"],
                           ["更激进 TTA（ms2b/ms2c）", "扰动越大越低分，仅最克制档为正"],
                       ])
            callout(pdf, "结论",
                    "后期最大增量来自“改变学生看到的信息”——1024 全图（+1.10）与物理 batch 8（+0.27）"
                    "合计 +1.37，量级远超损失函数与类权重的微调；"
                    "前期根因修复的价值在于把增益从基础模型一路贯穿到域适应。")

        elif num == "5":
            num_item(pdf, 1, "主线洞察“改变学生看到的信息”：把裁剪视野由 640 提到 1024 全图并配合尺度增广，"
                             "视野覆盖率由 47~94% 提升到 75~100%，取得最大单项增益 +1.10。")
            num_item(pdf, 2, "揭示“物理 batch 是被硬件隐式锁死的旋钮”：6GB 笔记本长期把 batch 钉在 2，"
                             "换到 32GB 卡提到 8 即得 +0.2703；batch16 反转，峰值在 8。")
            num_item(pdf, 3, "定位并修复“亮度捷径 → 暗色农田塌缩”这一根本瓶颈：噪声剔除 + 暗化增广"
                             "让基础模型 val +0.95，增益贯穿 PL / 蒸馏 / DACS 全链"
                             "（根因级修复 > 后处理拼救）。")
            num_item(pdf, 4, "建立可复现的域适应 SOP 与快照选拔判据体系；并提出“以强参照包做逐像素诊断”"
                             "的提分定位方法 —— 用一致率与净偏差，定位自己相对强参照的偏移方向与过冲量。")

        elif num == "6":
            body(pdf, "本方案全部代码、权重、噪声剔除样本清单、Dockerfile 与复现脚本已封装于复现材料中：")
            bullet(pdf, "环境：torch 2.6.0+cu124、smp 0.5.0、timm 1.0.28（Docker 镜像已固化）。")
            bullet(pdf, "快速验证路径（约 5 分钟 / 100 张）：加载最终权重 "
                        "ckpt_kan_dacs_cw7_1024su_relaxbg_b8_i4000.pth 跑 ms2x 推理，"
                        "与输出包逐像素比对 —— 已实测像素一致率 99.9948%（差异属 GPU 浮点级）。")
            bullet(pdf, "完整路径（两阶段）：scripts/00_reproduce_all.sh 从公开 ImageNet Swin-B 一键复现——"
                        "阶段A（基础模型 16k → 伪标签 → PL 微调 → 蒸馏，约 12h 量级）"
                        "+ 阶段B（DACS 训练 + 推理，约 4.5h）。支持 full / stage-a / stage-b 三种模式。")
            body(pdf, "交付物清单：", indent=False)
            data_table(pdf,
                       ["类别", "文件", "说明"],
                       [26, 66, 82],
                       [
                           ["容器", "Dockerfile", "镜像固化 Py3.13 / torch 2.6.0 / CUDA 12.4"],
                           ["复现", "scripts/00_reproduce_all.sh", "两阶段一键复现（full / stage-a / stage-b）"],
                           ["链路", "从公开权重复现_完整链路.md", "公开权重 → 69.904990 全链说明"],
                           ["清单", "内置 kan_swin.py:390", "30 张疑似错标图 ID（硬编码，无需外部文件）"],
                           ["权重", "阶段A/B 关键权重", "base → PL → 蒸馏 → DACS 全链可加载"],
                           ["快验", "…_relaxbg_b8_i4000.pth", "约 5 分钟 / 100 张 比对输出包"],
                           ["文档", "本方案 PDF + README", "技术说明、消融记录与操作手册"],
                       ])
            callout(pdf, "合规声明",
                    "① 预训练权重仅使用学术公开的 ImageNet Swin-B（timm 托管），符合“只能采用学术领域"
                    "权重公开的预训练模型做微调”。② 训练与推理仅使用赛事方提供的数据集，无额外数据。"
                    "③ 最终输出为单模型，ms2x 属同一模型的测试时增强，不构成多模型集成。"
                    "④ 队友教师标签与锚定预测为同队共享的模型预测衍生，仅用于训练期，"
                    "不进入最终模型结构。",
                    warn=True)

    if pages is None:
        return starts
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "技术方案.pdf")
    pdf.output(out)
    print("已生成:", out, f"{os.path.getsize(out)/1024:.1f} KB, 共 {pdf.page_no()} 页")


if __name__ == "__main__":
    # 两遍生成：第一遍记录各章起始页码，第二遍回填目录页码
    first = build(None)
    print("各章起始页:", first)
    build(first)
