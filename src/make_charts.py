# -*- coding: utf-8 -*-
"""生成 README 中引用的两张主图：人群坐标图 + ROI 对比"""
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
SRC = os.path.join(ROOT_DIR, 'outputs', 'user_segments_result.csv')
OUT = os.path.join(ROOT_DIR, 'docs')
os.makedirs(OUT, exist_ok=True)

BLUE = '#2F5597'
CORAL = '#D85A30'
GRAY = '#B7B7B7'
INK = '#2C2C2A'

df = pd.read_csv(SRC)

# ---------- 图1：1000 用户的人群坐标 ----------
CORE = ['核心VIP', '重要价值用户', '重要保持用户']
BLIND = ['纠结土豪', '犹豫型潜力用户', '隐形活跃者', '高潜沉睡用户']
CHURN = ['高潜流失客', '重要挽留用户']


def grp(s):
    if s in CORE:
        return '核心价值'
    if s in BLIND:
        return 'RFM 盲区'
    if s in CHURN:
        return '待挽留'
    return '其他人群'


df['Group'] = df['User_Segment'].apply(grp)

fig, ax = plt.subplots(figsize=(7.6, 5.4))
for g, c, z, a in [('其他人群', GRAY, 1, 0.40), ('待挽留', '#C9A227', 2, 0.60),
                   ('RFM 盲区', CORAL, 3, 0.80), ('核心价值', BLUE, 4, 0.85)]:
    s = df[df['Group'] == g]
    ax.scatter(s['M_Score'], s['I_Score'], s=26, c=c, alpha=a, edgecolors='none',
               label=f'{g}  {len(s)} 人', zorder=z)

ax.axvline(60, color='#9A9A9A', lw=0.8, ls=(0, (4, 4)), zorder=0)
ax.axhline(60, color='#9A9A9A', lw=0.8, ls=(0, (4, 4)), zorder=0)

ax.set_xlabel('历史价值 M（成交金额归一化）', fontsize=10, color=INK, labelpad=6)
ax.set_ylabel('行为意向 I（浏览 / 停留归一化）', fontsize=10, color=INK, labelpad=6)
ax.set_xlim(-2, 105)
ax.set_ylim(-2, 105)
ax.tick_params(colors='#666666', labelsize=9)
for sp in ['top', 'right']:
    ax.spines[sp].set_visible(False)
for sp in ['left', 'bottom']:
    ax.spines[sp].set_color('#CCCCCC')

box = dict(boxstyle='round,pad=0.28', fc='white', ec='none', alpha=0.82)
ax.text(3, 101, '高意向 · 低成交\n传统 RFM 判为低价值', fontsize=9, color=CORAL, va='top', ha='left',
        fontweight='bold', bbox=box, zorder=10)
ax.text(102, 101, '高价值 · 高意向', fontsize=9, color=BLUE, va='top', ha='right', bbox=box, zorder=10)
ax.text(102, 3, '高价值 · 低意向', fontsize=9, color='#8A6D10', va='bottom', ha='right', bbox=box, zorder=10)
ax.text(3, 3, '低价值 · 低意向', fontsize=9, color='#7A7A7A', va='bottom', ha='left', bbox=box, zorder=10)

ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.28), ncol=4, frameon=False, fontsize=9,
          handletextpad=0.4, columnspacing=1.1, markerscale=1.3)

plt.tight_layout()
plt.savefig(f'{OUT}/fig1_segment_map.png', dpi=200, bbox_inches='tight', facecolor='white')
plt.close()

# ---------- 图2：ROI 双面板 ----------
fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.7), layout='constrained',
                         gridspec_kw={'width_ratios': [1, 1.3]})

# 左：两方案边际 ROI
names = ['传统 RFM\n前 20% 投放', '分层策略\n核心 + 潜力']
vals = [4.0, 33.5]
bars = axes[0].bar(names, vals, color=[GRAY, BLUE], width=0.52, zorder=3)
axes[0].set_ylabel('边际 ROI', fontsize=10, color=INK)
axes[0].yaxis.set_major_formatter(lambda v, p: f'{v:.0f}%')
axes[0].set_ylim(0, 42)
axes[0].grid(axis='y', color='#EEEEEE', lw=0.8, zorder=0)
axes[0].set_axisbelow(True)
for b, v in zip(bars, vals):
    axes[0].text(b.get_x() + b.get_width() / 2, v + 1.2, f'{v}%', ha='center', fontsize=12,
                 color='#333333', fontweight='bold')
for sp in ['top', 'right']:
    axes[0].spines[sp].set_visible(False)
for sp in ['left', 'bottom']:
    axes[0].spines[sp].set_color('#CCCCCC')
axes[0].tick_params(colors='#666666', labelsize=9.5)

# 右：单券边际利润
labels = ['核心用户\nlift 5pt', 'RFM 前 20%\nlift 10pt', '潜力用户\nlift 19pt']
profit = [-4.80, 0.41, 9.77]
colors = [CORAL, '#E8C15A', BLUE]
bb = axes[1].barh(labels, profit, color=colors, height=0.5, zorder=3)
axes[1].axvline(0, color='#555555', lw=1, zorder=4)
axes[1].set_xlabel('每张券的边际利润（元）', fontsize=10, color=INK)
axes[1].set_xlim(-8, 13)
axes[1].grid(axis='x', color='#EEEEEE', lw=0.8, zorder=0)
axes[1].set_axisbelow(True)
for b, v in zip(bb, profit):
    axes[1].text(v + (0.5 if v >= 0 else -0.5), b.get_y() + b.get_height() / 2,
                 f'{v:+.2f}', va='center', ha='left' if v >= 0 else 'right',
                 fontsize=10.5, color='#333333', fontweight='bold')
for sp in ['top', 'right']:
    axes[1].spines[sp].set_visible(False)
for sp in ['left', 'bottom']:
    axes[1].spines[sp].set_color('#CCCCCC')
axes[1].tick_params(colors='#666666', labelsize=9)

plt.savefig(f'{OUT}/fig2_roi.png', dpi=200, facecolor='white')
plt.close()

print('已生成:', os.listdir(OUT))
