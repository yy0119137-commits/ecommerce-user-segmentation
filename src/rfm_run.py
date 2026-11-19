# -*- coding: utf-8 -*-
"""
电商用户价值分层与精准营销策略（RFM-I 模型）

在传统 RFM 基础上引入站内行为数据，构造两个新指标：
  I        意向深度 = 停留时长与浏览页数等权归一化（0–100）
  Friction 决策摩擦 = 浏览页数 /（购买频次 + 1）

再以意向分作乘数修正 RFM 综合分，输出 15 类人群，并用边际收益法测算投放 ROI。

运行：
    pip install -r requirements.txt
    python src/rfm_run.py

输出：outputs/ 下的 5 张图与人群明细 CSV
"""
import os
import sys
import warnings

import numpy as np
import pandas as pd
import matplotlib

SHOW_PLOTS = False

if not SHOW_PLOTS:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
input_path = os.path.join(ROOT_DIR, 'data', 'user_personalized_features.csv')
output_dir = os.path.join(ROOT_DIR, 'outputs')
os.makedirs(output_dir, exist_ok=True)

print(f"[环境] 解释器: {sys.executable}")
print(f"[环境] pandas {pd.__version__} / numpy {np.__version__} / matplotlib {matplotlib.__version__}")
print(f"[路径] 数据: {input_path}")
print(f"[路径] 输出: {output_dir}\n")


# ---------- 1. 数据加载与预处理 ----------
def load_and_preprocess(filepath):
    df = pd.read_csv(filepath)
    df = df.drop(columns=[c for c in df.columns if str(c).startswith('Unnamed') or c == ''], errors='ignore')

    print("=" * 60)
    print("第一阶段：数据质量检查")
    print("=" * 60)
    print(f"数据规模: {df.shape[0]} 行 x {df.shape[1]} 列")
    print(f"缺失值: {df.isnull().sum().sum()}")
    age_outliers = df[(df['Age'] < 10) | (df['Age'] > 100)].shape[0]
    print(f"年龄异常值: {age_outliers}")
    return df


df = load_and_preprocess(input_path)
print(df.head())


# ---------- 2. 特征工程 ----------
def min_max_normalize(series, reverse=False):
    min_val, max_val = series.min(), series.max()
    if max_val == min_val:
        return pd.Series([50] * len(series), index=series.index)
    if reverse:
        return ((max_val - series) / (max_val - min_val)) * 100
    return ((series - min_val) / (max_val - min_val)) * 100


def feature_engineering(df):
    print("\n" + "=" * 60)
    print("第二阶段：特征工程 - 构建优化指标")
    print("=" * 60)

    # I：意向深度
    df['Time_Spent_Norm'] = min_max_normalize(df['Time_Spent_on_Site_Minutes'])
    df['Pages_Viewed_Norm'] = min_max_normalize(df['Pages_Viewed'])
    df['I_Score'] = 0.5 * df['Time_Spent_Norm'] + 0.5 * df['Pages_Viewed_Norm']
    # 决策摩擦
    df['Friction'] = df['Pages_Viewed'] / (df['Purchase_Frequency'] + 1)

    # L：活跃连接度
    def calculate_loyalty(row):
        if row['Newsletter_Subscription'] and row['Last_Login_Days_Ago'] < 7:
            return 3
        elif (not row['Newsletter_Subscription']) and row['Last_Login_Days_Ago'] < 7:
            return 2
        return 1

    df['L_Score'] = df.apply(calculate_loyalty, axis=1)

    # 购买力分层
    income_33, income_66 = df['Income'].quantile(0.33), df['Income'].quantile(0.66)

    def income_level(income):
        if income <= income_33:
            return 'Low'
        elif income <= income_66:
            return 'Medium'
        return 'High'

    df['Income_Level'] = df['Income'].apply(income_level)

    # 人货匹配度
    df['Interest_Match'] = (df['Interests'] == df['Product_Category_Preference']).astype(int)

    print(f"I_Score 范围: [{df['I_Score'].min():.2f}, {df['I_Score'].max():.2f}]")
    print(f"Friction 范围: [{df['Friction'].min():.2f}, {df['Friction'].max():.2f}]")
    print(f"L_Score 分布: {df['L_Score'].value_counts().to_dict()}")
    print(f"Income_Level 分布: {df['Income_Level'].value_counts().to_dict()}")
    print(f"Interest_Match 匹配率: {df['Interest_Match'].mean() * 100:.1f}%")
    return df


df = feature_engineering(df)


# ---------- 3. EDA ----------
def perform_eda(df, output_dir):
    print("\n" + "=" * 60)
    print("第三阶段：EDA探索性数据分析")
    print("=" * 60)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes[0, 0].hist(df['Total_Spending'], bins=50, edgecolor='black', alpha=0.7, color='#0B5CAD')
    axes[0, 0].set_title('Total_Spending Distribution (Long Tail Pattern)', fontsize=12)
    axes[0, 0].set_xlabel('Total Spending')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].axvline(df['Total_Spending'].median(), color='red', linestyle='--',
                       label=f'Median: {df["Total_Spending"].median():.0f}')
    axes[0, 0].legend()

    colors = {'Low': '#FF6B6B', 'Medium': '#4ECDC4', 'High': '#0B5CAD'}
    for level in ['Low', 'Medium', 'High']:
        subset = df[df['Income_Level'] == level]
        axes[0, 1].scatter(subset['Income'], subset['Total_Spending'],
                           alpha=0.5, label=level, c=colors[level], s=20)
    axes[0, 1].set_title('Income vs Total_Spending by Income Level', fontsize=12)
    axes[0, 1].set_xlabel('Income')
    axes[0, 1].set_ylabel('Total Spending')
    axes[0, 1].legend()

    df.boxplot(column='Last_Login_Days_Ago', by='Newsletter_Subscription', ax=axes[1, 0])
    axes[1, 0].set_title('Login Days by Subscription Status', fontsize=12)
    axes[1, 0].set_xlabel('Newsletter Subscription')
    axes[1, 0].set_ylabel('Last Login Days Ago')

    axes[1, 1].hist(df['I_Score'], bins=30, edgecolor='black', alpha=0.7, color='#4ECDC4')
    axes[1, 1].set_title('Intent Score (I_Score) Distribution', fontsize=12)
    axes[1, 1].set_xlabel('I_Score')
    axes[1, 1].set_ylabel('Frequency')

    plt.tight_layout()
    plt.savefig(f'{output_dir}/eda_distributions.png', dpi=150, bbox_inches='tight')
    if SHOW_PLOTS:
        plt.show()
    plt.close()

    numeric_cols = ['Age', 'Income', 'Last_Login_Days_Ago', 'Purchase_Frequency',
                    'Average_Order_Value', 'Total_Spending', 'Time_Spent_on_Site_Minutes',
                    'Pages_Viewed', 'I_Score', 'Friction']
    corr_matrix = df[numeric_cols].corr()

    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(corr_matrix, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
    ax.set_xticks(range(len(numeric_cols)))
    ax.set_yticks(range(len(numeric_cols)))
    ax.set_xticklabels(numeric_cols, rotation=45, ha='right', fontsize=9)
    ax.set_yticklabels(numeric_cols, fontsize=9)
    for i in range(len(numeric_cols)):
        for j in range(len(numeric_cols)):
            ax.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}', ha='center', va='center', fontsize=7,
                    color='white' if abs(corr_matrix.iloc[i, j]) > 0.5 else 'black')
    ax.set_title('Feature Correlation Matrix', fontsize=14)
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/correlation_matrix.png', dpi=150, bbox_inches='tight')
    if SHOW_PLOTS:
        plt.show()
    plt.close()

    time_purchase_corr = df['Time_Spent_on_Site_Minutes'].corr(df['Purchase_Frequency'])
    pages_purchase_corr = df['Pages_Viewed'].corr(df['Purchase_Frequency'])
    print("\n关键相关性分析:")
    print(f"Time_Spent vs Purchase_Frequency 相关系数: {time_purchase_corr:.3f}")
    print(f"Pages_Viewed vs Purchase_Frequency 相关系数: {pages_purchase_corr:.3f}")

    high_friction = df[df['Friction'] > df['Friction'].quantile(0.75)]
    print("\n高摩擦用户(前25%)特征:")
    print(f"  - 平均浏览页数: {high_friction['Pages_Viewed'].mean():.1f}")
    print(f"  - 平均购买频率: {high_friction['Purchase_Frequency'].mean():.1f}")
    print(f"  - 平均消费金额: {high_friction['Total_Spending'].mean():.0f}")

    hi_low = df[(df['Income_Level'] == 'High') & (df['Total_Spending'] < df['Total_Spending'].median())]
    print("\n高收入低消费群体:")
    print(f"  - 用户数: {len(hi_low)}")
    print(f"  - 占高收入群体比例: {len(hi_low) / len(df[df['Income_Level'] == 'High']) * 100:.1f}%")
    return high_friction, hi_low


perform_eda(df, output_dir)


# ---------- 4. RFM-I 模型 ----------
df['R_Score'] = min_max_normalize(df['Last_Login_Days_Ago'], reverse=True)
df['F_Score'] = min_max_normalize(df['Purchase_Frequency'])
df['M_Score'] = min_max_normalize(df['Total_Spending'])
df['RFM_Score'] = 0.2 * df['R_Score'] + 0.3 * df['F_Score'] + 0.5 * df['M_Score']
df['I_Weight'] = df['I_Score'] / 500
df['Final_Score'] = df['RFM_Score'] * (1 + df['I_Weight'])

print("\n" + "=" * 60)
print("第四阶段：RFM-I模型构建")
print("=" * 60)
for c in ['R_Score', 'F_Score', 'M_Score', 'RFM_Score', 'Final_Score']:
    print(f"{c} 范围: [{df[c].min():.2f}, {df[c].max():.2f}]")


# ---------- 5. 用户分层 ----------
def classify_user(row):
    r, f, m, i = row['R_Score'], row['F_Score'], row['M_Score'], row['I_Score']
    income, friction = row['Income_Level'], row['Friction']

    if r > 60 and f > 60 and m > 60:
        base = "重要价值用户"
    elif r > 60 and m > 60 and f < 40:
        base = "重要发展用户"
    elif r < 40 and f > 60 and m > 60:
        base = "重要保持用户"
    elif r < 40 and m > 60:
        base = "重要挽留用户"
    elif r > 60 and f > 40 and m < 40:
        base = "一般发展用户"
    elif r > 60 and f < 40 and m < 40:
        base = "一般维持用户"
    elif r < 40 and f < 40 and m < 40:
        base = "低价值用户"
    else:
        base = "一般用户"

    if base in ["低价值用户", "一般维持用户"] and income == "High" and i > 60:
        return "高潜沉睡用户"
    if income == "High" and i > 70 and f < 40 and m < 50:
        return "纠结土豪"
    if base in ["一般维持用户", "一般发展用户"] and i > 80 and friction > df['Friction'].quantile(0.6):
        return "犹豫型潜力用户"
    if income == "Low" and i > 70 and f < 40:
        return "隐形活跃者"
    if income == "High" and r < 40 and m > 50:
        return "高潜流失客"
    if m > 70 and f > 70 and i > 60:
        return "核心VIP"
    if income == "Low" and i < 40 and m < 40:
        return "羊毛党/低值"
    return base


df['User_Segment'] = df.apply(classify_user, axis=1)

segment_stats = df.groupby('User_Segment').agg({
    'User_ID': 'count', 'Total_Spending': 'mean', 'Purchase_Frequency': 'mean',
    'I_Score': 'mean', 'Income': 'mean', 'Final_Score': 'mean'
}).round(2)
segment_stats.columns = ['用户数', '平均消费', '平均购买频率', '平均意向分', '平均收入', '平均综合分']
segment_stats['占比(%)'] = (segment_stats['用户数'] / len(df) * 100).round(1)
segment_stats = segment_stats.sort_values('用户数', ascending=False)

print("\n" + "=" * 60)
print("第五阶段：用户分层与画像")
print("=" * 60)
print(segment_stats.to_string())


# ---------- 6. 画像可视化 ----------
def plot_portrait(df, output_dir):
    segments = df['User_Segment'].unique()
    metrics = ['R_Score', 'F_Score', 'M_Score', 'I_Score']
    segment_means = df.groupby('User_Segment')[metrics].mean()

    fig, axes = plt.subplots(2, 4, figsize=(16, 8), subplot_kw=dict(polar=True))
    axes = axes.flatten()
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]
    colors = plt.cm.Set3(np.linspace(0, 1, len(segments)))

    for idx, segment in enumerate(segments[:8]):
        ax = axes[idx]
        values = segment_means.loc[segment].tolist()
        values += values[:1]
        ax.fill(angles, values, alpha=0.3, color=colors[idx])
        ax.plot(angles, values, 'o-', color=colors[idx], linewidth=2)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(['R', 'F', 'M', 'I'], fontsize=10)
        ax.set_title(segment, fontsize=10, pad=10)
    for idx in range(len(segments), len(axes)):
        axes[idx].set_visible(False)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/segment_radar.png', dpi=150, bbox_inches='tight')
    if SHOW_PLOTS:
        plt.show()
    plt.close()

    counts = df['User_Segment'].value_counts()
    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.barh(counts.index, counts.values, color='#0B5CAD', alpha=0.8)
    ax.set_xlabel('User Count')
    ax.set_title('User Segment Distribution', fontsize=14)
    for bar, count in zip(bars, counts.values):
        ax.text(bar.get_width() + 2, bar.get_y() + bar.get_height() / 2,
                f'{count} ({count / len(df) * 100:.1f}%)', va='center', fontsize=9)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/segment_distribution.png', dpi=150, bbox_inches='tight')
    if SHOW_PLOTS:
        plt.show()
    plt.close()


plot_portrait(df, output_dir)


# ---------- 7. ROI 测算 ----------
def calculate_roi(df, output_dir):
    print("\n" + "=" * 60)
    print("第六阶段：ROI测算")
    print("=" * 60)

    total_budget, coupon_cost = 10000, 10
    aov = df['Average_Order_Value'].mean()

    core_natural, core_with_coupon = 0.25, 0.30
    potential_natural, potential_with_coupon = 0.01, 0.20
    rfm20_natural, rfm20_with_coupon = 0.20, 0.30

    core_lift = core_with_coupon - core_natural
    potential_lift = potential_with_coupon - potential_natural
    rfm20_lift = rfm20_with_coupon - rfm20_natural

    # 方案A：传统 RFM 前 20%
    rfm_top20 = df[df['RFM_Score'] >= df['RFM_Score'].quantile(0.80)]
    count_a = min(len(rfm_top20), total_budget // coupon_cost)
    cost_a = count_a * coupon_cost
    revenue_a = count_a * rfm20_lift * aov
    roi_a = (revenue_a - cost_a) / cost_a * 100 if cost_a else 0

    # 方案B：核心 80% 预算 + 潜力 20%
    core_users = df[df['User_Segment'].isin(['核心VIP', '重要价值用户'])]
    potential_users = df[df['User_Segment'].isin(
        ['纠结土豪', '高潜沉睡用户', '犹豫型潜力用户', '高潜流失客'])]
    core_count = min(len(core_users), int(total_budget * 0.8 / coupon_cost))
    remaining = total_budget - core_count * coupon_cost
    potential_count = min(len(potential_users), remaining // coupon_cost)
    count_b = core_count + potential_count
    cost_b = count_b * coupon_cost
    revenue_b = core_count * core_lift * aov + potential_count * potential_lift * aov
    roi_b = (revenue_b - cost_b) / cost_b * 100 if cost_b else 0

    print(f"\n【假设参数】总预算 {total_budget}元 / 券成本 {coupon_cost}元 / 客单价 {aov:.0f}元")
    print(f"【方案A - 传统RFM】目标 {count_a} 人, 成本 {cost_a} 元, "
          f"增量收益 {revenue_a:.0f} 元, 边际ROI {roi_a:.1f}%")
    print(f"【方案B - 优化RFI】核心 {core_count} 人 + 潜力 {potential_count} 人 = {count_b} 人, "
          f"成本 {cost_b} 元 (预算使用率 {cost_b / total_budget * 100:.1f}%), "
          f"增量收益 {revenue_b:.0f} 元, 边际ROI {roi_b:.1f}%")
    print(f"\n>>> 成本变化: {(cost_b - cost_a) / cost_a * 100:+.1f}%  "
          f"ROI 倍数: {roi_b / roi_a:.1f}x")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    strategies = ['Traditional RFM', 'Optimized RFI']
    rois = [roi_a, roi_b]
    bars = axes[0].bar(strategies, rois, color=['#FF6B6B', '#0B5CAD'], alpha=0.8, edgecolor='black')
    axes[0].set_ylabel('边际ROI (%)')
    axes[0].set_title('边际ROI对比：传统RFM vs 优化RFI', fontsize=12)
    for bar, roi in zip(bars, rois):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                     f'{roi:.1f}%', ha='center', fontsize=12)

    x = np.arange(2)
    width = 0.35
    core_marginal = [revenue_a, core_count * core_lift * aov]
    potential_marginal = [0, potential_count * potential_lift * aov]
    axes[1].bar(x, core_marginal, width, label='核心用户边际收益', color='#0B5CAD', alpha=0.8)
    axes[1].bar(x, potential_marginal, width, bottom=core_marginal,
                label='潜力用户边际收益', color='#4ECDC4', alpha=0.8)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(strategies)
    axes[1].set_ylabel('边际收益 (元)')
    axes[1].set_title('边际收益构成对比', fontsize=12)
    axes[1].legend()
    plt.tight_layout()
    plt.savefig(f'{output_dir}/roi_comparison.png', dpi=150, bbox_inches='tight')
    if SHOW_PLOTS:
        plt.show()
    plt.close()

    return roi_a, roi_b


calculate_roi(df, output_dir)

df.to_csv(f'{output_dir}/user_segments_result.csv', index=False, encoding='utf-8-sig')
print(f"\n[OK] 全部完成，图表与明细已保存至: {output_dir}")
