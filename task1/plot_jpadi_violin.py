import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
import numpy as np

# 设置绘图风格
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'Solarize_Light2')
plt.rcParams['figure.dpi'] = 300

# 尝试设置中文字体
try:
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'PingFang SC', 'sans-serif']
    plt.rcParams['axes.unicode_minus'] = False
except:
    pass

def plot_jpadi_violin():
    # 读取数据
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, 'task1_final_results.csv')
    output_dir = os.path.join(base_dir, 'figures')
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    df = pd.read_csv(data_path)
    
    # 准备数据
    seasons = sorted(df['Season'].unique())
    data_to_plot = [df[df['Season'] == s]['JPADI'].values for s in seasons]
    
    # 创建画布
    fig, ax = plt.subplots(figsize=(15, 8))
    
    # 绘制小提琴图 (Matplotlib)
    parts = ax.violinplot(data_to_plot, positions=seasons, showmeans=False, showmedians=True, showextrema=False)
    
    # 自定义颜色 (Bright Red for violin)
    for pc in parts['bodies']:
        pc.set_facecolor('#FF0000') # Bright Red
        pc.set_edgecolor('black')
        pc.set_alpha(0.8)
    
    # 添加中位数线样式
    parts['cmedians'].set_edgecolor('yellow') # Contrast with red body
    parts['cmedians'].set_linewidth(1.5)

    # 绘制参考区间背景 (High distinction colors)
    # Judge Dominant (0-0.3): Light Blue
    ax.axhspan(0, 0.3, color='#2196F3', alpha=0.15, label='Judge Dominant (0~0.3)')
    # Balanced Zone (0.3-0.5): Light Green
    ax.axhspan(0.3, 0.5, color='#4CAF50', alpha=0.15, label='Balanced Zone (0.3~0.5)')
    # Fan Dominant (>0.5): Gold/Amber (Very distinct from Green and Red violin)
    ax.axhspan(0.5, 2.0, color='#FFC107', alpha=0.15, label='Fan Dominant / Discrepancy (>0.5)')
    
    # 设置标题和标签
    ax.set_title('Distribution of Judge-Public Agreement Drift Index (JPADI) Across Seasons', fontsize=16, pad=20)
    ax.set_xlabel('Season', fontsize=12)
    ax.set_ylabel('JPADI Value\n(1 - Spearman Correlation)', fontsize=12)
    
    # 设置X轴刻度
    ax.set_xticks(seasons)
    ax.set_xticklabels(seasons)
    
    # 设置Y轴范围
    ax.set_ylim(-0.2, 2.2)
    
    # 自定义图例 (Explanation of JPADI Ranges)
    legend_handles = [
        mpatches.Patch(color='#2196F3', alpha=0.2, label='0.0 - 0.3: Judge Dominant'),
        mpatches.Patch(color='#4CAF50', alpha=0.2, label='0.3 - 0.5: Balanced Zone'),
        mpatches.Patch(color='#FFC107', alpha=0.2, label='0.5 - 2.0: Fan Dominant')
    ]
    
    ax.legend(handles=legend_handles, loc='upper left', title='JPADI Interpretation', 
              frameon=True, fancybox=True, shadow=True, fontsize=10)
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图片 (PNG and PDF)
    png_path = os.path.join(output_dir, 'jpadi_variation_violin.png')
    pdf_path = os.path.join(output_dir, 'jpadi_variation_violin.pdf')
    
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.savefig(pdf_path, bbox_inches='tight') # PDF vector format
    
    print(f"Chart saved to:\n- {png_path}\n- {pdf_path}")

if __name__ == "__main__":
    plot_jpadi_violin()
