import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

# 设置绘图风格
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'Solarize_Light2')
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans', 'Bitstream Vera Sans', 'sans-serif']

def plot_combo_accuracy_certainty():
    # 读取数据
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, 'task1_final_results.csv')
    output_dir = os.path.join(base_dir, 'figures')
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    df = pd.read_csv(data_path)
    
    # 按赛季聚合计算平均准确率和平均置信度
    season_stats = df.groupby('Season').agg({
        'Correct': 'mean',    # True/False converts to 1/0, mean gives accuracy
        'Certainty': 'mean'
    }).reset_index()
    
    seasons = season_stats['Season']
    accuracy = season_stats['Correct']
    certainty = season_stats['Certainty']
    
    # 创建画布和双轴
    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax2 = ax1.twinx()
    
    # 绘制准确率 (Bar Chart - Background)
    # 使用深天蓝色 (Darker Sky Blue / Steel Blue)
    # width=0.9 makes bars denser
    bars = ax1.bar(seasons, accuracy, color='#4682B4', alpha=0.8, label='Model Accuracy', width=0.9)
    
    # 绘制置信度 (Line Chart - Overlay)
    line = ax2.plot(seasons, certainty, color='#FF5722', marker='o', linewidth=2.5, label='Fan Vote Certainty')
    
    # 设置标题和标签 (Bold)
    ax1.set_title('Model Accuracy and Fan Vote Certainty Across Seasons', fontsize=16, pad=20, fontweight='bold')
    ax1.set_xlabel('Season', fontsize=12, fontweight='bold')
    
    # 左轴 (黑色加粗)
    ax1.set_ylabel('Model Accuracy', fontsize=12, color='black', fontweight='bold')
    ax1.tick_params(axis='y', colors='black', labelsize=10)
    for label in ax1.get_yticklabels():
        label.set_fontweight('bold')
        
    # 右轴 (加粗)
    ax2.set_ylabel('Mean Certainty (Bootstrap)', fontsize=12, color='#FF5722', fontweight='bold')
    ax2.tick_params(axis='y', colors='#FF5722', labelsize=10)
    for label in ax2.get_yticklabels():
        label.set_fontweight('bold')
    
    # 设置X轴 (每5个赛季显示一次，紧凑)
    # 生成 5, 10, 15... 的刻度
    major_ticks = np.arange(5, max(seasons)+1, 5)
    # 如果第1季没显示，强制加上1
    if 1 not in major_ticks:
        major_ticks = np.insert(major_ticks, 0, 1)
        
    ax1.set_xticks(major_ticks)
    ax1.set_xticklabels(major_ticks, rotation=0, fontsize=10, fontweight='bold')
    ax1.set_xlim(min(seasons)-1, max(seasons)+1)
    
    # 设置Y轴范围 (Accuracy 0-1.1 to leave room for legend/labels, Certainty scaled appropriately)
    ax1.set_ylim(0, 1.1)
    ax2.set_ylim(0, 1.1)
    
    # 添加数值标签 (可选，为了清晰可以给Line加)
    # for x, y in zip(seasons, certainty):
    #     ax2.text(x, y + 0.02, f'{y:.2f}', ha='center', va='bottom', fontsize=8, color='#FF5722')

    # 合并图例
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    # 放在顶部或者合适的位置
    ax1.legend(lines + lines2, labels + labels2, loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=2, frameon=False)
    
    # 调整Y轴颜色以匹配数据
    ax1.tick_params(axis='y', colors='gray')
    ax2.tick_params(axis='y', colors='#FF5722')
    
    # 优化布局
    plt.tight_layout()
    
    # 保存图片 (PDF format)
    pdf_path = os.path.join(output_dir, 'accuracy_certainty_combo.pdf')
    png_path = os.path.join(output_dir, 'accuracy_certainty_combo.png')
    
    plt.savefig(pdf_path, bbox_inches='tight')
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    
    print(f"Combo chart saved to:\n- {pdf_path}\n- {png_path}")

if __name__ == "__main__":
    plot_combo_accuracy_certainty()
