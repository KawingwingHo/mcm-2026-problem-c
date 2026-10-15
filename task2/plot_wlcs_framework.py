import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

# 设置绘图风格
plt.style.use('default')
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.family'] = 'sans-serif'
# 尝试设置中文字体
try:
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'PingFang SC', 'sans-serif']
    plt.rcParams['axes.unicode_minus'] = False
except:
    pass

def draw_box(ax, x, y, width, height, text, color='#E0F7FA', edge_color='#006064', fontsize=12, bold=True):
    """均一化绘制方框 - Aggressively Bold"""
    rect = patches.FancyBboxPatch((x, y), width, height, boxstyle="round,pad=0.03", 
                                 linewidth=2.5, edgecolor=edge_color, facecolor=color)
    ax.add_patch(rect)
    fw = 'heavy' if bold else 'normal'
    ax.text(x + width/2, y + height/2, text, ha='center', va='center', fontsize=fontsize, fontweight=fw, wrap=True)
    return rect

def draw_arrow(ax, x1, y1, x2, y2, color='#546E7A'):
    """绘制箭头 - Precise endpoints"""
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->", color=color, lw=2.5, shrinkA=0, shrinkB=5))

def plot_wlcs_framework():
    # Compact Figure Size
    fig, ax = plt.subplots(figsize=(10, 10)) # Reduced width
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10.5)
    ax.axis('off')
    
    # Coordinates Configuration
    y_l1 = 8.5      # Input Layer Box Y
    y_l2_grp = 5.0  # Core Group Y (bottom)
    y_l2_box1 = 6.8 # Standardization
    y_l2_box2 = 5.5 # Noise
    y_l3_rule = 3.2 # Rule Boxes Y
    y_l3_conv = 1.8 # Convergence Box Y
    y_l4 = 0.2      # Output Layer Y
    
    # Layer Separators & Titles
    # Input
    ax.text(5, 10.2, "WLCS Simulation Framework Architecture", ha='center', fontsize=18, fontweight='heavy')
    ax.text(0.5, 9.8, "Input Layer", fontsize=12, fontweight='heavy', color='#1565C0', ha='left')
    ax.plot([0.5, 9.5], [9.7, 9.7], color='#1565C0', lw=2.0, linestyle='--')
    
    # Core
    ax.text(0.5, 8.2, "Core Layer: Parallel Universe Generator", fontsize=12, fontweight='heavy', color='#2E7D32', ha='left')
    ax.plot([0.5, 9.5], [8.1, 8.1], color='#2E7D32', lw=2.0, linestyle='--')
    
    # Rule
    ax.text(0.5, 4.8, "Rule Layer: Dual-Track Evaluation", fontsize=12, fontweight='heavy', color='#EF6C00', ha='left')
    ax.plot([0.5, 9.5], [4.7, 4.7], color='#EF6C00', lw=2.0, linestyle='--')
    
    # Output
    ax.text(0.5, 1.6, "Output Layer: Metrics & Insights", fontsize=12, fontweight='heavy', color='#6A1B9A', ha='left')
    ax.plot([0.5, 9.5], [1.5, 1.5], color='#6A1B9A', lw=2.0, linestyle='--')

    # === DRAWING COMPONENTS (Smaller Boxes) ===
    
    # Layer 1 Boxes (Width 2.6, Height 0.9)
    # Centers: 2.5, 7.5 -> x = 1.2, 6.2
    draw_box(ax, 1.2, y_l1, 2.6, 0.9, "Judge Score Matrix\n$S_{judge}$", color='#E3F2FD', fontsize=12)
    draw_box(ax, 6.2, y_l1, 2.6, 0.9, "Fan Strength\n$\\hat{S}_{fan}$", color='#E3F2FD', fontsize=12)
    
    # Layer 2 Group 
    rect = patches.Rectangle((1.5, y_l2_grp), 7, 2.9, linewidth=2.0, edgecolor='#2E7D32', facecolor='#E8F5E9', linestyle=':')
    ax.add_patch(rect)
    ax.text(5, 7.7, "Monte Carlo Simulation Engine\n(N Iterations)", ha='center', fontsize=11, fontweight='heavy', color='#1B5E20')
    
    # Layer 2 Boxes (Width 2.6, Height 0.7)
    # Center 5 -> x = 3.7
    draw_box(ax, 3.7, y_l2_box1, 2.6, 0.7, "Standardization", color='#C8E6C9', fontsize=12)
    draw_box(ax, 3.7, y_l2_box2, 2.6, 0.7, "Noise Injection\n$\\epsilon \\sim N(0, \\sigma^2)$", color='#C8E6C9', fontsize=12)
    
    # Layer 3 Boxes (Width 3.0, Height 1.0)
    # Centers 2.0, 8.0 -> x = 0.5, 6.5
    draw_box(ax, 0.5, y_l3_rule, 3.0, 1.0, "Percentage Rule\n$wS_j + (1-w)S_f$", color='#FFE0B2', edge_color='#E65100', fontsize=11)
    draw_box(ax, 6.5, y_l3_rule, 3.0, 1.0, "Rank Rule\nRank -> Point", color='#FFE0B2', edge_color='#E65100', fontsize=11)
    
    # Convergence Box (Width 2.6, Height 0.7)
    # Center 5 -> x = 3.7
    draw_box(ax, 3.7, y_l3_conv - 0.5, 2.6, 0.7, "Elimination Logic", color='#FFCCBC', fontsize=12) 
    
    # Layer 4 Boxes (Width 2.2, Height 0.8)
    # Centers 2.0, 5.0, 8.0 -> x = 0.9, 3.9, 6.9
    draw_box(ax, 0.9, y_l4, 2.2, 0.8, "Simulated\nSequences", color='#E1BEE7', fontsize=11)
    draw_box(ax, 3.9, y_l4, 2.2, 0.8, "Controversy Rate", color='#E1BEE7', fontsize=11)
    draw_box(ax, 6.9, y_l4, 2.2, 0.8, "Champion\nGold Content", color='#E1BEE7', fontsize=11)

    # === ARROWS (Calculated for new sizes) ===
    
    # L1 -> L2
    # L1 Bottoms: (2.5, 8.5), (7.5, 8.5)
    # L2 Top: (5, 7.5)
    draw_arrow(ax, 2.5, 8.5, 5.0, 7.5)
    draw_arrow(ax, 7.5, 8.5, 5.0, 7.5)
    
    # Std -> Noise
    # Std Bot: (5, 6.8). Noise Top: (5, 6.2)
    draw_arrow(ax, 5.0, 6.8, 5.0, 6.2)
    
    # Noise -> Rules
    # Noise Bot: (5, 5.5)
    # Rules Tops: Pct(2.0, 4.2), Rank(8.0, 4.2)
    draw_arrow(ax, 5.0, 5.5, 2.0, 4.2)
    draw_arrow(ax, 5.0, 5.5, 8.0, 4.2)
    
    # Rules -> Convergence
    # Rules Bots: Pct(2.0, 3.2), Rank(8.0, 3.2)
    # Conv Top: (5.0, 2.0)
    draw_arrow(ax, 2.0, 3.2, 5.0, 2.0)
    draw_arrow(ax, 8.0, 3.2, 5.0, 2.0)
    
    # Convergence -> Output
    # Conv Bot: (5.0, 1.3)
    # Out Tops: (2.0, 1.0), (5.0, 1.0), (8.0, 1.0)
    draw_arrow(ax, 5.0, 1.3, 2.0, 1.0)
    draw_arrow(ax, 5.0, 1.3, 5.0, 1.0)
    draw_arrow(ax, 5.0, 1.3, 8.0, 1.0)
    
    # Feedback Loop (MLE)
    # Pct Rule Box Right Edge is 3.5. Rank Left is 6.5.
    ax.annotate("Calibration", 
                xy=(8.8, 9.0), xytext=(9.7, 3.7),
                arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.4", color='#D81B60', lw=2.5, ls='--'),
                ha='center', fontsize=11, color='#D81B60', fontweight='heavy')
    
    plt.tight_layout()
    
    # Save
    base_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(base_dir, 'figures')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    pdf_path = os.path.join(output_dir, 'wlcs_framework.pdf')
    png_path = os.path.join(output_dir, 'wlcs_framework.png')
    
    plt.savefig(pdf_path, bbox_inches='tight')
    plt.savefig(png_path, dpi=300, bbox_inches='tight') 
    
    print(f"Framework diagram saved to:\n- {pdf_path}\n- {png_path}")

if __name__ == "__main__":
    plot_wlcs_framework()
